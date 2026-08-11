"""基于 OpenILT ICCAD13 Hopkins 核实现可批处理三工艺角光刻仿真。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Windows 直接执行环境内 python.exe 时不会像 `conda run` 那样把 `<env>/bin`
# 加入 DLL 搜索目录，而 PyTorch 的 NVRTC 运行时位于该目录。句柄必须在进程
# 生命周期内保留，否则目录会立即从搜索路径移除；非 Windows 平台不执行此分支。
_DLL_DIRECTORY = None
if os.name == "nt":
    _cuda_dll_dir = Path(sys.prefix) / "bin"
    if _cuda_dll_dir.is_dir():
        _DLL_DIRECTORY = os.add_dll_directory(str(_cuda_dll_dir))
        os.environ["PATH"] = f"{_cuda_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class ICCAD13Config:
    """保存 ICCAD13 光刻模型的数值配置和固定画布尺寸。"""

    kernel_count: int
    target_density: float
    print_threshold: float
    print_steepness: float
    dose_max: float
    dose_min: float
    dose_nominal: float
    canvas: int
    resolution: int

    @classmethod
    def from_file(cls, path: str | Path) -> ICCAD13Config:
        """读取简单的“名称 值”配置文件并校验全部必需字段。"""
        source = Path(path).expanduser().resolve()
        values: dict[str, str] = {}
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"光刻配置行必须包含名称和值：{line}")
            values[parts[0]] = parts[1]
        required = (
            "KernelNum", "TargetDensity", "PrintThresh", "PrintSteepness",
            "DoseMax", "DoseMin", "DoseNom", "Canvas", "Resolution",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"光刻配置缺少字段：{', '.join(missing)}")
        config = cls(
            int(values["KernelNum"]), float(values["TargetDensity"]),
            float(values["PrintThresh"]), float(values["PrintSteepness"]),
            float(values["DoseMax"]), float(values["DoseMin"]),
            float(values["DoseNom"]), int(values["Canvas"]),
            int(values["Resolution"]),
        )
        if (config.kernel_count <= 0 or config.canvas <= 0 or config.resolution <= 0 or
                not 0.0 < config.print_threshold < 1.0 or
                not config.dose_min <= config.dose_nominal <= config.dose_max):
            raise ValueError("光刻配置的核数量、尺寸、阈值或剂量顺序无效")
        return config


@dataclass(frozen=True, slots=True)
class LithographyResult:
    """保存标称、最大和最小三个工艺角的连续光刻胶图像。"""

    nominal: torch.Tensor
    maximum: torch.Tensor
    minimum: torch.Tensor


class ICCAD13Lithography(nn.Module):
    """使用预计算 Hopkins 核把二值 mask 转换为三工艺角 printed image。"""

    def __init__(self, config_path: str | Path | None = None,
                 asset_dir: str | Path | None = None,
                 device: str | torch.device | None = None) -> None:
        """加载配置和四个实际使用的 kernel/scale 资产到指定设备。"""
        super().__init__()
        module_dir = Path(__file__).resolve().parent
        config_file = module_dir / "config" / "iccad13.txt" if config_path is None else Path(config_path)
        assets = module_dir / "assets" / "iccad13" if asset_dir is None else Path(asset_dir)
        self.config = ICCAD13Config.from_file(config_file)
        if device is None or str(device) == "auto":
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
        # 四个张量注册为 buffer 后会跟随 Module.to()，又不会被优化器当作可训练参数。
        # 文件路径完全相对本模块解析，保证从任意工作目录直接运行根脚本都能加载。
        focus = self._load_kernels(assets / "focus.pt", resolved_device)
        defocus = self._load_kernels(assets / "defocus.pt", resolved_device)
        focus_scales = self._load_scales(assets / "focus_scale.pt", resolved_device)
        defocus_scales = self._load_scales(assets / "defocus_scale.pt", resolved_device)
        count = self.config.kernel_count
        if (focus.shape[0] < count or defocus.shape[0] < count or
                len(focus_scales) < count or len(defocus_scales) < count):
            raise ValueError("光刻资产包含的 Hopkins 核数量少于配置要求")
        self.register_buffer("focus_kernels", focus, persistent=True)
        self.register_buffer("defocus_kernels", defocus, persistent=True)
        self.register_buffer("focus_scales", focus_scales, persistent=True)
        self.register_buffer("defocus_scales", defocus_scales, persistent=True)
        self.to(resolved_device)

    @property
    def device(self) -> torch.device:
        """返回当前模型 buffer 所在设备。"""
        return self.focus_kernels.device

    @staticmethod
    def _load_kernels(path: Path, device: torch.device) -> torch.Tensor:
        """读取 H×W×K 资产并转换为连续 K×H×W 复数计算布局。"""
        if not path.is_file():
            raise FileNotFoundError(f"找不到光刻 kernel：{path}")
        tensor = torch.load(path, map_location=device, weights_only=True)
        if tensor.ndim != 3:
            raise ValueError(f"光刻 kernel 必须是三维张量：{path}")
        # OpenILT 资产把 kernel 维放在最后；核的空间尺寸远大于 24 个核时可稳定识别。
        if tensor.shape[-1] < tensor.shape[0] and tensor.shape[-1] < tensor.shape[1]:
            tensor = tensor.permute(2, 0, 1)
        return tensor.to(dtype=torch.complex64, device=device).contiguous()

    @staticmethod
    def _load_scales(path: Path, device: torch.device) -> torch.Tensor:
        """读取一维 Hopkins 权重并转换为 float32。"""
        if not path.is_file():
            raise FileNotFoundError(f"找不到光刻 scale：{path}")
        tensor = torch.load(path, map_location=device, weights_only=True)
        if tensor.ndim != 1:
            raise ValueError(f"光刻 scale 必须是一维张量：{path}")
        return tensor.to(dtype=torch.float32, device=device).contiguous()

    @staticmethod
    def _kernel_multiply(kernels: torch.Tensor, mask_fft: torch.Tensor,
                         kernel_count: int) -> torch.Tensor:
        """把中心原点 kernel 的四个象限映射到 FFT 频谱四角。"""
        height, width = kernels.shape[-2:]
        half_height, half_width = height // 2, width // 2
        output = torch.zeros(
            (mask_fft.shape[0], kernel_count, mask_fft.shape[-2], mask_fft.shape[-1]),
            dtype=mask_fft.dtype, device=mask_fft.device)
        # 这里保持 OpenILT 的频域象限约定，不显式执行 fftshift；四次批量赋值避免
        # 对 kernel 或像素使用 Python 循环，也是 GPU 热路径的主要性能不变量。
        output[:, :, :half_height + 1, :half_width + 1] = (
            mask_fft[:, :, :half_height + 1, :half_width + 1] *
            kernels[None, :kernel_count, -(half_height + 1):, -(half_width + 1):])
        output[:, :, :half_height + 1, -half_width:] = (
            mask_fft[:, :, :half_height + 1, -half_width:] *
            kernels[None, :kernel_count, -(half_height + 1):, :half_width])
        output[:, :, -half_height:, :half_width + 1] = (
            mask_fft[:, :, -half_height:, :half_width + 1] *
            kernels[None, :kernel_count, :half_height, -(half_width + 1):])
        output[:, :, -half_height:, -half_width:] = (
            mask_fft[:, :, -half_height:, -half_width:] *
            kernels[None, :kernel_count, :half_height, :half_width])
        return output

    def _aerial_from_spectrum(self, spectrum: torch.Tensor, kernels: torch.Tensor,
                              scales: torch.Tensor) -> torch.Tensor:
        """从共享 mask 频谱计算单位剂量下的 Hopkins 部分相干强度。"""
        fields = torch.fft.ifft2(
            self._kernel_multiply(kernels, spectrum, self.config.kernel_count),
            norm="forward")
        weights = scales[:self.config.kernel_count][None, :, None, None]
        return torch.sum(weights * torch.abs(fields).square(), dim=1)

    def _prepare_mask(self, mask: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """把单张或批量 mask 居中补零并缩放到模型分辨率。"""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("光刻 mask 必须具有 [H,W] 或 [B,H,W] 形状")
        mask = mask.to(device=self.device, dtype=torch.float32)
        height, width = mask.shape[-2:]
        canvas = self.config.canvas
        if height > canvas or width > canvas:
            raise ValueError(f"光刻 mask 尺寸 {height}x{width} 超过 canvas {canvas}")
        top = (canvas - height) // 2
        bottom = canvas - height - top
        left = (canvas - width) // 2
        right = canvas - width - left
        padded = functional.pad(mask, (left, right, top, bottom))
        if self.config.resolution != canvas:
            padded = functional.interpolate(
                padded[:, None], size=(self.config.resolution, self.config.resolution),
                mode="nearest")[:, 0]
        return padded, (top, bottom, left, right)

    def _restore_size(self, image: torch.Tensor,
                      padding: tuple[int, int, int, int]) -> torch.Tensor:
        """把仿真分辨率恢复到 canvas，并删除输入时添加的零填充。"""
        if self.config.resolution != self.config.canvas:
            image = functional.interpolate(
                image[:, None], size=(self.config.canvas, self.config.canvas),
                mode="nearest")[:, 0]
        top, bottom, left, right = padding
        y_end = image.shape[-2] - bottom if bottom else image.shape[-2]
        x_end = image.shape[-1] - right if right else image.shape[-1]
        return image[:, top:y_end, left:x_end]

    def forward(self, mask: torch.Tensor) -> LithographyResult:
        """执行 mask→aerial image→连续光刻胶的三工艺角前向计算。"""
        squeeze = mask.ndim == 2
        prepared, padding = self._prepare_mask(mask)
        # 三个工艺角使用同一个 mask。FFT 是线性变换，剂量乘在振幅上最终使强度
        # 按 dose² 缩放，因此只需一次 FFT；focus 的单位剂量强度还能同时供 nominal
        # 和 maximum 使用。这样减少两次大批量 FFT 和一次 focus kernel 传播，同时
        # 不缓存跨调用张量，反向传播仍沿共享频谱返回原 mask。
        spectrum = torch.fft.fft2(prepared.to(torch.complex64).unsqueeze(1), norm="forward")
        focus = self._aerial_from_spectrum(
            spectrum, self.focus_kernels, self.focus_scales)
        nominal = focus * (self.config.dose_nominal ** 2)
        maximum = focus * (self.config.dose_max ** 2)
        del focus
        minimum = self._aerial_from_spectrum(
            spectrum, self.defocus_kernels, self.defocus_scales)
        minimum = minimum * (self.config.dose_min ** 2)
        del spectrum
        steepness, density = self.config.print_steepness, self.config.target_density
        nominal = self._restore_size(torch.sigmoid(steepness * (nominal - density)), padding)
        maximum = self._restore_size(torch.sigmoid(steepness * (maximum - density)), padding)
        minimum = self._restore_size(torch.sigmoid(steepness * (minimum - density)), padding)
        if squeeze:
            nominal, maximum, minimum = nominal[0], maximum[0], minimum[0]
        return LithographyResult(nominal, maximum, minimum)
