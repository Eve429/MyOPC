"""基于 OpenILT ICCAD13 Hopkins 核实现可批处理三工艺角光刻仿真。"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
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
class ProcessCondition:
    """描述一次独立光刻仿真的 kernel bank、剂量和结果名称。"""

    name: str
    kernel: str
    dose: float

    def __post_init__(self) -> None:
        """拒绝空名称、未知 kernel bank 或非正有限剂量。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工艺条件名称不能为空")
        if self.kernel not in ("focus", "defocus"):
            raise ValueError("工艺条件 kernel 必须是 focus 或 defocus")
        if not isfinite(self.dose) or self.dose <= 0.0:
            raise ValueError("工艺条件剂量必须是正有限数")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "dose", float(self.dose))


class ICCAD13Lithography(nn.Module):
    """使用预计算 Hopkins 核执行可微、可批处理的独立工艺条件仿真。"""

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

    def condition(self, name: str) -> ProcessCondition:
        """按名称构造一个默认 ICCAD13 工艺条件，不绑定其他条件。"""
        if name == "nominal":
            return ProcessCondition(name, "focus", self.config.dose_nominal)
        if name == "dose_max":
            return ProcessCondition(name, "focus", self.config.dose_max)
        if name == "defocus_min":
            return ProcessCondition(name, "defocus", self.config.dose_min)
        raise ValueError(f"未知默认工艺条件：{name}")

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

    def _kernel_bank(self, condition: ProcessCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """返回条件指定的 kernel/scale buffer，不复制设备张量。"""
        if condition.kernel == "focus":
            return self.focus_kernels, self.focus_scales
        return self.defocus_kernels, self.defocus_scales

    def forward_many(self, mask: torch.Tensor,
                     conditions: Sequence[ProcessCondition]) -> dict[str, torch.Tensor]:
        """一次准备 mask 并计算任意独立条件，复用相同 kernel bank 的传播。"""
        requested = tuple(conditions)
        if not requested:
            raise ValueError("至少需要一个光刻工艺条件")
        if any(not isinstance(condition, ProcessCondition) for condition in requested):
            raise TypeError("conditions 必须全部是 ProcessCondition")
        names = [condition.name for condition in requested]
        if len(set(names)) != len(names):
            raise ValueError("同一次仿真的工艺条件名称不能重复")
        squeeze = mask.ndim == 2
        prepared, padding = self._prepare_mask(mask)
        # 所有条件共享同一 mask FFT；相同 kernel bank 的单位剂量强度也只传播一次。
        # 剂量乘在复振幅上，因此强度严格按 dose² 缩放。缓存仅存在于本次调用的
        # autograd 图中，不跨 mask 保存，MB-OPC no_grad 和 ILT backward 使用同一路径。
        spectrum = torch.fft.fft2(prepared.to(torch.complex64).unsqueeze(1), norm="forward")
        steepness, density = self.config.print_steepness, self.config.target_density
        intensities: dict[str, torch.Tensor] = {}
        results: dict[str, torch.Tensor] = {}
        for condition in requested:
            unit = intensities.get(condition.kernel)
            if unit is None:
                kernels, scales = self._kernel_bank(condition)
                unit = self._aerial_from_spectrum(spectrum, kernels, scales)
                intensities[condition.kernel] = unit
            printed = torch.sigmoid(
                steepness * (unit * (condition.dose ** 2) - density))
            restored = self._restore_size(printed, padding)
            results[condition.name] = restored[0] if squeeze else restored
        return results

    def forward(self, mask: torch.Tensor, condition: ProcessCondition) -> torch.Tensor:
        """执行单个独立工艺条件的 mask→aerial→连续光刻胶前向计算。"""
        if not isinstance(condition, ProcessCondition):
            raise TypeError("condition 必须是 ProcessCondition")
        return self.forward_many(mask, (condition,))[condition.name]
