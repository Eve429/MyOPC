"""基于固定 ICCAD13 Hopkins 资产实现可微批量光刻仿真。"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

# Windows 直接执行环境内 python.exe 时不会像 conda run 那样把 <env>/bin 加入
# DLL 搜索目录，而 PyTorch 的 NVRTC JIT 运行时（nvrtc-builtins64_*.dll）位于
# 该目录；缺目录时 torch.cuda.is_available() 仍为 True，直到首次 CUDA FFT 才
# 抛 nvrtc 错误（迁移实测复现后授权的最小修复）。句柄必须在
# 进程生命周期内保留，否则目录会立即从搜索路径移除；非 Windows 不执行。
_DLL_DIRECTORY = None
if os.name == "nt":
    _cuda_dll_dir = Path(sys.prefix) / "bin"
    if _cuda_dll_dir.is_dir():
        _DLL_DIRECTORY = os.add_dll_directory(str(_cuda_dll_dir))
        os.environ["PATH"] = (
            f"{_cuda_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}")

# torch 导入必须保持在上述 DLL 目录注册之后：先注册后导入才能保证
# 首次 CUDA 操作在任意启动方式下都找得到 NVRTC 运行时。
import torch
from torch import nn
from torch.nn import functional

# 当前 OpenILT 资产每个 bank 固定提供 24 个 Hopkins 核；配置请求超过该上限说明
# 配置与资产世代不符，必须在解析期拒绝，而不是运行期得到错误截断的结果。
_MAX_ASSET_KERNELS = 24

# 画布与仿真分辨率按 ICCAD13 契约冻结为 256：其他尺寸需要新的 kernel 采样契约，
# 本模型不做插值伪装支持。
_FIXED_CANVAS_PIXELS = 256

# 配置文件的九个必需字段；缺失、重复或未知字段都在解析期失败。
_REQUIRED_FIELDS = (
    "KernelNum", "TargetDensity", "PrintThresh", "PrintSteepness",
    "DoseMax", "DoseMin", "DoseNom", "Canvas", "Resolution",
)


@dataclass(frozen=True, slots=True)
class ICCAD13Config:
    """保存 ICCAD13 Hopkins 模型的固定数值配置。"""

    kernel_count: int       # 使用的 Hopkins 核数量，不得超过资产核数
    target_density: float   # 连续胶 sigmoid 的强度阈值
    print_threshold: float  # 后续二值化阈值，不直接参与 forward
    print_steepness: float  # 连续胶 sigmoid 陡峭度
    dose_max: float         # 最大剂量
    dose_min: float         # 最小剂量
    dose_nominal: float     # 标称剂量
    canvas: int             # 固定画布边长（像素）
    resolution: int         # 固定仿真分辨率（像素）

    @classmethod
    def from_file(cls, path: str | Path) -> ICCAD13Config:
        """读取“字段 值”配置并返回完成数值校验的冻结配置。"""
        source = Path(path).expanduser().resolve()
        values: dict[str, str] = {}
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"光刻配置行必须包含名称和值：{line}")
            if parts[0] in values:
                raise ValueError(f"光刻配置字段重复出现：{parts[0]}")
            values[parts[0]] = parts[1]
        # 未知字段按拼写错误处理直接拒绝；缺字段同样不给默认值。
        unknown = sorted(set(values) - set(_REQUIRED_FIELDS))
        if unknown:
            raise ValueError(f"光刻配置含未知字段：{', '.join(unknown)}")
        missing = [name for name in _REQUIRED_FIELDS if name not in values]
        if missing:
            raise ValueError(f"光刻配置缺少字段：{', '.join(missing)}")

        # 数值转换失败时把字段名带进异常，避免裸 ValueError 定位困难。
        def as_integer(name: str) -> int:
            """把配置值转换为整数，失败时报字段名。"""
            try:
                return int(values[name])
            except ValueError as exc:
                raise ValueError(
                    f"光刻配置字段 {name} 必须是整数：{values[name]}") from exc

        def as_finite_float(name: str) -> float:
            """把配置值转换为有限浮点数，失败或非有限时报字段名。"""
            try:
                number = float(values[name])
            except ValueError as exc:
                raise ValueError(
                    f"光刻配置字段 {name} 必须是数字：{values[name]}") from exc
            if not isfinite(number):
                raise ValueError(f"光刻配置字段 {name} 必须是有限数：{values[name]}")
            return number

        config = cls(
            kernel_count=as_integer("KernelNum"),
            target_density=as_finite_float("TargetDensity"),
            print_threshold=as_finite_float("PrintThresh"),
            print_steepness=as_finite_float("PrintSteepness"),
            dose_max=as_finite_float("DoseMax"),
            dose_min=as_finite_float("DoseMin"),
            dose_nominal=as_finite_float("DoseNom"),
            canvas=as_integer("Canvas"),
            resolution=as_integer("Resolution"),
        )
        # 数值契约：核数量区间、画布冻结、阈值/陡峭度为正、
        # 剂量单调递增顺序；任何一条不满足都说明配置不可信。
        if not 1 <= config.kernel_count <= _MAX_ASSET_KERNELS:
            raise ValueError(
                f"KernelNum 必须在 1 到 {_MAX_ASSET_KERNELS} 之间：{config.kernel_count}")
        if (config.canvas != _FIXED_CANVAS_PIXELS or
                config.resolution != _FIXED_CANVAS_PIXELS):
            raise ValueError(
                f"Canvas 与 Resolution 当前固定为 {_FIXED_CANVAS_PIXELS}")
        if not 0.0 < config.target_density:
            raise ValueError(f"TargetDensity 必须为正数：{config.target_density}")
        if not 0.0 < config.print_threshold < 1.0:
            raise ValueError(
                f"PrintThresh 必须在 0 到 1 开区间内：{config.print_threshold}")
        if not config.print_steepness > 0.0:
            raise ValueError(f"PrintSteepness 必须为正数：{config.print_steepness}")
        if not 0.0 < config.dose_min <= config.dose_nominal <= config.dose_max:
            raise ValueError(
                "剂量必须满足 0 < DoseMin <= DoseNom <= DoseMax："
                f"{config.dose_min} / {config.dose_nominal} / {config.dose_max}")
        return config


@dataclass(frozen=True, slots=True)
class ProcessCondition:
    """描述一次独立工艺条件的名称、kernel bank 和剂量。"""

    name: str                                  # 同一次 forward_many 内唯一的结果名称
    kernel: Literal["focus", "defocus"]        # 选择已经加载的 kernel bank
    dose: float                                # 正有限振幅剂量，强度按 dose² 缩放

    def __post_init__(self) -> None:
        """拒绝空名称、未知 kernel bank 或非正有限剂量。"""
        # kernel 名拼错时若不在这里拦截，forward_many 的 bank 选择会把未知名
        # 静默落到 defocus 分支，产生难以定位的错误结果。
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工艺条件名称不能为空")
        if self.kernel not in ("focus", "defocus"):
            raise ValueError("工艺条件 kernel 必须是 focus 或 defocus")
        if not isfinite(self.dose) or self.dose <= 0.0:
            raise ValueError("工艺条件剂量必须是正有限数")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "dose", float(self.dose))


class ICCAD13Lithography(nn.Module):
    """使用固定 ICCAD13 Hopkins 资产执行可微批量光刻仿真。"""

    def __init__(self, config_path: str | Path | None = None,
                 asset_dir: str | Path | None = None,
                 device: str | torch.device | None = None) -> None:
        """加载配置与 focus/defocus kernel bank，并移动到指定设备。"""
        super().__init__()
        # 配置与资产路径缺省相对本模块解析，保证从任意工作目录直接运行根脚本
        # 时也能加载，与 main/*.py 的仓库根 sys.path 引导互补。
        module_dir = Path(__file__).resolve().parent
        config_file = (module_dir / "config" / "iccad13.txt"
                       if config_path is None else Path(config_path))
        assets = (module_dir / "assets" / "iccad13"
                  if asset_dir is None else Path(asset_dir))
        self.config = ICCAD13Config.from_file(config_file)
        # device=None 与 "auto" 同义：有 CUDA 用 CUDA，否则退回 CPU。
        if device is None or str(device) == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        focus_kernels, focus_scales = self._load_kernel_bank(assets, "focus", resolved)
        defocus_kernels, defocus_scales = self._load_kernel_bank(assets, "defocus", resolved)
        # 配置核数不得超过任一 bank 的实际核数，加载期即失败。
        count = self.config.kernel_count
        if (focus_kernels.shape[0] < count or defocus_kernels.shape[0] < count or
                focus_scales.shape[0] < count or defocus_scales.shape[0] < count):
            raise ValueError("光刻资产包含的 Hopkins 核数量少于配置要求")
        # 四个张量注册为 buffer：跟随 Module.to() 移动并进入 state_dict，但不会
        # 被优化器当作可训练参数——kernel/scale 是物理资产，不是学习对象。
        self.register_buffer("focus_kernels", focus_kernels, persistent=True)
        self.register_buffer("defocus_kernels", defocus_kernels, persistent=True)
        self.register_buffer("focus_scales", focus_scales, persistent=True)
        self.register_buffer("defocus_scales", defocus_scales, persistent=True)
        self.to(resolved)

    @property
    def device(self) -> torch.device:
        """返回模型 buffer 当前所在设备。"""
        return self.focus_kernels.device

    def condition(self, name: str) -> ProcessCondition:
        """按稳定名称返回一个默认且独立的 ICCAD13 工艺条件。"""
        if name == "nominal":
            return ProcessCondition(name, "focus", self.config.dose_nominal)
        if name == "dose_max":
            return ProcessCondition(name, "focus", self.config.dose_max)
        if name == "defocus_min":
            return ProcessCondition(name, "defocus", self.config.dose_min)
        raise ValueError(f"未知默认工艺条件：{name}")

    @staticmethod
    def _load_kernel_bank(
            asset_dir: Path, name: Literal["focus", "defocus"],
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """加载并校验一组 [H,W,K] kernel 与 [K] scale，返回 [K,H,W] 计算布局。"""
        kernel_path = asset_dir / f"{name}.pt"
        scale_path = asset_dir / f"{name}_scale.pt"
        if not kernel_path.is_file():
            raise FileNotFoundError(f"找不到光刻 kernel：{kernel_path}")
        if not scale_path.is_file():
            raise FileNotFoundError(f"找不到光刻 scale：{scale_path}")
        kernels = torch.load(kernel_path, map_location=device, weights_only=True)
        scales = torch.load(scale_path, map_location=device, weights_only=True)
        # 只接受已声明的 OpenILT 资产布局：方阵 [H,W,K] 且 kernel 维在最后；
        # 已是 [K,H,W] 计算布局或其他第三方布局一律拒绝，不做 shape 猜测
        if kernels.ndim != 3 or kernels.shape[0] != kernels.shape[1]:
            raise ValueError(f"光刻 kernel 必须是 [H,W,K] 方阵张量：{kernel_path}")
        if scales.ndim != 1:
            raise ValueError(f"光刻 scale 必须是一维张量：{scale_path}")
        if kernels.shape[2] != scales.shape[0]:
            raise ValueError(
                f"光刻 kernel 与 scale 数量不符：{kernel_path} 有 "
                f"{kernels.shape[2]} 个核，{scale_path} 有 {scales.shape[0]} 个权重")
        return (kernels.permute(2, 0, 1).to(
                    dtype=torch.complex64, device=device).contiguous(),
                scales.to(dtype=torch.float32, device=device).contiguous())

    def _prepare_mask(
            self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int], bool]:
        """规范化单张/批量 mask，并居中补零到固定 canvas。"""
        # 输入只接受单张 [H,W] 或批量 [B,H,W]；其他维度在频域大数组分配
        # 之前拒绝，错误不会留下等待回收的大张量。
        was_single = mask.ndim == 2
        if was_single:
            mask = mask.unsqueeze(0)  # 统一为 [B,H,W] 处理
        if mask.ndim != 3:
            raise ValueError("光刻 mask 必须具有 [H,W] 或 [B,H,W] 形状")
        mask = mask.to(device=self.device, dtype=torch.float32)
        height, width = mask.shape[-2:]
        canvas = self.config.canvas
        if height > canvas or width > canvas:
            raise ValueError(f"光刻 mask 尺寸 {height}x{width} 超过 canvas {canvas}")
        # 差值平均分配到低/高两侧，奇数余量归高坐标侧——与 opc.input.raster
        # 的居中 padding 约定逐位一致，同一几何永远得到同一 canvas 布局。
        top = (canvas - height) // 2
        bottom = canvas - height - top
        left = (canvas - width) // 2
        right = canvas - width - left
        padded = functional.pad(mask, (left, right, top, bottom))
        return padded, (top, bottom, left, right), was_single

    @staticmethod
    def _kernel_multiply(
            kernels: torch.Tensor, spectrum: torch.Tensor,
            kernel_count: int) -> torch.Tensor:
        """把中心原点 kernel 的四个象限批量映射到 FFT 频谱四角。"""
        # kernel 自身尺寸决定象限块大小（35→18/17）：Hopkins 核只覆盖中心
        # ±17 个频率采样点，因此频谱只有四角低频块与 kernel 相乘，其余
        # 频率保持零，模型天然抑制高频伪影。
        height, width = kernels.shape[-2:]
        half_height, half_width = height // 2, width // 2
        output = torch.zeros(
            (spectrum.shape[0], kernel_count,
             spectrum.shape[-2], spectrum.shape[-1]),
            dtype=spectrum.dtype, device=spectrum.device)
        # 保持 OpenILT 的频域象限约定，不显式执行 fftshift；四次批量赋值
        # 避免对 kernel 或像素的 Python 循环，是 GPU 热路径的主要性能
        # 不变量。赋值顺序固定左上→右上→左下→右下：DC 与 Nyquist 的重叠
        # 行/列由后写的象限覆盖。
        output[:, :, :half_height + 1, :half_width + 1] = (
            spectrum[:, :, :half_height + 1, :half_width + 1] *
            kernels[None, :kernel_count, -(half_height + 1):, -(half_width + 1):])
        output[:, :, :half_height + 1, -half_width:] = (
            spectrum[:, :, :half_height + 1, -half_width:] *
            kernels[None, :kernel_count, -(half_height + 1):, :half_width])
        output[:, :, -half_height:, :half_width + 1] = (
            spectrum[:, :, -half_height:, :half_width + 1] *
            kernels[None, :kernel_count, :half_height, -(half_width + 1):])
        output[:, :, -half_height:, -half_width:] = (
            spectrum[:, :, -half_height:, -half_width:] *
            kernels[None, :kernel_count, :half_height, :half_width])
        return output

    def _propagate(
            self, spectrum: torch.Tensor, kernels: torch.Tensor,
            scales: torch.Tensor) -> torch.Tensor:
        """从共享 mask 频谱计算一个 kernel bank 的单位剂量强度。"""
        # 每 kernel 一次 ifft2 回到空间域，取模平方后按 scale 加权求和，
        # 得到单位剂量下的 Hopkins 部分相干强度 [B,canvas,canvas]。
        fields = torch.fft.ifft2(
            self._kernel_multiply(kernels, spectrum, self.config.kernel_count),
            norm="forward")
        weights = scales[:self.config.kernel_count][None, :, None, None]
        return torch.sum(weights * torch.abs(fields).square(), dim=1)

    def forward_many(
            self, mask: torch.Tensor,
            conditions: Sequence[ProcessCondition]) -> dict[str, torch.Tensor]:
        """一次计算多个独立工艺条件，并保留 mask 的 autograd 图。"""
        requested = tuple(conditions)
        if not requested:
            raise ValueError("至少需要一个光刻工艺条件")
        if any(not isinstance(condition, ProcessCondition)
               for condition in requested):
            raise TypeError("conditions 必须全部是 ProcessCondition")
        names = [condition.name for condition in requested]
        if len(set(names)) != len(names):
            raise ValueError("同一次仿真的工艺条件名称不能重复")
        prepared, (top, bottom, left, right), was_single = self._prepare_mask(mask)
        # 所有条件共享同一 mask FFT；相同 kernel bank 的单位剂量强度也只
        # 传播一次。剂量乘在振幅上，强度严格按 dose² 缩放，因此 dose 只
        # 通过 dose² 因子复用同一 unit。缓存仅存在于本次调用的 autograd
        # 图中，不跨 mask 保存——MB-OPC 的 no_grad 推理与 ILT 的 backward
        # 走同一条路径。
        spectrum = torch.fft.fft2(
            prepared.to(torch.complex64).unsqueeze(1), norm="forward")
        steepness = self.config.print_steepness
        density = self.config.target_density
        intensities: dict[str, torch.Tensor] = {}  # bank 名 → 单位剂量强度
        results: dict[str, torch.Tensor] = {}
        for condition in requested:
            unit = intensities.get(condition.kernel)
            if unit is None:  # 该 bank 首次出现才传播，后续条件直接复用
                if condition.kernel == "focus":
                    kernels, scales = self.focus_kernels, self.focus_scales
                else:
                    kernels, scales = self.defocus_kernels, self.defocus_scales
                unit = self._propagate(spectrum, kernels, scales)
                intensities[condition.kernel] = unit
            printed = torch.sigmoid(
                steepness * (unit * (condition.dose ** 2) - density))
            # 内联 crop：去掉居中补零的四边恢复输入 H×W（top+高度恰等于
            # canvas−bottom，因此统一切片公式对零 padding 同样正确）；
            # 单张输入再压回 [H,W]。
            restored = printed[
                :, top:printed.shape[-2] - bottom,
                left:printed.shape[-1] - right]
            results[condition.name] = restored[0] if was_single else restored
        return results

    def forward(self, mask: torch.Tensor,
                condition: ProcessCondition) -> torch.Tensor:
        """执行单工艺条件的 mask 到连续 printed image 前向。"""
        if not isinstance(condition, ProcessCondition):
            raise TypeError("condition 必须是 ProcessCondition")
        return self.forward_many(mask, (condition,))[condition.name]
