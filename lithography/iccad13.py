"""基于固定 ICCAD13 Hopkins 资产实现可微批量光刻仿真。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import torch
from torch import nn

# 当前 OpenILT 资产每个 bank 固定提供 24 个 Hopkins 核；配置请求超过该上限说明
# 配置与资产世代不符，必须在解析期拒绝，而不是运行期得到错误截断的结果。
_MAX_ASSET_KERNELS = 24

# 画布与仿真分辨率按 ICCAD13 契约冻结为 256：其他尺寸需要新的 kernel 采样契约，
# 本模型不做插值伪装支持（设计文档 §7.1）。
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
        # 数值契约（设计文档 §7.1）：核数量区间、画布冻结、阈值/陡峭度为正、
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
        # （设计文档 §7.3）。
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
