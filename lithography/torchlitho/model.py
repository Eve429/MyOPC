"""TorchLitho 物理参数化光刻模型：Abbe 逐源点叠加与 Hopkins 本征核两方法。

满足 lithography.contracts.LithographyModel 协议（conditions 为不透明令牌，
求解器不访问其字段，故本模型使用自有条件类型）；mask 语义与 ICCAD13 一致：
输入透光率 tensor（1=透光），输出连续胶 printed image，形状与输入相同。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from .source import SOURCE_SHAPES, frequency_grid, pupil_function, source_points
from .tcc import build_tcc_kernels

# Abbe 源点分块上限：单次广播乘的 [块长,B,canvas,canvas] complex64 不超过该
# 规模（B=8、canvas=256 时约 128 MiB）；源点更多时分块累加，数值不变。
_MAX_ELEMENTS_PER_PASS = 64 * 256 * 256


@dataclass(frozen=True, slots=True)
class TorchLithoConfig:
    """[torchlitho] 段：物理参数、源形状与胶模型阈值（全默认值）。"""

    method: Literal["abbe", "hopkins"] = "abbe"
    source_shape: Literal["point", "disk", "dipole", "quadrupole"] = "point"
    sigma: float = 0.05  # 盘/极盘半径（NA 归一化）
    pole_center: float = 0.0  # 极心偏移（NA 归一化），dipole/quadrupole 必须为正
    wavelength_nm: float = 193.0
    na: float = 1.35
    refractive_index: float = 1.44  # 介质折射率，原库两处硬编码统一提参（NA=1.35 时同 1.44）
    defocus_min_nm: float = 40.0
    dose_nominal: float = 1.0
    dose_max: float = 1.02
    dose_min: float = 0.98
    print_steepness: float = 50.0
    target_density: float = 0.225
    print_threshold: float = 0.5

    def __post_init__(self) -> None:
        """物理与剂量契约：枚举合法、NA 小于介质折射率、σ/pole_center/dose 区间。"""
        if self.method not in ("abbe", "hopkins"):
            raise ValueError(f"method 必须是 abbe 或 hopkins：{self.method}")
        if self.source_shape not in SOURCE_SHAPES:
            raise ValueError(f"source_shape 必须是 {'/'.join(SOURCE_SHAPES)}：{self.source_shape}")
        if not 0.0 < self.sigma <= 1.0:
            raise ValueError(f"sigma 必须在 (0,1] 内：{self.sigma}")
        if self.source_shape in ("dipole", "quadrupole") and self.pole_center <= 0.0:
            raise ValueError(f"{self.source_shape} 源要求 pole_center > 0：{self.pole_center}")
        if self.pole_center < 0.0:
            raise ValueError(f"pole_center 不能为负：{self.pole_center}")
        if not 0.0 < self.na < self.refractive_index:
            raise ValueError(f"必须满足 0 < NA < 折射率：{self.na} vs {self.refractive_index}")
        for name in ("wavelength_nm", "refractive_index", "defocus_min_nm"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是正有限数：{value}")
        if not 0.0 < self.dose_min <= self.dose_nominal <= self.dose_max:
            raise ValueError(
                "剂量必须满足 0 < dose_min <= dose_nominal <= dose_max："
                f"{self.dose_min} / {self.dose_nominal} / {self.dose_max}"
            )
        if not self.print_steepness > 0.0 or not self.target_density > 0.0:
            raise ValueError("print_steepness 与 target_density 必须为正数")
        if not 0.0 < self.print_threshold < 1.0:
            raise ValueError(f"print_threshold 必须在 (0,1) 开区间内：{self.print_threshold}")


@dataclass(frozen=True, slots=True)
class TorchLithoCondition:
    """TorchLitho 工艺条件令牌：defocus 值决定瞳/TCC，dose 决定出口剂量。"""

    name: str
    defocus_nm: float
    dose: float

    def __post_init__(self) -> None:
        """拒绝空名称、非有限 defocus 与非正剂量。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工艺条件名称不能为空")
        if not isfinite(self.defocus_nm):
            raise ValueError("工艺条件 defocus 必须是有限数")
        if not isfinite(self.dose) or self.dose <= 0.0:
            raise ValueError("工艺条件剂量必须是正有限数")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "defocus_nm", float(self.defocus_nm))
        object.__setattr__(self, "dose", float(self.dose))


@dataclass(frozen=True, slots=True)
class _ConfigView:
    """向 LithographyConfigView 协议暴露画布与二值阈值。"""

    canvas: int
    print_threshold: float


class TorchLithoLithography(nn.Module):
    """物理参数化可微光刻仿真（迁移自 TorchLitho-2.0，满足 LithographyModel 协议）。"""

    def __init__(
        self,
        config: TorchLithoConfig,
        canvas: int,
        pixel_nm: float,
        device: str | torch.device | None = None,
    ) -> None:
        """预计算频率网格与源点，解析设备并就位。"""
        super().__init__()
        if not isinstance(canvas, int) or canvas <= 0:
            raise ValueError(f"canvas 必须是正整数：{canvas}")
        if not isfinite(pixel_nm) or pixel_nm <= 0.0:
            raise ValueError(f"pixel_nm 必须是正有限数：{pixel_nm}")
        self._config = config
        self._canvas = canvas
        self._pixel_nm = float(pixel_nm)
        if device is None or str(device) == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        # 频率网格（cycles/nm）与源点坐标注册为 buffer：跟随 to() 移动、随 state_dict
        # 持久化；Abbe 前向按源点坐标平移光瞳，Hopkins 前向只在 TCC 网格重建源掩膜。
        fx, fy, freq = frequency_grid(canvas, pixel_nm, resolved)
        coords, _ = source_points(
            config.source_shape, fx, fy, config.sigma, config.pole_center, config.na, config.wavelength_nm
        )
        self.register_buffer("freq_x", fx.contiguous(), persistent=True)
        self.register_buffer("freq_y", fy.contiguous(), persistent=True)
        self.register_buffer("freq_norm", freq.contiguous(), persistent=True)
        self.register_buffer("source_xy", coords.to(torch.float32).contiguous(), persistent=True)
        # Hopkins 本征核按 defocus 惰性构造（秒级），缓存于普通 dict：不进 state_dict，
        # 契约是构造后不再调用 to() 换设备（与 ICCAD13 构造尾定设备的用法一致）。
        self._tcc_cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
        self.to(resolved)

    @property
    def device(self) -> torch.device:
        """返回模型 buffer 当前所在设备。"""
        return self.freq_x.device

    @property
    def config(self) -> _ConfigView:
        """返回满足 LithographyConfigView 的画布与阈值视图。"""
        return _ConfigView(canvas=self._canvas, print_threshold=self._config.print_threshold)

    def condition(self, name: str) -> TorchLithoCondition:
        """按稳定名称返回默认工艺条件（nominal/dose_max 共享 defocus=0）。"""
        if name == "nominal":
            return TorchLithoCondition(name, 0.0, self._config.dose_nominal)
        if name == "dose_max":
            return TorchLithoCondition(name, 0.0, self._config.dose_max)
        if name == "defocus_min":
            return TorchLithoCondition(name, self._config.defocus_min_nm, self._config.dose_min)
        raise ValueError(f"未知默认工艺条件：{name}")

    def _prepare_mask(self, mask: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int], bool]:
        """规范化单张/批量 mask，并居中补零到固定 canvas（约定与 ICCAD13 逐位一致）。"""
        was_single = mask.ndim == 2
        if was_single:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("光刻 mask 必须具有 [H,W] 或 [B,H,W] 形状")
        mask = mask.to(device=self.device, dtype=torch.float32)
        height, width = mask.shape[-2:]
        if height > self._canvas or width > self._canvas:
            raise ValueError(f"光刻 mask 尺寸 {height}x{width} 超过 canvas {self._canvas}")
        # 差值平均分配到低/高两侧，奇数余量归高坐标侧（与 opc.input.raster 及
        # ICCAD13 的居中 padding 约定一致，同一几何得到同一 canvas 布局）。
        top = (self._canvas - height) // 2
        bottom = self._canvas - height - top
        left = (self._canvas - width) // 2
        right = self._canvas - width - left
        padded = functional.pad(mask, (left, right, top, bottom))
        return padded, (top, bottom, left, right), was_single

    def _abbe_aerial(self, prepared: torch.Tensor, defocus_nm: float) -> torch.Tensor:
        """Abbe 逐源点相干成像叠加：瞳按源点向量平移后与谱相乘，强度对源点平均。

        原库 getSourcePoints 返回频率范数标量导致瞳同心放大（R2 缺陷）；本实现
        按源点 (cx,cy) 二维坐标平移光瞳，离焦相位同样取平移后频率（光瞳内传播角）。
        """
        # 居中谱：fftshift(fft2(mask))，与原库 getMaskFFT 同构。
        mask_fft = torch.fft.fftshift(torch.fft.fft2(prepared.to(torch.complex64)), dim=(-2, -1))
        total = self.source_xy.shape[0]
        aerial = None
        # 源点分块：限制单次 [块长,B,canvas,canvas] complex64 的内存上界。
        batch = mask_fft.shape[0]
        pixels = self._canvas * self._canvas
        chunk = max(1, _MAX_ELEMENTS_PER_PASS // max(batch * pixels, 1))
        for begin in range(0, total, chunk):
            points = self.source_xy[begin : begin + chunk]  # [s,2]
            dx = self.freq_x[None] - points[:, 0].view(-1, 1, 1)  # [s,H,W]
            dy = self.freq_y[None] - points[:, 1].view(-1, 1, 1)
            dist = torch.sqrt(dx * dx + dy * dy)
            pupil = pupil_function(
                dist, self._config.na, self._config.wavelength_nm, defocus_nm, self._config.refractive_index
            )
            # [s,1,H,W] 光瞳与 [1,B,H,W] 谱广播相乘；ifftshift 回标准布局再 ifft2。
            shifted = pupil[:, None] * mask_fft[None]
            fields = torch.fft.ifft2(torch.fft.ifftshift(shifted, dim=(-2, -1)), dim=(-2, -1))
            block = torch.sum(fields.real.square() + fields.imag.square(), dim=0)
            aerial = block if aerial is None else aerial + block
        return aerial / total

    def _hopkins_aerial(self, prepared: torch.Tensor, defocus_nm: float) -> torch.Tensor:
        """Hopkins 本征核前向：谱乘核谱、逆变换居中后按权重累加模平方。"""
        kernels, weights = self._tcc_for(defocus_nm)
        mask_fft = torch.fft.fft2(prepared.to(torch.complex64), dim=(-2, -1))
        pixels = self._canvas * self._canvas
        aerial = torch.zeros_like(prepared)
        for index in range(kernels.shape[0]):
            # 与原库 HopkinsFunc 同构：ifft2 谱乘 → fftshift → 除以像素数（原库按
            # prod(含 B) 归一，B>1 时是隐患；此处按 H·W 修正，B=1 时两者相同）。
            phi_fft = torch.fft.fft2(kernels[index])[None]
            conved = torch.fft.ifft2(mask_fft * phi_fft, dim=(-2, -1))
            conved = torch.fft.fftshift(conved, dim=(-2, -1)) / pixels
            aerial = aerial + weights[index] * (conved.real.square() + conved.imag.square())
        return aerial

    def _tcc_for(self, defocus_nm: float) -> tuple[torch.Tensor, torch.Tensor]:
        """按 defocus 取本征核缓存，缺失时构造（构造期一次性、秒级）。"""
        key = float(defocus_nm)
        cached = self._tcc_cache.get(key)
        if cached is None:
            config = self._config
            kernels, weights = build_tcc_kernels(
                config.source_shape,
                config.sigma,
                config.pole_center,
                config.na,
                config.wavelength_nm,
                config.refractive_index,
                self._canvas,
                self._pixel_nm,
                key,
            )
            cached = (kernels.to(self.device), weights.to(self.device))
            self._tcc_cache[key] = cached
        return cached

    def forward_many(self, mask: torch.Tensor, conditions: Sequence[TorchLithoCondition]) -> dict[str, torch.Tensor]:
        """一次计算多个独立工艺条件（同 defocus 共享一次成像），保留 autograd 图。"""
        requested = tuple(conditions)
        if not requested:
            raise ValueError("至少需要一个光刻工艺条件")
        if any(not isinstance(condition, TorchLithoCondition) for condition in requested):
            raise TypeError("conditions 必须全部是 TorchLithoCondition")
        names = [condition.name for condition in requested]
        if len(set(names)) != len(names):
            raise ValueError("同一次仿真的工艺条件名称不能重复")
        prepared, (top, bottom, left, right), was_single = self._prepare_mask(mask)
        # 同 defocus 的单位剂量成像只算一次，剂量以平方因子复用（与 ICCAD13 同构）。
        aerials: dict[float, torch.Tensor] = {}
        steepness = self._config.print_steepness
        density = self._config.target_density
        results: dict[str, torch.Tensor] = {}
        for condition in requested:
            key = float(condition.defocus_nm)
            aerial = aerials.get(key)
            if aerial is None:
                if self._config.method == "abbe":
                    aerial = self._abbe_aerial(prepared, condition.defocus_nm)
                else:
                    aerial = self._hopkins_aerial(prepared, condition.defocus_nm)
                aerials[key] = aerial
            printed = torch.sigmoid(steepness * (aerial * (condition.dose**2) - density))
            # 内联 crop：去掉居中补零的四边恢复输入 H×W；单张输入再压回 [H,W]。
            restored = printed[:, top : printed.shape[-2] - bottom, left : printed.shape[-1] - right]
            results[condition.name] = restored[0] if was_single else restored
        return results

    def forward(self, mask: torch.Tensor, condition: TorchLithoCondition) -> torch.Tensor:
        """执行单工艺条件的 mask 到连续 printed image 前向。"""
        if not isinstance(condition, TorchLithoCondition):
            raise TypeError("condition 必须是 TorchLithoCondition")
        return self.forward_many(mask, (condition,))[condition.name]
