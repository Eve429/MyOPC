"""TorchLitho 频率网格、四种源形状格点与光瞳函数。"""

from __future__ import annotations

import math

import torch

# 源形状枚举；point 为原库 TCC 的默认行为（golden 一致性锚点）。
SOURCE_SHAPES = ("point", "disk", "dipole", "quadrupole")


def frequency_grid(
    canvas: int, pixel_nm: float, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造居中频率坐标网格（单位 cycles/nm），返回 (fx, fy, |f|) 三张 [canvas,canvas]。

    模型前向用默认 float32；TCC 构造链传 float64 以对齐原库 numpy 精度。
    """
    # fftshift 后 DC 位于中心，与原库 getFreqSupport 同构：x 沿列、y 沿行。
    basic = torch.fft.fftshift(torch.fft.fftfreq(canvas, d=pixel_nm, device=device)).to(dtype)
    fx = basic.reshape(1, -1).expand(canvas, canvas)
    fy = basic.reshape(-1, 1).expand(canvas, canvas)
    return fx, fy, torch.sqrt(fx * fx + fy * fy)


def source_points(
    shape: str,
    fx: torch.Tensor,
    fy: torch.Tensor,
    sigma: float,
    pole_center: float,
    na: float,
    wavelength_nm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按源形状生成离散源点，返回格点坐标 [S,2] 与同域 0/1 掩膜 [N,N]。

    Abbe 前向消费坐标列表（每个源点平移光瞳），Hopkins 的 TCC 消费掩膜形态；
    两者来自同一判定，保证两方法描述同一物理光源。
    """
    # 盘/极盘半径与极心偏移都按 NA 归一化定义：r = 因子 × NA/λ。
    pole_radius = sigma * na / wavelength_nm
    offset = pole_center * na / wavelength_nm
    if shape == "point":
        # 点源只含 DC：fftshift 网格中心格点的频率坐标恰为 (0, 0)。
        coords = torch.zeros((1, 2), dtype=torch.float32, device=fx.device)
        mask = torch.zeros(fx.shape, dtype=torch.bool, device=fx.device)
        mask[fx.shape[0] // 2, fx.shape[1] // 2] = True
        return coords, mask
    if shape == "disk":
        # 圆盘源：|f| <= σNA/λ（含等号，与原库 getSourcePoints 的判定一致）。
        mask = torch.sqrt(fx * fx + fy * fy) <= pole_radius
    elif shape == "dipole":
        # 双极沿 X 轴两极 (±c, 0)；方向固定，任意旋转角属后续增量。
        mask = _pole_union_mask(fx, fy, ((-offset, 0.0), (offset, 0.0)), pole_radius)
    elif shape == "quadrupole":
        # 四极分布在 ±45° 对角方向，极心 (±c, ±c)/√2。
        half = offset / math.sqrt(2.0)
        centers = ((half, half), (half, -half), (-half, half), (-half, -half))
        mask = _pole_union_mask(fx, fy, centers, pole_radius)
    else:
        raise ValueError(f"未知源形状：{shape}（只接受 {'/'.join(SOURCE_SHAPES)}）")
    coords = torch.stack((fx[mask], fy[mask]), dim=1)
    if coords.shape[0] == 0:
        # 极盘完全落在频率网格外（pole_center 过大或网格过粗）时源为空，构造期拒绝。
        raise ValueError(f"源形状 {shape} 在当前频率网格上没有任何格点，无法构造离散光源")
    return coords, mask


def _pole_union_mask(
    fx: torch.Tensor, fy: torch.Tensor, centers: tuple[tuple[float, float], ...], pole_radius: float
) -> torch.Tensor:
    """计算多个极盘并集的 0/1 掩膜：任一极心距格点 <= 极盘半径即选中。"""
    device = fx.device
    mask = torch.zeros(fx.shape, dtype=torch.bool, device=device)
    for center_x, center_y in centers:
        dx = fx - center_x
        dy = fy - center_y
        mask |= dx * dx + dy * dy <= pole_radius * pole_radius
    return mask


def pupil_function(
    freq: torch.Tensor,
    na: float,
    wavelength_nm: float,
    defocus_nm: float,
    refractive_index: float,
) -> torch.Tensor:
    """圆光瞳 1{|f| < NA/λ} 乘离焦相位，返回 [N,N] complex64。

    离焦相位 exp(i·(2π/λ)·Δz·(n − sqrt(n² − λ²f²))) 中的根式在瞳内恒正
    （配置校验保证 NA < n），瞳外被 0/1 掩膜归零；clamp(min=0) 仅防御瞳外
    负数开方产生的 NaN 进入未选中分支。输出精度跟随输入：float32 →
    complex64（模型前向），float64 → complex128（TCC 构造链对齐原库 numpy）。
    """
    limit = na / wavelength_nm
    inside = freq < limit
    out_dtype = torch.promote_types(freq.dtype, torch.complex64)
    if defocus_nm == 0.0:
        return inside.to(out_dtype)
    # 光程差按光瞳内传播角计算；2π/λ 相位因子与原库 getDefocus 同构。
    squared = refractive_index * refractive_index - (wavelength_nm * wavelength_nm) * freq * freq
    opd = defocus_nm * (refractive_index - torch.sqrt(squared.clamp(min=0.0)))
    phase = torch.exp(1j * (2.0 * torch.pi / wavelength_nm) * opd).to(out_dtype)
    return torch.where(inside, phase, torch.zeros_like(phase))
