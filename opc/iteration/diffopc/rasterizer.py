"""实现不依赖 KLayout 的解析软边段栅格化。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def _segment_occupancy_delta(
        points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
        normals: torch.Tensor, displacements: torch.Tensor,
        temperature: float) -> torch.Tensor:
    """计算一个边段 chunk 在全部像素上的平滑占据变化。"""
    vector = ends - starts
    length = torch.linalg.vector_norm(vector, dim=1)
    tangent = vector / length[:, None]
    delta = points[..., None, :] - starts
    along = torch.sum(delta * tangent, dim=-1)
    normal_distance = torch.sum(delta * normals, dim=-1)
    # 正外法向一侧的 signed distance 为正；边界外移 d 后 occupancy 从
    # H(-q) 变为 H(d-q)。两条 sigmoid 的差在 d=0 时严格为零，因此无论
    # 参考像素是否为部分覆盖，零位移都逐像素保持原值。切向双 sigmoid 只让
    # 当前有限 segment 及其平滑端帽参与，避免无限直线污染相邻图形。
    window = (torch.sigmoid(along / temperature) *
              torch.sigmoid((length - along) / temperature))
    shifted = torch.sigmoid((displacements - normal_distance) / temperature)
    reference = torch.sigmoid(-normal_distance / temperature)
    return torch.sum(window * (shifted - reference), dim=-1)


def rasterize_soft_edges(
        reference_mask: object, starts: object, ends: object, normals: object,
        displacements: torch.Tensor, *, pixel_dbu: int, temperature: float,
        origin_dbu: tuple[int, int], chunk_size: int = 32,
        relative_pixel_centers: torch.Tensor | None = None) -> torch.Tensor:
    """用有限边段的平滑占据差生成法向位移后的连续 mask。"""
    if (not isinstance(pixel_dbu, int) or pixel_dbu <= 0 or
            not isinstance(chunk_size, int) or chunk_size <= 0 or
            not np.isfinite(temperature) or temperature <= 0.0):
        raise ValueError("pixel_dbu、temperature 和 chunk_size 必须为正")
    if len(origin_dbu) != 2 or not all(np.isfinite(value) for value in origin_dbu):
        raise ValueError("origin_dbu 必须包含两个有限坐标")
    if not isinstance(displacements, torch.Tensor) or not displacements.is_floating_point():
        raise TypeError("displacements 必须是浮点 Tensor")
    dtype = displacements.dtype
    base = torch.as_tensor(reference_mask, device=displacements.device, dtype=dtype)
    if (base.ndim != 2 or displacements.ndim != 1 or
            not bool(torch.all(torch.isfinite(base)).item()) or
            not bool(torch.all(torch.isfinite(displacements)).item())):
        raise ValueError("reference_mask 必须是有限二维数组且 displacements 必须是有限一维 Tensor")
    start = torch.as_tensor(np.asarray(starts), device=base.device, dtype=dtype)
    end = torch.as_tensor(np.asarray(ends), device=base.device, dtype=dtype)
    normal = torch.as_tensor(np.asarray(normals), device=base.device, dtype=dtype)
    if (start.ndim != 2 or start.shape[1:] != (2,) or start.shape != end.shape or
            start.shape != normal.shape or start.shape[0] != displacements.numel() or
            not bool(torch.all(torch.isfinite(start)).item()) or
            not bool(torch.all(torch.isfinite(end)).item()) or
            not bool(torch.all(torch.isfinite(normal)).item())):
        raise ValueError("边段数组必须与 displacement 数量一致")
    lengths = torch.linalg.vector_norm(end - start, dim=1)
    normal_lengths = torch.linalg.vector_norm(normal, dim=1)
    if (bool(torch.any(lengths <= 0.0).item()) or
            bool(torch.any(torch.abs(normal_lengths - 1.0) > 1e-4).item())):
        raise ValueError("边段长度必须为正且法向必须是单位向量")
    height, width = base.shape
    if relative_pixel_centers is None:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=base.device, dtype=dtype),
            torch.arange(width, device=base.device, dtype=dtype), indexing="ij")
        relative_pixel_centers = torch.stack(
            ((xx + 0.5) * pixel_dbu, (yy + 0.5) * pixel_dbu), dim=-1)
    elif (relative_pixel_centers.shape != (height, width, 2) or
          relative_pixel_centers.device != base.device or
          not relative_pixel_centers.is_floating_point()):
        raise ValueError("relative_pixel_centers 必须与 mask 同设备且形状为 [H,W,2]")
    points = relative_pixel_centers.to(dtype=dtype) + torch.tensor(
        origin_dbu, device=base.device, dtype=dtype)
    influence = torch.zeros_like(base)
    for begin in range(0, displacements.numel(), chunk_size):
        end_index = min(begin + chunk_size, displacements.numel())
        inputs = (points, start[begin:end_index], end[begin:end_index],
                  normal[begin:end_index], displacements[begin:end_index])
        if torch.is_grad_enabled() and displacements.requires_grad:
            # checkpoint 不保存 H×W×chunk 的 sigmoid/投影中间量，backward 时按
            # chunk 重算；因此峰值受 chunk_size 控制，而不是只把前向循环分块。
            change = checkpoint(
                _segment_occupancy_delta, *inputs, temperature,
                use_reentrant=False)
        else:
            change = _segment_occupancy_delta(*inputs, temperature)
        influence = influence + change
    # 空 membership 仍通过零系数连接全局位移 Tensor，使空 tile 的统一 backward
    # 路径安全；clamp 仅约束多个拐角局部叠加后的物理 occupancy 范围。
    return (base + influence + displacements.sum() * 0.0).clamp(0.0, 1.0)
