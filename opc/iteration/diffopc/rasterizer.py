"""实现不依赖 KLayout 的解析软边段栅格化。"""

from __future__ import annotations

import numpy as np
import torch


def rasterize_soft_edges(
        reference_mask: object, starts: object, ends: object, normals: object,
        displacements: torch.Tensor, *, pixel_dbu: int, temperature: float,
        origin_dbu: tuple[int, int], chunk_size: int = 256) -> torch.Tensor:
    """用边段法向位移对参考 mask 生成可微连续 occupancy。"""
    base = torch.as_tensor(reference_mask, device=displacements.device, dtype=torch.float32)
    if base.ndim != 2 or displacements.ndim != 1:
        raise ValueError("reference_mask 必须二维且 displacements 必须一维")
    start = torch.as_tensor(np.asarray(starts, dtype=np.float32), device=base.device)
    end = torch.as_tensor(np.asarray(ends, dtype=np.float32), device=base.device)
    normal = torch.as_tensor(np.asarray(normals, dtype=np.float32), device=base.device)
    if start.shape != end.shape or start.shape != normal.shape or start.shape[0] != displacements.numel():
        raise ValueError("边段数组必须与 displacement 数量一致")
    height, width = base.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=base.device, dtype=torch.float32),
        torch.arange(width, device=base.device, dtype=torch.float32), indexing="ij")
    points = torch.stack((xx * pixel_dbu + origin_dbu[0] + 0.5 * pixel_dbu,
                          yy * pixel_dbu + origin_dbu[1] + 0.5 * pixel_dbu), dim=-1)
    influence = torch.zeros_like(base)
    for begin in range(0, displacements.numel(), chunk_size):
        end_index = min(begin + chunk_size, displacements.numel())
        vector = end[begin:end_index] - start[begin:end_index]
        length2 = torch.sum(vector.square(), dim=1).clamp_min(1e-6)
        delta = points[..., None, :] - start[begin:end_index]
        projection = torch.sum(delta * vector, dim=-1) / length2
        projection = projection.clamp(0.0, 1.0)
        nearest = start[begin:end_index] + projection[..., None] * vector
        distance2 = torch.sum((points[..., None, :] - nearest).square(), dim=-1)
        weight = torch.exp(-distance2 / (2.0 * temperature * temperature))
        sign = torch.sum((points[..., None, :] - nearest) * normal[begin:end_index], dim=-1)
        influence = influence + torch.sum(weight * displacements[begin:end_index] * torch.sign(sign + 1e-6), dim=-1)
    return (base + influence / max(float(pixel_dbu), 1.0)).clamp(0.0, 1.0)
