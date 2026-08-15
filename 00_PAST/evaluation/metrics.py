"""计算二值 L2、PVBand、边段 EPE 和确定性矩形 shot 估计。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class EPEEvaluation:
    """保存逐边段 EPE 有效性、违规类型和法向移动方向。"""

    valid: torch.Tensor
    inner_violations: torch.Tensor
    outer_violations: torch.Tensor
    ambiguous: torch.Tensor
    directions: torch.Tensor

    @property
    def violation_count(self) -> int:
        """返回至少一个有效探针违规的边段数量。"""
        return int(torch.count_nonzero(self.inner_violations | self.outer_violations).item())


def _aligned_images(*images: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """把二维图提升为 batch，并拒绝形状或设备不一致。"""
    normalized = tuple(image.unsqueeze(0) if image.ndim == 2 else image for image in images)
    if any(image.ndim != 3 for image in normalized):
        raise ValueError("评价图像必须具有 [H,W] 或 [B,H,W] 形状")
    first = normalized[0]
    if any(image.shape != first.shape or image.device != first.device for image in normalized[1:]):
        raise ValueError("评价图像的形状和设备必须一致")
    return normalized


def _selected_pixels(reference: torch.Tensor,
                     ownership_mask: torch.Tensor | None) -> torch.Tensor:
    """构造与评价图同形的布尔选择区，只让唯一 core 统计边界像素。"""
    if ownership_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    selected = ownership_mask.unsqueeze(0) if ownership_mask.ndim == 2 else ownership_mask
    if selected.shape != reference.shape or selected.device != reference.device:
        raise ValueError("ownership_mask 必须与评价图像形状和设备一致")
    return selected.to(torch.bool)


def evaluate_binary_l2(target: torch.Tensor, nominal: torch.Tensor,
                       threshold: float = 0.5,
                       ownership_mask: torch.Tensor | None = None) -> int:
    """统计标称二值 wafer 与目标二值图不一致的 ownership 像素数。"""
    target, nominal = _aligned_images(target, nominal)
    selected = _selected_pixels(target, ownership_mask)
    # OpenILT 的 binary MSE(sum) 对 0/1 图等价于异或像素数。直接比较布尔值不创建
    # 两张 float 差值/平方中间量；halo 仅供卷积读取，不参与跨 tile 的重复计数。
    mismatch = (nominal >= threshold) != (target >= threshold)
    return int(torch.count_nonzero(mismatch & selected).item())


def evaluate_pvband(maximum: torch.Tensor, minimum: torch.Tensor,
                    threshold: float = 0.5,
                    ownership_mask: torch.Tensor | None = None) -> int:
    """统计两个独立工艺条件二值 wafer 不一致的 ownership 像素数。"""
    maximum, minimum = _aligned_images(maximum, minimum)
    selected = _selected_pixels(maximum, ownership_mask)
    band = (maximum >= threshold) != (minimum >= threshold)
    return int(torch.count_nonzero(band & selected).item())


def estimate_rectangular_shots(mask: torch.Tensor, threshold: float = 0.5,
                               shape: tuple[int, int] = (512, 512)) -> int:
    """在固定评价分辨率上用确定性水平 run 合并估计矩形 shot 数。"""
    if mask.ndim == 2:
        images = mask.unsqueeze(0)
    elif mask.ndim == 3:
        images = mask
    else:
        raise ValueError("shot mask 必须具有 [H,W] 或 [B,H,W] 形状")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shot 评价尺寸必须是两个正整数")
    # Shot 是昂贵且非梯度的最终诊断项：先用最近邻统一到显式评价分辨率，再一次性
    # 转 CPU bool。算法逐行提取连续前景 run；仅当相邻行的左右端完全相同才延续
    # 同一矩形，结果确定、无随机搜索，也不引入 OpenCV/adabox 运行时依赖。
    resized = functional.interpolate(
        images[:, None].to(dtype=torch.float32), size=shape, mode="nearest")[:, 0]
    binary = resized.detach().cpu().numpy() >= threshold
    total = 0
    for image in binary:
        active: set[tuple[int, int]] = set()
        for row in image:
            padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
            changes = np.diff(padded)
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            current = set(zip(starts.tolist(), ends.tolist(), strict=True))
            total += len(current - active)
            active = current
    return total


def evaluate_edge_probes(target: torch.Tensor, nominal: torch.Tensor,
                         batch_indices: torch.Tensor, inner_xy: torch.Tensor,
                         outer_xy: torch.Tensor, threshold: float = 0.499) -> EPEEvaluation:
    """批量评价 inner/outer 探针并生成 -1/0/+1 法向移动方向。"""
    target, nominal = _aligned_images(target, nominal)
    device = target.device
    batches = batch_indices.to(device=device, dtype=torch.long)
    inner = torch.round(inner_xy.to(device=device)).to(torch.long)
    outer = torch.round(outer_xy.to(device=device)).to(torch.long)
    if (batches.ndim != 1 or inner.ndim != 2 or inner.shape[1] != 2 or
            outer.shape != inner.shape or len(batches) != len(inner)):
        raise ValueError("探针索引和坐标必须按边段对齐")
    height, width = target.shape[-2:]
    in_bounds = ((batches >= 0) & (batches < target.shape[0]) &
                 (inner[:, 0] >= 0) & (inner[:, 0] < width) &
                 (inner[:, 1] >= 0) & (inner[:, 1] < height) &
                 (outer[:, 0] >= 0) & (outer[:, 0] < width) &
                 (outer[:, 1] >= 0) & (outer[:, 1] < height))
    distinct = torch.any(inner != outer, dim=1)
    safe_batches = batches.clamp(0, max(target.shape[0] - 1, 0))
    safe_inner_x = inner[:, 0].clamp(0, max(width - 1, 0))
    safe_inner_y = inner[:, 1].clamp(0, max(height - 1, 0))
    safe_outer_x = outer[:, 0].clamp(0, max(width - 1, 0))
    safe_outer_y = outer[:, 1].clamp(0, max(height - 1, 0))
    target_inner = target[safe_batches, safe_inner_y, safe_inner_x] >= threshold
    target_outer = target[safe_batches, safe_outer_y, safe_outer_x] < threshold
    valid = in_bounds & distinct & target_inner & target_outer
    printed_inner = nominal[safe_batches, safe_inner_y, safe_inner_x] >= threshold
    printed_outer = nominal[safe_batches, safe_outer_y, safe_outer_x] >= threshold
    inner_vios = valid & ~printed_inner
    outer_vios = valid & printed_outer
    # inner 和 outer 是两个不同探针；若两者同时违规，它们分别要求外移和内移。
    # 简单单边动作无法可靠解决该局部状态，因此方向保持 0 并显式记录 ambiguous。
    ambiguous = inner_vios & outer_vios
    directions = torch.zeros(len(inner), dtype=torch.int8, device=device)
    directions[inner_vios & ~outer_vios] = 1
    directions[outer_vios & ~inner_vios] = -1
    return EPEEvaluation(valid, inner_vios, outer_vios, ambiguous, directions)
