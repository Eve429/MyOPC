"""向量化计算工艺窗口质量指标和基于边段探针的 EPE。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """保存 ownership 像素范围内的 L2 和 PVBand。"""

    l2: float
    pvband: float


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


def evaluate_process_window(target: torch.Tensor, nominal: torch.Tensor,
                            maximum: torch.Tensor, minimum: torch.Tensor,
                            ownership_mask: torch.Tensor | None = None) -> QualityMetrics:
    """只在唯一 ownership 像素上累计连续 L2 和 PVBand。"""
    target, nominal, maximum, minimum = _aligned_images(target, nominal, maximum, minimum)
    if ownership_mask is None:
        selected = torch.ones_like(target, dtype=torch.bool)
    else:
        selected = ownership_mask.unsqueeze(0) if ownership_mask.ndim == 2 else ownership_mask
        if selected.shape != target.shape or selected.device != target.device:
            raise ValueError("ownership_mask 必须与评价图像形状和设备一致")
        selected = selected.to(torch.bool)
    # halo 像素只提供光学上下文，不能重复计入全局指标；布尔索引在 GPU 上一次
    # 完成筛选，返回 CPU 的仅有两个标量和像素数，不会传回整张曝光图。
    l2 = torch.sum((nominal[selected] - target[selected]).square()).item()
    pvband = torch.sum((maximum[selected] - minimum[selected]).square()).item()
    # 求解器不使用像素数量，避免为无消费方增加一次 GPU→CPU 标量同步。
    return QualityMetrics(float(l2), float(pvband))


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
