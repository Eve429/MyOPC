"""计算二值 L2、PVBand 与边段 EPE 三项求解器消费的评价指标。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
        return int(torch.count_nonzero(
            self.inner_violations | self.outer_violations).item())


def _aligned_images(*images: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """把二维图提升为 batch，并拒绝形状或设备不一致。"""
    normalized = tuple(
        image.unsqueeze(0) if image.ndim == 2 else image for image in images)
    if any(image.ndim != 3 for image in normalized):
        raise ValueError("评价图像必须具有 [H,W] 或 [B,H,W] 形状")
    first = normalized[0]
    if any(image.shape != first.shape or image.device != first.device
           for image in normalized[1:]):
        raise ValueError("评价图像的形状和设备必须一致")
    return normalized


def _selected_pixels(reference: torch.Tensor,
                     ownership_mask: torch.Tensor | None) -> torch.Tensor:
    """构造与评价图同形的布尔选择区，只让唯一 core 统计边界像素。"""
    if ownership_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    selected = (ownership_mask.unsqueeze(0) if ownership_mask.ndim == 2
                else ownership_mask)
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


def evaluate_edge_probes(target: torch.Tensor, nominal: torch.Tensor,
                         batch_indices: torch.Tensor, inner_xy: torch.Tensor,
                         outer_xy: torch.Tensor,
                         threshold: float = 0.5) -> EPEEvaluation:
    """批量评价 inner/outer 探针并生成 -1/0/+1 法向移动方向。

    默认阈值与 L2/PVBand 统一为 0.5；求解器必须显式传入
    model.config.print_threshold，保证同一状态的三类指标共用同一
    "打印轮廓"定义。
    """
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
    printed_inner = nominal[
        safe_batches, safe_inner_y, safe_inner_x] >= threshold
    printed_outer = nominal[
        safe_batches, safe_outer_y, safe_outer_x] >= threshold
    inner_vios = valid & ~printed_inner
    outer_vios = valid & printed_outer
    # inner 和 outer 是两个不同探针；若两者同时违规，它们分别要求外移和内移。
    # 简单单边动作无法可靠解决该局部状态，因此方向保持 0 并显式记录 ambiguous。
    ambiguous = inner_vios & outer_vios
    directions = torch.zeros(len(inner), dtype=torch.int8, device=device)
    directions[inner_vios & ~outer_vios] = 1
    directions[outer_vios & ~inner_vios] = -1
    return EPEEvaluation(valid, inner_vios, outer_vios, ambiguous, directions)
