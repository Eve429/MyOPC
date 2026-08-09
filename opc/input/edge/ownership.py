"""为控制边段分配唯一 owner，并构造稀疏 halo context membership。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from opc.errors import OwnershipError
from opc.input import CoreSpec, RectilinearCoreGrid

from .types import OwnershipBatch, SegmentBatch


class OwnershipPolicy(Protocol):
    """允许后续替换跨 core 协调方法的最小归属策略接口。"""

    def assign(self, segments: SegmentBatch,
               cores: RectilinearCoreGrid | Sequence[CoreSpec]) -> OwnershipBatch:
        """返回每段唯一 owner 和所有 context membership。"""
        ...


def _grid_membership(segments: SegmentBatch, grid: RectilinearCoreGrid) -> OwnershipBatch:
    """利用网格切线直接展开每段覆盖的有限 core 范围。"""
    geometry = segments.materialize()
    midpoints = (geometry.starts + geometry.ends) * 0.5
    owners = grid.locate_points(midpoints)
    halo = grid.halo_dbu
    left = np.minimum(geometry.starts[:, 0], geometry.ends[:, 0]) - halo
    right = np.maximum(geometry.starts[:, 0], geometry.ends[:, 0]) + halo
    bottom = np.minimum(geometry.starts[:, 1], geometry.ends[:, 1]) - halo
    top = np.maximum(geometry.starts[:, 1], geometry.ends[:, 1]) + halo
    # 目标 core 的 ownership box 与扩展后的 segment bbox 接触即可成为 context。
    # 对 x/y 分别定位首尾候选，再展开每段通常很小的二维 core 范围；复杂度取决于
    # 实际 halo 邻居数量，不随全局 core 总数乘法增长，也不会建立 S×C 矩阵。
    ix0 = np.searchsorted(grid.x_cuts[1:], left, side="left")
    ix1 = np.searchsorted(grid.x_cuts[:-1], right, side="right") - 1
    iy0 = np.searchsorted(grid.y_cuts[1:], bottom, side="left")
    iy1 = np.searchsorted(grid.y_cuts[:-1], top, side="right") - 1
    ix0, ix1 = np.clip(ix0, 0, grid.column_count - 1), np.clip(ix1, 0, grid.column_count - 1)
    iy0, iy1 = np.clip(iy0, 0, grid.row_count - 1), np.clip(iy1, 0, grid.row_count - 1)
    x_spans = np.maximum(ix1 - ix0 + 1, 0)
    y_spans = np.maximum(iy1 - iy0 + 1, 0)
    membership_counts = x_spans * y_spans
    membership_offsets = np.empty(segments.segment_count + 1, dtype=np.int64)
    membership_offsets[0] = 0
    np.cumsum(membership_counts, out=membership_offsets[1:])
    members = np.repeat(np.arange(segments.segment_count, dtype=np.int32), membership_counts)
    local = np.arange(int(membership_offsets[-1]), dtype=np.int64) - np.repeat(
        membership_offsets[:-1], membership_counts)
    columns = ix0[members] + local % x_spans[members]
    rows = iy0[members] + local // x_spans[members]
    core_indices = rows * grid.column_count + columns
    order = np.argsort(core_indices, kind="stable")
    sorted_cores = core_indices[order]
    core_offsets = np.empty(grid.core_count + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(np.bincount(sorted_cores, minlength=grid.core_count), out=core_offsets[1:])
    return OwnershipBatch(grid.cores(), owners, core_offsets, members[order])


def _validate_explicit_cores(cores: tuple[CoreSpec, ...]) -> None:
    """拒绝重复 ID 和有正面积重叠的显式 ownership box。"""
    if not cores:
        raise OwnershipError("at least one core is required")
    if len({core.core_id for core in cores}) != len(cores):
        raise OwnershipError("core IDs must be unique")
    # 显式列表用于不规则的少量局部 core；规则大网格应使用 RectilinearCoreGrid。
    # 这里的 O(C²) 只发生在准备阶段，并换取明确拒绝歧义 ownership 的错误信息。
    for left_index, left in enumerate(cores):
        for right in cores[left_index + 1:]:
            if left.ownership_box.overlaps(right.ownership_box):
                raise OwnershipError(f"core ownership boxes overlap: {left.core_id}, {right.core_id}")


def _explicit_membership(segments: SegmentBatch,
                         cores: tuple[CoreSpec, ...]) -> OwnershipBatch:
    """为少量不规则 core 使用分块向量筛选构造归属。"""
    _validate_explicit_cores(cores)
    geometry = segments.materialize()
    midpoints = (geometry.starts + geometry.ends) * 0.5
    owners = np.full(segments.segment_count, -1, dtype=np.int32)
    memberships: list[np.ndarray] = []
    left = np.minimum(geometry.starts[:, 0], geometry.ends[:, 0])
    right = np.maximum(geometry.starts[:, 0], geometry.ends[:, 0])
    bottom = np.minimum(geometry.starts[:, 1], geometry.ends[:, 1])
    top = np.maximum(geometry.starts[:, 1], geometry.ends[:, 1])
    for core_index, core in enumerate(cores):
        box = core.ownership_box
        owned = ((midpoints[:, 0] >= box.left) & (midpoints[:, 0] < box.right) &
                 (midpoints[:, 1] >= box.bottom) & (midpoints[:, 1] < box.top))
        owners[owned] = core_index
        context = core.context_box
        memberships.append(np.flatnonzero(
            (left <= context.right) & (right >= context.left) &
            (bottom <= context.top) & (top >= context.bottom)).astype(np.int32))
    # 只有整体最右/最上外沿会在半开规则后无 owner；按输入 core 的稳定顺序对闭区间
    # 候选补一次归属，内部边界已经由右侧/上侧 core 接管，不会进入此兜底。
    for core_index, core in enumerate(cores):
        box = core.ownership_box
        fallback = ((owners < 0) & (midpoints[:, 0] >= box.left) &
                    (midpoints[:, 0] <= box.right) & (midpoints[:, 1] >= box.bottom) &
                    (midpoints[:, 1] <= box.top))
        owners[fallback] = core_index
    core_offsets = np.empty(len(cores) + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum([len(indices) for indices in memberships], out=core_offsets[1:])
    members = np.concatenate(memberships) if memberships else np.empty(0, dtype=np.int32)
    return OwnershipBatch(cores, owners, core_offsets, members)


class MidpointOwnerPolicy:
    """整段不切断、由中点所在 core 唯一决策的默认策略。"""

    def assign(self, segments: SegmentBatch,
               cores: RectilinearCoreGrid | Sequence[CoreSpec]) -> OwnershipBatch:
        """按 core 表示选择规则网格快速路径或显式局部路径。"""
        if isinstance(cores, RectilinearCoreGrid):
            return _grid_membership(segments, cores)
        return _explicit_membership(segments, tuple(cores))
