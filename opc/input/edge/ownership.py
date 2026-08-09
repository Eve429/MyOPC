"""在规则 core 网格上分配唯一 owner，并构造稀疏 halo membership。"""

from __future__ import annotations

import numpy as np

from opc.input import RectilinearCoreGrid

from .types import OwnershipBatch, SegmentBatch


def build_ownership(segments: SegmentBatch, grid: RectilinearCoreGrid) -> OwnershipBatch:
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
