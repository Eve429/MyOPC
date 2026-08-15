"""在规则 core 网格上分配唯一 owner，并构造稀疏 halo membership。"""

from __future__ import annotations

import numpy as np

from opc.input import RectilinearCoreGrid

from .fragmentation import SegmentBatch


def _build_ownership(
        segments: SegmentBatch, grid: RectilinearCoreGrid,
        max_memberships: int | None = None) -> tuple[np.ndarray, ...]:
    """利用网格切线展开有限 core 范围，并在大数组分配前执行硬上限。"""
    # ownership 只需要参考端点，不使用迭代阶段才需要的外法向。这里直接由数学边和
    # 参数区间批量生成端点，避免 materialize 额外复制 Sx2 normals 和构造临时对象；
    # 全部数组仍按全局 segment 顺序对齐，因此 owner 与 CSR membership 语义不变。
    edge_ids = segments.edge_ids
    vertices = segments.contours.vertices
    edge_starts = vertices[edge_ids].astype(np.float64)
    vectors = (vertices[segments.edge_next_ids[edge_ids]] - vertices[edge_ids]).astype(np.float64)
    starts = edge_starts + vectors * segments.t0[:, None]
    ends = edge_starts + vectors * segments.t1[:, None]
    # 端点生成后立即释放两张 Sx2 中间表；否则 Python 局部引用会一直保留到 CSR
    # 构造结束，使峰值内存反而高于旧 materialize 路径。
    del edge_starts, vectors
    midpoints = (starts + ends) * 0.5
    owners = grid.locate_points(midpoints)
    halo = grid.halo_dbu
    left = np.minimum(starts[:, 0], ends[:, 0]) - halo
    right = np.maximum(starts[:, 0], ends[:, 0]) + halo
    bottom = np.minimum(starts[:, 1], ends[:, 1]) - halo
    top = np.maximum(starts[:, 1], ends[:, 1]) + halo
    # 后续只依赖 owner 和四条 bbox 边界，尽早释放三张 Sx2 数组，把准备阶段峰值
    # 限制在 CSR 展开本身，而不是让几何临时量与 membership 临时量重叠常驻。
    del starts, ends, midpoints
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
    membership_total = int(np.sum(membership_counts, dtype=np.int64))
    int32_limit = int(np.iinfo(np.int32).max)
    if grid.core_count > int32_limit:
        raise OverflowError("core 数量超过紧凑 int32 owner 容量")
    if max_memberships is not None and (
            not isinstance(max_memberships, int) or isinstance(max_memberships, bool) or
            max_memberships < 0):
        raise ValueError("max_memberships 必须是非负整数或 None")
    limit = int32_limit if max_memberships is None else min(max_memberships, int32_limit)
    # 必须在 np.repeat/arange/argsort 之前拒绝；这些数组会同时达到 membership
    # 数量级，若先分配再由 Problem 校验，容量保护已经失去意义。
    if membership_total > limit:
        raise MemoryError(
            f"membership 数量 {membership_total:,} 超过构造上限 {limit:,}")
    membership_offsets = np.empty(segments.segment_count + 1, dtype=np.int64)
    membership_offsets[0] = 0
    np.cumsum(membership_counts, out=membership_offsets[1:])
    members = np.repeat(np.arange(segments.segment_count, dtype=np.int32), membership_counts)
    local = np.arange(membership_total, dtype=np.int64) - np.repeat(
        membership_offsets[:-1], membership_counts)
    columns = ix0[members] + local % x_spans[members]
    rows = iy0[members] + local // x_spans[members]
    core_indices = rows * grid.column_count + columns
    order = np.argsort(core_indices, kind="stable")
    sorted_cores = core_indices[order]
    core_offsets = np.empty(grid.core_count + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(np.bincount(sorted_cores, minlength=grid.core_count), out=core_offsets[1:])
    return owners, core_offsets, members[order]
