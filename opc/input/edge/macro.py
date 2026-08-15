"""按 CPU macro 准备未裁剪候选边段及其 tile 归属。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from geometry import extract_contour
from layout import DbuBox, LayerSpec, RegionBatch
from opc.input import (
    MaskPolarity,
    PhysicalMask,
    RectilinearCoreGrid,
    normalize_physical_mask,
)

from .fragmentation import FragmentationConfig, SegmentBatch, fragment_edges


@dataclass(frozen=True, slots=True)
class MacroPreparation:
    """保存一个 macro 生命周期内的完整候选几何和局部稀疏索引。"""

    physical_mask: PhysicalMask
    segments: SegmentBatch
    active_segment_indices: NDArray[np.int32]
    active_owner_indices: NDArray[np.int32]
    core_indices: NDArray[np.int32]
    core_offsets: NDArray[np.int64]
    member_segment_indices: NDArray[np.int32]

    def segments_for_core(self, local_core_index: int) -> NDArray[np.int32]:
        """返回一个 tile 可读取的当前 Macro SegmentBatch 局部下标视图。"""
        if local_core_index < 0 or local_core_index >= len(self.core_indices):
            raise IndexError("macro local core index 超出范围")
        start, end = self.core_offsets[local_core_index:local_core_index + 2]
        return self.member_segment_indices[start:end]

    def owned_segments(self) -> NDArray[np.int32]:
        """按需返回 owner tile 落在当前 macro 内的局部 segment 下标。"""
        # owned 集合只在发布或诊断时需要；不常驻一份可由两个短向量推导的副本。
        # core 数通常远小于 segment 数，np.isin 在 C 端批量完成，不增加 Python 热循环。
        return self.active_segment_indices[np.isin(
            self.active_owner_indices, self.core_indices)]


def _contained_core_indices(grid: RectilinearCoreGrid,
                            box: DbuBox) -> NDArray[np.int32]:
    """返回 ownership box 完整落在 macro 内的全局 tile 索引。"""
    x0 = int(np.searchsorted(grid.x_cuts, box.left, side="left"))
    x1 = int(np.searchsorted(grid.x_cuts, box.right, side="left"))
    y0 = int(np.searchsorted(grid.y_cuts, box.bottom, side="left"))
    y1 = int(np.searchsorted(grid.y_cuts, box.top, side="left"))
    if (x0 >= len(grid.x_cuts) or x1 >= len(grid.x_cuts) or
            y0 >= len(grid.y_cuts) or y1 >= len(grid.y_cuts) or
            grid.x_cuts[x0] != box.left or grid.x_cuts[x1] != box.right or
            grid.y_cuts[y0] != box.bottom or grid.y_cuts[y1] != box.top):
        raise ValueError("macro 边界必须与 tile 切线严格对齐")
    columns = np.arange(x0, x1, dtype=np.int32)
    rows = np.arange(y0, y1, dtype=np.int32)
    return np.ascontiguousarray(
        (rows[:, None] * grid.column_count + columns[None, :]).reshape(-1))


def prepare_macro(
        batch: RegionBatch, layer: LayerSpec, config: FragmentationConfig,
        tile_grid: RectilinearCoreGrid, ownership_box: DbuBox,
        polarity: MaskPolarity | str = MaskPolarity.CLEAR, *,
        max_memberships: int | None = None) -> MacroPreparation:
    """从完整相交图形准备当前 macro 的真实边、全局 owner 和局部 membership。"""
    if batch.query_box.intersection(ownership_box) != ownership_box:
        raise ValueError("macro 物化 context 必须完整包含 ownership box")
    physical = normalize_physical_mask(batch, layer, polarity)
    # 未裁剪 occurrence 先在 KLayout Region 中统一合并，随后仅提取一次真实物理
    # 轮廓；macro 查询框从未参与布尔相交，所以它的四条边不会进入 SegmentBatch。
    segments = fragment_edges(extract_contour(physical.region), config, physical.polarity)
    # 这里只需要参考端点 bbox，不需要外法向。直接从紧凑数学边和参数区间
    # 生成两张 S×2 端点，避免 materialize 额外复制第三张 S×2 normals。
    edge_ids, vertices = segments.edge_ids, segments.contours.vertices
    edge_starts = vertices[edge_ids].astype(np.float64)
    vectors = (vertices[segments.edge_next_ids[edge_ids]] - vertices[edge_ids]).astype(
        np.float64)
    starts = edge_starts + vectors * segments.t0[:, None]
    ends = edge_starts + vectors * segments.t1[:, None]
    del edge_starts, vectors
    left = np.minimum(starts[:, 0], ends[:, 0])
    right = np.maximum(starts[:, 0], ends[:, 0])
    bottom = np.minimum(starts[:, 1], ends[:, 1])
    top = np.maximum(starts[:, 1], ends[:, 1])
    context = batch.query_box
    active_mask = ((right >= context.left) & (left <= context.right) &
                   (top >= context.bottom) & (bottom <= context.top))
    active = np.flatnonzero(active_mask).astype(np.int32)
    midpoints = (starts[active] + ends[active]) * 0.5
    owners = tile_grid.locate_points(midpoints)
    # 端点在 active bbox 与 owner 计算后不再需要，须在 membership 数组分配前
    # 释放。ROI 外真实边可以落入边缘 tile halo：owner=-1 表示固定只读上下文，
    # 仍保留 membership，但 owned_segments 永远不会发布它。
    del starts, ends, midpoints
    cores = _contained_core_indices(tile_grid, ownership_box)
    if (max_memberships is not None and
            (not isinstance(max_memberships, int) or isinstance(max_memberships, bool) or
             max_memberships < 0)):
        raise ValueError("max_memberships 必须是非负整数或 None")
    counts = np.empty(len(cores), dtype=np.int64)
    active_left, active_right = left[active], right[active]
    active_bottom, active_top = bottom[active], top[active]
    for local_index, core_index in enumerate(cores):
        tile_context = tile_grid.core(int(core_index)).context_box
        # 每个 tile 一次 NumPy 批量过滤，不逐 segment 进入 Python。membership 只
        # 覆盖当前 macro 的 tile，因此临时内存上界由 macro 活跃边段数控制。
        counts[local_index] = int(np.count_nonzero(
            (active_right >= tile_context.left) & (active_left <= tile_context.right) &
            (active_top >= tile_context.bottom) & (active_bottom <= tile_context.top)))
    membership_total = int(np.sum(counts, dtype=np.int64))
    limit = int(np.iinfo(np.int32).max)
    if max_memberships is not None:
        limit = min(limit, max_memberships)
    # 必须在创建每个 tile 的 selected 副本和最终拼接数组前拒绝；否则 macro
    # 预检虽报告安全上限，生产构造仍可能因布尔合并新增边而越过预算。
    if membership_total > limit:
        raise MemoryError(
            f"macro membership 数量 {membership_total:,} 超过构造上限 {limit:,}")
    offsets = np.empty(len(cores) + 1, dtype=np.int64); offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    members = np.empty(membership_total, dtype=np.int32)
    for local_index, core_index in enumerate(cores):
        tile_context = tile_grid.core(int(core_index)).context_box
        selected = active[
            (active_right >= tile_context.left) & (active_left <= tile_context.right) &
            (active_top >= tile_context.bottom) & (active_bottom <= tile_context.top)]
        members[offsets[local_index]:offsets[local_index + 1]] = selected
    # 唯一归属只以全局 tile owner 为真源；macro 仅选取一组完整 tile，不能再用
    # ownership_box 重算第二份归属，否则边界闭开规则会形成两个可能分歧的定义。
    return MacroPreparation(
        physical, segments, active, owners, cores, offsets, members)
