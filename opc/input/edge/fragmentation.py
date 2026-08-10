"""把物理数学边批量切分为按全局拓扑顺序排列的 MB-OPC 控制边段。"""

from __future__ import annotations

import numpy as np

from geometry import ContourBatch

from .types import FragmentationConfig, SegmentBatch


def _outward_normals(contours: ContourBatch, edge_next_ids: np.ndarray,
                     edge_ring_ids: np.ndarray, ring_is_hole: np.ndarray,
                     lengths: np.ndarray) -> np.ndarray:
    """根据环绕向和 hole 标志解析计算从材料指向空区的单位法向。"""
    starts = contours.vertices.astype(np.float64)
    ends = contours.vertices[edge_next_ids].astype(np.float64)
    vectors = ends - starts
    cross = starts[:, 0] * ends[:, 1] - starts[:, 1] * ends[:, 0]
    area2 = np.bincount(edge_ring_ids, weights=cross, minlength=contours.ring_count)
    ccw = area2 > 0.0
    # 外轮廓顺时针时材料位于边右侧，外法向是左法向；hole 逆时针时材料位于
    # 环外侧，指向孔洞同样是左法向。其余两种绕向对称使用右法向。
    use_left = ccw[edge_ring_ids] == ring_is_hole[edge_ring_ids]
    left = np.column_stack((-vectors[:, 1], vectors[:, 0])) / lengths[:, None]
    right = -left
    return np.ascontiguousarray(np.where(use_left[:, None], left, right))


def fragment_edges(contours: ContourBatch, config: FragmentationConfig) -> SegmentBatch:
    """按角部短段和均衡中段策略向量化切分全部物理数学边。"""
    edge_count = len(contours.vertices)
    edge_next_ids = np.arange(edge_count, dtype=np.int32) + 1
    if edge_count:
        edge_next_ids[contours.ring_offsets[1:] - 1] = contours.ring_offsets[:-1]
    ring_lengths = np.diff(contours.ring_offsets)
    edge_ring_ids = np.repeat(np.arange(contours.ring_count, dtype=np.int32), ring_lengths)
    ring_polygon_ids = np.repeat(
        np.arange(contours.polygon_count, dtype=np.int32),
        np.diff(contours.polygon_ring_offsets))
    edge_polygon_ids = np.repeat(ring_polygon_ids, ring_lengths)
    ring_is_hole = np.ones(contours.ring_count, dtype=np.bool_)
    ring_is_hole[contours.polygon_ring_offsets[:-1]] = False
    vectors = (contours.vertices[edge_next_ids] - contours.vertices).astype(np.float64)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    if np.any(lengths <= 0.0):
        raise ValueError("physical boundary contains zero-length edges")
    maximum, corner = config.max_segment_length_dbu, config.corner_length_dbu
    counts = np.ceil(lengths / maximum).astype(np.int64)
    long_edges = lengths > 2.0 * maximum
    counts[long_edges] = 2 + np.ceil((lengths[long_edges] - 2.0 * corner) /
                                     maximum).astype(np.int64)
    if len(counts) and (int(counts.max()) > np.iinfo(np.int32).max or
                        int(counts.sum()) > np.iinfo(np.int32).max):
        raise OverflowError("segment count exceeds compact int32 index capacity")
    edge_offsets = np.empty(len(counts) + 1, dtype=np.int64)
    edge_offsets[0] = 0
    np.cumsum(counts, out=edge_offsets[1:])
    segment_count = int(edge_offsets[-1])
    edge_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    ordinal64 = np.arange(segment_count, dtype=np.int64) - np.repeat(edge_offsets[:-1], counts)
    fragment_counts = np.repeat(counts.astype(np.int32), counts)
    t0 = ordinal64 / fragment_counts
    t1 = (ordinal64 + 1) / fragment_counts
    long_segments = long_edges[edge_ids]
    if np.any(long_segments):
        selected_edges = edge_ids[long_segments]
        selected_ordinals = ordinal64[long_segments]
        selected_counts = fragment_counts[long_segments].astype(np.int64)
        middle_counts = selected_counts - 2
        corner_fraction = corner / lengths[selected_edges]
        middle_fraction = (1.0 - 2.0 * corner_fraction) / middle_counts
        first = selected_ordinals == 0
        last = selected_ordinals == selected_counts - 1
        middle_index = selected_ordinals - 1
        long_t0 = corner_fraction + middle_index * middle_fraction
        long_t1 = long_t0 + middle_fraction
        long_t0[first], long_t1[first] = 0.0, corner_fraction[first]
        long_t0[last], long_t1[last] = 1.0 - corner_fraction[last], 1.0
        t0[long_segments], t1[long_segments] = long_t0, long_t1
    # 数学边在 ContourBatch 中与顶点一一对应且按 ring 连续，因此可直接用原始
    # ring_offsets 索引构造期的 edge_offsets，避免为每个 segment 重复保存 ring ID。
    # edge_offsets 和 lengths 到此完成使命，不进入常驻 SegmentBatch；诊断若需要
    # 分组或长度，可分别从 edge_ids 和物化端点向量化推导，不拖慢多轮热路径。
    ring_offsets = edge_offsets[contours.ring_offsets]
    normals = _outward_normals(
        contours, edge_next_ids, edge_ring_ids, ring_is_hole, lengths)
    return SegmentBatch(contours, edge_next_ids, edge_polygon_ids, normals,
                        ring_offsets, edge_ids, t0, t1)
