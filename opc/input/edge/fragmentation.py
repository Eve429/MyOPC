"""把物理数学边批量切分为带稳定身份的 MB-OPC 控制边段。"""

from __future__ import annotations

import numpy as np

from opc.input import PhysicalMask

from .types import FragmentationConfig, SegmentBatch


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """使用固定整数混合器生成跨进程、跨 Python 版本稳定的 uint64 数据。"""
    values = np.ascontiguousarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def _edge_keys(mask: PhysicalMask) -> np.ndarray:
    """从 layer 和有向数学边端点一次性派生 128 位边 key。"""
    edge_count = mask.edges.edge_count
    high = np.full(edge_count, np.uint64(0x243F6A8885A308D3), dtype=np.uint64)
    low = np.full(edge_count, np.uint64(0x13198A2E03707344), dtype=np.uint64)
    fields = (
        np.full(edge_count, mask.layer.layer, dtype=np.int64),
        np.full(edge_count, mask.layer.datatype, dtype=np.int64),
        mask.edges.starts[:, 0], mask.edges.starts[:, 1],
        mask.edges.ends[:, 0], mask.edges.ends[:, 1],
    )
    # 每个数学边只混合六个标量；segment key 随后仅组合边 key 和分段序号，避免
    # 对数十万 segment 重复散列四个坐标。负坐标按 int64 二进制位稳定解释。
    for index, field in enumerate(fields):
        bits = np.ascontiguousarray(field, dtype=np.int64).view(np.uint64)
        high = _splitmix64(high ^ _splitmix64(bits + np.uint64(index * 2 + 1)))
        low = _splitmix64(low ^ _splitmix64(bits + np.uint64(index * 2 + 2)))
    return np.column_stack((high, low))


def _outward_normals(mask: PhysicalMask, lengths: np.ndarray) -> np.ndarray:
    """根据环绕向和 hole 标志解析计算从材料指向空区的单位法向。"""
    starts = mask.edges.starts.astype(np.float64)
    ends = mask.edges.ends.astype(np.float64)
    vectors = ends - starts
    cross = starts[:, 0] * ends[:, 1] - starts[:, 1] * ends[:, 0]
    area2 = np.bincount(mask.edges.ring_ids, weights=cross,
                        minlength=mask.contours.ring_count)
    ccw = area2 > 0.0
    # 外轮廓顺时针时材料位于边右侧，外法向是左法向；hole 逆时针时材料位于
    # 环外侧，指向孔洞同样是左法向。其余两种绕向对称使用右法向。
    use_left = ccw[mask.edges.ring_ids] == mask.edges.is_hole
    left = np.column_stack((-vectors[:, 1], vectors[:, 0])) / lengths[:, None]
    right = -left
    return np.ascontiguousarray(np.where(use_left[:, None], left, right))


def fragment_edges(mask: PhysicalMask, config: FragmentationConfig) -> SegmentBatch:
    """按角部短段和均衡中段策略向量化切分全部物理数学边。"""
    vectors = (mask.edges.ends - mask.edges.starts).astype(np.float64)
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
    fragment_indices = ordinal64.astype(np.int32)
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
    edge_keys = _edge_keys(mask)
    ordinal_bits = fragment_indices.astype(np.uint64)
    count_bits = fragment_counts.astype(np.uint64)
    keys = np.empty((segment_count, 2), dtype=np.uint64)
    keys[:, 0] = _splitmix64(edge_keys[edge_ids, 0] ^ ordinal_bits ^
                             (count_bits << np.uint64(32)))
    keys[:, 1] = _splitmix64(edge_keys[edge_ids, 1] ^ count_bits ^
                             (ordinal_bits << np.uint64(32)))
    # 数学边在 ContourBatch 中与顶点一一对应且按 ring 连续，因此可直接用原始
    # ring_offsets 索引 edge_offsets，避免为每个 segment 重复保存 ring ID。
    ring_offsets = edge_offsets[mask.contours.ring_offsets]
    # ordinal/count 已编入稳定 key，分段参数已保存为 t0/t1；构建完成后
    # 不再常驻两列重复整数，使每轮 MB-OPC 保持更小的内存工作集。
    return SegmentBatch(mask.contours, mask.edges, lengths,
                        _outward_normals(mask, lengths), edge_keys, edge_offsets,
                        ring_offsets, edge_ids, t0, t1, keys)
