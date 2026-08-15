"""把物理数学边批量切分为按全局拓扑顺序排列的 MB-OPC 控制边段。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from geometry import ContourBatch
from opc.input._arrays import as_matrix, as_vector
from opc.input._fragmentation import count_edge_fragments
from opc.input.mask import MaskPolarity

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FragmentationConfig:
    """控制边段长度、允许位移和拐角重建的 DBU 配置。"""

    corner_length_dbu: float
    max_segment_length_dbu: float
    max_displacement_dbu: float
    miter_limit: float = 4.0

    def __post_init__(self) -> None:
        """统一浮点类型并拒绝会产生过短中段或无界位移的配置。"""
        names = ("corner_length_dbu", "max_segment_length_dbu",
                 "max_displacement_dbu", "miter_limit")
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value):
                raise TypeError(f"{name} must be a finite real number")
            object.__setattr__(self, name, float(value))
        if self.corner_length_dbu <= 0.0 or self.max_segment_length_dbu <= 0.0:
            raise ValueError("fragment lengths must be positive")
        if self.max_segment_length_dbu < 2.0 * self.corner_length_dbu:
            raise ValueError("max segment length must be at least twice corner length")
        if self.max_displacement_dbu < 0.0 or self.miter_limit < 1.0:
            raise ValueError("maximum displacement and miter limit are invalid")


@dataclass(frozen=True, slots=True)
class SegmentGeometry:
    """按需物化的控制边段端点和外法向。"""

    starts: FloatArray
    ends: FloatArray
    normals: FloatArray

    def __post_init__(self) -> None:
        """校验三个几何数组逐项对齐且使用连续浮点布局。"""
        starts = as_matrix(self.starts, np.dtype(np.float64), 2, "starts")
        ends = as_matrix(self.ends, np.dtype(np.float64), 2, "ends")
        normals = as_matrix(self.normals, np.dtype(np.float64), 2, "normals")
        if ends.shape != starts.shape or normals.shape != starts.shape:
            raise ValueError("materialized segment arrays must have equal length")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)
        object.__setattr__(self, "normals", normals)


@dataclass(frozen=True, slots=True)
class SegmentBatch:
    """通过数学边索引和参数区间保存控制边段，避免重复几何元数据。"""

    contours: ContourBatch
    edge_next_ids: Int32Array
    edge_polygon_ids: Int32Array
    edge_normals: FloatArray
    ring_segment_offsets: IntArray
    edge_ids: Int32Array
    t0: FloatArray
    t1: FloatArray

    def __post_init__(self) -> None:
        """规范化重建与迭代真正使用的紧凑数组并校验拓扑边界。"""
        next_ids = as_vector(self.edge_next_ids, np.dtype(np.int32), "edge_next_ids")
        polygon_ids = as_vector(self.edge_polygon_ids, np.dtype(np.int32), "edge_polygon_ids")
        edge_normals = as_matrix(self.edge_normals, np.dtype(np.float64), 2, "edge_normals")
        ring_offsets = as_vector(self.ring_segment_offsets, np.dtype(np.int64),
                                 "ring_segment_offsets")
        edge_ids = as_vector(self.edge_ids, np.dtype(np.int32), "edge_ids")
        t0 = as_vector(self.t0, np.dtype(np.float64), "t0")
        t1 = as_vector(self.t1, np.dtype(np.float64), "t1")
        edge_count, segment_count = len(self.contours.vertices), len(edge_ids)
        if (len(next_ids) != edge_count or len(polygon_ids) != edge_count or
                len(edge_normals) != edge_count):
            raise ValueError("edge-level metadata must match mathematical edge count")
        expected_next = np.arange(edge_count, dtype=np.int32) + 1
        if edge_count:
            expected_next[self.contours.ring_offsets[1:] - 1] = self.contours.ring_offsets[:-1]
        ring_polygon_ids = np.repeat(
            np.arange(self.contours.polygon_count, dtype=np.int32),
            np.diff(self.contours.polygon_ring_offsets))
        expected_polygons = np.repeat(ring_polygon_ids, np.diff(self.contours.ring_offsets))
        # 两组边级缓存必须与轮廓拓扑完全一致；离线恢复时同步检查，避免损坏
        # 缓存把边接到其他 ring，或让 tile 漏选、错选整个 Polygon。
        if (not np.array_equal(next_ids, expected_next) or
                not np.array_equal(polygon_ids, expected_polygons)):
            raise ValueError("edge caches do not match contour topology")
        if (len(ring_offsets) != self.contours.ring_count + 1 or ring_offsets[0] != 0 or
                ring_offsets[-1] != segment_count or np.any(np.diff(ring_offsets) < 1)):
            raise ValueError("every contour ring must own at least one segment")
        if len(t0) != segment_count or len(t1) != segment_count:
            raise ValueError("segment-level metadata vectors must have equal length")
        if (segment_count and (np.any(edge_ids < 0) or np.any(edge_ids >= edge_count) or
                               np.any(t0 < 0.0) or np.any(t1 > 1.0) or np.any(t1 <= t0))):
            raise ValueError("segment edge IDs or parametric intervals are invalid")
        if not np.all(np.isfinite(edge_normals)):
            raise ValueError("mathematical edges must have finite normals")
        for name, value in (
            ("edge_next_ids", next_ids), ("edge_polygon_ids", polygon_ids),
            ("edge_normals", edge_normals), ("ring_segment_offsets", ring_offsets),
            ("edge_ids", edge_ids), ("t0", t0), ("t1", t1),
        ):
            object.__setattr__(self, name, value)

    @property
    def segment_count(self) -> int:
        """返回控制边段数量。"""
        return len(self.edge_ids)

    @property
    def persistent_nbytes(self) -> int:
        """返回当前批次新增常驻 NumPy 数组的总字节数。"""
        arrays = (self.edge_next_ids, self.edge_polygon_ids, self.edge_normals,
                  self.ring_segment_offsets, self.edge_ids, self.t0, self.t1)
        return sum(array.nbytes for array in arrays)

    def materialize(self, displacements: object | None = None) -> SegmentGeometry:
        """按全局稳定顺序生成全部 segment 的当前端点和外法向。"""
        edge_ids = self.edge_ids
        vertices = self.contours.vertices
        edge_starts = vertices[edge_ids].astype(np.float64)
        vectors = (vertices[self.edge_next_ids[edge_ids]] - vertices[edge_ids]).astype(np.float64)
        starts = edge_starts + vectors * self.t0[:, None]
        ends = edge_starts + vectors * self.t1[:, None]
        normals = np.ascontiguousarray(self.edge_normals[edge_ids])
        if displacements is not None:
            values = as_vector(displacements, np.dtype(np.float64), "displacements")
            if len(values) != self.segment_count or not np.all(np.isfinite(values)):
                raise ValueError("displacements must be a finite segment-aligned vector")
            shift = normals * values[:, None]
            starts += shift
            ends += shift
        return SegmentGeometry(starts, ends, normals)


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


def fragment_edges(contours: ContourBatch, config: FragmentationConfig,
                   polarity: MaskPolarity | str = MaskPolarity.CLEAR) -> SegmentBatch:
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
    counts = count_edge_fragments(lengths, corner, maximum)
    long_edges = lengths > 2.0 * maximum
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
    normalized = polarity if isinstance(polarity, MaskPolarity) else MaskPolarity(polarity)
    # `_outward_normals` 从源多边形内部指向外部。clear 时这正是透光→不透光；
    # opaque 时透光位于外部，反向后仍保持公共法向不变量，迭代器无需极性分支。
    if normalized is MaskPolarity.OPAQUE:
        normals = np.ascontiguousarray(-normals)
    return SegmentBatch(contours, edge_next_ids, edge_polygon_ids, normals,
                        ring_offsets, edge_ids, t0, t1)
