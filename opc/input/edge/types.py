"""MB-OPC 紧凑控制边段、物化几何和配置数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from geometry import ContourBatch
from opc.input import PhysicalMask, RectilinearCoreGrid

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
FloatArray = NDArray[np.float64]


def _vector(value: object, dtype: np.dtype, name: str) -> NDArray:
    """把一维字段转换为指定类型的连续数组。"""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _matrix(value: object, dtype: np.dtype, columns: int, name: str) -> NDArray:
    """把二维字段转换为固定列数的连续数组。"""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    return array


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
        starts = _matrix(self.starts, np.dtype(np.float64), 2, "starts")
        ends = _matrix(self.ends, np.dtype(np.float64), 2, "ends")
        normals = _matrix(self.normals, np.dtype(np.float64), 2, "normals")
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
        next_ids = _vector(self.edge_next_ids, np.dtype(np.int32), "edge_next_ids")
        polygon_ids = _vector(self.edge_polygon_ids, np.dtype(np.int32), "edge_polygon_ids")
        edge_normals = _matrix(self.edge_normals, np.dtype(np.float64), 2, "edge_normals")
        ring_offsets = _vector(self.ring_segment_offsets, np.dtype(np.int64),
                               "ring_segment_offsets")
        edge_ids = _vector(self.edge_ids, np.dtype(np.int32), "edge_ids")
        t0 = _vector(self.t0, np.dtype(np.float64), "t0")
        t1 = _vector(self.t1, np.dtype(np.float64), "t1")
        edge_count, segment_count = len(self.contours.vertices), len(edge_ids)
        if len(next_ids) != edge_count or len(polygon_ids) != edge_count or len(edge_normals) != edge_count:
            raise ValueError("edge-level metadata must match mathematical edge count")
        expected_next = np.arange(edge_count, dtype=np.int32) + 1
        if edge_count:
            expected_next[self.contours.ring_offsets[1:] - 1] = self.contours.ring_offsets[:-1]
        ring_polygon_ids = np.repeat(
            np.arange(self.contours.polygon_count, dtype=np.int32),
            np.diff(self.contours.polygon_ring_offsets))
        expected_polygons = np.repeat(ring_polygon_ids, np.diff(self.contours.ring_offsets))
        # 这两组缓存由轮廓拓扑唯一决定。加载离线问题时必须核对其内容，防止损坏
        # 缓存把一条边接到其他 ring，或让 tile 漏选/错选整个 Polygon。
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
                  self.ring_segment_offsets,
                  self.edge_ids, self.t0, self.t1)
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
            values = _vector(displacements, np.dtype(np.float64), "displacements")
            if len(values) != self.segment_count or not np.all(np.isfinite(values)):
                raise ValueError("displacements must be a finite segment-aligned vector")
            shift = normals * values[:, None]
            starts += shift
            ends += shift
        return SegmentGeometry(starts, ends, normals)


@dataclass(frozen=True, slots=True)
class MBOPCProblem:
    """多轮 MB-OPC 可重复使用的参考边界、重建配置和归属索引。"""

    physical_mask: PhysicalMask
    config: FragmentationConfig
    segments: SegmentBatch
    grid: RectilinearCoreGrid
    owner_indices: Int32Array
    core_offsets: IntArray
    member_segment_indices: Int32Array

    def __post_init__(self) -> None:
        """校验问题级 owner 与 context membership CSR 的交叉不变量。"""
        owners = _vector(self.owner_indices, np.dtype(np.int32), "owner_indices")
        offsets = _vector(self.core_offsets, np.dtype(np.int64), "core_offsets")
        members = _vector(self.member_segment_indices, np.dtype(np.int32),
                          "member_segment_indices")
        segment_count, core_count = self.segments.segment_count, self.grid.core_count
        if len(owners) != segment_count:
            raise ValueError("owner_indices must match segment count")
        if len(owners) and (np.any(owners < 0) or np.any(owners >= core_count)):
            raise ValueError("每个 segment 必须具有一个有效 owner")
        if (len(offsets) != core_count + 1 or offsets[0] != 0 or
                offsets[-1] != len(members) or np.any(np.diff(offsets) < 0)):
            raise ValueError("core membership offsets are invalid")
        if len(members) and (np.any(members < 0) or np.any(members >= segment_count)):
            raise ValueError("member_segment_indices 超出 segment 范围")
        object.__setattr__(self, "owner_indices", owners)
        object.__setattr__(self, "core_offsets", offsets)
        object.__setattr__(self, "member_segment_indices", members)

    @property
    def core_count(self) -> int:
        """返回规则网格中的 core 总数。"""
        return self.grid.core_count

    @property
    def persistent_nbytes(self) -> int:
        """返回问题中不重复计数的常驻 NumPy 数组字节数。"""
        contours = self.segments.contours
        arrays = (contours.vertices, contours.ring_offsets, contours.polygon_ring_offsets,
                  self.grid.x_cuts, self.grid.y_cuts, self.owner_indices,
                  self.core_offsets, self.member_segment_indices)
        return self.segments.persistent_nbytes + sum(array.nbytes for array in arrays)

    def segments_for_core(self, core_index: int) -> Int32Array:
        """返回指定 core 的 owner 与只读 halo context segment 索引视图。"""
        if core_index < 0 or core_index >= self.core_count:
            raise IndexError("core index is out of range")
        start, end = self.core_offsets[core_index:core_index + 2]
        return self.member_segment_indices[start:end]
