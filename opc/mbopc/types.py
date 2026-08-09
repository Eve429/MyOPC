"""MB-OPC 紧凑控制边段、物化几何和配置数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from geometry import ContourBatch, EdgeBatch
from opc.common import BoundarySampleTemplate, CoreSpec, PhysicalMask
from opc.errors import OPCError

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
UIntArray = NDArray[np.uint64]
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
    """按需物化的控制边段端点、外法向和长度。"""

    segment_indices: Int32Array
    starts: FloatArray
    ends: FloatArray
    normals: FloatArray
    lengths: FloatArray

    def __post_init__(self) -> None:
        """校验所有几何数组与 segment 索引逐项对齐。"""
        indices = _vector(self.segment_indices, np.dtype(np.int32), "segment_indices")
        starts = _matrix(self.starts, np.dtype(np.float64), 2, "starts")
        ends = _matrix(self.ends, np.dtype(np.float64), 2, "ends")
        normals = _matrix(self.normals, np.dtype(np.float64), 2, "normals")
        lengths = _vector(self.lengths, np.dtype(np.float64), "lengths")
        if any(len(array) != len(indices) for array in (starts, ends, normals, lengths)):
            raise ValueError("materialized segment arrays must have equal length")
        object.__setattr__(self, "segment_indices", indices)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "lengths", lengths)


@dataclass(frozen=True, slots=True)
class SegmentBatch:
    """通过数学边索引和参数区间保存控制边段，避免重复几何元数据。"""

    contours: ContourBatch
    edges: EdgeBatch
    edge_lengths: FloatArray
    edge_normals: FloatArray
    edge_keys: UIntArray
    edge_segment_offsets: IntArray
    ring_segment_offsets: IntArray
    edge_ids: Int32Array
    t0: FloatArray
    t1: FloatArray
    fragment_indices: Int32Array
    fragment_counts: Int32Array
    keys: UIntArray
    _key_order: IntArray = field(init=False, repr=False)
    _sorted_tokens: UIntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """规范化紧凑数组，并一次性建立后续更新复用的 key 查找索引。"""
        edge_lengths = _vector(self.edge_lengths, np.dtype(np.float64), "edge_lengths")
        edge_normals = _matrix(self.edge_normals, np.dtype(np.float64), 2, "edge_normals")
        edge_keys = _matrix(self.edge_keys, np.dtype(np.uint64), 2, "edge_keys")
        edge_offsets = _vector(self.edge_segment_offsets, np.dtype(np.int64),
                               "edge_segment_offsets")
        ring_offsets = _vector(self.ring_segment_offsets, np.dtype(np.int64),
                               "ring_segment_offsets")
        edge_ids = _vector(self.edge_ids, np.dtype(np.int32), "edge_ids")
        t0 = _vector(self.t0, np.dtype(np.float64), "t0")
        t1 = _vector(self.t1, np.dtype(np.float64), "t1")
        fragment_indices = _vector(self.fragment_indices, np.dtype(np.int32),
                                   "fragment_indices")
        fragment_counts = _vector(self.fragment_counts, np.dtype(np.int32), "fragment_counts")
        keys = _matrix(self.keys, np.dtype(np.uint64), 2, "keys")
        edge_count, segment_count = self.edges.edge_count, len(edge_ids)
        if (len(edge_lengths) != edge_count or len(edge_normals) != edge_count or
                len(edge_keys) != edge_count):
            raise ValueError("edge-level metadata must match mathematical edge count")
        if (len(edge_offsets) != edge_count + 1 or edge_offsets[0] != 0 or
                edge_offsets[-1] != segment_count or np.any(np.diff(edge_offsets) < 1)):
            raise ValueError("every mathematical edge must own at least one segment")
        if (len(ring_offsets) != self.contours.ring_count + 1 or ring_offsets[0] != 0 or
                ring_offsets[-1] != segment_count or np.any(np.diff(ring_offsets) < 1)):
            raise ValueError("every contour ring must own at least one segment")
        fields = (t0, t1, fragment_indices, fragment_counts, keys)
        if any(len(array) != segment_count for array in fields):
            raise ValueError("segment-level metadata vectors must have equal length")
        if (segment_count and (np.any(edge_ids < 0) or np.any(edge_ids >= edge_count) or
                               np.any(t0 < 0.0) or np.any(t1 > 1.0) or np.any(t1 <= t0))):
            raise ValueError("segment edge IDs or parametric intervals are invalid")
        if np.any(edge_lengths <= 0.0) or not np.all(np.isfinite(edge_normals)):
            raise ValueError("mathematical edges must have finite positive geometry")
        # 128 位 key 保留在拓扑顺序中。查找索引只额外保存一个 int64 顺序和一个
        # uint64 token；这样每轮更新不建立 Python dict，也不重复复制完整双列 key。
        tokens = keys[:, 0] ^ ((keys[:, 1] << np.uint64(17)) |
                              (keys[:, 1] >> np.uint64(47)))
        order = np.argsort(tokens, kind="stable")
        sorted_tokens = np.ascontiguousarray(tokens[order])
        if len(sorted_tokens) > 1 and np.any(sorted_tokens[1:] == sorted_tokens[:-1]):
            raise OPCError("segment lookup token collision detected")
        for name, value in (
            ("edge_lengths", edge_lengths), ("edge_normals", edge_normals),
            ("edge_keys", edge_keys), ("edge_segment_offsets", edge_offsets),
            ("ring_segment_offsets", ring_offsets), ("edge_ids", edge_ids), ("t0", t0),
            ("t1", t1), ("fragment_indices", fragment_indices),
            ("fragment_counts", fragment_counts), ("keys", keys),
            ("_key_order", order.astype(np.int64, copy=False)),
            ("_sorted_tokens", sorted_tokens),
        ):
            object.__setattr__(self, name, value)

    @property
    def segment_count(self) -> int:
        """返回控制边段数量。"""
        return len(self.edge_ids)

    @property
    def persistent_nbytes(self) -> int:
        """返回当前批次新增常驻 NumPy 数组的总字节数。"""
        arrays = (self.edge_lengths, self.edge_normals, self.edge_keys,
                  self.edge_segment_offsets, self.ring_segment_offsets, self.edge_ids,
                  self.t0, self.t1, self.fragment_indices, self.fragment_counts, self.keys,
                  self._key_order, self._sorted_tokens)
        return sum(array.nbytes for array in arrays)

    def materialize(self, displacements: object | None = None,
                    indices: object | None = None) -> SegmentGeometry:
        """按需生成全部或指定 segment 的当前几何坐标。"""
        if indices is None:
            segment_indices = np.arange(self.segment_count, dtype=np.int32)
        else:
            segment_indices = _vector(indices, np.dtype(np.int32), "indices")
            if len(segment_indices) and (np.any(segment_indices < 0) or
                                         np.any(segment_indices >= self.segment_count)):
                raise IndexError("segment index is out of range")
        edge_ids = self.edge_ids[segment_indices]
        edge_starts = self.edges.starts[edge_ids].astype(np.float64)
        vectors = (self.edges.ends[edge_ids] - self.edges.starts[edge_ids]).astype(np.float64)
        starts = edge_starts + vectors * self.t0[segment_indices, None]
        ends = edge_starts + vectors * self.t1[segment_indices, None]
        normals = np.ascontiguousarray(self.edge_normals[edge_ids])
        if displacements is not None:
            values = _vector(displacements, np.dtype(np.float64), "displacements")
            if len(values) != self.segment_count or not np.all(np.isfinite(values)):
                raise ValueError("displacements must be a finite segment-aligned vector")
            shift = normals * values[segment_indices, None]
            starts += shift
            ends += shift
        lengths = self.edge_lengths[edge_ids] * (self.t1[segment_indices] -
                                                 self.t0[segment_indices])
        return SegmentGeometry(segment_indices, starts, ends, normals, lengths)

    def lookup_keys(self, keys: object) -> Int32Array:
        """通过稳定 128 位 key 批量查找 segment，未知 key 返回 -1。"""
        requested = _matrix(keys, np.dtype(np.uint64), 2, "keys")
        tokens = requested[:, 0] ^ ((requested[:, 1] << np.uint64(17)) |
                                    (requested[:, 1] >> np.uint64(47)))
        positions = np.searchsorted(self._sorted_tokens, tokens)
        result = np.full(len(requested), -1, dtype=np.int32)
        valid = positions < len(self._sorted_tokens)
        valid_indices = np.flatnonzero(valid)
        matched = self._sorted_tokens[positions[valid]] == tokens[valid]
        valid_indices = valid_indices[matched]
        candidates = self._key_order[positions[valid][matched]]
        exact = np.all(self.keys[candidates] == requested[valid_indices], axis=1)
        result[valid_indices[exact]] = candidates[exact].astype(np.int32)
        return result


@dataclass(frozen=True, slots=True)
class OwnershipBatch:
    """segment 的唯一 owner 及按 core 排列的稀疏 context membership。"""

    cores: tuple[CoreSpec, ...]
    owner_indices: Int32Array
    core_offsets: IntArray
    member_segment_indices: Int32Array

    def __post_init__(self) -> None:
        """校验 owner 范围和 core CSR 索引结构。"""
        owners = _vector(self.owner_indices, np.dtype(np.int32), "owner_indices")
        offsets = _vector(self.core_offsets, np.dtype(np.int64), "core_offsets")
        members = _vector(self.member_segment_indices, np.dtype(np.int32),
                          "member_segment_indices")
        if len({core.core_id for core in self.cores}) != len(self.cores):
            raise ValueError("core IDs must be unique")
        if (len(offsets) != len(self.cores) + 1 or offsets[0] != 0 or
                offsets[-1] != len(members) or np.any(np.diff(offsets) < 0)):
            raise ValueError("core membership offsets are invalid")
        if len(owners) and (np.any(owners < -1) or np.any(owners >= len(self.cores))):
            raise ValueError("owner index is out of range")
        if len(members) and np.any(members < 0):
            raise ValueError("member segment indices must be non-negative")
        object.__setattr__(self, "owner_indices", owners)
        object.__setattr__(self, "core_offsets", offsets)
        object.__setattr__(self, "member_segment_indices", members)

    def segments_for_core(self, core_index: int) -> Int32Array:
        """返回指定 core 的全部 ownership 和 halo context segment 索引视图。"""
        if core_index < 0 or core_index >= len(self.cores):
            raise IndexError("core index is out of range")
        start, end = self.core_offsets[core_index:core_index + 2]
        return self.member_segment_indices[start:end]


@dataclass(frozen=True, slots=True)
class SegmentUpdateBatch:
    """一个或多个 core 提交的绝对法向位移更新。"""

    keys: UIntArray
    source_core_indices: Int32Array
    normal_displacements: FloatArray

    def __post_init__(self) -> None:
        """规范化更新字段，并在归属匹配前拒绝非有限数值。"""
        keys = _matrix(self.keys, np.dtype(np.uint64), 2, "keys")
        sources = _vector(self.source_core_indices, np.dtype(np.int32),
                          "source_core_indices")
        values = _vector(self.normal_displacements, np.dtype(np.float64),
                         "normal_displacements")
        if len(keys) != len(sources) or len(keys) != len(values):
            raise ValueError("segment update fields must have equal length")
        if not np.all(np.isfinite(values)):
            raise ValueError("segment updates must be finite")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "source_core_indices", sources)
        object.__setattr__(self, "normal_displacements", values)


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """归属校验后与全局 segment 对齐的位移及脏对象索引。"""

    displacements: FloatArray
    changed_segment_indices: Int32Array
    dirty_polygon_ids: Int32Array


@dataclass(frozen=True, slots=True)
class MBOPCProblem:
    """多轮 MB-OPC 可重复使用的参考边界、索引和采样模板。"""

    physical_mask: PhysicalMask
    config: FragmentationConfig
    segments: SegmentBatch
    ownership: OwnershipBatch
    sample_template: BoundarySampleTemplate
