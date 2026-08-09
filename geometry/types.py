"""紧凑的 NumPy 几何数据契约以及不可变 Patch 描述。"""

from __future__ import annotations

from dataclasses import dataclass

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from layout.types import DbuBox, LayerSpec

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _bool_vector(value: object, name: str) -> BoolArray:
    """把布尔元数据规范化为连续的一维数组。"""
    array = np.ascontiguousarray(value, dtype=np.bool_)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


@dataclass(frozen=True, slots=True)
class ContourBatch:
    """使用连续整数数组保存单个 Layer 中的全部 Polygon 环。"""

    layer: LayerSpec
    vertices: IntArray
    ring_offsets: IntArray
    ring_polygon_ids: IntArray
    ring_is_hole: BoolArray

    def __post_init__(self) -> None:
        """规范化内存布局，并校验 CSR 风格的环编码是否自洽。"""
        vertices = np.ascontiguousarray(self.vertices, dtype=np.int64)
        offsets = np.ascontiguousarray(self.ring_offsets, dtype=np.int64)
        polygon_ids = np.ascontiguousarray(self.ring_polygon_ids, dtype=np.int64)
        holes = _bool_vector(self.ring_is_hole, "ring_is_hole")
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2)")
        if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("ring_offsets must be a vector starting at zero")
        if offsets[-1] != len(vertices) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("ring_offsets must be monotonic and end at vertex count")
        ring_count = len(offsets) - 1
        if len(polygon_ids) != ring_count or len(holes) != ring_count:
            raise ValueError("ring metadata length must equal ring count")
        if ring_count and np.any(np.diff(offsets) < 3):
            raise ValueError("every ring must contain at least three vertices")
        if np.any(polygon_ids < 0):
            raise ValueError("polygon IDs must be non-negative")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "ring_offsets", offsets)
        object.__setattr__(self, "ring_polygon_ids", polygon_ids)
        object.__setattr__(self, "ring_is_hole", holes)

    @property
    def ring_count(self) -> int:
        """返回外轮廓与孔洞环的总数。"""
        return len(self.ring_offsets) - 1

    @property
    def polygon_count(self) -> int:
        """返回当前局部批次的 Polygon 数量。"""
        return 0 if not len(self.ring_polygon_ids) else int(self.ring_polygon_ids.max()) + 1


@dataclass(frozen=True, slots=True)
class EdgeBatch:
    """单个 Layer 的数学边集合；OPC 分段策略仍由算法层负责。"""

    layer: LayerSpec
    starts: IntArray
    ends: IntArray
    ring_ids: IntArray
    polygon_ids: IntArray
    is_hole: BoolArray

    def __post_init__(self) -> None:
        """规范化数组，使其可直接交给 NumPy、C++ 或 CUDA 使用。"""
        starts = np.ascontiguousarray(self.starts, dtype=np.int64)
        ends = np.ascontiguousarray(self.ends, dtype=np.int64)
        ring_ids = np.ascontiguousarray(self.ring_ids, dtype=np.int64)
        polygon_ids = np.ascontiguousarray(self.polygon_ids, dtype=np.int64)
        holes = _bool_vector(self.is_hole, "is_hole")
        if starts.ndim != 2 or starts.shape[1] != 2 or ends.shape != starts.shape:
            raise ValueError("starts and ends must have equal shape (N, 2)")
        count = len(starts)
        if any(array.ndim != 1 or len(array) != count for array in (ring_ids, polygon_ids, holes)):
            raise ValueError("edge metadata vectors must match edge count")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)
        object.__setattr__(self, "ring_ids", ring_ids)
        object.__setattr__(self, "polygon_ids", polygon_ids)
        object.__setattr__(self, "is_hole", holes)

    @property
    def edge_count(self) -> int:
        """返回数学边数量。"""
        return len(self.starts)

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """一条顺序稳定、便于测试和报告的校验问题。"""

    code: str
    message: str
    layer: LayerSpec
    indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """不修改输入数据的几何校验汇总结果。"""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """未发现任何问题时返回 True。"""
        return not self.issues


@dataclass(frozen=True, slots=True)
class GeometryPatch:
    """由一个 core 矩形负责、位于全局坐标系的单层结果。"""

    patch_id: str
    layer: LayerSpec
    region: kdb.Region
    ownership_box: DbuBox

    def __post_init__(self) -> None:
        """尽早拒绝空标识或非 KLayout Region 数据。"""
        if not self.patch_id.strip():
            raise ValueError("patch_id must be non-empty")
        if not isinstance(self.region, kdb.Region):
            raise TypeError("region must be a KLayout Region")
