"""紧凑的 NumPy 几何数据契约以及不可变 Patch 描述。"""

from __future__ import annotations

from dataclasses import dataclass

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from layout.types import DbuBox, LayerSpec

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ContourBatch:
    """用两级 CSR 连续数组保存 Polygon、轮廓环和整数顶点。"""

    vertices: IntArray
    ring_offsets: IntArray
    polygon_ring_offsets: IntArray

    def __post_init__(self) -> None:
        """规范化内存布局，并校验 Polygon/Ring 两级 CSR 编码。"""
        vertices = np.ascontiguousarray(self.vertices, dtype=np.int64)
        ring_offsets = np.ascontiguousarray(self.ring_offsets, dtype=np.int64)
        polygon_offsets = np.ascontiguousarray(self.polygon_ring_offsets, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2)")
        if ring_offsets.ndim != 1 or len(ring_offsets) == 0 or ring_offsets[0] != 0:
            raise ValueError("ring_offsets must be a vector starting at zero")
        if ring_offsets[-1] != len(vertices) or np.any(np.diff(ring_offsets) < 0):
            raise ValueError("ring_offsets must be monotonic and end at vertex count")
        ring_count = len(ring_offsets) - 1
        if ring_count and np.any(np.diff(ring_offsets) < 3):
            raise ValueError("every ring must contain at least three vertices")
        if (polygon_offsets.ndim != 1 or len(polygon_offsets) == 0 or
                polygon_offsets[0] != 0):
            raise ValueError("polygon_ring_offsets must be a vector starting at zero")
        # 每个 Polygon 必须至少有一个 ring；约定首 ring 为 hull，后续 ring 为 hole。
        # 该结构直接表达原先 polygon_id/is_hole 两组重复元数据，避免为每个 ring
        # 常驻保存可由连续范围推导的信息。
        if (polygon_offsets[-1] != ring_count or
                (len(polygon_offsets) > 1 and np.any(np.diff(polygon_offsets) < 1))):
            raise ValueError("every polygon must own at least one contour ring")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "ring_offsets", ring_offsets)
        object.__setattr__(self, "polygon_ring_offsets", polygon_offsets)

    @property
    def ring_count(self) -> int:
        """返回外轮廓与孔洞环的总数。"""
        return len(self.ring_offsets) - 1

    @property
    def polygon_count(self) -> int:
        """返回当前局部批次的 Polygon 数量。"""
        return len(self.polygon_ring_offsets) - 1

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
