"""原生 Region 与连续轮廓数组之间的一次性转换。"""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from layout.types import LayerSpec, RegionBatch




@dataclass(frozen=True, slots=True)
class ContourBatch:
    """用两级 CSR 连续数组保存 Polygon、轮廓环和整数顶点。"""

    # vertices = [[0,0], [10,0], [10,10], [0,10], [20,0], [40,0], [40,20], [20,20], [25,5], [25,15], [35,15], [35,5]]
    # ring_offsets = [0, 4, 8, 12]
    # polygon_ring_offsets = [0, 1, 3]
    
    vertices: NDArray[np.int64] # 所有 ring 的顶点连续存储
    ring_offsets: NDArray[np.int64] # 每个 ring 在 vertices 中的起止位置
    polygon_ring_offsets: NDArray[np.int64] # 每个 polygon 在 ring 集合中的起止位置

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
        # 每个 Polygon 至少含一个 ring；首 ring 为 hull，后续 ring 为 hole。
        # 连续范围已经能表达 polygon_id/is_hole，避免为每个 ring 重复常驻元数据。
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


def extract_contours(batch: RegionBatch) -> Mapping[LayerSpec, ContourBatch]:
    """仅物化一次局部 Region 边界，供后续多轮数值 OPC 重复使用。"""
    return MappingProxyType({layer: extract_contour(region)
                             for layer, region in batch.regions.items()})


def extract_contour(region: kdb.Region) -> ContourBatch:
    """把一个原生 Region 转换为 Polygon/Ring 两级 CSR 轮廓。"""
    # KLayout 迭代器只能逐 ring 给出点；直接追加到 64 位连续缓冲，可避免为每个
    # 小 ring 创建 Python 列表和 NumPy 数组，最后也不需要一次峰值很高的 concatenate。
    # `np.frombuffer` 返回的数组会持有缓冲引用，ContourBatch 的连续化不会复制已满足
    # dtype/布局的视图，因此常驻内存只有最终三组 CSR 数据。
    vertices_buffer = array("q")
    ring_offsets_buffer = array("q", [0])
    polygon_offsets_buffer = array("q", [0])
    ring_count = vertex_count = 0
    for polygon in region:
        for point in polygon.each_point_hull():
            vertices_buffer.extend((point.x, point.y))
            vertex_count += 1
        ring_offsets_buffer.append(vertex_count)
        ring_count += 1
        for hole_index in range(polygon.holes()):
            for point in polygon.each_point_hole(hole_index):
                vertices_buffer.extend((point.x, point.y))
                vertex_count += 1
            ring_offsets_buffer.append(vertex_count)
            ring_count += 1
        # KLayout 已经给出 Polygon 边界；每个范围首 ring 是 hull，剩余 ring 是 hole。
        # 只记录范围端点即可恢复 polygon_id 和 hole 属性，避免逐 ring 重复保存两列元数据。
        polygon_offsets_buffer.append(ring_count)
    vertices = np.frombuffer(vertices_buffer, dtype=np.int64).reshape(-1, 2)
    ring_offsets = np.frombuffer(ring_offsets_buffer, dtype=np.int64)
    polygon_offsets = np.frombuffer(polygon_offsets_buffer, dtype=np.int64)
    return ContourBatch(vertices, ring_offsets, polygon_offsets)


def contours_to_region(contours: ContourBatch) -> kdb.Region:
    """在保留 Polygon 孔洞拓扑的前提下重建原生 Region。"""
    region = kdb.Region()
    for polygon_id in range(contours.polygon_count):
        ring_start, ring_end = contours.polygon_ring_offsets[polygon_id:polygon_id + 2]
        polygon = kdb.Polygon(_ring_points(contours, int(ring_start)))
        for ring_id in range(int(ring_start) + 1, int(ring_end)):
            polygon.insert_hole(_ring_points(contours, ring_id))
        region.insert(polygon)
    return region


def _ring_points(contours: ContourBatch, ring_id: int) -> list[kdb.Point]:
    """仅在明确的原生重建边界上，把一个环转换为 KLayout Point。"""
    start, end = contours.ring_offsets[ring_id:ring_id + 2]
    return [kdb.Point(int(x), int(y)) for x, y in contours.vertices[start:end]]
