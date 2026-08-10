"""原生 Region 与连续轮廓数组之间的一次性转换。"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import klayout.db as kdb
import numpy as np

from layout.types import LayerSpec, RegionBatch

from .types import ContourBatch


def extract_contours(batch: RegionBatch) -> Mapping[LayerSpec, ContourBatch]:
    """仅物化一次局部 Region 边界，供后续多轮数值 OPC 重复使用。"""
    return MappingProxyType({layer: extract_contour(region)
                             for layer, region in batch.regions.items()})


def extract_contour(region: kdb.Region) -> ContourBatch:
    """把一个原生 Region 转换为 Polygon/Ring 两级 CSR 轮廓。"""
    rings: list[np.ndarray] = []
    polygon_ring_offsets = [0]
    for polygon in region:
        rings.append(np.asarray([(point.x, point.y) for point in polygon.each_point_hull()],
                                dtype=np.int64))
        for hole_index in range(polygon.holes()):
            rings.append(np.asarray([(point.x, point.y)
                                     for point in polygon.each_point_hole(hole_index)],
                                    dtype=np.int64))
        # KLayout 已经给出 Polygon 边界；每个范围首 ring 是 hull，剩余 ring 是 hole。
        # 只记录范围端点即可恢复 polygon_id 和 hole 属性，避免逐 ring 重复保存两列元数据。
        polygon_ring_offsets.append(len(rings))
    lengths = np.asarray([len(ring) for ring in rings], dtype=np.int64)
    offsets = np.empty(len(rings) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    vertices = np.concatenate(rings) if rings else np.empty((0, 2), dtype=np.int64)
    return ContourBatch(vertices, offsets, np.asarray(polygon_ring_offsets, dtype=np.int64))


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
