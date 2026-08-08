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
    return MappingProxyType({layer: _extract_layer(layer, region)
                             for layer, region in batch.regions.items()})


def _extract_layer(layer: LayerSpec, region: kdb.Region) -> ContourBatch:
    """仅把已经局部化的单层 Region 转换为 CSR 风格的外轮廓/孔洞环。"""
    rings: list[np.ndarray] = []
    polygon_ids: list[int] = []
    holes: list[bool] = []
    for polygon_id, polygon in enumerate(region):
        rings.append(np.asarray([(point.x, point.y) for point in polygon.each_point_hull()],
                                dtype=np.int64))
        polygon_ids.append(polygon_id)
        holes.append(False)
        for hole_index in range(polygon.holes()):
            rings.append(np.asarray([(point.x, point.y)
                                     for point in polygon.each_point_hole(hole_index)],
                                    dtype=np.int64))
            polygon_ids.append(polygon_id)
            holes.append(True)
    lengths = np.asarray([len(ring) for ring in rings], dtype=np.int64)
    offsets = np.empty(len(rings) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    vertices = np.concatenate(rings) if rings else np.empty((0, 2), dtype=np.int64)
    return ContourBatch(layer, vertices, offsets,
                        np.asarray(polygon_ids, dtype=np.int64), np.asarray(holes, dtype=np.bool_))


def contours_to_region(contours: ContourBatch) -> kdb.Region:
    """在保留 Polygon 孔洞拓扑的前提下重建原生 Region。"""
    region = kdb.Region()
    polygon_ids = sorted({int(value) for value in contours.ring_polygon_ids})
    for polygon_id in polygon_ids:
        ring_ids = np.flatnonzero(contours.ring_polygon_ids == polygon_id)
        hull_ids = [index for index in ring_ids if not contours.ring_is_hole[index]]
        if len(hull_ids) != 1:
            raise ValueError(f"polygon {polygon_id} must have exactly one hull")
        polygon = kdb.Polygon(_ring_points(contours, hull_ids[0]))
        for ring_id in ring_ids:
            if contours.ring_is_hole[ring_id]:
                polygon.insert_hole(_ring_points(contours, int(ring_id)))
        region.insert(polygon)
    return region


def _ring_points(contours: ContourBatch, ring_id: int) -> list[kdb.Point]:
    """仅在明确的原生重建边界上，把一个环转换为 KLayout Point。"""
    start, end = contours.ring_offsets[ring_id:ring_id + 2]
    return [kdb.Point(int(x), int(y)) for x, y in contours.vertices[start:end]]
