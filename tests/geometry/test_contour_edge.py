"""原生 Region 与两级 CSR 连续轮廓数组的转换测试。"""

import klayout.db as kdb
import numpy as np

from geometry import contours_to_region, extract_contour, extract_contours
from layout import LayerSpec

from .helpers import region_batch


def test_polygon_hole_round_trip_and_nested_offsets() -> None:
    """外轮廓与孔洞经过数组转换和原生重建后必须完全一致。"""
    layer = LayerSpec(7, 1)
    donut = kdb.Region(kdb.Box(0, 0, 100, 100)) - kdb.Region(kdb.Box(20, 20, 80, 80))
    contours = extract_contours(region_batch({layer: donut}))[layer]
    assert contours.vertices.dtype == np.int64
    assert contours.vertices.flags.c_contiguous
    assert (contours.polygon_count, contours.ring_count) == (1, 2)
    assert contours.ring_offsets.tolist() == [0, 4, 8]
    assert contours.polygon_ring_offsets.tolist() == [0, 2]
    assert (contours_to_region(contours) ^ donut).area() == 0


def test_empty_region_produces_well_shaped_empty_arrays() -> None:
    """空局部窗口在 OPC 流程中应保持为合法的零长度批次。"""
    layer = LayerSpec(1, 0)
    contours = extract_contours(region_batch({layer: kdb.Region()}))[layer]
    assert contours.vertices.shape == (0, 2)
    assert contours.ring_offsets.tolist() == [0]
    assert contours.polygon_ring_offsets.tolist() == [0]


def test_two_polygons_keep_distinct_ring_ranges() -> None:
    """两个 Polygon 通过连续 ring 范围保持稳定的局部分组。"""
    layer = LayerSpec(1, 0)
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 10, 10))
    region.insert(kdb.Box(20, 0, 30, 10))
    contours = extract_contours(region_batch({layer: region}))[layer]
    assert contours.polygon_count == 2
    assert contours.polygon_ring_offsets.tolist() == [0, 1, 2]


def test_single_region_extraction_matches_layer_mapping() -> None:
    """单 Region 快速入口必须与多层映射入口产生完全相同的数组。"""
    layer = LayerSpec(1, 0)
    region = kdb.Region(kdb.Box(0, 0, 10, 20))
    direct = extract_contour(region)
    mapped = extract_contours(region_batch({layer: region}))[layer]
    assert np.array_equal(direct.vertices, mapped.vertices)
    assert np.array_equal(direct.ring_offsets, mapped.ring_offsets)
    assert np.array_equal(direct.polygon_ring_offsets, mapped.polygon_ring_offsets)
