"""原生 Region 到连续轮廓/边数组的转换测试。"""

import klayout.db as kdb
import numpy as np

from geometry import contours_to_region, extract_contours, extract_edges
from layout import LayerSpec

from .helpers import region_batch


def test_polygon_hole_round_trip_and_edge_metadata() -> None:
    """外轮廓与孔洞经过数组转换和原生重建后必须完全一致。"""
    layer = LayerSpec(7, 1)
    donut = kdb.Region(kdb.Box(0, 0, 100, 100)) - kdb.Region(kdb.Box(20, 20, 80, 80))
    contours = extract_contours(region_batch({layer: donut}))[layer]
    assert contours.vertices.dtype == np.int64
    assert contours.vertices.flags.c_contiguous
    assert (contours.polygon_count, contours.ring_count) == (1, 2)
    assert contours.ring_is_hole.tolist() == [False, True]
    edges = extract_edges(contours)
    assert edges.edge_count == 8
    assert edges.is_hole.tolist() == [False] * 4 + [True] * 4
    assert (contours_to_region(contours) ^ donut).area() == 0


def test_empty_region_produces_well_shaped_empty_arrays() -> None:
    """空局部窗口在 OPC 流程中应保持为合法的零长度批次。"""
    layer = LayerSpec(1, 0)
    contours = extract_contours(region_batch({layer: kdb.Region()}))[layer]
    edges = extract_edges(contours)
    assert contours.vertices.shape == (0, 2)
    assert contours.ring_offsets.tolist() == [0]
    assert edges.starts.shape == (0, 2)
    assert edges.edge_count == 0


def test_two_polygons_keep_distinct_local_ids() -> None:
    """局部 Polygon ID 在轮廓和边元数据之间保持稳定。"""
    layer = LayerSpec(1, 0)
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 10, 10))
    region.insert(kdb.Box(20, 0, 30, 10))
    contours = extract_contours(region_batch({layer: region}))[layer]
    edges = extract_edges(contours)
    assert contours.polygon_count == 2
    assert set(edges.polygon_ids.tolist()) == {0, 1}
