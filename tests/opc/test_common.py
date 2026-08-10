"""物理 mask、core 网格和通用边界采样测试。"""

import tempfile
from pathlib import Path

import klayout.db as kdb
import numpy as np

from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.input import RectilinearCoreGrid, normalize_physical_mask
from opc.input.edge import FragmentationConfig, edge_probe_points, prepare_problem


def _batch(region: kdb.Region, layer: LayerSpec | None = None) -> RegionBatch:
    """为公共层测试构造不依赖磁盘的单层批次。"""
    layer = layer or LayerSpec(1, 0)
    return RegionBatch({layer: region}, DbuBox(-100, -100, 300, 300), CellRef("TOP", 0))


def test_physical_mask_removes_overlap_cut_lines_and_keeps_corner_touch_separate() -> None:
    """物理合并应消除内部边，同时不把仅角点接触区域粘成一个 Polygon。"""
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 100, 100))
    region.insert(kdb.Box(50, 0, 150, 100))
    region.insert(kdb.Box(150, 100, 200, 150))
    mask = normalize_physical_mask(_batch(region), LayerSpec(1, 0))
    assert mask.region.area() == 17_500
    assert mask.region.count() == 2
    problem = prepare_problem(_batch(region), LayerSpec(1, 0), FragmentationConfig(5, 20, 4))
    assert len(problem.segments.contours.vertices) == 8


def test_gds_keyhole_bridge_is_removed_before_edge_extraction() -> None:
    """GDS 孔洞桥接线不得成为可采样或可移动的物理边。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top, layer_index = layout.create_cell("TOP"), layout.layer(1, 0)
    polygon = kdb.Polygon(kdb.Box(0, 0, 100, 100))
    polygon.insert_hole(kdb.Box(25, 25, 75, 75))
    top.shapes(layer_index).insert(polygon)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "keyhole.gds"
        layout.write(str(path))
        loaded = kdb.Layout()
        loaded.read(str(path))
        cell = loaded.top_cell()
        raw = kdb.Region(cell.begin_shapes_rec(loaded.layer(1, 0)))
        assert next(iter(raw)).num_points_hull() == 10
        problem = prepare_problem(
            _batch(raw), LayerSpec(1, 0), FragmentationConfig(5, 20, 4))
    assert problem.segments.contours.ring_count == 2
    assert problem.segments.contours.polygon_ring_offsets.tolist() == [0, 2]
    assert len(problem.segments.contours.vertices) == 8


def test_rectilinear_grid_uses_half_open_internal_and_closed_outer_boundaries() -> None:
    """内部共享线归右上 core，整体最大边界仍必须得到确定 owner。"""
    grid = RectilinearCoreGrid(np.array([0, 50, 100]), np.array([0, 40, 100]), 10)
    points = np.array([[0, 0], [49.5, 20], [50, 20], [100, 100], [-1, 0]])
    assert grid.locate_points(points).tolist() == [0, 0, 1, 3, -1]
    cores = grid.cores()
    assert len(cores) == 4
    assert cores[0].context_box == DbuBox(-10, -10, 60, 50)


def test_edge_probe_points_follow_midpoint_and_outward_normal() -> None:
    """inner/outer 探针应位于边段中点的负/正外法向两侧。"""
    starts = np.array([[0, 0], [10, 0]], dtype=np.float64)
    ends = np.array([[10, 0], [10, 10]], dtype=np.float64)
    normals = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    inner, outer = edge_probe_points(starts, ends, normals, 2.0)
    assert inner.tolist() == [[5.0, -2.0], [12.0, 5.0]]
    assert outer.tolist() == [[5.0, 2.0], [8.0, 5.0]]
