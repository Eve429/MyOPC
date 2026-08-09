"""MB-OPC 紧凑分段、法向、拓扑顺序和按需物化测试。"""

import klayout.db as kdb
import numpy as np

from layout import LayerSpec
from opc.input import normalize_physical_mask
from opc.input.edge import FragmentationConfig, fragment_edges

from .test_common import _batch


def test_corner_and_middle_fragments_cover_edges_with_bounded_lengths() -> None:
    """长矩形边应保留角部短段，中段均衡且完整覆盖原数学边。"""
    layer = LayerSpec(1, 0)
    mask = normalize_physical_mask(_batch(kdb.Region(kdb.Box(0, 0, 100, 50)), layer), layer)
    segments = fragment_edges(mask, FragmentationConfig(10, 20, 8))
    geometry = segments.materialize()
    lengths = np.linalg.norm(geometry.ends - geometry.starts, axis=1)
    assert segments.segment_count == 20
    assert lengths.max() <= 20.0 + 1e-12
    assert np.isclose(lengths.sum(), 300.0)
    counts = np.bincount(segments.edge_ids, minlength=segments.edges.edge_count)
    assert sorted(counts.tolist()) == [4, 4, 6, 6]
    edge_lengths = np.linalg.norm(segments.edges.ends - segments.edges.starts, axis=1)
    indices = np.flatnonzero(segments.edge_ids == int(np.argmax(edge_lengths)))
    assert np.allclose(lengths[indices], [10, 20, 20, 20, 20, 10])


def test_hull_and_hole_normals_both_point_from_material_to_clear_space() -> None:
    """外轮廓法向应指向外部，孔洞法向应指向孔内。"""
    layer = LayerSpec(1, 0)
    donut = kdb.Region(kdb.Box(0, 0, 100, 100)) - kdb.Region(kdb.Box(25, 25, 75, 75))
    mask = normalize_physical_mask(_batch(donut, layer), layer)
    segments = fragment_edges(mask, FragmentationConfig(5, 20, 8))
    geometry = segments.materialize()
    midpoints = (geometry.starts + geometry.ends) * 0.5
    inner = midpoints - geometry.normals * 2.0
    outer = midpoints + geometry.normals * 2.0
    # 用 1 DBU 小方框做原生点侧判定；所有负法向点在材料中，正法向点在空区。
    for point in inner:
        x, y = np.rint(point).astype(np.int64)
        assert (mask.region & kdb.Region(kdb.Box(int(x), int(y), int(x + 1), int(y + 1)))).area()
    for point in outer:
        x, y = np.rint(point).astype(np.int64)
        assert not (mask.region & kdb.Region(kdb.Box(int(x), int(y), int(x + 1), int(y + 1)))).area()


def test_diagonal_segments_are_deterministic_and_move_along_unit_normals() -> None:
    """斜边应按确定拓扑顺序切分，并沿单位外法向移动。"""
    layer = LayerSpec(2, 1)
    triangle = kdb.Region(kdb.Polygon([kdb.Point(0, 0), kdb.Point(30, 0), kdb.Point(0, 40)]))
    mask = normalize_physical_mask(_batch(triangle, layer), layer)
    config = FragmentationConfig(5, 20, 6)
    first, second = fragment_edges(mask, config), fragment_edges(mask, config)
    np.testing.assert_array_equal(first.edge_ids, second.edge_ids)
    np.testing.assert_allclose(first.t0, second.t0)
    np.testing.assert_allclose(first.t1, second.t1)
    base = first.materialize()
    moved = first.materialize(np.full(first.segment_count, 3.0))
    assert np.allclose(moved.starts - base.starts, base.normals * 3.0)
    assert np.allclose(np.linalg.norm(base.normals, axis=1), 1.0)
    assert np.linalg.norm(base.ends - base.starts, axis=1).max() <= 20.0


def test_compact_batch_does_not_persist_expanded_segment_geometry() -> None:
    """大量分段的常驻数组应明显小于持久保存端点、法向和父字段的方案。"""
    layer = LayerSpec(1, 0)
    mask = normalize_physical_mask(_batch(kdb.Region(kdb.Box(0, 0, 10_000, 1_000)), layer), layer)
    segments = fragment_edges(mask, FragmentationConfig(10, 20, 8))
    expanded_extra = segments.segment_count * (2 * 2 * 8 + 2 * 8 + 3 * 8)
    assert segments.persistent_nbytes < expanded_extra
    # PhysicalMask 与 SegmentBatch 各自保留完整领域语义，但共享同一不可变批量对象，
    # 不允许为了接口便利复制轮廓或数学边数组。
    assert segments.contours is mask.contours
    assert segments.edges is mask.edges
    assert not hasattr(segments, "keys")
    assert not hasattr(segments, "edge_segment_offsets")
