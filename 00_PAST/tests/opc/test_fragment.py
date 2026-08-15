"""MB-OPC 紧凑分段、法向、拓扑顺序和按需物化测试。"""

import klayout.db as kdb
import numpy as np

from geometry import extract_contour
from layout import LayerSpec
from opc.input import normalize_physical_mask
from opc.input._fragmentation import count_edge_fragments
from opc.input.edge import FragmentationConfig, fragment_edges

from .test_common import _batch


def test_corner_and_middle_fragments_cover_edges_with_bounded_lengths() -> None:
    """长矩形边应保留角部短段，中段均衡且完整覆盖原数学边。"""
    layer = LayerSpec(1, 0)
    mask = normalize_physical_mask(_batch(kdb.Region(kdb.Box(0, 0, 100, 50)), layer), layer)
    segments = fragment_edges(extract_contour(mask.region), FragmentationConfig(10, 20, 8))
    geometry = segments.materialize()
    lengths = np.linalg.norm(geometry.ends - geometry.starts, axis=1)
    assert segments.segment_count == 20
    assert lengths.max() <= 20.0 + 1e-12
    assert np.isclose(lengths.sum(), 300.0)
    edge_count = len(segments.contours.vertices)
    counts = np.bincount(segments.edge_ids, minlength=edge_count)
    assert sorted(counts.tolist()) == [4, 4, 6, 6]
    vertices = segments.contours.vertices
    edge_lengths = np.linalg.norm(
        vertices[segments.edge_next_ids] - vertices, axis=1)
    indices = np.flatnonzero(segments.edge_ids == int(np.argmax(edge_lengths)))
    assert np.allclose(lengths[indices], [10, 20, 20, 20, 20, 10])


def test_hull_and_hole_normals_both_point_from_material_to_clear_space() -> None:
    """外轮廓法向应指向外部，孔洞法向应指向孔内。"""
    layer = LayerSpec(1, 0)
    donut = kdb.Region(kdb.Box(0, 0, 100, 100)) - kdb.Region(kdb.Box(25, 25, 75, 75))
    mask = normalize_physical_mask(_batch(donut, layer), layer)
    segments = fragment_edges(extract_contour(mask.region), FragmentationConfig(5, 20, 8))
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
    contours = extract_contour(mask.region)
    first, second = fragment_edges(contours, config), fragment_edges(contours, config)
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
    segments = fragment_edges(extract_contour(mask.region), FragmentationConfig(10, 20, 8))
    expanded_extra = segments.segment_count * (2 * 2 * 8 + 2 * 8 + 3 * 8)
    assert segments.persistent_nbytes < expanded_extra
    # PhysicalMask 只保存原生 Region，数值轮廓由 SegmentBatch 唯一持有；两组
    # int32 edge cache 代替完整起止点、ring 和 hole 派生数组。
    assert not hasattr(mask, "contours")
    assert segments.edge_next_ids.dtype == np.int32
    assert segments.edge_polygon_ids.dtype == np.int32
    assert not hasattr(segments, "edges")
    assert not hasattr(segments, "keys")
    assert not hasattr(segments, "edge_segment_offsets")


def test_shared_fragment_count_formula_matches_real_randomized_batches() -> None:
    """共享纯数组公式应对随机边长逐边等于真实 SegmentBatch 计数。"""
    random = np.random.default_rng(20260812)
    corner, maximum = 7.0, 23.0
    for lengths in random.integers(1, 2000, size=(20, 8)):
        # 把随机整数边长构成彼此分离的矩形，避免 Region 合并改变输入数学边；
        # 生产切分得到的 edge_ids bincount 是最直接的逐边实际分配数量。
        region = kdb.Region()
        offset = 0
        for width, height in lengths.reshape(-1, 2):
            region.insert(kdb.Box(offset, 0, offset + int(width), int(height)))
            offset += int(width) + 10
        layer = LayerSpec(1, 0)
        contours = extract_contour(normalize_physical_mask(_batch(region, layer), layer).region)
        segments = fragment_edges(
            contours, FragmentationConfig(corner, maximum, 4.0))
        vertices = contours.vertices
        edge_next = np.arange(len(vertices), dtype=np.int32) + 1
        edge_next[contours.ring_offsets[1:] - 1] = contours.ring_offsets[:-1]
        edge_lengths = np.linalg.norm(vertices[edge_next] - vertices, axis=1)
        expected = count_edge_fragments(edge_lengths, corner, maximum)
        actual = np.bincount(segments.edge_ids, minlength=len(edge_lengths))
        np.testing.assert_array_equal(actual, expected)
