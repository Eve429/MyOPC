"""MB-OPC 跨 core 归属、更新同步和轮廓重建测试。"""

import klayout.db as kdb
import numpy as np
import pytest

from layout import LayerSpec
from opc.input import RectilinearCoreGrid
from opc.input.edge import (
    FragmentationConfig,
    SegmentBatch,
    prepare_problem,
    reconstruct_region,
)

from .test_common import _batch


def _rectangle_problem() -> tuple:
    """构造边段会跨越非分段点 core 边界的矩形问题。"""
    layer = LayerSpec(1, 0)
    batch = _batch(kdb.Region(kdb.Box(0, 0, 100, 60)), layer)
    grid = RectilinearCoreGrid(np.array([0, 53, 100]), np.array([0, 60]), 10)
    config = FragmentationConfig(10, 20, 8)
    return prepare_problem(batch, layer, config, grid), config


def test_cross_core_segment_has_one_owner_and_both_context_memberships() -> None:
    """跨过 x=53 的控制段只能有一个 owner，但应被两个 halo core 看到。"""
    problem, _ = _rectangle_problem()
    geometry = problem.segments.materialize()
    crossing = np.flatnonzero(
        (np.minimum(geometry.starts[:, 0], geometry.ends[:, 0]) < 53) &
        (np.maximum(geometry.starts[:, 0], geometry.ends[:, 0]) > 53))
    assert len(crossing)
    left = set(problem.segments_for_core(0).tolist())
    right = set(problem.segments_for_core(1).tolist())
    for segment_index in crossing:
        assert int(problem.owner_indices[segment_index]) in (0, 1)
        assert int(segment_index) in left & right


def test_ownership_does_not_materialize_unused_segment_normals(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """ownership 准备不得物化只供迭代使用的逐段法向，且结果必须保持一致。"""
    problem, config = _rectangle_problem()
    grid = RectilinearCoreGrid(np.array([0, 53, 100]), np.array([0, 60]), 10)

    def reject_materialization(*args: object, **kwargs: object) -> None:
        """只要 ownership 误走完整几何物化路径就立即暴露回归。"""
        raise AssertionError("ownership 不应调用 SegmentBatch.materialize")

    monkeypatch.setattr(SegmentBatch, "materialize", reject_materialization)
    batch = _batch(kdb.Region(kdb.Box(0, 0, 100, 60)), LayerSpec(1, 0))
    actual = prepare_problem(batch, LayerSpec(1, 0), config, grid)
    assert np.array_equal(actual.owner_indices, problem.owner_indices)
    assert np.array_equal(actual.core_offsets, problem.core_offsets)
    assert np.array_equal(actual.member_segment_indices, problem.member_segment_indices)


def test_owner_index_update_synchronizes_all_context_views() -> None:
    """唯一 owner 写入全局位移后，所有 context 应读取同一移动边段。"""
    problem, _ = _rectangle_problem()
    segment_index = int(problem.segments_for_core(0)[0])
    owner = int(problem.owner_indices[segment_index])
    assert segment_index in problem.segments_for_core(owner)
    values = np.zeros(problem.segments.segment_count)
    values[segment_index] = 2.0
    current = problem.segments.materialize(values)
    reference = problem.segments.materialize()
    assert np.allclose(current.starts[segment_index] - reference.starts[segment_index],
                       reference.normals[segment_index] * 2.0)


def test_zero_and_uniform_outward_reconstruction_match_reference_and_sizing() -> None:
    """零位移应精确回环，矩形统一正位移应等于原生向外 sizing。"""
    problem, _ = _rectangle_problem()
    zeros = np.zeros(problem.segments.segment_count)
    reference = reconstruct_region(problem, zeros)
    assert (reference ^ problem.physical_mask.region).area() == 0
    outward = reconstruct_region(problem, np.full(problem.segments.segment_count, 5.0))
    assert (outward ^ problem.physical_mask.region.sized(5)).area() == 0


def test_independent_fragment_move_creates_valid_jogs_without_mutating_reference() -> None:
    """只移动直边中段应生成连接 jog，并保持参考物理 Region 不变。"""
    problem, _ = _rectangle_problem()
    values = np.zeros(problem.segments.segment_count)
    edge_counts = np.bincount(problem.segments.edge_ids,
                              minlength=len(problem.segments.contours.vertices))
    edge_id = int(np.argmax(edge_counts))
    indices = np.flatnonzero(problem.segments.edge_ids == edge_id)
    values[int(indices[len(indices) // 2])] = 4.0
    moved = reconstruct_region(problem, values)
    assert moved.has_valid_polygons()
    assert (moved ^ problem.physical_mask.region).area() > 0
    assert problem.physical_mask.region.area() == 6_000


def test_hole_and_diagonal_polygon_reconstruct_at_zero_displacement() -> None:
    """孔洞和非 Manhattan 拐角均应在零位移下保持拓扑与面积。"""
    layer = LayerSpec(3, 0)
    outer = kdb.Polygon([kdb.Point(0, 0), kdb.Point(120, 0),
                         kdb.Point(100, 100), kdb.Point(0, 80)])
    region = kdb.Region(outer) - kdb.Region(kdb.Box(30, 20, 60, 50))
    config = FragmentationConfig(5, 20, 6)
    problem = prepare_problem(_batch(region, layer), layer, config)
    result = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
    assert (result ^ region).area() == 0


def test_long_non_grid_aligned_diagonal_drops_equal_displacement_split_points() -> None:
    """斜边内部参数点不得因 DBU 取整变成零位移重建毛刺。"""
    layer = LayerSpec(4, 0)
    region = kdb.Region(kdb.Polygon([kdb.Point(230, 130), kdb.Point(140, 140),
                                     kdb.Point(150, 220), kdb.Point(210, 210)]))
    config = FragmentationConfig(5, 20, 6)
    problem = prepare_problem(_batch(region, layer), layer, config)
    result = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
    assert (result ^ region).area() == 0
