"""MB-OPC 跨 core 归属、更新同步和轮廓重建测试。"""

import klayout.db as kdb
import numpy as np
import pytest

from layout import LayerSpec
from opc.common import RectilinearCoreGrid
from opc.errors import OwnershipError
from opc.mbopc import (
    FragmentationConfig,
    SegmentUpdateBatch,
    merge_owner_updates,
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
    left = set(problem.ownership.segments_for_core(0).tolist())
    right = set(problem.ownership.segments_for_core(1).tolist())
    for segment_index in crossing:
        assert int(problem.ownership.owner_indices[segment_index]) in (0, 1)
        assert int(segment_index) in left & right


def test_only_owner_updates_and_stable_key_synchronizes_all_views() -> None:
    """非 owner 更新必须拒绝，owner 更新应通过全局向量同步到任意 core 视图。"""
    problem, _ = _rectangle_problem()
    segment_index = int(problem.ownership.segments_for_core(0)[0])
    owner = int(problem.ownership.owner_indices[segment_index])
    invalid = SegmentUpdateBatch(problem.segments.keys[[segment_index]],
                                 np.array([1 - owner]), np.array([2.0]))
    with pytest.raises(OwnershipError, match="unique owner"):
        merge_owner_updates(problem, [invalid])
    valid = SegmentUpdateBatch(problem.segments.keys[[segment_index]],
                               np.array([owner]), np.array([2.0]))
    result = merge_owner_updates(problem, [valid])
    assert result.displacements[segment_index] == 2.0
    assert result.changed_segment_indices.tolist() == [segment_index]
    assert result.dirty_polygon_ids.tolist() == [0]
    current = problem.segments.materialize(result.displacements, [segment_index])
    reference = problem.segments.materialize(indices=[segment_index])
    assert np.allclose(current.starts - reference.starts, reference.normals * 2.0)


def test_zero_and_uniform_outward_reconstruction_match_reference_and_sizing() -> None:
    """零位移应精确回环，矩形统一正位移应等于原生向外 sizing。"""
    problem, config = _rectangle_problem()
    zeros = np.zeros(problem.segments.segment_count)
    reference = reconstruct_region(problem.segments, zeros, config)
    assert (reference ^ problem.physical_mask.region).area() == 0
    outward = reconstruct_region(problem.segments, np.full(problem.segments.segment_count, 5.0), config)
    assert (outward ^ problem.physical_mask.region.sized(5)).area() == 0


def test_independent_fragment_move_creates_valid_jogs_without_mutating_reference() -> None:
    """只移动直边中段应生成连接 jog，并保持参考物理 Region 不变。"""
    problem, config = _rectangle_problem()
    values = np.zeros(problem.segments.segment_count)
    edge_counts = np.diff(problem.segments.edge_segment_offsets)
    edge_id = int(np.argmax(edge_counts))
    start, end = problem.segments.edge_segment_offsets[edge_id:edge_id + 2]
    values[int(start + (end - start) // 2)] = 4.0
    moved = reconstruct_region(problem.segments, values, config)
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
    result = reconstruct_region(problem.segments, np.zeros(problem.segments.segment_count), config)
    assert (result ^ region).area() == 0


def test_long_non_grid_aligned_diagonal_drops_equal_displacement_split_points() -> None:
    """斜边内部参数点不得因 DBU 取整变成零位移重建毛刺。"""
    layer = LayerSpec(4, 0)
    region = kdb.Region(kdb.Polygon([kdb.Point(230, 130), kdb.Point(140, 140),
                                     kdb.Point(150, 220), kdb.Point(210, 210)]))
    config = FragmentationConfig(5, 20, 6)
    problem = prepare_problem(_batch(region, layer), layer, config)
    result = reconstruct_region(problem.segments, np.zeros(problem.segments.segment_count), config)
    assert (result ^ region).area() == 0
