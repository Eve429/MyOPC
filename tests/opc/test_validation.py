"""OPC 数据契约、归属和重建拒绝路径测试。"""

from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

from opc.diagnostics import render_boundary_overlay, save_problem_npz, write_debug_gds
from opc.errors import ReconstructionError
from opc.input.edge import (
    FragmentationConfig,
    edge_probe_points,
    reconstruct_region,
)

from .test_ownership_reconstruct import _rectangle_problem


@pytest.mark.parametrize("arguments, error", [
    ((0, 20, 8), ValueError), ((10, 15, 8), ValueError),
    ((10, 20, -1), ValueError), ((10, 20, 8, 0.5), ValueError),
    ((float("nan"), 20, 8), TypeError), (("10", 20, 8), TypeError),
])
def test_fragmentation_config_rejects_invalid_values(arguments: tuple, error: type[Exception]) -> None:
    """非数值、非有限值及会破坏分段约束的配置必须立即拒绝。"""
    with pytest.raises(error):
        FragmentationConfig(*arguments)


def test_probe_sampling_rejects_malformed_lines_and_distance() -> None:
    """探针入口必须阻止形状广播以及非正、非有限距离。"""
    starts = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="equal shape"):
        edge_probe_points([0, 0], starts, starts, 1.0)
    for distance in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="distance_dbu"):
            edge_probe_points(starts, starts, starts, distance)


def test_ownership_membership_rejects_bad_core_index() -> None:
    """规则网格必须为全部边段分配 owner，并拒绝越界 membership 查询。"""
    problem, _ = _rectangle_problem()
    assert np.all(problem.ownership.owner_indices >= 0)
    with pytest.raises(IndexError, match="core index"):
        problem.ownership.segments_for_core(len(problem.ownership.cores))


def test_reconstruction_and_artifacts_reject_invalid_vectors_and_paths(tmp_path: Path) -> None:
    """重建与产物输出必须拒绝非有限、超限、错长向量及错误扩展名。"""
    problem, config = _rectangle_problem()
    count = problem.segments.segment_count
    with pytest.raises(ValueError, match="segment count"):
        reconstruct_region(problem, np.zeros(count - 1))
    invalid = np.zeros(count)
    invalid[0] = np.nan
    with pytest.raises(ReconstructionError, match="finite"):
        reconstruct_region(problem, invalid)
    invalid[0] = config.max_displacement_dbu + 1
    with pytest.raises(ReconstructionError, match="maximum"):
        reconstruct_region(problem, invalid)
    with pytest.raises(ValueError, match="displacements"):
        save_problem_npz(problem, np.zeros(count - 1), tmp_path / "bad.npz")
    with pytest.raises(ValueError, match=".gds"):
        write_debug_gds(kdb.Region(), kdb.Region(), tmp_path / "bad.oas", 0.001, 1)


def test_visualization_rejects_bad_arrays_owners_and_limits(tmp_path: Path) -> None:
    """诊断图对外部数组的形状、owner 数量和输出限制执行一致检查。"""
    problem, _ = _rectangle_problem()
    geometry = problem.segments.materialize()
    arguments = (problem.physical_mask.region, problem.physical_mask.layer,
                 problem.physical_mask.query_box, 0.001)
    with pytest.raises(ValueError, match="equal shape"):
        render_boundary_overlay(*arguments, [0, 0], geometry.ends, geometry.normals,
                                tmp_path / "bad.png")
    with pytest.raises(ValueError, match="owners"):
        render_boundary_overlay(*arguments, geometry.starts, geometry.ends,
                                geometry.normals, tmp_path / "bad.png", owners=[0])
    with pytest.raises(ValueError, match="limits"):
        render_boundary_overlay(*arguments, geometry.starts, geometry.ends,
                                geometry.normals, tmp_path / "bad.png", max_dimension=10)


def test_segment_materialization_validates_displacement_vector() -> None:
    """紧凑 segment 批次必须在热路径前拒绝错长和非有限位移。"""
    problem, _ = _rectangle_problem()
    with pytest.raises(ValueError, match="finite"):
        problem.segments.materialize(np.zeros(problem.segments.segment_count - 1))
    with pytest.raises(ValueError, match="finite"):
        problem.segments.materialize(np.full(problem.segments.segment_count, np.nan))
