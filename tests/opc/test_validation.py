"""OPC 数据契约、归属和重建拒绝路径测试。"""

from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

from layout import DbuBox
from opc.common import (
    CoreSpec,
    build_sample_template,
    render_boundary_overlay,
    sample_lines,
)
from opc.errors import OwnershipError, ReconstructionError
from opc.mbopc import (
    FragmentationConfig,
    MidpointOwnerPolicy,
    SegmentUpdateBatch,
    merge_owner_updates,
    reconstruct_region,
    save_problem_npz,
    write_debug_gds,
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


def test_sampling_rejects_malformed_templates_lines_and_buffers() -> None:
    """采样入口必须阻止广播、越界索引和错误复用缓冲区。"""
    with pytest.raises(ValueError, match="line_count"):
        build_sample_template(-1)
    with pytest.raises(ValueError, match="non-empty"):
        build_sample_template(1, (), (0,))
    starts = np.array([[0.0, 0.0]])
    template = build_sample_template(2)
    with pytest.raises(ValueError, match="unknown line"):
        sample_lines(np.vstack((starts, starts)), np.vstack((starts, starts)),
                     np.vstack((starts, starts)), build_sample_template(3))
    with pytest.raises(ValueError, match="equal shape"):
        sample_lines([0, 0], starts, starts, template)
    with pytest.raises(ValueError, match="out must"):
        sample_lines(np.vstack((starts, starts)), np.vstack((starts, starts)),
                     np.vstack((starts, starts)), template, np.empty((2, 2), dtype=np.float32))


def test_owner_update_rejects_unknown_duplicate_range_and_bad_base() -> None:
    """更新汇聚必须拒绝未知 key、重复写入、越界 core 和超限位移。"""
    problem, _ = _rectangle_problem()
    index, key = 0, problem.segments.keys[[0]]
    owner = int(problem.ownership.owner_indices[index])
    unknown = key.copy()
    unknown[0, 0] ^= np.uint64(0xFFFF)
    cases = [
        SegmentUpdateBatch(unknown, np.array([owner]), np.array([1.0])),
        SegmentUpdateBatch(key, np.array([99]), np.array([1.0])),
        SegmentUpdateBatch(np.repeat(key, 2, axis=0), np.array([owner, owner]),
                           np.array([1.0, 2.0])),
        SegmentUpdateBatch(key, np.array([owner]),
                           np.array([problem.config.max_displacement_dbu + 1])),
    ]
    for update in cases:
        with pytest.raises(OwnershipError):
            merge_owner_updates(problem, [update])
    with pytest.raises(ValueError, match="base_displacements"):
        merge_owner_updates(problem, [], np.array([np.nan]))
    unchanged = merge_owner_updates(problem, [])
    assert not len(unchanged.changed_segment_indices)


def test_explicit_core_validation_and_membership_bounds() -> None:
    """显式 core 列表必须非空、不重叠，并对访问索引严格检查。"""
    problem, _ = _rectangle_problem()
    policy = MidpointOwnerPolicy()
    with pytest.raises(OwnershipError, match="at least one"):
        policy.assign(problem.segments, ())
    left = CoreSpec("left", DbuBox(0, 0, 60, 60), DbuBox(-5, -5, 65, 65))
    right = CoreSpec("right", DbuBox(50, 0, 100, 60), DbuBox(45, -5, 105, 65))
    with pytest.raises(OwnershipError, match="overlap"):
        policy.assign(problem.segments, (left, right))
    explicit = policy.assign(problem.segments, (
        CoreSpec("all", DbuBox(0, 0, 100, 60), DbuBox(-10, -10, 110, 70)),))
    with pytest.raises(IndexError, match="core index"):
        explicit.segments_for_core(1)


def test_reconstruction_and_artifacts_reject_invalid_vectors_and_paths(tmp_path: Path) -> None:
    """重建与产物输出必须拒绝非有限、超限、错长向量及错误扩展名。"""
    problem, config = _rectangle_problem()
    count = problem.segments.segment_count
    with pytest.raises(ValueError, match="segment count"):
        reconstruct_region(problem.segments, np.zeros(count - 1), config)
    invalid = np.zeros(count)
    invalid[0] = np.nan
    with pytest.raises(ReconstructionError, match="finite"):
        reconstruct_region(problem.segments, invalid, config)
    invalid[0] = config.max_displacement_dbu + 1
    with pytest.raises(ReconstructionError, match="maximum"):
        reconstruct_region(problem.segments, invalid, config)
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


def test_segment_materialization_and_update_batch_validate_shapes() -> None:
    """紧凑 segment 批次必须在数组进入热路径前拒绝越界与形状错误。"""
    problem, _ = _rectangle_problem()
    with pytest.raises(IndexError, match="out of range"):
        problem.segments.materialize(indices=[problem.segments.segment_count])
    with pytest.raises(ValueError, match="finite"):
        problem.segments.materialize(np.full(problem.segments.segment_count, np.nan))
    with pytest.raises(ValueError, match="shape"):
        problem.segments.lookup_keys(np.array([1, 2], dtype=np.uint64))
    with pytest.raises(ValueError, match="equal length"):
        SegmentUpdateBatch(problem.segments.keys[:2], np.array([0]), np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        SegmentUpdateBatch(problem.segments.keys[:1], np.array([0]), np.array([np.nan]))
