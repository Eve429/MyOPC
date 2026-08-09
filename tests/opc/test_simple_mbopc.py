"""流式同步 simple MB-OPC 的缓存、归属、回滚和实际模型集成测试。"""

from __future__ import annotations

import klayout.db as kdb
import numpy as np
import pytest
import torch

from lithography import ICCAD13Lithography, LithographyResult
from opc.errors import ReconstructionError
from opc.input import RectilinearCoreGrid
from opc.input.edge import FragmentationConfig, prepare_problem
from opc.iteration.mbopc.solver import (
    _preserves_reference_topology,
    _subset_contours,
    _target_tile,
    _TargetCache,
    optimize,
)
from opc.iteration.mbopc.types import SimpleMBOPCConfig

from .test_common import _batch


class _RecordingZeroModel:
    """记录每个流式 batch 的 mask，并返回全零曝光图制造稳定 inner 违规。"""

    def __init__(self) -> None:
        """在 CPU 上创建空输入记录，避免测试依赖 CUDA。"""
        self.device = torch.device("cpu")
        self.inputs: list[np.ndarray] = []

    def __call__(self, mask: torch.Tensor) -> LithographyResult:
        """复制本批 mask 到 CPU，并返回与其同形的三张全零图。"""
        self.inputs.append(mask.detach().cpu().numpy().copy())
        zeros = torch.zeros_like(mask)
        return LithographyResult(zeros, zeros, zeros)


def _rectangle_problem() -> tuple:
    """构造横跨两个 core 且每个 context 可放入 96 像素画布的矩形。"""
    region = kdb.Region(kdb.Box(5, 5, 95, 55))
    grid = RectilinearCoreGrid(np.array([0, 53, 100]), np.array([0, 60]), 10)
    config = FragmentationConfig(4, 16, 8)
    return prepare_problem(_batch(region), _batch(region).layers[0], config, grid), config


def _solver_config(iterations: int = 2, cache_bytes: int = 1 << 20) -> SimpleMBOPCConfig:
    """返回适合小型 CPU 回归用例的确定性迭代参数。"""
    return SimpleMBOPCConfig(
        iterations=iterations, initial_step_dbu=2.0, decay_every=2,
        max_displacement_dbu=8.0, epe_distance_dbu=3.0, pixel_dbu=1,
        canvas=96, batch_size=1, target_cache_bytes=cache_bytes)


def test_target_cache_hit_restores_unit_interval() -> None:
    """uint8 target 缓存命中后必须恢复到 [0,1]，不能把 255 送入光刻模型。"""
    problem, _ = _rectangle_problem()
    cache = _TargetCache(1 << 20)
    first = _target_tile(problem, 0, _solver_config(), cache)
    second = _target_tile(problem, 0, _solver_config(), cache)
    assert np.array_equal(first, second)
    assert float(second.min()) >= 0.0
    assert float(second.max()) <= 1.0


def test_target_lru_replaces_existing_value_and_evicts_oldest() -> None:
    """target LRU 应正确更新字节计数，并在超限时只驱逐最旧 tile。"""
    cache = _TargetCache(8)
    cache.put(0, np.zeros(4, dtype=np.uint8))
    cache.put(0, np.zeros(6, dtype=np.uint8))
    assert cache.current_bytes == 6
    cache.put(1, np.zeros(4, dtype=np.uint8))
    assert list(cache.values) == [1]
    assert cache.current_bytes == 4
    cache.put(2, np.zeros(9, dtype=np.uint8))
    assert list(cache.values) == [1]


def test_contour_subset_selects_only_requested_polygon_and_all_holes() -> None:
    """局部提取应跳过无关顶点，同时完整保留目标 Polygon 的 hull 与 hole。"""
    first = kdb.Region(kdb.Box(0, 0, 20, 20))
    second = (kdb.Region(kdb.Box(40, 0, 80, 40)) -
              kdb.Region(kdb.Box(50, 10, 70, 30)))
    problem = prepare_problem(_batch(first + second), _batch(first + second).layers[0],
                              FragmentationConfig(2, 8, 2))
    selected = _subset_contours(problem.segments.contours, np.array([1]))
    assert selected.ring_polygon_ids.tolist() == [1, 1]
    assert selected.ring_is_hole.tolist() == [False, True]
    assert selected.ring_count == 2


def test_all_batches_read_same_state_before_barrier_and_owner_moves_once() -> None:
    """同轮两个 core 应读取同一旧状态，每个 segment 只由 owner 写一次。"""
    problem, _ = _rectangle_problem()
    model = _RecordingZeroModel()
    result = optimize(problem, model, _solver_config())
    assert len(model.inputs) == 4
    # 第一轮两个 batch 都来自零位移；完成整轮并通过重建后，第二轮才看到外移 mask。
    assert not np.array_equal(model.inputs[0], model.inputs[2])
    assert not np.array_equal(model.inputs[1], model.inputs[3])
    assert result.records[0].moved_segments == problem.segments.segment_count
    assert result.records[0].rejected_segments == 0


def test_reconstruction_failure_rolls_back_whole_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选轮廓非法时整轮必须回滚，不能留下半个 Polygon 的局部位移。"""
    from opc.iteration.mbopc import solver

    problem, _ = _rectangle_problem()
    original = solver.reconstruct_contours
    calls = 0

    def fail_candidate(*args: object, **kwargs: object):
        """允许初始零位移重建，并让第一次候选发布稳定复现失败。"""
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ReconstructionError("测试候选非法")
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "reconstruct_contours", fail_candidate)
    result = solver.optimize(problem, _RecordingZeroModel(), _solver_config())
    assert result.stop_reason == "no_legal_update"
    assert result.records[0].rejected_segments == problem.segments.segment_count
    assert np.count_nonzero(result.best_displacements) == 0


def test_topology_guard_rejects_opposite_edge_crossing_and_hull_inside_hole() -> None:
    """发布屏障必须拒绝矩形对边穿越，以及外轮廓越过 hole 边界。"""
    from opc.input.edge import reconstruct_contours

    rectangle = kdb.Region(kdb.Box(0, 0, 20, 20))
    rectangle_config = FragmentationConfig(2, 10, 40)
    rectangle_problem = prepare_problem(
        _batch(rectangle), _batch(rectangle).layers[0], rectangle_config)
    rectangle_geometry = rectangle_problem.segments.materialize()
    rectangle_values = np.zeros(rectangle_problem.segments.segment_count)
    left = ((rectangle_geometry.starts[:, 0] == 0) &
            (rectangle_geometry.ends[:, 0] == 0))
    rectangle_values[left] = -30.0
    crossed = reconstruct_contours(
        rectangle_problem.segments, rectangle_values, rectangle_config)
    assert not _preserves_reference_topology(
        rectangle_problem.segments.contours, crossed)

    hollow = (kdb.Region(kdb.Box(0, 0, 40, 40)) -
              kdb.Region(kdb.Box(10, 10, 30, 30)))
    hollow_config = FragmentationConfig(2, 10, 30)
    hollow_problem = prepare_problem(_batch(hollow), _batch(hollow).layers[0], hollow_config)
    hollow_values = np.zeros(hollow_problem.segments.segment_count)
    segment_holes = hollow_problem.segments.edges.is_hole[hollow_problem.segments.edge_ids]
    hollow_values[~segment_holes] = -25.0
    inverted = reconstruct_contours(
        hollow_problem.segments, hollow_values, hollow_config)
    assert not _preserves_reference_topology(hollow_problem.segments.contours, inverted)


def test_two_dbu_hollow_wall_invalid_long_edge_probes_are_not_published() -> None:
    """壁宽 2 DBU 而探针距 8 DBU 时，无效长边不得进入最终最佳位移。"""
    region = (kdb.Region(kdb.Box(0, 0, 40, 40)) -
              kdb.Region(kdb.Box(2, 2, 38, 38)))
    config = FragmentationConfig(1, 10, 2)
    grid = RectilinearCoreGrid(np.array([0, 40]), np.array([0, 40]))
    problem = prepare_problem(_batch(region), _batch(region).layers[0], config, grid)
    iteration = SimpleMBOPCConfig(
        iterations=2, initial_step_dbu=1.0, decay_every=1,
        max_displacement_dbu=2.0, epe_distance_dbu=8.0, pixel_dbu=1,
        canvas=256, batch_size=1)
    result = optimize(problem, _RecordingZeroModel(), iteration)
    # 长边内探针已经越过 2 DBU 窄壁进入孔洞，因 target_inner=False 被排除；靠
    # 拐角的极短段沿法向仍可能落在相邻壁内，属于局部有效探针，不能一概判无效。
    assert 0 < result.records[0].valid_probes < problem.segments.segment_count
    assert np.count_nonzero(result.best_displacements) == 0


def test_actual_iccad13_model_completes_one_streaming_round() -> None:
    """真实 ICCAD13 光刻模型应能完成 target、评价和 owner 方向的端到端一轮。"""
    region = kdb.Region(kdb.Box(80, 80, 176, 176))
    config = FragmentationConfig(8, 32, 8)
    grid = RectilinearCoreGrid(np.array([0, 256]), np.array([0, 256]))
    problem = prepare_problem(_batch(region), _batch(region).layers[0], config, grid)
    iteration = SimpleMBOPCConfig(
        iterations=1, initial_step_dbu=2.0, decay_every=1,
        max_displacement_dbu=8.0, epe_distance_dbu=8.0, pixel_dbu=1,
        canvas=256, batch_size=1, target_cache_bytes=0)
    result = optimize(problem, ICCAD13Lithography(device="cpu"), iteration)
    assert len(result.records) == 1
    assert result.records[0].valid_probes == problem.segments.segment_count
    assert np.isfinite(result.records[0].l2)
    assert np.isfinite(result.records[0].pvband)


@pytest.mark.parametrize("overrides", [
    {"iterations": 0}, {"pixel_dbu": 0}, {"target_cache_bytes": -1},
    {"initial_step_dbu": float("nan")}, {"epe_distance_dbu": 0.0},
    {"print_threshold": 1.0},
])
def test_iteration_config_rejects_invalid_limits(overrides: dict[str, object]) -> None:
    """空轮次、非法内存上限和非有限/越界浮点参数必须在运行前拒绝。"""
    values: dict[str, object] = {
        "iterations": 1, "initial_step_dbu": 1.0, "decay_every": 1,
        "max_displacement_dbu": 2.0, "epe_distance_dbu": 1.0,
        "pixel_dbu": 1,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        SimpleMBOPCConfig(**values)


def test_optimize_rejects_displacement_and_canvas_outside_frontend_contract() -> None:
    """求解位移超过前端上限或 context 超出画布时不得开始模型计算。"""
    problem, _ = _rectangle_problem()
    with pytest.raises(ValueError, match="最大位移"):
        optimize(problem, _RecordingZeroModel(), SimpleMBOPCConfig(
            1, 1.0, 1, 9.0, 3.0, 1, canvas=96, batch_size=1))
    with pytest.raises(ValueError, match="超过固定光刻画布"):
        optimize(problem, _RecordingZeroModel(), SimpleMBOPCConfig(
            1, 1.0, 1, 8.0, 3.0, 1, canvas=32, batch_size=1))
