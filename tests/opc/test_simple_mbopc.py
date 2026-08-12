"""流式同步 simple MB-OPC 的缓存、归属、回滚和实际模型集成测试。"""

from __future__ import annotations

from types import SimpleNamespace

import klayout.db as kdb
import numpy as np
import pytest
import torch

from lithography import ICCAD13Lithography, ProcessCondition
from opc.errors import ReconstructionError
from opc.input import RectilinearCoreGrid
from opc.input.edge import FragmentationConfig, prepare_problem
from opc.input.raster import rasterize_region_canvas
from opc.iteration._cache import ArrayTileCache
from opc.iteration.mbopc.contracts import SimpleMBOPCConfig
from opc.iteration.mbopc.solver import (
    _current_tile,
    _owner_indices,
    _polygon_ids_for_core,
    _subset_contours,
    _target_tile,
    optimize,
)

from .test_common import _batch


class _RecordingZeroModel:
    """记录每个流式 batch 的 mask，并返回全零曝光图制造稳定 inner 违规。"""

    def __init__(self, canvas: int = 256) -> None:
        """在 CPU 上创建空输入记录，避免测试依赖 CUDA。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas, print_threshold=0.499)
        self.inputs: list[np.ndarray] = []

    def condition(self, name: str) -> ProcessCondition:
        """构造名称独立的伪工艺条件，kernel 和剂量仅满足求解器接口。"""
        return ProcessCondition(name, "focus", 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """复制本批 mask 到 CPU，并按请求名称返回同形全零曝光图。"""
        self.inputs.append(mask.detach().cpu().numpy().copy())
        zeros = torch.zeros_like(mask)
        return {condition.name: zeros for condition in conditions}


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
        epe_distance_dbu=3.0, pixel_dbu=1,
        canvas=96, batch_size=1, target_cache_bytes=cache_bytes)


def test_target_cache_hit_keeps_compact_uint8_until_batch_transfer() -> None:
    """target 命中与未命中都应保持 uint8，避免 CPU 批次提前展开为 float32。"""
    problem, _ = _rectangle_problem()
    cache = ArrayTileCache(1 << 20)
    core = problem.grid.cores()[0]
    first = _target_tile(problem, 0, core, _solver_config(), cache)
    second = _target_tile(problem, 0, core, _solver_config(), cache)
    assert np.array_equal(first, second)
    assert second.dtype == np.uint8
    assert (int(second.min()), int(second.max())) == (0, 255)


def test_owner_indices_match_global_reference_scan() -> None:
    """membership CSR 过滤必须与旧的全局 owner 扫描逐 core 完全一致。"""
    problem, _ = _rectangle_problem()
    actual = _owner_indices(problem)
    expected = tuple(np.flatnonzero(problem.owner_indices == core).astype(np.int32)
                     for core in range(problem.core_count))
    assert len(actual) == len(expected)
    assert all(np.array_equal(left, right)
               for left, right in zip(actual, expected, strict=True))


def test_target_lru_replaces_existing_value_and_evicts_oldest() -> None:
    """target LRU 应正确更新字节计数，并在超限时只驱逐最旧 tile。"""
    cache = ArrayTileCache(8)
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
    assert selected.polygon_ring_offsets.tolist() == [0, 2]
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


def test_diagnostic_metrics_do_not_select_mbopc_best_state(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """L2/PVBand 即使改善也不得在 EPE 相同时替换更早的最佳几何。"""
    from opc.iteration.mbopc import solver

    problem, _ = _rectangle_problem()
    l2_values = iter((100, 100, 0, 0))

    def diagnostic_l2(*args: object, **kwargs: object) -> int:
        """制造第二轮明显更好的诊断 L2，但不改变 EPE 探针结果。"""
        return next(l2_values)

    def diagnostic_pvband(*args: object, **kwargs: object) -> int:
        """返回固定 PVBand，隔离本测试只关注最佳状态选择规则。"""
        return 0

    monkeypatch.setattr(solver, "evaluate_binary_l2", diagnostic_l2)
    monkeypatch.setattr(solver, "evaluate_pvband", diagnostic_pvband)
    result = solver.optimize(problem, _RecordingZeroModel(), _solver_config())
    assert result.records[1].l2 < result.records[0].l2
    assert result.records[1].epe == result.records[0].epe
    assert result.best_iteration == 0


def test_zero_state_skips_global_and_local_contour_reconstruction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """单轮零位移求解应共享参考轮廓，并跳过局部轮廓和 Region 差分构造。"""
    from opc.iteration.mbopc import solver

    problem, _ = _rectangle_problem()

    def reject_reconstruction(*args: object, **kwargs: object) -> None:
        """零位移热路径若触发全局或局部重建则立即失败。"""
        raise AssertionError("零位移状态不应重建轮廓或 Region")

    monkeypatch.setattr(solver, "reconstruct_contours", reject_reconstruction)
    monkeypatch.setattr(solver, "contours_to_region", reject_reconstruction)
    result = solver.optimize(problem, _RecordingZeroModel(), _solver_config(iterations=1))
    assert len(result.records) == 1


def test_zero_local_tile_preserves_unquantized_reference_raster() -> None:
    """局部零位移快路必须保留原 current mask 覆盖率，不得偷换成 uint8 target。"""
    problem, _ = _rectangle_problem()
    # 参考矩形从 5 DBU 起始，4 DBU 像素会产生 1/16 等分数覆盖；这些值通常不能
    # 被 uint8/255 精确表达，因此可以真实区分 current raster 与缓存 target。
    config = SimpleMBOPCConfig(
        iterations=1, initial_step_dbu=2.0, decay_every=1,
        epe_distance_dbu=3.0, pixel_dbu=4,
        canvas=96, batch_size=1, target_cache_bytes=1 << 20)
    core_index = 0
    core = problem.grid.cores()[core_index]
    context = core.context_box
    expected = rasterize_region_canvas(
        problem.physical_mask.region, context, config.pixel_dbu, config.canvas)
    target = _target_tile(problem, core_index, core, config, ArrayTileCache(1 << 20))
    actual = _current_tile(
        problem, problem.segments.contours,
        np.zeros(problem.segments.segment_count), core_index, core,
        _polygon_ids_for_core(problem, core_index), target, config)
    assert not np.array_equal(target, expected)
    assert np.array_equal(actual, expected)


def test_reconstruction_failure_rolls_back_whole_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选轮廓非法时整轮必须回滚，不能留下半个 Polygon 的局部位移。"""
    from opc.iteration.mbopc import solver

    problem, _ = _rectangle_problem()
    calls = 0

    def fail_candidate(*args: object, **kwargs: object):
        """让第一次候选发布稳定复现失败；零位移初态不得调用本函数。"""
        nonlocal calls
        calls += 1
        raise ReconstructionError("测试候选非法")

    monkeypatch.setattr(solver, "reconstruct_contours", fail_candidate)
    result = solver.optimize(problem, _RecordingZeroModel(), _solver_config())
    assert result.stop_reason == "no_legal_update"
    assert result.records[0].rejected_segments == problem.segments.segment_count
    assert np.count_nonzero(result.best_displacements) == 0
    assert calls == 1


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
    with pytest.raises(ReconstructionError, match="orientation"):
        reconstruct_contours(rectangle_problem, rectangle_values)

    hollow = (kdb.Region(kdb.Box(0, 0, 40, 40)) -
              kdb.Region(kdb.Box(10, 10, 30, 30)))
    hollow_config = FragmentationConfig(2, 10, 30)
    hollow_problem = prepare_problem(_batch(hollow), _batch(hollow).layers[0], hollow_config)
    hollow_values = np.zeros(hollow_problem.segments.segment_count)
    contours = hollow_problem.segments.contours
    ring_holes = np.ones(contours.ring_count, dtype=np.bool_)
    ring_holes[contours.polygon_ring_offsets[:-1]] = False
    edge_ring_ids = np.repeat(
        np.arange(contours.ring_count, dtype=np.int32), np.diff(contours.ring_offsets))
    segment_holes = ring_holes[edge_ring_ids[hollow_problem.segments.edge_ids]]
    hollow_values[~segment_holes] = -25.0
    with pytest.raises(ReconstructionError, match="escaped"):
        reconstruct_contours(hollow_problem, hollow_values)


def test_two_dbu_hollow_wall_invalid_long_edge_probes_are_not_published() -> None:
    """壁宽 2 DBU 而探针距 8 DBU 时，无效长边不得进入最终最佳位移。"""
    region = (kdb.Region(kdb.Box(0, 0, 40, 40)) -
              kdb.Region(kdb.Box(2, 2, 38, 38)))
    config = FragmentationConfig(1, 10, 2)
    grid = RectilinearCoreGrid(np.array([0, 40]), np.array([0, 40]))
    problem = prepare_problem(_batch(region), _batch(region).layers[0], config, grid)
    iteration = SimpleMBOPCConfig(
        iterations=2, initial_step_dbu=1.0, decay_every=1,
        epe_distance_dbu=8.0, pixel_dbu=1,
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
        epe_distance_dbu=8.0, pixel_dbu=1,
        canvas=256, batch_size=1, target_cache_bytes=0)
    result = optimize(problem, ICCAD13Lithography(device="cpu"), iteration)
    assert len(result.records) == 1
    assert result.records[0].valid_probes == problem.segments.segment_count
    assert np.isfinite(result.records[0].l2)
    assert np.isfinite(result.records[0].pvband)


@pytest.mark.parametrize("overrides", [
    {"iterations": 0}, {"pixel_dbu": 0}, {"target_cache_bytes": -1},
    {"initial_step_dbu": float("nan")}, {"epe_distance_dbu": 0.0},
])
def test_iteration_config_rejects_invalid_limits(overrides: dict[str, object]) -> None:
    """空轮次、非法内存上限和非有限/越界浮点参数必须在运行前拒绝。"""
    values: dict[str, object] = {
        "iterations": 1, "initial_step_dbu": 1.0, "decay_every": 1,
        "epe_distance_dbu": 1.0, "pixel_dbu": 1,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        SimpleMBOPCConfig(**values)


def test_optimize_rejects_canvas_outside_model_or_context_contract() -> None:
    """tile canvas 超过模型上限或装不下 context 时不得开始计算。"""
    problem, _ = _rectangle_problem()
    with pytest.raises(ValueError, match="不能超过光刻模型"):
        optimize(problem, _RecordingZeroModel(canvas=64), SimpleMBOPCConfig(
            1, 1.0, 1, 3.0, 1, canvas=96, batch_size=1))
    with pytest.raises(ValueError, match="超过固定光刻画布"):
        optimize(problem, _RecordingZeroModel(), SimpleMBOPCConfig(
            1, 1.0, 1, 3.0, 1, canvas=32, batch_size=1))
