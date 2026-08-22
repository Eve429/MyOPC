"""最简 MB-OPC 求解器的评价、提案、迭代与图形矩阵生成式测试。"""

import klayout.db as kdb
import numpy as np
import pytest
import torch

from layout import DbuBox, LayerSpec, RegionBatch
from lithography import ICCAD13Lithography, ProcessCondition
from opc.errors import ReconstructionError
from opc.input import plan_macros
from opc.input.edge import prepare_macro_problem, reconstruct_region
from opc.input.edge.fragmentation import FragmentationConfig
from opc.iteration.mbopc import (
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    SimpleMBOPCStep,
    TargetCanvasCache,
    evaluate_state,
    optimize_simple_macro,
    simple,
)

LAYER = LayerSpec(1, 0)
# 廉价契约：80² 版图、单 macro、core 40、context 20、pixel 4 → 2×2 core。
BOUNDS = DbuBox(0, 0, 80, 80)
FRAG = FragmentationConfig(corner_length_dbu=8.0, max_segment_length_dbu=16.0,
                           max_displacement_dbu=10.0, miter_limit=4.0)
_SOLVER_DEFAULTS = {"iterations": 2, "initial_step_dbu": 2.0, "decay_every": 2,
                    "epe_distance_dbu": 4.0, "batch_size": 2,
                    "target_cache_bytes": 256 * 256 * 8}


def _macro(**overrides):
    """返回单 macro 规划（默认 80² 版图 2×2 core）。"""
    values = {"macro_grid": (1, 1), "core_size_dbu": 40, "context_dbu": 20,
              "pixel_dbu": 4, "canvas_pixels": 256}
    values.update(overrides)
    return plan_macros(BOUNDS, **values)[0]


def _problem(region, macro=None, polarity="clear", frag=FRAG,
             data_bounds=BOUNDS):
    """把原生 Region 直接包装为 RegionBatch 并生成单 macro problem。

    data_bounds 默认为规划包络 BOUNDS（负板补铬的减区基准）；自定义
    bounds 规划的用例须随规划传入同一包络。
    """
    macro = macro if macro is not None else _macro()
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_macro_problem(batch, LAYER, polarity, frag, macro,
                                 data_bounds=data_bounds)


def _config(**overrides):
    """按默认值组装求解器配置，允许覆盖。"""
    return SimpleMBOPCConfig(**{**_SOLVER_DEFAULTS, **overrides})


@pytest.fixture(scope="session")
def cpu_model():
    """共享一个真实 CPU ICCAD13 模型（资产加载昂贵）。"""
    return ICCAD13Lithography(device="cpu")


class _StubConfig:
    """提供求解器消费的 canvas 与二值阈值的最小配置视图。"""

    def __init__(self, canvas=256, threshold=0.5):
        self.canvas = canvas
        self.print_threshold = threshold


class _PhaseModel:
    """按调用序返回预设变换的假光刻模型，用于确定性方向/停止测试。"""

    def __init__(self, transforms):
        self.device = torch.device("cpu")  # 与契约一致的 CPU 设备
        self.config = _StubConfig()  # 与 problem 一致的 256 画布
        self._transforms = list(transforms)  # 每次调用的输出变换
        self._calls = 0  # 已调用次数
        self.conditions_seen = []  # 每次调用的条件名

    def condition(self, name):
        """返回固定标称条件（假模型不区分工艺角）。"""
        return ProcessCondition(name, "focus", 1.0)

    def forward_many(self, mask, conditions):
        """按第 N 次调用返回第 N 个变换的批量输出。"""
        self.conditions_seen.append([c.name for c in conditions])  # 记录
        transform = self._transforms[min(self._calls, len(self._transforms) - 1)]
        self._calls += 1  # 计数
        return {c.name: transform(mask) for c in conditions}


def _identity(mask):
    """直通变换：输出等于输入（零位移时与 target 全同 → 无违规）。"""
    return mask


def _zero(mask):
    """全暗变换：输出全 0（印刷不足 → inner 违规 → +1 外移）。"""
    return torch.zeros_like(mask)


def _invert(mask):
    """反相变换：inner 不打印且 outer 打印 → 全部 ambiguous → 方向 0。"""
    return 1.0 - mask


def _ones(mask):
    """全亮变换：印刷过量（outer 也打印）→ 全部 -1 内移。"""
    return torch.ones_like(mask)


class TestTargetCache:
    """TargetCanvasCache 的命中、驱逐与禁用路径。"""

    def test_miss_then_hit(self):
        """put 后命中返回同一数组，未命中返回 None。"""
        cache = TargetCanvasCache(256 * 256)  # 恰容一张 canvas
        canvas = np.zeros((256, 256), dtype=np.uint8)
        assert cache.get("mr0c0", 3) is None  # 初始未命中
        cache.put("mr0c0", 3, canvas)  # 写入
        assert cache.get("mr0c0", 3) is canvas  # 命中同对象
        assert cache.get("mr0c0", 3).dtype == np.uint8  # dtype 不变

    def test_replace_same_key(self):
        """同 key 二次 put 替换旧值且字节账目正确。"""
        cache = TargetCanvasCache(256 * 256 * 2)
        first = np.zeros((256, 256), dtype=np.uint8)
        second = np.ones((256, 256), dtype=np.uint8)
        cache.put("m", 0, first)
        cache.put("m", 0, second)  # 替换
        assert cache.get("m", 0) is second  # 新值生效

    def test_lru_evicts_oldest(self):
        """超上限时从最旧端驱逐，刚访问过的保留。"""
        item = 256 * 256  # 每项字节数
        cache = TargetCanvasCache(item * 2)  # 恰容两项
        cache.put("m", 0, np.zeros((256, 256), np.uint8))
        cache.put("m", 1, np.zeros((256, 256), np.uint8))
        assert cache.get("m", 0) is not None  # 触发访问，m0 变最新
        cache.put("m", 2, np.zeros((256, 256), np.uint8))  # 驱逐最旧 m1
        assert cache.get("m", 1) is None  # m1 被驱逐
        assert cache.get("m", 0) is not None  # m0 保留
        assert cache.get("m", 2) is not None  # m2 在

    def test_single_oversized_item_not_cached(self):
        """单项超过总上限时不缓存，也不驱逐既有条目。"""
        cache = TargetCanvasCache(256 * 256 * 2)
        cache.put("m", 0, np.zeros((256, 256), np.uint8))  # 占一半
        cache.put("m", 1, np.zeros((256, 256, 3), np.uint8))  # 超限项
        assert cache.get("m", 1) is None  # 未缓存
        assert cache.get("m", 0) is not None  # 既有条目未被驱逐

    def test_zero_budget_disables(self):
        """0 上限完全禁用缓存。"""
        cache = TargetCanvasCache(0)
        cache.put("m", 0, np.zeros((4, 4), np.uint8))
        assert cache.get("m", 0) is None  # 永不命中

    def test_macro_id_is_part_of_key(self):
        """不同 macro 的同 core 编号互不串用。"""
        cache = TargetCanvasCache(1024)
        cache.put("mr0c0", 0, np.zeros((4, 4), np.uint8))
        assert cache.get("mr0c1", 0) is None  # 跨 macro 不命中

    def test_negative_budget_fails(self):
        """负上限在构造期失败。"""
        with pytest.raises(ValueError):
            TargetCanvasCache(-1)


class TestConfigValidation:
    """SimpleMBOPCConfig 的数值契约。"""

    def test_valid_defaults(self):
        """默认参数集构造成功。"""
        config = _config()
        assert config.iterations == 2
        assert config.initial_step_dbu == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "overrides",
        [{"iterations": 0}, {"iterations": 1.5},
         {"initial_step_dbu": 0.0}, {"initial_step_dbu": float("nan")},
         {"decay_every": 0}, {"epe_distance_dbu": -1.0},
         {"batch_size": 0}, {"target_cache_bytes": -8}],
        ids=["iter=0", "iter=小数", "step=0", "step=nan", "decay=0",
             "epe<0", "batch=0", "cache<0"])
    def test_invalid_values_fail(self, overrides):
        """越界参数在构造期失败。"""
        with pytest.raises(ValueError):
            _config(**overrides)


class TestThresholdPropagation:
    """L2/PVBand/EPE 三类指标显式跟随模型 PrintThresh（审查问题 3）。"""

    def test_all_metrics_receive_model_threshold(self, monkeypatch):
        """模型阈值 0.45 时三个 evaluate_* 都收到 threshold=0.45。"""
        model = _PhaseModel([_identity])  # 直通变换
        model.config = _StubConfig(threshold=0.45)  # 非默认打印阈值
        captured = {}  # 指标名 → 收到的 threshold 列表
        for name in ("evaluate_binary_l2", "evaluate_pvband",
                     "evaluate_edge_probes"):
            real = getattr(simple, name)  # 真实现（透传）

            def spy(*args, _name=name, _real=real, **kwargs):
                """记录 threshold 关键字并透传。"""
                captured.setdefault(_name, []).append(kwargs.get("threshold"))
                return _real(*args, **kwargs)

            monkeypatch.setattr(simple, name, spy)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        optimize_simple_macro(problem, model, _config(iterations=1),
                       TargetCanvasCache(256 * 256 * 8))
        for name, values in captured.items():  # 全部收到模型阈值
            assert values, name  # 至少一次
            assert all(value == 0.45 for value in values), name
        assert len(captured) == 3  # L2/PVBand/EPE 齐全


class TestEntryContracts:
    """evaluate_state 的入口契约拦截。"""

    def _rectangle(self):
        """返回单矩形 problem 与零位移。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        return problem, np.zeros(problem.segments.segment_count)

    def test_wrong_length_fails(self):
        """位移长度不等于段数时失败。"""
        problem, _ = self._rectangle()
        with pytest.raises(ValueError, match="有限向量"):
            evaluate_state(problem, kdb.Region(), np.zeros(3),
                                 _PhaseModel([_identity]), _config(), 2.0,
                                 TargetCanvasCache(0), can_update=True)

    def test_nonfinite_fails(self):
        """含 nan 的位移失败。"""
        problem, zeros = self._rectangle()
        zeros[0] = float("nan")  # 注入 nan
        with pytest.raises(ValueError, match="有限向量"):
            evaluate_state(problem, kdb.Region(), zeros,
                                 _PhaseModel([_identity]), _config(), 2.0,
                                 TargetCanvasCache(0), can_update=True)

    def test_context_displacement_fails(self):
        """context 段（owner=-1）位移非零时失败。"""
        # bbox 外、query_box 内的小块产生 owner=-1 的纯 context 段。
        region = (kdb.Region(kdb.Box(20, 20, 60, 60)) +
                  kdb.Region(kdb.Box(-15, -15, -5, -5)))
        problem = _problem(region)
        context = np.flatnonzero(problem.owner_indices < 0)
        assert len(context) > 0  # 外部小块必为 context 段
        zeros = np.zeros(problem.segments.segment_count)
        zeros[context[0]] = 1.0  # context 段注入位移
        with pytest.raises(ValueError, match="恒为 0"):
            evaluate_state(problem, kdb.Region(), zeros,
                                 _PhaseModel([_identity]), _config(), 2.0,
                                 TargetCanvasCache(0), can_update=True)

    def test_canvas_mismatch_fails(self):
        """模型画布与 problem 画布不一致时失败。"""
        problem, zeros = self._rectangle()
        model = _PhaseModel([_identity])
        model.config = _StubConfig(canvas=128)  # 与 problem 的 256 不一致
        with pytest.raises(ValueError, match="画布"):
            evaluate_state(problem, kdb.Region(), zeros, model,
                                 _config(), 2.0, TargetCanvasCache(0),
                                 can_update=True)

    def test_negative_step_fails(self):
        """负步长失败。"""
        problem, zeros = self._rectangle()
        with pytest.raises(ValueError, match="step_dbu"):
            evaluate_state(problem, kdb.Region(), zeros,
                                 _PhaseModel([_identity]), _config(), -1.0,
                                 TargetCanvasCache(0), can_update=True)


class TestEvaluatePropose:
    """evaluate_state 的指标、方向与批语义。"""

    RECT = kdb.Region(kdb.Box(20, 20, 60, 60))

    def _evaluate(self, model, problem=None, **overrides):
        """按默认配置评价单矩形零位移状态并返回 (step, problem, zeros)。"""
        step_dbu = overrides.pop("step_dbu", 2.0)  # 非配置参数单独取出
        can_update = overrides.pop("can_update", True)  # 同上
        problem = problem if problem is not None else _problem(self.RECT)
        zeros = np.zeros(problem.segments.segment_count)
        region = reconstruct_region(problem, zeros)  # 零位移候选
        step = evaluate_state(
            problem, region, zeros, model, _config(**overrides),
            step_dbu, TargetCanvasCache(0), can_update=can_update)
        return step, problem, zeros

    def test_stub_identity_gives_zero_epe_and_no_move(self):
        """直通模型下零位移与 target 全同：无违规、方向全 0。"""
        step, _, zeros = self._evaluate(_PhaseModel([_identity]))
        assert isinstance(step, SimpleMBOPCStep)
        assert step.epe == 0  # 无违规段
        assert step.valid_probes > 0  # 探针确有生效
        assert step.moved_segments == 0  # 方向全 0 → 不移动
        np.testing.assert_array_equal(step.next_displacements, zeros)

    def test_stub_zero_output_moves_outward(self):
        """全暗输出（印刷不足）使全部 owner 段沿 +法向外移一个步长。"""
        step, problem, _ = self._evaluate(_PhaseModel([_zero]), step_dbu=2.0)
        owned = problem.owner_indices >= 0
        assert step.epe == int(owned.sum())  # 每段一个 inner 违规
        values = step.next_displacements[owned]
        assert np.all(values == 2.0)  # +1 方向 × 步长 2
        assert np.all(step.next_displacements[~owned] == 0.0)  # context 恒 0

    def test_stub_invert_gives_all_ambiguous_zero_direction(self):
        """反相输出（inner 暗且 outer 亮）全为歧义：方向 0 但 EPE 非零。"""
        step, _, _ = self._evaluate(_PhaseModel([_invert]))
        assert step.epe > 0  # 歧义段计入违规数
        assert step.ambiguous_probes == step.valid_probes  # 全部歧义
        assert step.moved_segments == 0  # 方向 0 不移动

    def test_can_update_false_returns_current(self):
        """can_update=False 只评价不提案：next 等于 current、moved=0。"""
        step, _, zeros = self._evaluate(
            _PhaseModel([_zero]), can_update=False, step_dbu=0.0)
        assert step.epe > 0  # 指标照常计算
        np.testing.assert_array_equal(step.next_displacements, zeros)
        assert step.moved_segments == 0

    def test_direction_magnitude_is_step_and_clip(self):
        """方向幅度恰为步长；已在上限的段被裁回。"""
        problem = _problem(self.RECT)
        segment_count = problem.segments.segment_count
        current = np.zeros(segment_count)
        current[problem.owner_indices >= 0] = 10.0  # 全部顶到上限
        region = reconstruct_region(problem, current)
        step = evaluate_state(
            problem, region, current, _PhaseModel([_zero]), _config(),
            4.0, TargetCanvasCache(0), can_update=True)
        # 全暗 → +1 外移：10+4 超上限 10，裁回 10。
        assert np.all(step.next_displacements[problem.owner_indices >= 0] == 10.0)

    def test_batch_size_does_not_change_result(self):
        """batch_size=1 与 4（全 core 一批）产生相同指标与位移。"""
        single, _, _ = self._evaluate(_PhaseModel([_zero]), batch_size=1)
        full, _, _ = self._evaluate(_PhaseModel([_zero]), batch_size=4)
        assert single.epe == full.epe  # 指标一致
        assert single.l2 == full.l2
        np.testing.assert_array_equal(  # 提案一致（同步读同一 current）
            single.next_displacements, full.next_displacements)

    def test_forward_many_once_per_batch(self, monkeypatch, cpu_model):
        """每个 GPU batch 恰一次三条件 forward_many。"""
        calls = []
        real = ICCAD13Lithography.forward_many

        def _counting(self, mask, conditions):
            """记录每次调用的条件数并透传。"""
            calls.append(len(conditions))
            return real(self, mask, conditions)
        monkeypatch.setattr(ICCAD13Lithography, "forward_many", _counting)
        problem = _problem(self.RECT)
        zeros = np.zeros(problem.segments.segment_count)
        region = reconstruct_region(problem, zeros)
        evaluate_state(  # 4 core / batch 2 → 恰 2 次调用
            problem, region, zeros, cpu_model, _config(batch_size=2),
            2.0, TargetCanvasCache(0), can_update=True)
        assert calls == [3, 3]  # 每批一次、每次三条件


    def test_mask_canvas_receives_surround_geometry(self, monkeypatch):
        """负板组批栅格化的 Region 已含包络外补铬（覆盖到查询边界）。

        2026-08-22 几何方案：暗界参数已移除，环带语义由 prepare 阶段的
        补铬几何承载——这里断言该几何确实到达每一次栅格化。
        """
        problem = _problem(self.RECT, polarity="opaque")
        seen = []
        real = simple.rasterize_mask_canvas

        def spy(region_, box, pixel, canvas, *, polarity):
            """记录栅格化 Region 的包络后透传。"""
            seen.append(region_.bbox())
            return real(region_, box, pixel, canvas, polarity=polarity)

        monkeypatch.setattr(simple, "rasterize_mask_canvas", spy)
        zeros = np.zeros(problem.segments.segment_count)
        evaluate_state(
            problem, reconstruct_region(problem, zeros), zeros,
            _PhaseModel([_identity]), _config(), 2.0,
            TargetCanvasCache(0), can_update=False)
        query = problem.macro.query_box
        # 零位移候选含补铬：Region 包络覆盖到 query 四角（宽高由 context
        # 扩张与 BOUNDS 规划共同决定），clear 对照仅覆盖图形自身包络。
        assert seen and all(
            box.left <= query.left and box.bottom <= query.bottom
            and box.right >= query.right and box.top >= query.top
            for box in seen)

    def test_cache_hit_avoids_retarget_rasterization(self, monkeypatch):
        """第二次评价命中缓存：target 不再栅格化，只有当前 mask 栅格。"""
        problem = _problem(self.RECT)
        zeros = np.zeros(problem.segments.segment_count)
        region = reconstruct_region(problem, zeros)
        model = _PhaseModel([_identity])
        cache = TargetCanvasCache(256 * 256 * 8)  # 足够容纳 4 个 core
        counts = []
        real_raster = simple.rasterize_mask_canvas

        def _counting(region_, box, pixel, canvas, *, polarity):
            """计数栅格化调用并透传。"""
            counts.append(1)
            return real_raster(region_, box, pixel, canvas, polarity=polarity)
        monkeypatch.setattr(simple, "rasterize_mask_canvas", _counting)
        evaluate_state(problem, region, zeros, model, _config(),
                             2.0, cache, can_update=True)
        first = len(counts)  # 4 core × (target + mask) = 8 次
        evaluate_state(problem, region, zeros, model, _config(),
                             2.0, cache, can_update=True)
        second = len(counts) - first  # 第二次新增的栅格调用
        assert first == 8
        assert second == 4  # target 全命中 → 只剩 4 次 mask 栅格

    def test_empty_owner_core_still_counts_as_tile(self):
        """图形只落在一个 core 时，其余空 owner core 不炸且计入 tile 进度。"""
        macro = _macro(core_size_dbu=40)  # 2×2 core
        problem = _problem(kdb.Region(kdb.Box(0, 0, 20, 20)), macro)  # 只在 c0
        zeros = np.zeros(problem.segments.segment_count)
        region = reconstruct_region(problem, zeros)
        tiles = []
        evaluate_state(  # 全部 4 core 都要过一遍
            problem, region, zeros, _PhaseModel([_identity]), _config(),
            2.0, TargetCanvasCache(0), can_update=True,
            on_tiles_completed=tiles.append)
        assert sum(tiles) == problem.macro.core_count  # 空 core 也计入
        assert tiles == [2, 2]  # batch 2 → 每批报告 2 个 tile

    def test_real_model_metrics_finite(self, cpu_model):
        """真实 ICCAD13 CPU 模型输出有限指标与有限提案。"""
        step, problem, _ = self._evaluate(cpu_model)
        assert step.epe >= 0 and step.l2 >= 0 and step.pvband >= 0
        assert step.valid_probes > 0
        assert np.all(np.isfinite(step.next_displacements))
        owned = problem.owner_indices >= 0
        # 离散方向 × 步长：owner 段位移只能取 {-2, 0, +2}。
        assert set(np.unique(step.next_displacements[owned])) <= {-2.0, 0.0, 2.0}


class TestOptimizeMacro:
    """optimize_simple_macro 的轮次语义、停止原因与最佳状态选择。"""

    def _run(self, model, region=None, **overrides):
        """按默认配置跑单 macro 完整迭代。"""
        problem = _problem(region if region is not None
                           else kdb.Region(kdb.Box(20, 20, 60, 60)))
        return optimize_simple_macro(problem, model, _config(**overrides),
                              TargetCanvasCache(0))

    def test_baseline_only_when_zero_epe(self):
        """直通模型 baseline 无违规：只评价一次，best 是零位移。"""
        result = self._run(_PhaseModel([_identity]))
        assert isinstance(result, SimpleMBOPCResult)
        assert result.stop_reason == "zero_epe"
        assert len(result.records) == 1  # 无移动后状态
        assert result.best_state_index == 0  # baseline 最优
        np.testing.assert_array_equal(
            result.best_displacements,
            np.zeros_like(result.best_displacements))
        assert result.stop_detail is None

    def test_records_semantics_for_one_iteration(self):
        """iterations=1 恰两条记录：baseline 与一次移动后评价。"""
        result = self._run(_PhaseModel([_zero, _identity]), iterations=1)
        assert [r.state_index for r in result.records] == [0, 1]
        baseline = result.records[0]
        assert baseline.step_dbu == 0.0  # baseline 无步长
        assert baseline.moved_segments == 0  # baseline 无移动
        assert baseline.epe > 0  # 全暗输出必有违规
        first = result.records[1]
        assert first.step_dbu == pytest.approx(2.0)  # 产生 Round 1 的步长
        assert first.moved_segments > 0  # +1 方向外移了 owner 段
        # 直通输出使大部分违规消失（外移让印刷覆盖探针），少数边缘段受像素
        # 量化影响仍违规——只断言严格改善，不断言精确归零。
        assert 0 < first.epe < baseline.epe  # 移动后 EPE 改善
        assert result.stop_reason == "iteration_limit"  # 轮次用尽（EPE 未归零）
        assert result.best_state_index == 1  # EPE 更优的移动后状态胜出
        assert int(np.count_nonzero(result.best_displacements)) > 0

    def test_no_update_stops_with_ambiguous_only(self):
        """全歧义方向 0 且 EPE 非零时按 no_update 停止，不重复评价同状态。"""
        result = self._run(_PhaseModel([_invert, _invert]))
        assert result.stop_reason == "no_update"
        # 提案与当前一致时直接停止（同一状态再评无新信息），只有 baseline。
        assert len(result.records) == 1
        assert result.records[0].moved_segments == 0

    def test_zero_epe_during_iteration_with_pixel_aligned_step(self):
        """步长为像素整数倍时移动后直通输出达成零违规（循环内 zero_epe）。"""
        result = self._run(_PhaseModel([_zero, _identity, _identity]),
                           iterations=2, initial_step_dbu=4.0)
        assert result.stop_reason == "zero_epe"
        assert len(result.records) == 3  # baseline + 两轮（Round 2 归零）
        assert result.records[-1].epe == 0  # 最后一轮零违规
        assert result.best_state_index == 2

    def test_iteration_limit_records_all_rounds(self):
        """持续违规且持续移动时跑满轮次并按 iteration_limit 停止。"""
        result = self._run(_PhaseModel([_zero, _zero, _zero, _zero]),
                           iterations=2)
        assert result.stop_reason == "iteration_limit"
        assert [r.state_index for r in result.records] == [0, 1, 2]
        assert result.best_state_index == 0  # EPE 相同（全违规）保留较早 baseline

    def test_invalid_geometry_stops_and_keeps_last_best(self, monkeypatch):
        """非法候选终止迭代、保留最后合法 best、原因写入 stop_detail。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        real_reconstruct = simple.reconstruct_region
        calls = {"moved": 0}

        def _failing_on_first_moved(problem_, displacements):
            """首个非零位移候选（Round 1）抛出非法几何。"""
            if np.any(displacements != 0.0):
                calls["moved"] += 1
                if calls["moved"] == 1:
                    raise ReconstructionError("hole escaped its hull")
            return real_reconstruct(problem_, displacements)
        monkeypatch.setattr(simple, "reconstruct_region", _failing_on_first_moved)
        model = _PhaseModel([_zero, _zero, _zero])
        result = optimize_simple_macro(problem, model, _config(),
                                TargetCanvasCache(0))
        assert result.stop_reason == "invalid_geometry"
        assert "hole escaped" in result.stop_detail  # 原因不吞掉
        assert len(result.records) == 1  # Round 1 未能评价
        assert result.best_state_index == 0  # 保留 baseline best

    def test_step_decays_every_configured_rounds(self):
        """步长按 decay_every 周期减半（decay=1 即每轮减半）。"""
        result = self._run(  # 持续全暗保证每轮都移动（2+1+0.5 远离上限）
            _PhaseModel([_zero, _zero, _zero, _zero]),
            iterations=3, initial_step_dbu=2.0, decay_every=1)
        steps = [r.step_dbu for r in result.records]
        assert steps[0] == 0.0  # baseline 无步长
        assert steps[1] == pytest.approx(2.0)  # Round 1 用初始步长
        assert steps[2] == pytest.approx(1.0)  # Round 2 减半
        assert steps[3] == pytest.approx(0.5)  # Round 3 再减半

    def test_l2_does_not_break_epe_ties(self, monkeypatch):
        """L2 只是诊断：EPE 平局保留较早轮，被篡改的 L2 不能翻盘。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _PhaseModel([_zero, _zero, _zero])  # 持续全暗 → 各轮 EPE 相同
        monkeypatch.setattr(  # L2 恒 0（若参与比较也偏向不了任何轮）
            simple, "evaluate_binary_l2", lambda *a, **kw: 0)
        result = optimize_simple_macro(problem, model, _config(iterations=2),
                                TargetCanvasCache(0))
        assert result.stop_reason == "iteration_limit"
        assert result.best_state_index == 0  # EPE 平局 → 保留最早（baseline）

    def test_progress_counts_all_evaluated_tiles(self):
        """进度回调总数 = (iterations+1)×core_count，每次为批内真实 tile 数。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        tiles = []
        result = optimize_simple_macro(  # 全暗持续移动 → 3 次评价全跑
            problem, _PhaseModel([_zero, _zero, _zero]), _config(iterations=2),
            TargetCanvasCache(0), on_tiles_completed=tiles.append)
        core_count = problem.macro.core_count
        assert sum(tiles) == 3 * core_count  # baseline + 2 轮 × 全部 core
        assert all(count <= 2 for count in tiles)  # 每批不超 batch_size
        assert result.stop_reason == "iteration_limit"

    def test_entry_guard_rejects_oversized_step(self):
        """步长超过 problem 位移上限时入口失败。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        with pytest.raises(ValueError, match="位移上限"):
            optimize_simple_macro(problem, _PhaseModel([_identity]),
                           _config(initial_step_dbu=11.0), TargetCanvasCache(0))
        with pytest.raises(ValueError, match="context"):
            optimize_simple_macro(problem, _PhaseModel([_identity]),
                           _config(epe_distance_dbu=21.0), TargetCanvasCache(0))


class TestGeometryMatrix:
    """复杂图形下迭代完成或以合法停止收场（§16.3 矩阵）。"""

    def _run(self, region, polarity="clear", **overrides):
        """构造 problem 并以真实 CPU 模型完成一次小规模迭代。"""
        problem = _problem(region, polarity=polarity)
        return optimize_simple_macro(problem, self._model, _config(
            iterations=overrides.pop("iterations", 1),
            batch_size=4, **overrides), TargetCanvasCache(0)), problem

    @pytest.fixture(autouse=True)
    def _shared_model(self, cpu_model):
        """全部矩阵用例共享会话级真实模型。"""
        self._model = cpu_model

    def _assert_healthy(self, result, problem):
        """断言结果满足全部结构不变量。"""
        assert result.stop_reason in {  # 五种合法停止
            "zero_epe", "no_update", "invalid_geometry",
            "insufficient_probes", "iteration_limit"}
        assert result.records[0].state_index == 0  # baseline 在首位
        if result.stop_reason == "invalid_geometry":
            assert result.stop_detail  # 原因非空
        else:
            assert result.stop_detail is None
        assert 0 <= result.best_state_index < len(result.records)  # best 是已评价轮
        best = result.best_displacements
        assert np.all(np.isfinite(best))  # 位移有限
        assert np.all(best[problem.owner_indices < 0] == 0.0)  # context 归零
        limit = float(problem.fragmentation.max_displacement_dbu)
        assert np.all(np.abs(best) <= limit + 1e-9)  # 裁剪在上限内

    def test_plain_rectangle(self):
        """普通矩形完成一次移动后评价。"""
        result, problem = self._run(kdb.Region(kdb.Box(20, 20, 60, 60)))
        self._assert_healthy(result, problem)

    def test_narrow_wall_with_hole(self):
        """4nm 窄壁中空图形（探针距离 4nm）不产生非法状态。"""
        region = (kdb.Region(kdb.Box(10, 10, 70, 70)) -
                  kdb.Region(kdb.Box(14, 14, 66, 66)))  # 壁宽 4nm
        result, problem = self._run(region)
        self._assert_healthy(result, problem)

    def test_concave_polygon(self):
        """凹（L 形）多边形。"""
        polygon = kdb.Polygon([(10, 10), (70, 10), (70, 30), (30, 30),
                               (30, 70), (10, 70)])
        result, problem = self._run(kdb.Region(polygon))
        self._assert_healthy(result, problem)

    def test_multiple_polygons_and_holes(self):
        """多 polygon 且各带 hole。"""
        region = ((kdb.Region(kdb.Box(5, 5, 35, 35)) -
                   kdb.Region(kdb.Box(12, 12, 28, 28))) +
                  (kdb.Region(kdb.Box(45, 45, 75, 75)) -
                   kdb.Region(kdb.Box(52, 52, 68, 68))))
        result, problem = self._run(region)
        self._assert_healthy(result, problem)

    def test_diagonal_edge(self):
        """45° 斜边（单位外法向支持非 Manhattan 边）。"""
        polygon = kdb.Polygon([(10, 10), (70, 10), (10, 70)])  # 直角三角形
        result, problem = self._run(kdb.Region(polygon))
        self._assert_healthy(result, problem)
        # 斜边段存在且被评价：有效探针覆盖非水平/垂直段。
        assert result.records[0].valid_probes > 0

    def test_polygon_spans_three_cores(self):
        """横跨至少三个 core 的横条。"""
        macro = _macro(core_size_dbu=20)  # 80/20 = 4×4 core
        problem = _problem(kdb.Region(kdb.Box(5, 35, 75, 45)), macro)
        result = optimize_simple_macro(problem, self._model, _config(
            iterations=1, batch_size=4), TargetCanvasCache(0))
        self._assert_healthy(result, problem)

    def test_diagonal_spans_macro_boundary(self):
        """斜边跨 macro 切线的多 macro 场景（2×2 macro 各自独立求解）。"""
        bounds = DbuBox(0, 0, 160, 160)
        macros = plan_macros(bounds, macro_grid=(2, 2), core_size_dbu=40,
                             context_dbu=20, pixel_dbu=4, canvas_pixels=256)
        polygon = kdb.Polygon([(10, 10), (150, 10), (10, 150)])  # 斜边跨中心
        region = kdb.Region(polygon)
        for macro in macros:  # 每个 macro 独立跑一轮
            problem = _problem(region, macro, data_bounds=bounds)
            result = optimize_simple_macro(problem, self._model, _config(
                iterations=1, batch_size=4), TargetCanvasCache(0))
            self._assert_healthy(result, problem)

    def test_opaque_polarity(self):
        """opaque 极性同样完成迭代（法向翻转不进入求解器分支）。"""
        region = kdb.Region(kdb.Box(20, 20, 60, 60))
        result, problem = self._run(region, polarity="opaque")
        self._assert_healthy(result, problem)

    def test_empty_macro(self):
        """空 macro（无任何图形）零段零探针，baseline 即零违规。"""
        problem = _problem(kdb.Region())  # 空 Region
        assert problem.segments.segment_count == 0
        result = optimize_simple_macro(problem, self._model, _config(iterations=1),
                                TargetCanvasCache(0))
        assert result.stop_reason == "zero_epe"  # 无违规即停
        assert len(result.records) == 1
        assert result.best_displacements.shape == (0,)  # 空位移向量

    def test_narrow_wall_with_oversized_probes_is_insufficient(self):
        """2nm 窄壁 + 8nm 探针：全部探针无效时报告无法评价而非零违规。"""
        # 壁宽 2nm（外框 10..70、hole 12..68），探针距离 8nm 越过壁落入
        # 异侧，inner/outer 的 target 语义全部不成立 → valid_probes == 0。
        region = (kdb.Region(kdb.Box(10, 10, 70, 70)) -
                  kdb.Region(kdb.Box(12, 12, 68, 68)))  # 壁宽 2nm
        problem = _problem(region)
        result = optimize_simple_macro(problem, self._model, _config(
            iterations=1, epe_distance_dbu=8.0), TargetCanvasCache(0))
        assert result.stop_reason == "insufficient_probes"  # 不是 zero_epe
        assert result.records[0].valid_probes == 0  # 确无有效探针
        assert result.records[0].epe == 0  # epe 恒 0（无法评价 ≠ 零违规）
        assert "有效 EPE 探针 0 个" in result.stop_detail  # 原因在案
        assert result.best_state_index == 0  # 保留零位移 baseline
        assert len(result.records) == 1  # 无移动后状态

    def test_hull_shrinking_into_hole_is_invalid_geometry(self):
        """外轮廓内移 + hole 外扩同时进行时 hole 越出 hull，重建守卫拦截。"""
        # 壁 15nm；全亮输出（印刷过量）→ 全 -1：hull 边内移、hole 边外扩，
        # 一轮 10nm 后 hole(25..55) 越出 hull(30..50) → hole escaped its hull。
        region = (kdb.Region(kdb.Box(20, 20, 60, 60)) -
                  kdb.Region(kdb.Box(35, 35, 45, 45)))  # 壁 15nm
        problem = _problem(region)
        result = optimize_simple_macro(problem, _PhaseModel([_ones, _ones]),
                                _config(iterations=2, initial_step_dbu=10.0),
                                TargetCanvasCache(0))
        assert result.stop_reason == "invalid_geometry"  # 守卫拦截
        assert "重建失败" in result.stop_detail  # 原因在案
        assert result.best_state_index == 0  # 非法候选前的 baseline 保留

    def test_rectangle_edges_crossing_is_invalid_geometry(self):
        """矩形大幅内移到边交叉/共线退化时重建守卫拦截（真构造）。"""
        # 矩形 20..60 宽 40；全亮输出 → 全 -1，一步 20nm 内移使四边共线
        # 交叉：实测重建以「ring 少于三顶点」的 ValueError 形态失败，被
        # invalid_geometry 路径捕获（更大幅度翻转会被 miter 解析成反向
        # ring，共线退化是最先触发的守卫形态）。
        wide = FragmentationConfig(corner_length_dbu=8.0,
                                   max_segment_length_dbu=16.0,
                                   max_displacement_dbu=30.0, miter_limit=4.0)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), frag=wide)
        result = optimize_simple_macro(problem, _PhaseModel([_ones, _ones]),
                                _config(iterations=1, initial_step_dbu=20.0),
                                TargetCanvasCache(0))
        assert result.stop_reason == "invalid_geometry"  # 守卫拦截
        assert result.stop_detail is not None  # 原因在案
        assert result.best_state_index == 0  # 非法候选前的 baseline 保留
        assert len(result.records) == 1  # 退化候选未进入评价


class TestRealModelCuda:
    """CUDA 可用时的真实模型直通验证。"""

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="当前环境没有 CUDA")
    def test_cuda_optimize_simple_macro_completes(self):
        """CUDA 设备上完整迭代完成且指标有限。"""
        model = ICCAD13Lithography(device="cuda")
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        result = optimize_simple_macro(problem, model, _config(iterations=1),
                                TargetCanvasCache(0))
        assert result.stop_reason in {  # 五种合法停止
            "zero_epe", "no_update", "invalid_geometry",
            "insufficient_probes", "iteration_limit"}
        assert np.all(np.isfinite(result.best_displacements))
