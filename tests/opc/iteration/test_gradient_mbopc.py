"""梯度 MB-OPC 求解器的代理梯度、状态语义、几何矩阵与真实模型集成测试。"""

import inspect

import klayout.db as kdb
import numpy as np
import pytest
import torch

from layout import DbuBox, LayerSpec, RegionBatch
from lithography import ICCAD13Lithography, ProcessCondition
from opc.input import (
    ownership_canvas,
    plan_macros,
    points_to_canvas,
    rasterize_mask_canvas,
)
from opc.input.edge import (
    MacroProblem,
    prepare_macro_problem,
    reconstruct_region,
)
from opc.input.edge.fragmentation import FragmentationConfig
from opc.iteration.mbopc import (
    GradientMBOPCConfig,
    TargetCanvasCache,
    optimize_gradient_macro,
)
from opc.iteration.mbopc import gradient as gradient_module
from opc.iteration.mbopc.gradient import _EdgeGradientMask

LAYER = LayerSpec(1, 0)
# 廉价契约：80² 版图、单 macro、core 40、context 20、pixel 4 → 2×2 core。
BOUNDS = DbuBox(0, 0, 80, 80)
FRAG = FragmentationConfig(corner_length_dbu=8.0, max_segment_length_dbu=16.0,
                           max_displacement_dbu=10.0, miter_limit=4.0)
_GRADIENT_DEFAULTS = {"iterations": 2, "learning_rate_dbu": 1.0,
                      "weight_nominal_l2": 1.0, "weight_process_l2": 0.5,
                      "weight_pvband": 0.1, "epe_distance_dbu": 4.0,
                      "batch_size": 2, "target_cache_bytes": 256 * 256 * 8}
_CACHE_BUDGET = 256 * 256 * 8  # 恰容 8 张 canvas，测试几何足够全员命中


def _macro(**overrides):
    """返回单 macro 规划（默认 80² 版图 2×2 core）。"""
    values = {"macro_grid": (1, 1), "core_size_dbu": 40, "context_dbu": 20,
              "pixel_dbu": 4, "canvas_pixels": 256}
    values.update(overrides)
    return plan_macros(BOUNDS, **values)[0]


def _problem(region, macro=None, polarity="clear", frag=FRAG):
    """把原生 Region 直接包装为 RegionBatch 并生成单 macro problem。"""
    macro = macro if macro is not None else _macro()
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_macro_problem(batch, LAYER, polarity, frag, macro)


def _config(**overrides):
    """按默认值组装梯度求解器配置，允许覆盖。"""
    return GradientMBOPCConfig(**{**_GRADIENT_DEFAULTS, **overrides})


@pytest.fixture(scope="session")
def cpu_model():
    """共享一个真实 CPU ICCAD13 模型（资产加载昂贵）。"""
    return ICCAD13Lithography(device="cpu")


class _StubConfig:
    """提供求解器消费的 canvas 与二值阈值的最小配置视图。"""

    def __init__(self, canvas=256, threshold=0.5):
        self.canvas = canvas
        self.print_threshold = threshold


class _LinearModel:
    """线性可微假光刻：printed = mask×dose，逐像素独立、无饱和。"""

    def __init__(self, doses=None):
        default = {"nominal": 1.3, "dose_max": 1.4, "defocus_min": 1.2}
        doses = doses if doses is not None else default
        self.device = torch.device("cpu")  # 与契约一致的 CPU 设备
        self.config = _StubConfig()  # 与 problem 一致的 256 画布
        self.calls = 0  # forward_many 调用计数
        self._conditions = {
            name: ProcessCondition(name,
                                   "defocus" if name == "defocus_min"
                                   else "focus", dose)
            for name, dose in doses.items()}

    def condition(self, name):
        """返回固定剂量条件。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """输出等于 mask×dose（线性链路保证梯度方向可手推）。"""
        self.calls += 1  # 计数
        return {c.name: mask * self._conditions[c.name].dose
                for c in conditions}


class _StateWeightModel:
    """按状态序号切换剂量的假模型，用于确定性 best/record 测试。"""

    def __init__(self, weights, batches_per_state):
        self.device = torch.device("cpu")  # CPU 契约设备
        self.config = _StubConfig()  # 256 画布视图
        self._weights = list(weights)  # 每状态的统一剂量
        self._batches = batches_per_state  # 每状态的 forward 批数
        self._calls = 0  # 累计调用数
        self._conditions = {name: ProcessCondition(name, "focus", 1.0)
                            for name in ("nominal", "dose_max",
                                         "defocus_min")}

    def condition(self, name):
        """返回单位剂量条件（剂量经 forward 路径生效）。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """第 s 状态的全部批输出 mask×weights[s]。"""
        state = self._calls // self._batches  # 当前状态序号
        weight = self._weights[min(state, len(self._weights) - 1)]
        self._calls += 1  # 计数
        return {c.name: mask * weight for c in conditions}


class _NaNModel(_LinearModel):
    """输出含 NaN 的假模型，用于触发数值失效异常。"""

    def forward_many(self, mask, conditions):
        """输出全 NaN（loss 与梯度随之非有限）。"""
        self.calls += 1
        return {c.name: mask * float("nan") for c in conditions}


def _optimize(problem, model, config, cache=None):
    """以满预算缓存运行一次梯度求解（测试默认路径）。"""
    cache = cache if cache is not None else TargetCanvasCache(_CACHE_BUDGET)
    return optimize_gradient_macro(problem, model, config, cache)


class TestConfigValidation:
    """GradientMBOPCConfig 的数值契约。"""

    def test_valid_defaults(self):
        """默认参数集构造成功。"""
        config = _config()
        assert config.iterations == 2
        assert config.learning_rate_dbu == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "overrides",
        [{"iterations": 0}, {"iterations": True}, {"learning_rate_dbu": 0.0},
         {"learning_rate_dbu": float("nan")},
         {"weight_nominal_l2": -0.1}, {"weight_pvband": float("inf")},
         {"weight_nominal_l2": 0.0, "weight_process_l2": 0.0,
          "weight_pvband": 0.0},
         {"epe_distance_dbu": 0.0}, {"batch_size": 0},
         {"target_cache_bytes": -1}],
        ids=["iter=0", "iter=bool", "lr=0", "lr=nan", "w<0", "w=inf",
             "全零权重", "epe=0", "batch=0", "cache<0"])
    def test_invalid_values_fail(self, overrides):
        """越界参数在构造期失败。"""
        with pytest.raises(ValueError):
            _config(**overrides)


class TestEdgeGradientMask:
    """_EdgeGradientMask 的前向直通与 Algorithm 4 反向公式。"""

    def _apply(self, hard, local, slots, mids):
        """在 CPU 上执行一次 apply 并返回输出。"""
        return _EdgeGradientMask.apply(hard, local, slots, mids)

    def test_forward_preserves_exact_raster(self):
        """输出与输入逐位相同、dtype/shape/device 不变（REQ-003）。"""
        hard = torch.tensor([[[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]]])  # [1,2,3]
        hard = hard.repeat(2, 1, 1)  # 两批
        local = torch.zeros(2, requires_grad=True)  # 任意位移连接
        slots = torch.tensor([0, 1], dtype=torch.int64)
        mids = torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.float32)
        out = self._apply(hard, local, slots, mids)
        assert out.shape == hard.shape  # shape 不变
        assert out.dtype == hard.dtype  # dtype 不变
        assert out.device == hard.device  # device 不变
        assert torch.equal(out, hard)  # 数值逐位直通

    def test_backward_is_two_times_bilinear_midpoint(self):
        """单条目梯度恰为 2×双线性值；越界为 0；重复索引求和（REQ-004）。"""
        size = 8  # 8×8 图
        grad_output = torch.arange(2 * size * size, dtype=torch.float32)
        grad_output = grad_output.reshape(2, size, size)  # 图 1 基址 64
        local = torch.zeros(4, requires_grad=True)  # 四条独立 membership
        slots = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        mids = torch.tensor([[1.5, 2.5], [7.0, 7.0], [-3.0, 4.0], [2.0, 2.0]],
                            dtype=torch.float32)
        hard = torch.zeros(2, size, size)  # 数值不参与反向
        out = self._apply(hard, local, slots, mids)
        (out * grad_output).sum().backward()  # 直接以已知梯度图反传
        # 半像素点双线性四角均值：(17+18+25+26)/4 = 21.5；
        # 边界整点 (7,7)→127；越界点恒 0；内部整点 (2,2)→82。
        expect = torch.tensor([2 * 21.5, 2 * 127.0, 0.0, 2 * 82.0])
        assert torch.allclose(local.grad, expect)  # 2·g_mid 精确公式
        # 同一参数被两条 membership 引用时由 autograd 求和（图 0 整点）。
        shared = torch.zeros(1, requires_grad=True)
        gathered = shared[torch.tensor([0, 0])]  # 重复索引
        out2 = self._apply(
            torch.zeros(1, size, size), gathered,
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32))
        (out2 * grad_output[:1]).sum().backward()
        assert torch.allclose(shared.grad, torch.tensor([2 * (9.0 + 18.0)]))

    def test_backward_source_has_no_python_loop_or_item(self):
        """反向实现不含逐段 Python 循环与 .item() 同步（PERF-004）。"""
        source = inspect.getsource(_EdgeGradientMask.backward)
        assert ".item(" not in source  # 禁逐条目设备同步
        assert "\n        for " not in source  # 禁逐段 Python 循环


class TestEntryContracts:
    """optimize_gradient_macro 的入口契约拦截。"""

    def _rectangle(self):
        """返回单矩形 problem。"""
        return _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))

    def test_canvas_mismatch_fails(self):
        """模型画布与 problem 不一致在 GPU 前失败。"""
        model = _LinearModel()
        model.config = _StubConfig(canvas=128)  # 与 problem 的 256 不一致
        with pytest.raises(ValueError, match="画布"):
            _optimize(self._rectangle(), model, _config())

    def test_epe_beyond_context_fails(self):
        """探针距离超过 context 在 GPU 前失败。"""
        model = _LinearModel()
        with pytest.raises(ValueError, match="context"):
            _optimize(self._rectangle(), model,
                      _config(epe_distance_dbu=25.0))  # context 仅 20


class TestEmptyAndContext:
    """空问题与纯 context macro 的退化路径。"""

    def test_empty_macro_returns_baseline(self):
        """零段 macro 返回单条零记录与 no_owned_segments。"""
        problem = _problem(kdb.Region())  # 空几何
        result = _optimize(problem, _LinearModel(), _config())
        assert result.stop_reason == "no_owned_segments"
        assert len(result.records) == 1  # 只有一条 baseline 记录
        assert result.records[0].state_index == 0
        assert result.best_state_index == 0
        assert len(result.best_displacements) == 0  # 零段

    def test_context_only_macro_returns_baseline(self):
        """只含 context 段的 macro 不建参数、直接以零位移停止。"""
        region = kdb.Region(kdb.Box(-15, -15, -5, -5))  # 在 bounds 外 context 内
        problem = _problem(region)
        owner = problem.owner_indices >= 0
        assert owner.sum() == 0  # 全部段只作 context 可见
        assert problem.segments.segment_count > 0  # 有段但全部只读
        result = _optimize(problem, _LinearModel(), _config())
        assert result.stop_reason == "no_owned_segments"
        assert np.all(result.best_displacements == 0.0)  # context 恒 0
        assert len(result.best_displacements) == problem.segments.segment_count

    def test_context_segments_have_no_parameter(self, monkeypatch):
        """owner+context 混合时参数数恰为 owner 段数、context 恒 0。"""
        region = kdb.Region(kdb.Box(20, 20, 60, 60))
        region.insert(kdb.Box(-15, -15, -5, -5))  # 补充 context 段
        problem = _problem(region)
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        assert len(owner_ids) < problem.segments.segment_count  # 有 context 段
        observed = {}  # SpyAdam 记录的参数规模

        class _SpyAdam(torch.optim.Adam):
            """记录参数 numel 的 Adam 代理。"""

            def __init__(self, params, *args, **kwargs):
                super().__init__(params, *args, **kwargs)
                observed["numel"] = int(next(iter(params)).numel())

        monkeypatch.setattr(torch.optim, "Adam", _SpyAdam)
        result = _optimize(problem, _LinearModel(), _config())
        assert observed["numel"] == len(owner_ids)  # 参数只为 owner 分配
        context = problem.owner_indices < 0
        assert np.all(result.best_displacements[context] == 0.0)  # 恒 0


class TestLossFormula:
    """连续 loss 公式、ownership 屏蔽与全局分母（REQ-006/007）。"""

    @staticmethod
    def _expected_losses(problem, model, region):
        """独立 numpy 路径复算三分量 loss（不经求解器批循环）。"""
        conditions = (model.condition("nominal"), model.condition("dose_max"),
                      model.condition("defocus_min"))
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        reference = reconstruct_region(
            problem, np.zeros(problem.segments.segment_count, dtype=np.float64))
        nominal_sum = process_sum = pv_sum = 0.0
        total = 0
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu,
                                         canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(reference, spec.context_box,
                                           pixel_dbu, canvas,
                                           polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box,
                                   pixel_dbu, canvas)
            total += int(own.sum())  # 全局分母 P
            batch = torch.from_numpy(mask)[None]  # [1,H,W]
            printed = model.forward_many(batch, conditions)
            nom = printed["nominal"][0].numpy()
            dmax = printed["dose_max"][0].numpy()
            dmin = printed["defocus_min"][0].numpy()
            nominal_sum += float(((nom - target) ** 2 * own).sum())
            process_sum += float((((dmax - target) ** 2
                                  + (dmin - target) ** 2) * own).sum())
            pv_sum += float(((dmax - dmin) ** 2 * own).sum())
        return nominal_sum / total, process_sum / total, pv_sum / total, total

    def test_continuous_losses_match_independent_recompute(self):
        """三分量与加权 total 和独立复算一致（确定性线性模型）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _LinearModel()
        config = _config()
        result = _optimize(problem, model, config)
        expected = self._expected_losses(problem, model, reconstruct_region(
            problem, np.zeros(problem.segments.segment_count,
                              dtype=np.float64)))
        record = result.records[0]
        assert record.nominal_l2_loss == pytest.approx(expected[0], rel=1e-6)
        assert record.process_l2_loss == pytest.approx(expected[1], rel=1e-6)
        assert record.pvband_loss == pytest.approx(expected[2], rel=1e-6)
        total = (config.weight_nominal_l2 * expected[0]
                 + config.weight_process_l2 * expected[1]
                 + config.weight_pvband * expected[2])
        assert record.total_loss == pytest.approx(total, rel=1e-6)

    def test_halo_geometry_does_not_score(self):
        """线性逐像素模型下 halo 区几何变化不改变 loss（不直接计分）。"""
        plain = kdb.Region(kdb.Box(20, 20, 60, 60))
        with_halo = kdb.Region(kdb.Box(20, 20, 60, 60))
        with_halo.insert(kdb.Box(-15, -15, -5, -5))  # 只落在 context 区
        results = []
        for region in (plain, with_halo):
            problem = _problem(region)
            results.append(_optimize(problem, _LinearModel(), _config()))
        first, second = results
        assert second.records[0].total_loss == pytest.approx(
            first.records[0].total_loss, rel=1e-6)  # halo 不进入计分


class TestBatchAndBarrier:
    """批大小不变性与 optimizer 屏障（REQ-007/008，INV-003）。"""

    def test_batch_size_preserves_gradient_and_published_state(self):
        """批 1 与全 core 批的 loss 与发布位移一致（容差内）。"""
        results = []
        for batch_size in (1, 4):
            problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
            results.append(_optimize(problem, _LinearModel(),
                                     _config(batch_size=batch_size)))
        first, second = results
        assert len(first.records) == len(second.records)
        for record_a, record_b in zip(first.records, second.records):
            assert record_a.total_loss == pytest.approx(
                record_b.total_loss, rel=1e-5, abs=1e-7)
            assert record_a.nominal_l2_loss == pytest.approx(
                record_b.nominal_l2_loss, rel=1e-5, abs=1e-7)
            assert record_a.process_l2_loss == pytest.approx(
                record_b.process_l2_loss, rel=1e-5, abs=1e-7)
        assert np.allclose(first.best_displacements,
                           second.best_displacements, atol=1e-5)

    def test_first_optimizer_step_after_all_batch_backward(self, monkeypatch):
        """首个 step 发生在本状态全部批 backward 之后（屏障证据）。"""
        events = []  # 事件序列：forward（带批大小）与 step 交错记录
        model = _LinearModel()
        original = model.forward_many

        def counting_forward(mask, conditions):
            """记录每次前向的实际批大小。"""
            events.append(("forward", int(mask.shape[0])))
            return original(mask, conditions)

        model.forward_many = counting_forward  # 实例级包装

        class _SpyAdam(torch.optim.Adam):
            """记录 step 事件的 Adam 代理。"""

            def step(self, *args, **kwargs):
                events.append(("step", None))
                return super().step(*args, **kwargs)

        monkeypatch.setattr(torch.optim, "Adam", _SpyAdam)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        _optimize(problem, model, _config(batch_size=2))  # 4 core → 每状态 2 批
        steps = [index for index, event in enumerate(events)
                 if event[0] == "step"]
        assert len(steps) == 2  # iterations=2 → 每 state 恰一次 step
        # 首个 step 之前恰有本状态全部 2 个批的前向（无批内提前更新）。
        assert events[:steps[0]] == [("forward", 2), ("forward", 2)]
        # 完整序列：两个状态各「2 批前向→1 次 step」，末状态纯评价 2 批前向。
        assert events == ([("forward", 2), ("forward", 2), ("step", None)] * 2
                          + [("forward", 2), ("forward", 2)])


class TestStateRecords:
    """状态记录与 best 快照语义（REQ-009/011，INV-004）。"""

    def test_records_and_best_use_same_evaluated_snapshots(self):
        """三状态、第二状态最优：best 指向 state 1 且快照一致。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _StateWeightModel([1.5, 1.0, 1.5], batches_per_state=2)
        result = _optimize(problem, model, _config(iterations=2))
        assert [record.state_index for record in result.records] == [0, 1, 2]
        assert result.stop_reason == "iteration_limit"  # 正常走满
        losses = [record.total_loss for record in result.records]
        assert losses[1] < losses[0]  # 第二状态优于 baseline
        assert losses[2] > losses[1]  # 第三状态变差
        assert result.best_state_index == 1  # best 指向最优已评价状态
        best_record = result.records[result.best_state_index]
        displaced = int(np.count_nonzero(result.best_displacements))
        assert displaced == best_record.displaced_segments  # 同一快照的位移量
        assert best_record.total_loss == min(losses)

    def test_zero_loss_stops_immediately(self):
        """baseline 连续 loss 恰为零时以 zero_loss 停止。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        identity = _LinearModel(doses={"nominal": 1.0, "dose_max": 1.0,
                                       "defocus_min": 1.0})  # 输出==mask==target
        result = _optimize(problem, identity, _config())
        assert result.stop_reason == "zero_loss"
        assert len(result.records) == 1  # baseline 后立即停止
        assert result.records[0].total_loss == 0.0


class TestGeometryRejection:
    """候选几何拒绝与异常边界（REQ-010/015，ERR-003/004）。"""

    def test_invalid_reconstruction_keeps_last_legal_best(self):
        """四边内移恰共线：invalid_geometry 且保留 baseline 为 best。"""
        # 40×40 矩形、位移上限 20：印刷过量（剂量 >1）使 dL/dMask>0，Adam
        # 沿负梯度方向内移全部边；大学习率使首步超限后被 clamp 到恰 ±20 →
        # 四边同移到中心线、ring 顶点共线，KLayout 以 ValueError 冒出
        # （simple 轮实测的同款守卫形态）。
        frag = FragmentationConfig(corner_length_dbu=8.0,
                                   max_segment_length_dbu=16.0,
                                   max_displacement_dbu=20.0, miter_limit=4.0)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), frag=frag)
        model = _LinearModel(doses={"nominal": 1.5, "dose_max": 1.5,
                                    "defocus_min": 1.5})
        result = _optimize(problem, model, _config(iterations=1,
                                                   learning_rate_dbu=70.0))
        assert result.stop_reason == "invalid_geometry"
        assert result.stop_detail is not None  # 原始错误文本在案
        assert "重建失败" in result.stop_detail
        assert len(result.records) == 1  # 非法候选不产生记录
        assert result.best_state_index == 0  # 保留最后合法状态
        assert np.all(result.best_displacements == 0.0)

    def test_program_runtime_error_is_not_converted(self, monkeypatch):
        """程序错误（RuntimeError）原样传播，不被吞成 invalid_geometry。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))

        def exploding(problem_, displacements):
            """模拟非几何领域的程序缺陷。"""
            raise RuntimeError("程序缺陷")

        monkeypatch.setattr(gradient_module, "reconstruct_region", exploding)
        with pytest.raises(RuntimeError, match="程序缺陷"):
            _optimize(problem, _LinearModel(), _config())

    def test_nonfinite_loss_raises_floating_point_error(self):
        """非有限 loss 抛 FloatingPointError 且消息含定位信息。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        with pytest.raises(FloatingPointError, match="state 0"):
            _optimize(problem, _NaNModel(), _config())


class TestGeometryMatrix:
    """几何矩阵：各形态全部发布合法状态（REQ-005/007/010/012）。"""

    @staticmethod
    def _assert_valid_publication(problem, result, max_displacement=10.0):
        """矩阵共用断言：位移界、context 零、best 对应真实记录。"""
        if len(result.best_displacements):
            assert np.all(np.abs(result.best_displacements)
                          <= max_displacement + 1e-9)  # 位移上限
            context = problem.owner_indices < 0
            assert np.all(result.best_displacements[context] == 0.0)  # 恒 0
        best = result.records[result.best_state_index]
        losses = [record.total_loss for record in result.records]
        assert best.total_loss == min(losses)  # best 属于真实已评价记录

    def test_solid_rectangle(self):
        """实心矩形：正常迭代、位移有限。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        result = _optimize(problem, _LinearModel(), _config())
        assert result.stop_reason == "iteration_limit"
        assert len(result.records) == 3
        self._assert_valid_publication(problem, result)

    def test_narrow_wall_reports_unavailable_probes(self):
        """2 DBU 壁 + 8 DBU 探针：部分探针不可用，不触发 zero_loss。"""
        problem = _problem(kdb.Region(kdb.Box(30, 30, 50, 32)))  # 高 2 窄条
        result = _optimize(problem, _LinearModel(),
                           _config(epe_distance_dbu=8.0))
        owner_count = int((problem.owner_indices >= 0).sum())
        assert owner_count > 0  # 窄条有 owner 段
        assert result.records[0].valid_probes < owner_count  # 有不可用探针
        assert result.stop_reason != "zero_loss"  # 连续 loss 仍为正
        assert result.records[0].total_loss > 0.0

    def test_polygon_with_hole(self):
        """多 polygon 带 hole：重建守卫全程通过。"""
        region = kdb.Region(kdb.Box(10, 10, 70, 70))
        region.insert(kdb.Box(30, 30, 50, 50))  # 第二个多边形
        hole = kdb.Region(kdb.Box(40, 40, 44, 44))  # 小 hole
        region = region - hole
        problem = _problem(region)
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)

    def test_concave_polygon(self):
        """L 形凹多边形：正常发布。"""
        polygon = kdb.Polygon([kdb.Point(20, 20), kdb.Point(60, 20),
                               kdb.Point(60, 40), kdb.Point(40, 40),
                               kdb.Point(40, 60), kdb.Point(20, 60)])
        problem = _problem(kdb.Region(polygon))
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)

    def test_diagonal_edge(self):
        """45° 斜边：同一单位法向公式无 H/V 分支。"""
        polygon = kdb.Polygon([kdb.Point(20, 20), kdb.Point(60, 20),
                               kdb.Point(20, 60)])  # 斜边 (60,20)-(20,60)
        problem = _problem(kdb.Region(polygon))
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)

    def test_cross_core_geometry(self, monkeypatch):
        """跨全部 core 的图形：梯度按全部 membership 采样（40 条）。"""
        problem = _problem(kdb.Region(kdb.Box(10, 10, 70, 70)))  # 跨 2×2 core
        counts = []  # 每次 apply 实际进入采样路径的条目数

        class _CountingMask(_EdgeGradientMask):
            """记录采样条目数的代理（backward 逻辑继承不变）。"""

            @staticmethod
            def forward(ctx, hard_masks, local_displacements, batch_indices,
                        midpoints_xy):
                counts.append(int(midpoints_xy.shape[0]))  # 采样条目计数
                return _EdgeGradientMask.forward(
                    ctx, hard_masks, local_displacements, batch_indices,
                    midpoints_xy)

        monkeypatch.setattr(gradient_module, "_EdgeGradientMask", _CountingMask)
        result = _optimize(problem, _LinearModel(), _config(batch_size=1))
        self._assert_valid_publication(problem, result)
        # state0 共 4 次 apply（batch_size=1 逐 core）；40 条 = 全部 membership
        # 中 owner 段（修复前 owner-only 仅 24 条，丢跨 core 边界段邻 tile 贡献）。
        assert sum(counts[:4]) == 40

    def test_cross_core_contributions_sum(self, monkeypatch):
        """同一参数的跨 core 梯度贡献严格求和（防 owner-only/覆盖/平均）。"""
        problem = _problem(kdb.Region(kdb.Box(10, 10, 70, 70)))  # 恰跨 4 core 交界
        captured = {}  # Spy 捕获的首个 step 前完整累积梯度

        class _GradCapture(torch.optim.Adam):
            """在首个 step 前捕获 state0 的累积梯度。"""

            def step(self, *args, **kwargs):
                if "grad" not in captured:
                    parameters = self.param_groups[0]["params"][0]
                    captured["grad"] = parameters.grad.detach().clone()
                return super().step(*args, **kwargs)

        monkeypatch.setattr(torch.optim, "Adam", _GradCapture)

        def _gradient_with_cores(allowed):
            """只保留指定 core 的采样条目（前向照跑全部 core）并返回梯度。"""
            real = problem.segments_for_core  # 原始 membership 绑定方法

            def _filtered(self, core_index):
                """非允许 core 的 membership 返回空视图（采样为空）。"""
                members = np.asarray(real(core_index))
                return members if core_index in allowed else members[:0]

            captured.pop("grad", None)  # 重置捕获
            patcher = pytest.MonkeyPatch()  # 独立还原域，不影响 Adam 补丁
            # frozen slots 实例不可 setattr，补丁打在类上（本测试独占实例）。
            patcher.setattr(MacroProblem, "segments_for_core", _filtered)
            try:
                _optimize(problem, _LinearModel(),
                          _config(batch_size=4, iterations=1))
            finally:
                patcher.undo()  # 只还原本函数的类级补丁
            return captured["grad"].numpy()

        full = _gradient_with_cores({0, 1, 2, 3})  # 全部 membership 采样
        upper = _gradient_with_cores({0, 1})  # 只上行两 core 的采样贡献
        lower = _gradient_with_cores({2, 3})  # 只下行两 core 的采样贡献
        # 跨 core 边界段（恰在 4 core 交界）在上下行都有条目：SUM 聚合。
        np.testing.assert_allclose(full, upper + lower, rtol=1e-5, atol=1e-7)
        # owner-only（==单侧）/覆盖/平均 三种错误形态都必须与 full 不同。
        assert not np.allclose(full, upper)
        assert not np.allclose(full, lower)

    def test_cross_macro_geometries_merge_independently(self):
        """跨 macro：两个 problem 各自独立求解、context 恒 0。"""
        macros = plan_macros(BOUNDS, macro_grid=(2, 1), core_size_dbu=40,
                             context_dbu=20, pixel_dbu=4, canvas_pixels=256)
        region = kdb.Region(kdb.Box(20, 20, 60, 60))  # 跨 x=40 的 macro 边界
        results = []
        for macro in macros:
            batch = RegionBatch({LAYER: region}, macro.query_box)
            problem = prepare_macro_problem(batch, LAYER, "clear", FRAG, macro)
            results.append(_optimize(problem, _LinearModel(), _config()))
        for macro, result in zip(macros, results):
            self._assert_valid_publication(
                _problem(region, macro=macro), result)

    def test_opaque_polarity(self):
        """opaque 极性：同一正位移=扩大透光，无 solver 分支。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)),
                           polarity="opaque")
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)


class TestCallCounts:
    """热路径调用次数契约（PERF-003/004）。"""

    def test_forward_many_once_per_batch_per_state(self):
        """每状态每批恰一次三条件前向。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _LinearModel()
        _optimize(problem, model, _config(batch_size=2))  # 3 状态 × 2 批
        assert model.calls == 6

    def test_candidate_region_reconstructed_once_and_reused(
            self, monkeypatch):
        """候选每 state 恰重建一次并被下一状态栅格复用。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        counts = {"n": 0}
        real = gradient_module.reconstruct_region

        def spy(problem_, displacements):
            """计数重建调用。"""
            counts["n"] += 1
            return real(problem_, displacements)

        monkeypatch.setattr(gradient_module, "reconstruct_region", spy)
        _optimize(problem, _LinearModel(), _config(iterations=2))
        assert counts["n"] == 3  # 1 次 reference + 2 次候选（各发布一次）

    def test_target_cache_hit_avoids_target_raster(self, monkeypatch):
        """target cache 命中时 target 不重栅格化。"""
        counts = {"n": 0}
        real = gradient_module.rasterize_mask_canvas

        def spy(*args, **kwargs):
            """计数栅格化调用（target 与当前 mask 合计）。"""
            counts["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(gradient_module, "rasterize_mask_canvas", spy)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        _optimize(problem, _LinearModel(), _config())  # 3 状态 × 4 core
        assert counts["n"] == 16  # 12 次当前 mask + 4 次首次 target
        _optimize(problem, _LinearModel(), _config(),
                  cache=TargetCanvasCache(0))  # 禁用缓存对照
        assert counts["n"] == 16 + 24  # 每 state 的 target 都要重栅格


class TestRealModel:
    """真实 ICCAD13 的方向一致性与 CPU/CUDA 集成（TEST-003/011）。"""

    @staticmethod
    def _loss_numeric(problem, model, config, owner_index, delta):
        """对单段 ±delta 位移独立复算加权连续 loss（数值路径）。"""
        segments = problem.segments.segment_count
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        displacements = np.zeros(segments, dtype=np.float64)
        displacements[owner_ids[owner_index]] = delta
        region = reconstruct_region(problem, displacements)
        reference = reconstruct_region(
            problem, np.zeros(segments, dtype=np.float64))
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        conditions = (model.condition("nominal"), model.condition("dose_max"),
                      model.condition("defocus_min"))
        weighted = 0.0
        total = 0
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu,
                                         canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(reference, spec.context_box,
                                           pixel_dbu, canvas,
                                           polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box,
                                   pixel_dbu, canvas)
            total += int(own.sum())
            printed = model.forward_many(
                torch.from_numpy(mask)[None], conditions)
            nom = printed["nominal"][0]
            dmax = printed["dose_max"][0]
            dmin = printed["defocus_min"][0]
            t = torch.from_numpy(target)
            own_t = torch.from_numpy(own)
            nominal_part = float(((nom - t) ** 2 * own_t).sum())
            process_part = float((((dmax - t) ** 2
                                   + (dmin - t) ** 2) * own_t).sum())
            pv_part = float(((dmax - dmin) ** 2 * own_t).sum())
            weighted += (config.weight_nominal_l2 * nominal_part
                         + config.weight_process_l2 * process_part
                         + config.weight_pvband * pv_part)
        return weighted / total

    @staticmethod
    def _surrogate_gradients(problem, model):
        """直接经 _EdgeGradientMask 求 baseline 的 owner 梯度。"""
        segments = problem.segments.segment_count
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        segment_to_parameter = np.full(segments, -1, dtype=np.int32)
        segment_to_parameter[owner_ids] = np.arange(len(owner_ids))
        reference = problem.segments.materialize()
        midpoints = (reference.starts + reference.ends) * 0.5  # 零位移中点
        region = reconstruct_region(
            problem, np.zeros(segments, dtype=np.float64))
        parameters = torch.zeros(len(owner_ids), requires_grad=True)
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        conditions = (model.condition("nominal"), model.condition("dose_max"),
                      model.condition("defocus_min"))
        loss = torch.zeros((), dtype=torch.float32)
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu,
                                         canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(region, spec.context_box,
                                           pixel_dbu, canvas,
                                           polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box,
                                   pixel_dbu, canvas)
            hard = torch.from_numpy(mask)[None]
            # 与实现一致：采样集合 = 该 core 全部 membership 中的 owner 段
            # （P1-1 修复后语义；owner_segments_for_core 只覆盖 owner core）。
            members = np.asarray(problem.segments_for_core(core_index))
            members = members[segment_to_parameter[members] >= 0]
            if len(members):
                local = parameters[torch.from_numpy(
                    segment_to_parameter[members].astype(np.int64))]
                slots = torch.zeros(len(members), dtype=torch.int64)
                mids = torch.from_numpy(points_to_canvas(
                    midpoints[members], spec.context_box, pixel_dbu,
                    canvas)).to(dtype=torch.float32)
                mask_tensor = _EdgeGradientMask.apply(hard, local, slots,
                                                      mids)
            else:
                mask_tensor = hard
            printed = model.forward_many(mask_tensor, conditions)
            t = torch.from_numpy(target)[None]
            own_t = torch.from_numpy(own)[None]
            loss = loss + (((printed["nominal"] - t) ** 2
                            + 0.5 * ((printed["dose_max"] - t) ** 2
                                     + (printed["defocus_min"] - t) ** 2)
                            + 0.1 * (printed["dose_max"]
                                     - printed["defocus_min"]) ** 2)
                           * own_t).sum()
        loss.backward()
        return owner_ids, parameters.grad.numpy()

    @pytest.mark.parametrize("polarity", ["clear", "opaque"])
    def test_surrogate_direction_matches_integer_geometry_difference(
            self, cpu_model, polarity):
        """surrogate 梯度与 ±1 DBU 精确几何差分同号（真实 ICCAD13）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)),
                           polarity=polarity)
        config = _config()
        owner_ids, gradients = self._surrogate_gradients(problem, cpu_model)
        # 只抽验每条边最长的一段（矩形四边各一）控制真实前向次数。
        lengths = problem.segments.t1 - problem.segments.t0
        chosen = []
        seen_edges = set()
        for index in np.argsort(-lengths):
            edge = int(problem.segments.edge_ids[index])
            if index in set(owner_ids.tolist()) and edge not in seen_edges:
                seen_edges.add(edge)
                chosen.append(int(np.flatnonzero(owner_ids == index)[0]))
            if len(chosen) == 4:
                break
        checked = 0
        for owner_index in chosen:
            plus = self._loss_numeric(problem, cpu_model, config,
                                      owner_index, 1.0)
            minus = self._loss_numeric(problem, cpu_model, config,
                                       owner_index, -1.0)
            difference = (plus - minus) / 2.0
            if abs(difference) <= 1e-9:  # 数值零不比较方向
                continue
            checked += 1
            assert gradients[owner_index] * difference > 0.0, (
                f"段 {owner_index}：surrogate {gradients[owner_index]:.3e} "
                f"与有限差分 {difference:.3e} 方向相反")
        assert checked >= 2  # 至少两段完成了方向验证

    def test_real_iccad13_cpu_runs_nonzero_update_and_valid_best(
            self, cpu_model):
        """真实 CPU 模型：有限 loss、非零更新、best 不差于 baseline。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        result = _optimize(problem, cpu_model, _config(iterations=2))
        assert all(np.isfinite(record.total_loss)
                   for record in result.records)  # 全部有限
        assert result.records[-1].displaced_segments > 0  # 有实际更新
        best = result.records[result.best_state_index]
        assert best.total_loss <= result.records[0].total_loss  # 不劣化
        assert np.max(np.abs(result.best_displacements)) <= 10.0 + 1e-9

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_real_iccad13_cuda_matches_cpu_direction(self, cpu_model):
        """CUDA 与 CPU 的 baseline loss 一致、更新方向一致。"""
        cuda_model = ICCAD13Lithography(device="cuda")
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        config = _config(iterations=1)
        cpu_result = _optimize(problem, cpu_model, config)
        cuda_result = _optimize(problem, cuda_model, config)
        assert cpu_result.records[0].total_loss == pytest.approx(
            cuda_result.records[0].total_loss, rel=2e-4, abs=1e-6)
        moved = np.abs(cpu_result.best_displacements) > 1e-9  # 已更新段
        if moved.any():
            signs = (np.sign(cpu_result.best_displacements[moved])
                     == np.sign(cuda_result.best_displacements[moved]))
            assert signs.all()  # 选定段方向一致


class TestSimpleCompat:
    """simple 方法与共享缓存的兼容（REQ-016）。"""

    def test_cache_import_paths_preserved(self):
        """包级、simple 模块级与 _cache 三条导入路径同一对象。"""
        from opc.iteration.mbopc import TargetCanvasCache as package_cache
        from opc.iteration.mbopc._cache import TargetCanvasCache as underscore_cache
        from opc.iteration.mbopc.simple import TargetCanvasCache as simple_cache
        assert package_cache is simple_cache is underscore_cache
