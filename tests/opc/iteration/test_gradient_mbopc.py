"""梯度 MB-OPC 求解器的代理梯度、状态语义、几何矩阵与真实模型集成测试。"""

import inspect

import klayout.db as kdb
import numpy as np
import pytest
import torch

from layout import DbuBox, LayerSpec, RegionBatch
from lithography import ICCAD13Lithography, ProcessCondition
from opc.errors import ReconstructionError
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
    reconstruct_region_with_midpoints,
)
from opc.input.edge.fragmentation import FragmentationConfig
from opc.iteration.mbopc import (
    GradientMBOPCConfig,
    TargetCanvasCache,
    optimize_gradient_macro,
)
from opc.iteration.mbopc import gradient as gradient_module
from opc.iteration.mbopc.gradient import (
    _EdgeGradientMask,
    _evaluate_state,
    _prepare_gradient_context,
    _profile_d_s,
    _take_optimizer_step,
)

LAYER = LayerSpec(1, 0)
# 廉价契约：80² 版图、单 macro、core 40、context 20、pixel 4 → 2×2 core。
BOUNDS = DbuBox(0, 0, 80, 80)
FRAG = FragmentationConfig(
    corner_length_dbu=8.0, max_segment_length_dbu=16.0, max_displacement_dbu=10.0, miter_limit=4.0
)
_GRADIENT_DEFAULTS = {
    "iterations": 2,
    "learning_rate_dbu": 1.0,
    "weight_nominal_l2": 1.0,
    "weight_process_l2": 0.5,
    "weight_pvband": 0.1,
    "epe_distance_dbu": 4.0,
    "batch_size": 2,
    "target_cache_bytes": 256 * 256 * 8,
}
_CACHE_BUDGET = 256 * 256 * 8  # 恰容 8 张 canvas，测试几何足够全员命中


def _macro(**overrides):
    """返回单 macro 规划（默认 80² 版图 2×2 core）。"""
    values = {"macro_grid": (1, 1), "core_size_dbu": 40, "context_dbu": 20, "pixel_dbu": 4, "canvas_pixels": 256}
    values.update(overrides)
    return plan_macros(BOUNDS, **values)[0]


def _problem(region, macro=None, polarity="clear", frag=FRAG):
    """把原生 Region 直接包装为 RegionBatch 并生成单 macro problem。"""
    macro = macro if macro is not None else _macro()
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_macro_problem(batch, LAYER, polarity, frag, macro, data_bounds=BOUNDS)


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
            name: ProcessCondition(name, "defocus" if name == "defocus_min" else "focus", dose)
            for name, dose in doses.items()
        }

    def condition(self, name):
        """返回固定剂量条件。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """输出等于 mask×dose（线性链路保证梯度方向可手推）。"""
        self.calls += 1  # 计数
        return {c.name: mask * self._conditions[c.name].dose for c in conditions}


class _StateWeightModel:
    """按状态序号切换剂量的假模型，用于确定性 best/record 测试。"""

    def __init__(self, weights, batches_per_state):
        self.device = torch.device("cpu")  # CPU 契约设备
        self.config = _StubConfig()  # 256 画布视图
        self._weights = list(weights)  # 每状态的统一剂量
        self._batches = batches_per_state  # 每状态的 forward 批数
        self._calls = 0  # 累计调用数
        self._conditions = {
            name: ProcessCondition(name, "focus", 1.0) for name in ("nominal", "dose_max", "defocus_min")
        }

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
        [
            {"iterations": 0},
            {"iterations": True},
            {"learning_rate_dbu": 0.0},
            {"learning_rate_dbu": float("nan")},
            {"weight_nominal_l2": -0.1},
            {"weight_pvband": float("inf")},
            {"weight_nominal_l2": 0.0, "weight_process_l2": 0.0, "weight_pvband": 0.0},
            {"epe_distance_dbu": 0.0},
            {"batch_size": 0},
            {"target_cache_bytes": -1},
        ],
        ids=["iter=0", "iter=bool", "lr=0", "lr=nan", "w<0", "w=inf", "全零权重", "epe=0", "batch=0", "cache<0"],
    )
    def test_invalid_values_fail(self, overrides):
        """越界参数在构造期失败。"""
        with pytest.raises(ValueError):
            _config(**overrides)


class TestThresholdPropagation:
    """L2/PVBand/EPE 三类指标显式跟随模型 PrintThresh（审查问题 3）。"""

    def test_all_metrics_receive_model_threshold(self, monkeypatch):
        """模型阈值 0.45 时三个 evaluate_* 都收到 threshold=0.45。"""
        model = _LinearModel()  # 线性可微假模型
        model.config = _StubConfig(threshold=0.45)  # 非默认打印阈值
        captured = {}  # 指标名 → 收到的 threshold 列表
        for name in ("evaluate_binary_l2", "evaluate_pvband", "evaluate_edge_probes"):
            real = getattr(gradient_module, name)  # 真实现（透传）

            def spy(*args, _name=name, _real=real, **kwargs):
                """记录 threshold 关键字并透传。"""
                captured.setdefault(_name, []).append(kwargs.get("threshold"))
                return _real(*args, **kwargs)

            monkeypatch.setattr(gradient_module, name, spy)
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        optimize_gradient_macro(problem, model, _config(iterations=1), TargetCanvasCache(_CACHE_BUDGET))
        for name, values in captured.items():  # 全部收到模型阈值
            assert values, name  # 至少一次
            assert all(value == 0.45 for value in values), name
        assert len(captured) == 3  # L2/PVBand/EPE 齐全


class TestEdgeGradientMask:
    """_EdgeGradientMask 的前向直通与 Algorithm 4 反向公式。"""

    def _apply(self, hard, local, slots, mids, pixel_dbu=1):
        """在 CPU 上执行一次 apply 并返回输出（pixel_dbu 默认 1）。"""
        return _EdgeGradientMask.apply(hard, local, slots, mids, pixel_dbu)

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
        """单条目梯度恰为 2×双线性/pixel_dbu；越界为 0；重复索引求和。"""
        size = 8  # 8×8 图
        pixel_dbu = 4  # 非平凡换算尺度，锁定 DBU 单位契约
        grad_output = torch.arange(2 * size * size, dtype=torch.float32)
        grad_output = grad_output.reshape(2, size, size)  # 图 1 基址 64
        local = torch.zeros(4, requires_grad=True)  # 四条独立 membership
        slots = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        mids = torch.tensor([[1.5, 2.5], [7.0, 7.0], [-3.0, 4.0], [2.0, 2.0]], dtype=torch.float32)
        hard = torch.zeros(2, size, size)  # 数值不参与反向
        out = self._apply(hard, local, slots, mids, pixel_dbu)
        (out * grad_output).sum().backward()  # 直接以已知梯度图反传
        # 半像素点双线性四角均值：(17+18+25+26)/4 = 21.5；
        # 边界整点 (7,7)→127；越界点恒 0；内部整点 (2,2)→82。
        expect = torch.tensor([2 * 21.5, 2 * 127.0, 0.0, 2 * 82.0]) / pixel_dbu
        assert torch.allclose(local.grad, expect)  # 2·g_mid/pixel_dbu 公式
        # 同一参数被两条 membership 引用时由 autograd 求和（图 0 整点）。
        shared = torch.zeros(1, requires_grad=True)
        gathered = shared[torch.tensor([0, 0])]  # 重复索引
        out2 = self._apply(
            torch.zeros(1, size, size),
            gathered,
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            pixel_dbu,
        )
        (out2 * grad_output[:1]).sum().backward()
        assert torch.allclose(shared.grad, torch.tensor([2 * (9.0 + 18.0)]) / pixel_dbu)

    def test_backward_scales_inverse_with_pixel_dbu(self):
        """pixel_dbu=1/2/4 下位移梯度方向一致、幅值按 g、g/2、g/4 缩放。

        直接锁定单位契约（位移参数为 DBU，采样在 pixel 域），不受
        栅格化、光刻或 Adam 干扰。
        """
        size = 8
        # 单图、三条 membership：半像素点、内部整点、另一内部整点。
        grad_output = torch.arange(size * size, dtype=torch.float32)
        grad_output = grad_output.reshape(1, size, size)
        slots = torch.zeros(3, dtype=torch.int64)
        mids = torch.tensor([[1.5, 2.5], [6.0, 3.0], [2.0, 2.0]], dtype=torch.float32)
        hard = torch.zeros(1, size, size)
        grads = {}
        for pixel_dbu in (1, 2, 4):
            local = torch.zeros(3, requires_grad=True)
            out = self._apply(hard, local, slots, mids, pixel_dbu)
            (out * grad_output).sum().backward()
            grads[pixel_dbu] = local.grad.clone()
        base = grads[1]
        assert torch.all(base != 0)  # 采样点非零，缩放可观测
        for factor, pixel_dbu in ((2.0, 2), (4.0, 4)):
            scaled = grads[pixel_dbu]
            assert torch.allclose(scaled, base / factor, atol=1e-7)  # 幅值
            assert torch.all(scaled * base > 0)  # 方向完全一致

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
            _optimize(self._rectangle(), model, _config(epe_distance_dbu=25.0))  # context 仅 20


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
        conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        reference = reconstruct_region(problem, np.zeros(problem.segments.segment_count, dtype=np.float64))
        nominal_sum = process_sum = pv_sum = 0.0
        total = 0
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(reference, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box, pixel_dbu, canvas)
            total += int(own.sum())  # 全局分母 P
            batch = torch.from_numpy(mask)[None]  # [1,H,W]
            printed = model.forward_many(batch, conditions)
            nom = printed["nominal"][0].numpy()
            dmax = printed["dose_max"][0].numpy()
            dmin = printed["defocus_min"][0].numpy()
            nominal_sum += float(((nom - target) ** 2 * own).sum())
            process_sum += float((((dmax - target) ** 2 + (dmin - target) ** 2) * own).sum())
            pv_sum += float(((dmax - dmin) ** 2 * own).sum())
        return nominal_sum / total, process_sum / total, pv_sum / total, total

    def test_continuous_losses_match_independent_recompute(self):
        """三分量与加权 total 和独立复算一致（确定性线性模型）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _LinearModel()
        config = _config()
        result = _optimize(problem, model, config)
        expected = self._expected_losses(
            problem, model, reconstruct_region(problem, np.zeros(problem.segments.segment_count, dtype=np.float64))
        )
        record = result.records[0]
        assert record.nominal_l2_loss == pytest.approx(expected[0], rel=1e-6)
        assert record.process_l2_loss == pytest.approx(expected[1], rel=1e-6)
        assert record.pvband_loss == pytest.approx(expected[2], rel=1e-6)
        total = (
            config.weight_nominal_l2 * expected[0]
            + config.weight_process_l2 * expected[1]
            + config.weight_pvband * expected[2]
        )
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
        assert second.records[0].total_loss == pytest.approx(first.records[0].total_loss, rel=1e-6)  # halo 不进入计分


class TestBatchAndBarrier:
    """批大小不变性与 optimizer 屏障（REQ-007/008，INV-003）。"""

    def test_batch_size_preserves_gradient_and_published_state(self):
        """批 1 与全 core 批的 loss 与发布位移一致（容差内）。"""
        results = []
        for batch_size in (1, 4):
            problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
            results.append(_optimize(problem, _LinearModel(), _config(batch_size=batch_size)))
        first, second = results
        assert len(first.records) == len(second.records)
        for record_a, record_b in zip(first.records, second.records):
            assert record_a.total_loss == pytest.approx(record_b.total_loss, rel=1e-5, abs=1e-7)
            assert record_a.nominal_l2_loss == pytest.approx(record_b.nominal_l2_loss, rel=1e-5, abs=1e-7)
            assert record_a.process_l2_loss == pytest.approx(record_b.process_l2_loss, rel=1e-5, abs=1e-7)
        assert np.allclose(first.best_displacements, second.best_displacements, atol=1e-5)

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
        steps = [index for index, event in enumerate(events) if event[0] == "step"]
        assert len(steps) == 2  # iterations=2 → 每 state 恰一次 step
        # 首个 step 之前恰有本状态全部 2 个批的前向（无批内提前更新）。
        assert events[: steps[0]] == [("forward", 2), ("forward", 2)]
        # 完整序列：两个状态各「2 批前向→1 次 step」，末状态纯评价 2 批前向。
        assert events == ([("forward", 2), ("forward", 2), ("step", None)] * 2 + [("forward", 2), ("forward", 2)])


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
        identity = _LinearModel(doses={"nominal": 1.0, "dose_max": 1.0, "defocus_min": 1.0})  # 输出==mask==target
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
        frag = FragmentationConfig(
            corner_length_dbu=8.0, max_segment_length_dbu=16.0, max_displacement_dbu=20.0, miter_limit=4.0
        )
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), frag=frag)
        model = _LinearModel(doses={"nominal": 1.5, "dose_max": 1.5, "defocus_min": 1.5})
        result = _optimize(problem, model, _config(iterations=1, learning_rate_dbu=70.0))
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

        monkeypatch.setattr(gradient_module, "reconstruct_region_with_midpoints", exploding)
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
            assert np.all(np.abs(result.best_displacements) <= max_displacement + 1e-9)  # 位移上限
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
        result = _optimize(problem, _LinearModel(), _config(epe_distance_dbu=8.0))
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
        polygon = kdb.Polygon(
            [
                kdb.Point(20, 20),
                kdb.Point(60, 20),
                kdb.Point(60, 40),
                kdb.Point(40, 40),
                kdb.Point(40, 60),
                kdb.Point(20, 60),
            ]
        )
        problem = _problem(kdb.Region(polygon))
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)

    def test_diagonal_edge(self):
        """45° 斜边：同一单位法向公式无 H/V 分支。"""
        polygon = kdb.Polygon([kdb.Point(20, 20), kdb.Point(60, 20), kdb.Point(20, 60)])  # 斜边 (60,20)-(20,60)
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
            def forward(ctx, hard_masks, local_displacements, batch_indices, midpoints_xy, pixel_dbu):
                counts.append(int(midpoints_xy.shape[0]))  # 采样条目计数
                return _EdgeGradientMask.forward(
                    ctx, hard_masks, local_displacements, batch_indices, midpoints_xy, pixel_dbu
                )

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
                _optimize(problem, _LinearModel(), _config(batch_size=4, iterations=1))
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
        macros = plan_macros(
            BOUNDS, macro_grid=(2, 1), core_size_dbu=40, context_dbu=20, pixel_dbu=4, canvas_pixels=256
        )
        region = kdb.Region(kdb.Box(20, 20, 60, 60))  # 跨 x=40 的 macro 边界
        results = []
        for macro in macros:
            batch = RegionBatch({LAYER: region}, macro.query_box)
            problem = prepare_macro_problem(batch, LAYER, "clear", FRAG, macro, data_bounds=BOUNDS)
            results.append(_optimize(problem, _LinearModel(), _config()))
        for macro, result in zip(macros, results):
            self._assert_valid_publication(_problem(region, macro=macro), result)

    def test_opaque_polarity(self):
        """opaque 极性：同一正位移=扩大透光，无 solver 分支。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), polarity="opaque")
        result = _optimize(problem, _LinearModel(), _config())
        self._assert_valid_publication(problem, result)


class TestEpeLoss:
    """可微 EPE loss：公式、方向、owner、batch、关闭兼容与两个不变性。"""

    @staticmethod
    def _identity_model():
        """三条件剂量全 1 的线性模型：零位移下 printed == target。"""
        return _LinearModel({"nominal": 1.0, "dose_max": 1.0, "defocus_min": 1.0})

    def test_epe_profile_formula_and_zero_baseline(self):
        """双线性采样 + sum 聚合 + zero-based sigmoid + 长度加权的逐值公式。"""
        # 手算张量：1×6×6 误差图，两条 profile 采整数/半整数中心混合点
        error = torch.zeros(1, 6, 6)
        error[0, 2, 3] = 0.4  # (y=2, x=3)
        error[0, 3, 4] = 1.0  # (y=3, x=4)
        xy = torch.tensor(
            [
                [[3.0, 2.0], [4.5, 3.0]],  # 整数中心 + 半像素
                [[3.0, 2.5], [4.0, 3.0]],
            ]
        )  # 半像素 + 整数中心
        slots = torch.tensor([0, 0])
        d_s = _profile_d_s(error, slots, xy)
        # 段0：0.4 + bilinear(1.0@y3x4 于 (4.5,3)) = 0.4 + 0.5
        # 段1：bilinear(0.4@y2x3 于 y2.5) + 1.0 = 0.2 + 1.0
        assert float(d_s[0]) == pytest.approx(0.9, abs=1e-6)
        assert float(d_s[1]) == pytest.approx(1.2, abs=1e-6)
        # penalty = 2(σ(γ·d)−0.5)；L_epe = Σ len·pen / Σ len
        gamma = 4.0
        penalty = 2.0 * (torch.sigmoid(gamma * d_s) - 0.5)
        lengths = torch.tensor([8.0, 16.0])
        epe = float((lengths * penalty).sum() / lengths.sum())
        expected = (8.0 * float(penalty[0]) + 16.0 * float(penalty[1])) / 24.0
        assert epe == pytest.approx(expected, rel=1e-6)
        assert epe >= 0.0  # zero-based：值域 [0,1)
        # 零误差基线：恒等模型 + 零位移（mask==target）→ epe_loss 严格 0
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = _optimize(problem, self._identity_model(), _config(weight_epe=1.0))
        assert result.records[0].epe_loss == 0.0
        assert result.records[0].total_loss == 0.0

    def test_epe_profile_coordinates_all_directions(self):
        """H/V/45° 的 profile 坐标 = 参考中点 + q·单位法向（画布换算）。"""
        polygon = kdb.Polygon(
            [
                kdb.Point(8, 8),
                kdb.Point(48, 8),
                kdb.Point(48, 24),
                kdb.Point(24, 24),
                kdb.Point(24, 48),
                kdb.Point(8, 48),
            ]
        )
        problem = _problem(kdb.Region(polygon))  # L 形含 H/V 边；另加斜边
        problem2 = _problem(kdb.Region(kdb.Polygon([kdb.Point(8, 8), kdb.Point(48, 8), kdb.Point(8, 48)])))
        for target_problem in (problem, problem2):
            ctx = _prepare_gradient_context(target_problem, _LinearModel(), _config(weight_epe=1.0))
            assert ctx.epe_length_sum > 0.0
            reference = target_problem.segments.materialize()
            pixel = target_problem.macro.pixel_dbu
            radius = 1  # 默认 epe_distance_dbu=4 / pixel 4 → Q=2
            offsets = (np.arange(2 * radius) - radius + 0.5) * pixel
            for core_index in range(target_problem.macro.core_count):
                owner = ctx.pack.owner_members[core_index]
                profile = ctx.epe_profiles[core_index]
                if not len(owner):
                    assert profile is None
                    continue
                assert profile.shape == (len(owner), 2 * radius, 2)
                mids = (reference.starts[owner] + reference.ends[owner]) * 0.5
                normals = reference.normals[owner]
                lengths = np.linalg.norm(reference.ends[owner] - reference.starts[owner], axis=1)
                assert np.allclose(ctx.epe_lengths[core_index], lengths)
                expected = mids[:, None, :] + offsets[None, :, None] * normals[:, None, :]
                from opc.input import points_to_canvas as p2c

                spec = target_problem.macro.core(core_index)
                expected_xy = p2c(expected.reshape(-1, 2), spec.context_box, pixel, 256).reshape(len(owner), -1, 2)
                assert np.allclose(profile, expected_xy, atol=1e-9)
                # 结构性质：坐标在闭区间 [0, 255] 内
                assert profile.min() >= 0.0 and profile.max() <= 255.0
            assert np.isclose(
                sum(
                    np.sum(ctx.epe_lengths[c]) if ctx.epe_lengths[c] is not None else 0.0
                    for c in range(target_problem.macro.core_count)
                ),
                ctx.epe_length_sum,
            )
            del reference

    def test_epe_loss_backpropagates_through_midpoint_ste(self, monkeypatch):
        """EPE-only 梯度经既有 midpoint STE 到达唯一 owner 参数。"""
        counts = {"apply": 0}
        real_apply = _EdgeGradientMask.apply

        def counting_apply(*args, **kwargs):
            """计数 STE apply 调用并透传。"""
            counts["apply"] += 1
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(
            gradient_module, "_EdgeGradientMask", type("Spy", (), {"apply": staticmethod(counting_apply)})
        )
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        config = _config(weight_nominal_l2=0.0, weight_process_l2=0.0, weight_pvband=0.0, weight_epe=1.0)
        result = _optimize(problem, _LinearModel(), config)
        assert counts["apply"] > 0  # EPE 梯度路径经过 STE
        # EPE-only：梯度非零 → 参数移动 → 位移发布
        assert result.records[1].displaced_segments > 0
        assert all(np.isfinite(r.epe_loss) for r in result.records)

    def test_epe_owner_scores_once_membership_gradients_sum(self):
        """每段 profile 恰在其 owner core 计一次（分母 L_sum 全宏唯一）。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))  # 跨 2×2 core
        ctx = _prepare_gradient_context(problem, _LinearModel(), _config(weight_epe=1.0))
        owner_total = sum(len(ctx.pack.owner_members[c]) for c in range(problem.macro.core_count))
        profile_total = sum(
            ctx.epe_profiles[c].shape[0] if ctx.epe_profiles[c] is not None else 0
            for c in range(problem.macro.core_count)
        )
        assert profile_total == owner_total  # 无 membership 重复条目

    def test_epe_batch_size_invariant(self):
        """batch 1/2/4 下四项 loss 与位移更新一致（分母 L_sum 不变）。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))
        results = []
        for batch_size in (1, 2, 4):
            results.append(
                _optimize(problem, _LinearModel(), _config(iterations=2, batch_size=batch_size, weight_epe=1.0))
            )
        base = results[0]
        for other in results[1:]:
            assert other.best_state_index == base.best_state_index
            assert np.allclose(other.best_displacements, base.best_displacements, rtol=1e-5, atol=1e-6)
            for record_b, record_o in zip(base.records, other.records):
                assert record_o.epe_loss == pytest.approx(record_b.epe_loss, rel=1e-5, abs=1e-9)
                assert record_o.total_loss == pytest.approx(record_b.total_loss, rel=1e-5, abs=1e-9)

    def test_epe_disabled_is_exactly_compatible(self, monkeypatch):
        """weight_epe=0：不建 profile、不采样、epe 恒 0、total 为三项加权和。"""
        calls = {"profile": 0}
        real = gradient_module._profile_d_s

        def spy(*args, **kwargs):
            """采样器不得在关闭路径被调用。"""
            calls["profile"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(gradient_module, "_profile_d_s", spy)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        ctx = _prepare_gradient_context(problem, _LinearModel(), _config())
        assert ctx.epe_length_sum == 0.0  # 关闭：无 profile 无分母
        assert all(p is None for p in ctx.epe_profiles)
        result = _optimize(problem, self._identity_model(), _config())
        assert calls["profile"] == 0
        for record in result.records:
            assert record.epe_loss == 0.0
            expected = record.nominal_l2_loss + 0.5 * record.process_l2_loss + 0.1 * record.pvband_loss
            assert record.total_loss == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"weight_epe": -0.1}, "非负"),
            ({"epe_steepness": 0.0}, "epe_steepness"),
            ({"epe_steepness": float("nan")}, "epe_steepness"),
            ({"weight_nominal_l2": 0.0, "weight_process_l2": 0.0, "weight_pvband": 0.0}, "至少一个为正"),
        ],
        ids=["w_epe<0", "gamma=0", "gamma=nan", "全零权重"],
    )
    def test_epe_config_validation_fails(self, overrides, match):
        """非法权重/陡度在构造期失败。"""
        with pytest.raises(ValueError, match=match):
            _config(**overrides)

    def test_epe_distance_must_be_pixel_multiple_when_enabled(self):
        """启用时 epe_distance 非 pixel 整数倍在入口、空 owner 返回前失败。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        with pytest.raises(ValueError, match="正整数倍"):
            _optimize(problem, _LinearModel(), _config(weight_epe=1.0, epe_distance_dbu=6.0))
        # R≥1：距离小于一个 pixel 同样拒绝
        with pytest.raises(ValueError, match="正整数倍"):
            _optimize(problem, _LinearModel(), _config(weight_epe=1.0, epe_distance_dbu=2.0))

    def test_epe_profile_out_of_canvas_rejected(self, monkeypatch):
        """profile 越界守卫：构造期 ValueError，不裁剪不跳过。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        ctx_probe_rows = len(problem.owner_segments_for_core(0))
        real = gradient_module.points_to_canvas

        def shifted(points, *args, **kwargs):
            """profile 规模的调用（E·Q 行）平移出画布，探针调用透传。"""
            out = real(points, *args, **kwargs)
            if len(out) > ctx_probe_rows:  # E·Q > E（Q≥2 恒成立）
                return out + 1e4
            return out

        monkeypatch.setattr(gradient_module, "points_to_canvas", shifted)
        with pytest.raises(ValueError, match="越出画布"):
            _prepare_gradient_context(problem, _LinearModel(), _config(weight_epe=1.0))

    def test_epe_only_update_improves_evaluated_loss(self):
        """EPE-only：至少一次合法更新使已评价 epe_loss 严格下降。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        config = _config(weight_nominal_l2=0.0, weight_process_l2=0.0, weight_pvband=0.0, weight_epe=1.0)
        result = _optimize(problem, _LinearModel(), config)
        assert result.records[0].epe_loss > 0.0
        assert result.records[1].epe_loss < result.records[0].epe_loss
        best = result.records[result.best_state_index]
        losses = [record.epe_loss for record in result.records]
        assert best.epe_loss == min(losses)  # best 与记录一致

    def test_epe_forward_count_unchanged_and_profile_built_once(self, monkeypatch):
        """启用 EPE 不增加 forward；profile 每宏恰建一次（不随 state 重建）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        prepare_calls = {"n": 0}
        real = gradient_module._prepare_gradient_context

        def counting(*args, **kwargs):
            """计数上下文（含 profile 坐标）构造次数。"""
            prepare_calls["n"] += 1
            return real(*args, **kwargs)

        models = {}
        for enabled in (False, True):
            monkeypatch.setattr(gradient_module, "_prepare_gradient_context", counting)
            prepare_calls["n"] = 0
            model = _LinearModel()
            _optimize(problem, model, _config(iterations=2, weight_epe=1.0 if enabled else 0.0))
            models[enabled] = model
            # profile 坐标随上下文每宏恰构造一次，不随 state 重复
            assert prepare_calls["n"] == 1
        # forward 次数与关闭时完全一致（(N+1)×批数）
        assert models[True].calls == models[False].calls

    def test_epe_profile_width_invariant_d_s(self):
        """Q 不变性（DEC-002）：1px 阶跃偏移下 d_s 不随 R 衰减（sum 聚合）。"""
        # 理想阶跃：D 仅 2 槽非零（x=9、10），图像宽 20 保证 R=8 的
        # 坐标（x∈[2,17]）落在闭区间 [0,19] 内（采样器的构造期前提）
        error = torch.zeros(1, 1, 20)
        error[0, 0, 9] = 1.0
        error[0, 0, 10] = 1.0
        d_values = []
        for radius in (2, 4, 8):
            q_slots = 2 * radius
            offsets = (np.arange(q_slots) - radius + 0.5).astype(float)
            xy = np.stack([9.5 + offsets, np.zeros(q_slots)], axis=1)
            xy = torch.from_numpy(xy)[None, :, :]  # [1,Q,2]
            slots = torch.zeros(1, dtype=torch.long)
            d_values.append(float(_profile_d_s(error, slots, xy)[0]))
        # sum 聚合：三个 R 的 d_s 都是 2.0（mean 版本将是 1.0/0.5/0.25）
        for value in d_values:
            assert value == pytest.approx(2.0, abs=1e-6)

    def test_epe_loss_invariant_to_segmentation(self):
        """切段不变性（DEC-007）：同一直边不同 fragmentation 的 L_epe 一致。"""

        # 邻近第二矩形使光学沿线非均匀（否则任何归约都恒等、无判别力）
        def make_problem(corner, segment):
            """按 fragmentation 参数构造同一直边问题。"""
            frag = FragmentationConfig(
                corner_length_dbu=corner, max_segment_length_dbu=segment, max_displacement_dbu=10.0, miter_limit=4.0
            )
            region = kdb.Region(kdb.Box(8, 8, 72, 12))
            region.insert(kdb.Box(20, 20, 56, 40))  # 光学沿线变化的邻特征
            macro = _macro()
            batch = RegionBatch({LAYER: region}, macro.query_box)
            return prepare_macro_problem(batch, LAYER, "clear", frag, macro, data_bounds=BOUNDS)

        losses = []
        for corner, segment in ((8.0, 16.0), (4.0, 32.0), (2.0, 30.0)):
            problem = make_problem(corner, segment)
            assert problem.segments.segment_count > 4  # 确实多段且长短混合
            result = _optimize(problem, _LinearModel(), _config(iterations=1, weight_epe=1.0))
            losses.append(result.records[0].epe_loss)
        # 容差依据：midpoint 求积的一阶近似差，本几何实测 spread ≈0.8%
        # （等权归约在同组切法下随短段占比漂移，不具该稳定性——DEC-007）
        spread = max(losses) - min(losses)
        assert spread == pytest.approx(0.0, abs=2e-2 * max(abs(l) for l in losses))


class TestCallCounts:
    """热路径调用次数契约（PERF-003/004）。"""

    def test_forward_many_once_per_batch_per_state(self):
        """每状态每批恰一次三条件前向。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        model = _LinearModel()
        _optimize(problem, model, _config(batch_size=2))  # 3 状态 × 2 批
        assert model.calls == 6

    def test_candidate_geometry_reconstructed_once_and_reused(self, monkeypatch):
        """候选几何每 state 恰重建一次并被下一状态栅格/采样复用。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        counts = {"n": 0}
        real = gradient_module.reconstruct_region_with_midpoints

        def spy(problem_, displacements):
            """计数重建调用。"""
            counts["n"] += 1
            return real(problem_, displacements)

        monkeypatch.setattr(gradient_module, "reconstruct_region_with_midpoints", spy)
        _optimize(problem, _LinearModel(), _config(iterations=2))
        assert counts["n"] == 3  # 1 次 reference + 2 次候选（各发布一次）

    def test_sampling_midpoints_come_from_published_reconstruction(self, monkeypatch):
        """apply 收到的采样中点逐行可在已发布重构的中点里找到。

        刚体推算（参考中点+法向×位移）在 corner 邻段处对不上任何一次
        发布值——本测试锁死采样坐标只来自已发布重构几何。
        """
        problem = _problem(kdb.Region(kdb.Box(10, 10, 70, 70)))
        published = []  # 各次重构发布的段中点快照（DBU）
        real_reconstruct = gradient_module.reconstruct_region_with_midpoints

        def reconstruct_spy(problem_, displacements):
            """记录每次重构发布的 (Region, 中点)。"""
            region, midpoints = real_reconstruct(problem_, displacements)
            published.append(midpoints.copy())
            return region, midpoints

        monkeypatch.setattr(gradient_module, "reconstruct_region_with_midpoints", reconstruct_spy)
        captured = []  # apply 实收中点（canvas 坐标 float32）

        class _CaptureMask(_EdgeGradientMask):
            """捕获 apply 输入中点的代理（前向逻辑继承不变）。"""

            @staticmethod
            def forward(ctx, hard_masks, local_displacements, batch_indices, midpoints_xy, pixel_dbu):
                captured.append(midpoints_xy.detach().clone())
                return _EdgeGradientMask.forward(
                    ctx, hard_masks, local_displacements, batch_indices, midpoints_xy, pixel_dbu
                )

        monkeypatch.setattr(gradient_module, "_EdgeGradientMask", _CaptureMask)
        _optimize(problem, _LinearModel(), _config(iterations=1))
        assert len(published) >= 2  # reference + 至少一次候选
        # 允许集合 = 每次发布中点经任一 core context 换算的 canvas 坐标
        # （float32 精确比对：换算与类型转换都是确定性的）。
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        allowed = set()
        for snapshot in published:
            for core_index in range(problem.macro.core_count):
                spec = problem.macro.core(core_index)
                converted = points_to_canvas(snapshot, spec.context_box, pixel_dbu, canvas).astype(np.float32)
                allowed.update((float(x), float(y)) for x, y in converted)
        checked = 0
        for tensor in captured:
            rows = tensor.numpy()
            checked += len(rows)
            for row in rows:
                assert (float(row[0]), float(row[1])) in allowed, row
        assert checked > 0  # 确实捕获了采样条目

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
        _optimize(problem, _LinearModel(), _config(), cache=TargetCanvasCache(0))  # 禁用缓存对照
        assert counts["n"] == 16 + 24  # 每 state 的 target 都要重栅格


class TestStructuralSplit:
    """三段函数拆分的结构级单测（数值行为由既有端到端用例守护）。"""

    def test_prepare_gradient_context_static_mappings(self):
        """静态上下文的 owner 映射、membership 与计分像素直接来自 problem。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        config = _config()
        ctx = _prepare_gradient_context(problem, _LinearModel(), config)
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        assert np.array_equal(ctx.owner_ids, owner_ids)
        assert ctx.segment_to_parameter.shape == (problem.segments.segment_count,)
        assert np.all(ctx.segment_to_parameter[owner_ids] >= 0)
        assert np.all(ctx.segment_to_parameter[problem.owner_indices < 0] == -1)
        assert ctx.reference_segment_midpoints.shape == (problem.segments.segment_count, 2)
        for core_index in range(problem.macro.core_count):
            members = np.asarray(problem.segments_for_core(core_index))
            expected = members[ctx.segment_to_parameter[members] >= 0]
            assert np.array_equal(ctx.core_sampling_members[core_index], expected)
            assert np.array_equal(ctx.pack.owner_members[core_index], problem.owner_segments_for_core(core_index))
        assert ctx.pack.total_pixels > 0
        assert [c.name for c in ctx.conditions] == ["nominal", "dose_max", "defocus_min"]

    def test_evaluate_state_without_gradient_keeps_params_clean(self):
        """纯评价路径不为参数创建梯度（评价与更新职责分离）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        config = _config()
        model = _LinearModel()
        ctx = _prepare_gradient_context(problem, model, config)
        parameters = torch.zeros(len(ctx.owner_ids), dtype=torch.float32, requires_grad=True)
        evaluation = _evaluate_state(
            ctx,
            model,
            problem,
            config,
            TargetCanvasCache(_CACHE_BUDGET),
            parameters,
            ctx.pack.reference_region,
            ctx.reference_segment_midpoints,
            build_gradient=False,
        )
        assert parameters.grad is None
        assert evaluation.total_loss > 0.0

    def test_evaluate_state_with_gradient_accumulates_only(self):
        """建图路径只累积梯度，参数值保持调用前原值。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        config = _config()
        model = _LinearModel()
        ctx = _prepare_gradient_context(problem, model, config)
        parameters = torch.zeros(len(ctx.owner_ids), dtype=torch.float32, requires_grad=True)
        before = parameters.detach().clone()
        _evaluate_state(
            ctx,
            model,
            problem,
            config,
            TargetCanvasCache(_CACHE_BUDGET),
            parameters,
            ctx.pack.reference_region,
            ctx.reference_segment_midpoints,
            build_gradient=True,
        )
        assert parameters.grad is not None
        assert bool(torch.isfinite(parameters.grad).all())
        assert torch.equal(parameters.detach(), before)

    def test_take_optimizer_step_none_pair_and_raise(self, monkeypatch):
        """零梯度返回 None；有梯度返回同源二元组；重构异常原样上抛。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        parameters = torch.zeros(len(owner_ids), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([parameters], lr=1.0)
        candidate_full = np.zeros(problem.segments.segment_count, dtype=np.float64)
        # 梯度全零：Adam 更新量为零，按 no_update 返回 None
        parameters.grad = torch.zeros_like(parameters)
        assert (
            _take_optimizer_step(
                problem, parameters, optimizer, owner_ids, candidate_full, macro_id="mr0c0", state_index=0
            )
            is None
        )
        # 非零梯度：参数移动并重构出同源 Region+midpoints 二元组
        parameters.grad = torch.full_like(parameters, 0.5)
        candidate = _take_optimizer_step(
            problem, parameters, optimizer, owner_ids, candidate_full, macro_id="mr0c0", state_index=0
        )
        assert isinstance(candidate[0], kdb.Region)
        assert candidate[1].shape == (problem.segments.segment_count, 2)
        # 重构失败必须原样上抛，不得吞成返回值

        def _raise(problem, displacements):
            """模拟非法候选几何。"""
            raise ReconstructionError("boom")

        monkeypatch.setattr(gradient_module, "reconstruct_region_with_midpoints", _raise)
        parameters.grad = torch.full_like(parameters, 0.5)
        with pytest.raises(ReconstructionError):
            _take_optimizer_step(
                problem, parameters, optimizer, owner_ids, candidate_full, macro_id="mr0c0", state_index=0
            )


class TestRealModel:
    """真实 ICCAD13 的方向一致性与 CPU/CUDA 集成（TEST-003/011）。"""

    @staticmethod
    def _loss_numeric(problem, model, config, owner_index, delta, base=None):
        """在给定状态（默认零位移）上对单段 ±delta 复算加权连续 loss。"""
        segments = problem.segments.segment_count
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        displacements = np.zeros(segments, dtype=np.float64) if base is None else np.array(base, dtype=np.float64)
        displacements[owner_ids[owner_index]] += delta
        region = reconstruct_region(problem, displacements)
        reference = reconstruct_region(problem, np.zeros(segments, dtype=np.float64))
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
        weighted = 0.0
        total = 0
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(reference, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box, pixel_dbu, canvas)
            total += int(own.sum())
            printed = model.forward_many(torch.from_numpy(mask)[None], conditions)
            nom = printed["nominal"][0]
            dmax = printed["dose_max"][0]
            dmin = printed["defocus_min"][0]
            t = torch.from_numpy(target)
            own_t = torch.from_numpy(own)
            nominal_part = float(((nom - t) ** 2 * own_t).sum())
            process_part = float((((dmax - t) ** 2 + (dmin - t) ** 2) * own_t).sum())
            pv_part = float(((dmax - dmin) ** 2 * own_t).sum())
            weighted += (
                config.weight_nominal_l2 * nominal_part
                + config.weight_process_l2 * process_part
                + config.weight_pvband * pv_part
            )
        return weighted / total

    @staticmethod
    def _surrogate_gradients(problem, model, base=None):
        """直接经 _EdgeGradientMask 求指定状态（默认零位移）的 owner 梯度。"""
        segments = problem.segments.segment_count
        owner_ids = np.flatnonzero(problem.owner_indices >= 0)
        segment_to_parameter = np.full(segments, -1, dtype=np.int32)
        segment_to_parameter[owner_ids] = np.arange(len(owner_ids))
        values = np.zeros(segments, dtype=np.float64) if base is None else np.array(base, dtype=np.float64)
        # 与求解器同源：mask 来自当前状态重构，采样中点来自同一次重构。
        region, midpoints = reconstruct_region_with_midpoints(problem, values)
        reference = reconstruct_region(problem, np.zeros(segments, dtype=np.float64))
        parameters = torch.zeros(len(owner_ids), requires_grad=True)
        pixel_dbu = int(problem.macro.pixel_dbu)
        canvas = int(problem.macro.canvas_pixels)
        conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
        loss = torch.zeros((), dtype=torch.float32)
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            mask = rasterize_mask_canvas(region, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            target = rasterize_mask_canvas(reference, spec.context_box, pixel_dbu, canvas, polarity=problem.polarity)
            own = ownership_canvas(spec.ownership_box, spec.context_box, pixel_dbu, canvas)
            hard = torch.from_numpy(mask)[None]
            # 与实现一致：采样集合 = 该 core 全部 membership 中的 owner 段
            # （P1-1 修复后语义；owner_segments_for_core 只覆盖 owner core）。
            members = np.asarray(problem.segments_for_core(core_index))
            members = members[segment_to_parameter[members] >= 0]
            if len(members):
                local = parameters[torch.from_numpy(segment_to_parameter[members].astype(np.int64))]
                slots = torch.zeros(len(members), dtype=torch.int64)
                mids = torch.from_numpy(points_to_canvas(midpoints[members], spec.context_box, pixel_dbu, canvas)).to(
                    dtype=torch.float32
                )
                mask_tensor = _EdgeGradientMask.apply(hard, local, slots, mids, pixel_dbu)
            else:
                mask_tensor = hard
            printed = model.forward_many(mask_tensor, conditions)
            t = torch.from_numpy(target)[None]
            own_t = torch.from_numpy(own)[None]
            loss = (
                loss
                + (
                    (
                        (printed["nominal"] - t) ** 2
                        + 0.5 * ((printed["dose_max"] - t) ** 2 + (printed["defocus_min"] - t) ** 2)
                        + 0.1 * (printed["dose_max"] - printed["defocus_min"]) ** 2
                    )
                    * own_t
                ).sum()
            )
        loss.backward()
        return owner_ids, parameters.grad.numpy()

    @pytest.mark.parametrize("polarity", ["clear", "opaque"])
    def test_surrogate_direction_matches_integer_geometry_difference(self, cpu_model, polarity):
        """surrogate 梯度与 ±1 DBU 精确几何差分同号（真实 ICCAD13）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), polarity=polarity)
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
            plus = self._loss_numeric(problem, cpu_model, config, owner_index, 1.0)
            minus = self._loss_numeric(problem, cpu_model, config, owner_index, -1.0)
            difference = (plus - minus) / 2.0
            if abs(difference) <= 1e-9:  # 数值零不比较方向
                continue
            checked += 1
            assert gradients[owner_index] * difference > 0.0, (
                f"段 {owner_index}：surrogate {gradients[owner_index]:.3e} 与有限差分 {difference:.3e} 方向相反"
            )
        assert checked >= 2  # 至少两段完成了方向验证

    def test_surrogate_direction_matches_difference_at_nonuniform_state(self, cpu_model):
        """非均匀位移状态：corner 邻段 surrogate 与 ±1 DBU 差分仍同号。

        baseline 全零位移时 corner 邻段无切向失配，掩盖刚体推算中点的
        缺陷；先构造 横 +3/竖 −2 状态，再对 corner 邻段做数值差分。
        """
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        config = _config()
        geometry = problem.segments.materialize(None)
        base = np.zeros(problem.segments.segment_count, dtype=np.float64)
        bottom = (geometry.starts[:, 1] == 20) & (geometry.ends[:, 1] == 20)
        right = (geometry.starts[:, 0] == 60) & (geometry.ends[:, 0] == 60)
        base[bottom] = 3.0
        base[right] = -2.0
        owner_ids, gradients = self._surrogate_gradients(problem, cpu_model, base)
        owner_set = set(owner_ids.tolist())
        # 底边上最靠近右下角 (60,20) 的 owner fragment：corner 邻段。
        candidates = np.flatnonzero(bottom & (np.maximum(geometry.starts[:, 0], geometry.ends[:, 0]) == 60))
        corner_frag = int(next(i for i in candidates if i in owner_set))
        owner_index = int(np.flatnonzero(owner_ids == corner_frag)[0])
        plus = self._loss_numeric(problem, cpu_model, config, owner_index, 1.0, base)
        minus = self._loss_numeric(problem, cpu_model, config, owner_index, -1.0, base)
        difference = (plus - minus) / 2.0
        assert abs(difference) > 1e-9  # corner 邻段在非均匀状态下有实差分
        assert gradients[owner_index] * difference > 0.0, (
            f"corner 邻段 {corner_frag}：surrogate {gradients[owner_index]:.3e} 与有限差分 {difference:.3e} 方向相反"
        )

    def test_real_iccad13_cpu_runs_nonzero_update_and_valid_best(self, cpu_model):
        """真实 CPU 模型：有限 loss、非零更新、best 不差于 baseline。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 60, 60)))
        result = _optimize(problem, cpu_model, _config(iterations=2))
        assert all(np.isfinite(record.total_loss) for record in result.records)  # 全部有限
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
        assert cpu_result.records[0].total_loss == pytest.approx(cuda_result.records[0].total_loss, rel=2e-4, abs=1e-6)
        moved = np.abs(cpu_result.best_displacements) > 1e-9  # 已更新段
        if moved.any():
            signs = np.sign(cpu_result.best_displacements[moved]) == np.sign(cuda_result.best_displacements[moved])
            assert signs.all()  # 选定段方向一致


class TestSimpleCompat:
    """simple 方法与共享缓存的兼容（REQ-016）。"""

    def test_cache_import_paths_preserved(self):
        """包级、simple 模块级与 _cache 三条导入路径同一对象。"""
        from opc.iteration.mbopc import TargetCanvasCache as package_cache
        from opc.iteration.mbopc._cache import TargetCanvasCache as underscore_cache
        from opc.iteration.mbopc.simple import TargetCanvasCache as simple_cache

        assert package_cache is simple_cache is underscore_cache
