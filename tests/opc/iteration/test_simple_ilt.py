"""Simple ILT 的参数化、损失公式、状态语义、梯度累加与真实模型测试。"""

import klayout.db as kdb
import numpy as np
import pytest
import torch
from torch.nn import functional

from layout import DbuBox, LayerSpec, RegionBatch
from lithography import ICCAD13Lithography, ProcessCondition
from opc.input.pixel import prepare_pixel_macro_problem
from opc.input.raster import _center_padding
from opc.iteration.ilt import (
    SimpleILTConfig,
    build_simple_final_context_canvas,
    optimize_simple_macro,
)
from opc.iteration.ilt import simple as simple_module
from opc.iteration.ilt._common import (
    curvature_loss,
    owned_continuous_losses,
)

LAYER = LayerSpec(1, 0)
# 廉价契约：80² 版图、单 macro、core 40、context 20、pixel 4 → 2×2 core。
BOUNDS = DbuBox(0, 0, 80, 80)
_DEFAULTS = {"iterations": 1, "step_size": 0.5, "sigmoid_steepness": 4.0,
             "weight_process_l2": 1.0, "weight_pvband": 0.5,
             "curvature_weight": 0.0, "mask_threshold": 0.5, "batch_size": 2}


def _macro(**overrides):
    """返回单 macro 规划（默认 2×2 core；core_size 覆盖为 80 得单 core）。"""
    values = {"macro_grid": (1, 1), "core_size_dbu": 40, "context_dbu": 20,
              "pixel_dbu": 4, "canvas_pixels": 256}
    values.update(overrides)
    return plan_macros_local(BOUNDS, **values)[0]


def plan_macros_local(bounds, **kw):
    """薄包装：测试文件内统一导入网格规划。"""
    from opc.input import plan_macros
    return plan_macros(bounds, **kw)


def _problem(region, macro=None, polarity="clear"):
    """把原生 Region 包装为 RegionBatch 并生成像素 macro 问题。"""
    macro = macro if macro is not None else _macro()
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_pixel_macro_problem(
        batch, LAYER, polarity, macro, layout_bounds=BOUNDS)


def _config(**overrides):
    """按默认值组装 Simple ILT 配置。"""
    return SimpleILTConfig(**{**_DEFAULTS, **overrides})


def _soft0(target, beta=4.0):
    """OpenILT 初始化的 state0 soft 掩膜 σ(β(2T−1))。"""
    return 1.0 / (1.0 + np.exp(-beta * (2.0 * np.asarray(target) - 1.0)))


class _StubConfig:
    """提供求解器消费的 canvas 与二值阈值的最小配置视图。"""

    def __init__(self, canvas=256, threshold=0.5):
        self.canvas = canvas
        self.print_threshold = threshold


class _DoseModel:
    """线性可微假光刻：printed = mask×dose，逐像素独立。"""

    def __init__(self, doses=None):
        default = {"nominal": 1.0, "dose_max": 1.4, "defocus_min": 1.2}
        doses = doses if doses is not None else default
        self.device = torch.device("cpu")
        self.config = _StubConfig()
        self.calls = 0
        self._conditions = {
            name: ProcessCondition(
                name, "defocus" if name == "defocus_min" else "focus", dose)
            for name, dose in doses.items()}

    def condition(self, name):
        """返回固定剂量条件。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """输出等于 mask×dose（线性链路，梯度可手推）。"""
        self.calls += 1
        return {c.name: mask * self._conditions[c.name].dose
                for c in conditions}


class _IdentityModel(_DoseModel):
    """恒等假光刻：printed = mask（state0 公式检验用）。"""

    def __init__(self):
        super().__init__({"nominal": 1.0, "dose_max": 1.0, "defocus_min": 1.0})


class _ConstantModel:
    """输出恒为常数的假光刻：损失可手算且梯度恒零（参数冻结）。"""

    def __init__(self, value=0.5):
        self.value = value
        self.device = torch.device("cpu")
        self.config = _StubConfig()
        self.calls = 0
        self._conditions = {name: ProcessCondition(name, "focus", 1.0)
                            for name in ("nominal", "dose_max", "defocus_min")}

    def condition(self, name):
        """返回单位条件（值经 forward 生效）。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """三条件全部输出常数。"""
        self.calls += 1
        # +mask×0 保留对输入的零梯度依赖：常数输出且 backward 不失败
        return {c.name: mask * 0.0 + self.value for c in conditions}


class _ConstPerCallModel:
    """按调用序号输出常数的假光刻：逐 state/逐 core 损失可控且梯度恒零。"""

    def __init__(self, values):
        self.values = list(values)
        self.device = torch.device("cpu")
        self.config = _StubConfig()
        self.calls = 0
        self._conditions = {name: ProcessCondition(name, "focus", 1.0)
                            for name in ("nominal", "dose_max", "defocus_min")}

    def condition(self, name):
        """返回单位条件。"""
        return self._conditions[name]

    def forward_many(self, mask, conditions):
        """第 i 次调用输出 values[i]（越界取末值）。"""
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        # 同上：零梯度依赖保证常数输出可反向
        return {c.name: mask * 0.0 + float(value)
                for c in conditions}


class _LocalAverageModel(_DoseModel):
    """3×3 局部平均假光刻：制造跨 core 光学耦合（context 像素影响邻核损失）。"""

    def forward_many(self, mask, conditions):
        """printed = avgpool3(mask)×dose。"""
        self.calls += 1
        pooled = functional.avg_pool2d(mask[:, None], 3, stride=1,
                                       padding=1)[:, 0]
        return {c.name: pooled * self._conditions[c.name].dose
                for c in conditions}


class _NaNModel(_DoseModel):
    """输出全 NaN 的假光刻：触发非有限异常。"""

    def forward_many(self, mask, conditions):
        """输出全 NaN。"""
        self.calls += 1
        return {c.name: mask * float("nan") for c in conditions}


def _float64_reference(problem, config, model):
    """float64 独立镜像：单图全参数 autograd 复算逐 state 参数与损失。

    与被测实现完全不同的路径（不分组、不 scatter、双精度），
    用于锁定状态语义与一阶更新数值。
    """
    beta = float(config.sigmoid_steepness)
    pixel = problem.macro.pixel_dbu
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    mrow0 = (box.bottom - query.bottom) // pixel
    mcol0 = (box.left - query.left) // pixel
    t_own = (problem.target_u8[mrow0:mrow0 + hm, mcol0:mcol0 + wm]
             .astype(np.float64) / 255.0)
    core_count = problem.macro.core_count
    canvases = []
    for core_index in range(core_count):
        target = torch.from_numpy(
            problem.target_canvas(core_index)).to(torch.float64) / 255.0
        ownership = torch.from_numpy(problem.ownership_canvas(core_index))
        trainable = torch.from_numpy(
            problem.trainable_index_canvas(core_index))
        valid = torch.from_numpy(problem.context_valid_canvas(core_index))
        canvases.append((target, ownership, trainable, valid))
    parameters = torch.tensor(2.0 * t_own - 1.0, dtype=torch.float64)
    best_loss = float("inf")
    best = parameters.detach().clone()
    best_state = 0
    losses = []
    for state_index in range(config.iterations + 1):
        build = state_index < config.iterations
        if build:
            parameters = parameters.detach().requires_grad_(True)
        total = torch.zeros((), dtype=torch.float64)
        parts = {"nominal": 0.0, "process": 0.0, "pvband": 0.0}
        for target, ownership, trainable, valid in canvases:
            flat = parameters.reshape(-1)
            safe = trainable.clamp_min(0)
            local = flat[safe]
            soft = torch.sigmoid(beta * local)
            # 与实现同款三值语义：物理 context 初始 soft σ(β(2T−1))，
            # 数值 padding 恒 0
            context_soft = torch.sigmoid(beta * (target * 2.0 - 1.0))
            context = torch.where(valid, context_soft,
                                  torch.zeros_like(context_soft))
            mask = torch.where(trainable >= 0, soft, context)
            printed = model.forward_many(mask, (
                model.condition("nominal"), model.condition("dose_max"),
                model.condition("defocus_min")))
            l_nom, l_proc, l_pv = owned_continuous_losses(
                printed["nominal"], printed["dose_max"],
                printed["defocus_min"], target, ownership)
            total = total + weighted(
                l_nom, l_proc, l_pv, config)
            parts["nominal"] += float(l_nom.detach())
            parts["process"] += float(l_proc.detach())
            parts["pvband"] += float(l_pv.detach())
        losses.append((float(total.detach()), parts))
        if float(total.detach()) < best_loss:
            best_loss = float(total.detach())
            best_state = state_index
            best = parameters.detach().clone()
        if not build:
            break
        grad = torch.autograd.grad(total, parameters)[0]
        with torch.no_grad():
            parameters = parameters - config.step_size * grad
    return {"losses": losses, "best": best.numpy(), "best_state": best_state}


def weighted(l_nom, l_proc, l_pv, config):
    """镜像内的加权总损失（curvature=0 场景）。"""
    return (l_nom + config.weight_process_l2 * l_proc
            + config.weight_pvband * l_pv)


class TestConfigValidation:
    """SimpleILTConfig 的数值契约。"""

    @pytest.mark.parametrize(
        "overrides",
        [{"iterations": 0}, {"iterations": True}, {"batch_size": 0},
         {"step_size": 0.0}, {"step_size": float("nan")},
         {"sigmoid_steepness": -1.0}, {"weight_process_l2": -0.1},
         {"weight_pvband": float("inf")}, {"curvature_weight": -1.0},
         {"mask_threshold": 0.0}, {"mask_threshold": 1.0}],
        ids=["iter=0", "iter=bool", "batch=0", "step=0", "step=nan",
             "beta<0", "wp<0", "wpv=inf", "wc<0", "thr=0", "thr=1"])
    def test_invalid_values_fail(self, overrides):
        """越界参数在构造期失败。"""
        with pytest.raises(ValueError):
            _config(**overrides)

    def test_valid_defaults(self):
        """默认参数集构造成功且全字段在位。"""
        config = _config()
        assert config.iterations == 1
        assert config.mask_threshold == 0.5


class TestParameterizationAndLoss:
    """OpenILT 2T−1 初始化与四项损失公式（TEST-005）。"""

    def test_state0_soft_matches_openilt_initialization(self):
        """state0 soft = σ(β(2T−1))：恒等模型损失与 numpy 复算一致。"""
        # 二值一致性（REQ-B）：threshold 0.5 下 soft 与 T≥0.5 同判
        probe = _soft0(np.array([0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]))
        assert list(probe >= 0.5) == [False, False, True, True]
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))  # 含 1/4 覆盖格
        config = _config(weight_process_l2=1.0, weight_pvband=0.5)
        result = optimize_simple_macro(problem, _IdentityModel(), config)
        expected = 0.0
        for core in range(problem.macro.core_count):
            target = problem.target_canvas(core).astype(np.float64) / 255.0
            own = problem.ownership_canvas(core)
            expected += float(((_soft0(target) - target) ** 2 * own).sum())
        record = result.records[0]
        # 恒等模型：nominal=Σd²、process=2Σd²、pvband=0 → total=3Σd²
        assert record.total_loss == pytest.approx(3.0 * expected, rel=1e-5)
        assert record.pvband_loss == 0.0  # 三条件相同

    def test_hand_computed_losses_constant_model(self):
        """常数模型：nominal/process/pvband 与 numpy 复算逐项一致。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        model = _ConstantModel(0.5)
        config = _config(weight_process_l2=1.0, weight_pvband=0.5)
        result = optimize_simple_macro(problem, model, config)
        expected_nom = 0.0
        for core_index in range(problem.macro.core_count):
            target = problem.target_canvas(core_index).astype(
                np.float64) / 255.0
            own = problem.ownership_canvas(core_index)
            # 监督目标是 T：常数输出与 mask 无关，损失 = (c−T)²
            # 与初始化方案无关（state0 soft 只影响梯度路径）
            expected_nom += float(((0.5 - target) ** 2 * own).sum())
        expected_proc = 2.0 * expected_nom  # 两个 process 条件同常数
        record = result.records[0]
        assert record.nominal_l2 == pytest.approx(expected_nom, rel=1e-5)
        assert record.process_l2 == pytest.approx(expected_proc, rel=1e-5)
        assert record.pvband_loss == 0.0  # 常数差为 0
        assert record.total_loss == pytest.approx(
            expected_nom + 1.0 * expected_proc, rel=1e-5)

    def test_curvature_matches_numpy_reference(self):
        """曲率项与 numpy 3×3 零和核复算一致（权重计入总损失）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        config = _config(curvature_weight=2.0)
        model = _ConstantModel(0.5)
        result = optimize_simple_macro(problem, model, config)
        kernel = np.array([[-1, 5, -1], [5, -16, 5], [-1, 5, -1]],
                          dtype=np.float64) / 16.0
        expected = 0.0
        for core_index in range(problem.macro.core_count):
            target = problem.target_canvas(core_index).astype(
                np.float64) / 255.0  # [256,256]
            soft = _soft0(target)  # state0 掩膜是 σ(β(2T−1))
            own = problem.ownership_canvas(core_index)
            # 与实现同式：无 padding valid 卷积（相关形式）+ ownership 内圈
            convo = np.zeros((254, 254))
            for dy in range(3):
                for dx in range(3):
                    convo += kernel[dy, dx] * soft[dy:dy + 254,
                                                   dx:dx + 254]
            expected += float(((convo ** 2)
                               * own[1:-1, 1:-1]).sum())
        assert result.records[0].curvature_loss == pytest.approx(
            expected, rel=1e-3, abs=1e-6)
        assert result.records[0].total_loss == pytest.approx(
            result.records[0].nominal_l2
            + result.records[0].process_l2 + 2.0 * expected,
            rel=1e-3, abs=1e-6)

    def test_strict_zero_one_parameters_finite(self):
        """严格 0/1 target 的初值 ±1 参数有限。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))  # 纯 0/1
        result = optimize_simple_macro(problem, _DoseModel(), _config())
        assert np.all(np.isfinite(result.best_parameters))

    def test_owned_losses_count_ownership_only(self):
        """损失 helper 只统计 ownership 像素（context 差异不计分）。"""
        target = torch.zeros(1, 6, 6)
        printed_nom = torch.zeros(1, 6, 6)
        printed_max = torch.zeros(1, 6, 6)
        printed_min = torch.zeros(1, 6, 6)
        ownership = torch.zeros(1, 6, 6, dtype=torch.bool)
        printed_nom[0, 1, 1] = 0.7  # ownership 内一处偏差
        printed_nom[0, 4, 4] = 0.9  # context 内一处偏差（不计分）
        printed_max[0, 1, 1] = 0.7
        printed_min[0, 1, 1] = 0.7
        ownership[0, 1, 1] = True
        l_nom, l_proc, l_pv = owned_continuous_losses(
            printed_nom, printed_max, printed_min, target, ownership)
        assert float(l_nom) == pytest.approx(0.49)
        assert float(l_proc) == pytest.approx(0.98)  # 两个 process 条件
        assert float(l_pv) == 0.0
        # 曲率 helper 同样只统计卷积有效区∩ownership：spike 位于 ownership
        # 格 (1,1)，其自身核中心响应 -1；右邻格 (1,2) 以 5/16 边权看到 spike
        mask = torch.zeros(1, 6, 6)
        mask[0, 1, 1] = 1.0
        ownership[0, 1, 2] = True
        value = curvature_loss(mask, ownership)
        assert float(value) == pytest.approx(1.0 + (5.0 / 16.0) ** 2)


class TestStateAndBarrier:
    """N+1 状态、屏障 step 与 float64 镜像对照（TEST-006）。"""

    def test_state_count_and_forward_calls(self):
        """iterations=N → N+1 条记录；forward 恰每批每状态一次。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        model = _DoseModel()
        result = optimize_simple_macro(problem, model, _config(iterations=2))
        assert [r.state_index for r in result.records] == [0, 1, 2]
        assert all(r.stage_index == 0 and r.scale == 1 for r in result.records)
        assert model.calls == 3 * 2  # 4 core / batch 2 = 2 批 × 3 状态

    def test_single_core_matches_float64_reference(self):
        """单 core 全流程与 float64 镜像逐 state 一致（含一阶更新与 best）。"""
        macro = _macro(core_size_dbu=80)  # 单 core 覆盖全宏
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)), macro=macro)
        config = _config(iterations=2, weight_pvband=0.0)
        model = _DoseModel()
        result = optimize_simple_macro(problem, model, config)
        reference = _float64_reference(problem, config, model)
        for record, (total, parts) in zip(result.records, reference["losses"]):
            assert record.total_loss == pytest.approx(total, rel=1e-4)
            assert record.nominal_l2 == pytest.approx(
                parts["nominal"], rel=1e-4, abs=1e-9)
            assert record.process_l2 == pytest.approx(
                parts["process"], rel=1e-4, abs=1e-9)
        assert result.best_state_index == reference["best_state"]
        assert np.allclose(result.best_parameters, reference["best"],
                           rtol=1e-4, atol=1e-6)

    def test_barrier_no_param_update_mid_state(self):
        """同 state 全部 core 读同一快照：镜像核对 state0/1 损失。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))  # 跨 2×2 core
        config = _config(iterations=2, weight_pvband=0.0, batch_size=1)
        model = _DoseModel()
        result = optimize_simple_macro(problem, model, config)
        reference = _float64_reference(problem, config, model)
        for record, (total, _) in zip(result.records, reference["losses"]):
            # 若实现批内提前 step，state0 损失将偏离镜像快照值
            assert record.total_loss == pytest.approx(total, rel=1e-4)


class TestMacroBestAndBatch:
    """macro best 与 batch 切分不变（TEST-007）。"""

    def test_best_is_macro_total_not_per_core_patchwork(self):
        """各 core 局部最优 state 不同：best 只按宏总损失选择。"""
        # 2×2=4 core；材料占左列两核（9/10 列覆盖），右列两核全空
        problem = _problem(kdb.Region(kdb.Box(0, 0, 36, 80)))
        core_count = problem.macro.core_count
        assert core_count == 4
        config = _config(iterations=2, batch_size=1)
        # 逐 state 逐 core 常数（batch1 × 4 core × 3 state = 12 次调用）：
        # 材料核 argmin 在 s0，空核 argmin 在 s2，宏总和 argmin 在 s2——
        # 若实现按材料核局部最优或逐核拼贴选择，将得到 s0 而不是 s2。
        values = [0.9, 0.4, 0.9, 0.4,   # s0
                  1.0, 0.3, 1.0, 0.3,   # s1
                  0.8, 0.0, 0.8, 0.0]   # s2
        model = _ConstPerCallModel(values)
        result = optimize_simple_macro(problem, model, config)
        assert model.calls == 12

        def core_loss(core_index, value):
            """该 core 在常数输出下的 ownership 损失（监督为 T）。"""
            target = problem.target_canvas(core_index).astype(
                np.float64) / 255.0
            own = problem.ownership_canvas(core_index)
            return float(((value - target) ** 2 * own).sum())

        totals = [sum(core_loss(core, values[state * core_count + core])
                      for core in range(core_count)) for state in range(3)]
        assert result.best_state_index == int(np.argmin(totals))
        # 判别性：左列材料核与右列空核的局部最优必须不同，且宏最优
        # 与材料核局部最优不同（拼贴/局部最优都会选错）
        per_core_best = [
            int(np.argmin([core_loss(core, values[s * core_count + core])
                           for s in range(3)])) for core in range(core_count)]
        assert per_core_best[0] != per_core_best[1]
        assert result.best_state_index != per_core_best[0]

    def test_batch_size_invariance(self):
        """batch=1/2/4 下记录、best 与参数一致（拆分不改变语义）。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))
        model = _DoseModel()
        results = []
        for batch_size in (1, 2, 4):
            results.append(optimize_simple_macro(
                problem, model,
                _config(iterations=2, batch_size=batch_size)))
        base = results[0]
        for other in results[1:]:
            assert other.best_state_index == base.best_state_index
            assert np.allclose(other.best_parameters, base.best_parameters,
                               rtol=1e-5, atol=1e-6)
            for record_b, record_o in zip(base.records, other.records):
                assert record_o.total_loss == pytest.approx(
                    record_b.total_loss, rel=1e-5)


class TestCrossCoreGradient:
    """跨 core context 梯度求和（TEST-008）。"""

    @staticmethod
    def _boundary_shared_indices(problem):
        """返回同时在多个 core 批次出现过的宏像素索引 → 出现次数。"""
        counts: dict[int, int] = {}
        for core_index in range(problem.macro.core_count):
            canvas = problem.trainable_index_canvas(core_index)
            for value in np.unique(canvas[canvas >= 0]):
                counts[int(value)] = counts.get(int(value), 0) + 1
        return counts

    def test_shared_pixel_receives_summed_gradient(self, monkeypatch):
        """耦合模型下共享像素获得多 core 梯度之和（batch 1 与 2 一致）。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))
        counts = self._boundary_shared_indices(problem)
        shared = [index for index, n in counts.items() if n > 1]
        assert shared  # context 交叠区确有共享像素
        # batch=1：两个批分别累加；batch=2：单批内同时出现——求和语义下
        # 两者必须一致；若实现按出现次数平均，两种切分将给出不同结果。
        results = {}
        captures = []
        real_add_at = np.add.at

        def spy(dest, indices, values):
            """记录每次 scatter-add 的目标与载荷。"""
            captures.append((dest.sum(), len(indices)))
            real_add_at(dest, indices, values)

        monkeypatch.setattr(simple_module.np.add, "at", spy)
        for batch_size in (1, 2):
            captures.clear()
            model = _LocalAverageModel()
            results[batch_size] = optimize_simple_macro(
                problem, model, _config(iterations=1, batch_size=batch_size))
            assert captures  # scatter 确实发生
        base = results[1]
        other = results[2]
        assert other.best_state_index == base.best_state_index
        assert np.allclose(other.best_parameters, base.best_parameters,
                           rtol=1e-5, atol=1e-6)
        for record_b, record_o in zip(base.records, other.records):
            assert record_o.total_loss == pytest.approx(
                record_b.total_loss, rel=1e-5)

    def test_parameters_only_on_macro_ownership(self):
        """参数域形状恒等于宏 ownership 整像素形状（context 无参数）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = optimize_simple_macro(problem, _DoseModel(), _config())
        hm, wm = problem.ownership_shape
        assert result.best_parameters.shape == (hm, wm)
        assert result.soft_mask.shape == (hm, wm)
        assert result.binary_mask.shape == (hm, wm)
        assert np.logical_and(result.soft_mask >= 0.0,
                              result.soft_mask <= 1.0).all()


@pytest.fixture(scope="session")
def cpu_model():
    """共享一个真实 CPU ICCAD13 模型（资产加载昂贵）。"""
    return ICCAD13Lithography(device="cpu")


class _MaskCaptureModel(_IdentityModel):
    """记录每次 forward 输入 mask 的恒等模型（state0 画布捕获）。"""

    def __init__(self):
        super().__init__()
        self.captured = []

    def forward_many(self, mask, conditions):
        """克隆记录输入画布后透传恒等前向。"""
        self.captured.append(mask.detach().clone())
        return super().forward_many(mask, conditions)


class TestMacroSeamConsistency:
    """macro seam 初始 transmission 一致性（P1-1 后续修复）。"""

    @staticmethod
    def _state0_raster(problem):
        """运行一轮优化并返回 state0 mask 在 query 栅格上的重组数组。"""
        model = _MaskCaptureModel()
        optimize_simple_macro(
            problem, model,
            _config(iterations=1, batch_size=problem.macro.core_count))
        grid = np.full(problem.query_shape, np.nan, dtype=np.float32)
        for core_index in range(problem.macro.core_count):
            _, r0, r1, c0, c1 = problem._context_window(core_index)
            low_y, _, low_x, _ = _center_padding(r1 - r0, c1 - c0, 256)
            canvas = model.captured[0][core_index].numpy()  # state0 该 core
            grid[r0:r1, c0:c1] = canvas[
                low_y:low_y + (r1 - r0), low_x:low_x + (c1 - c0)]
        assert not np.isnan(grid).any()  # query 全部像素恰被 core 窗口覆盖
        return grid

    def test_context_matches_neighbor_state0(self):
        """A 的 context 中属于 B 的像素与 B 自身 state0 逐位一致。"""
        # 2×1 macro，矩形横跨 x=80 切线：B 区像素在 A 画布中是固定 context
        macros = plan_macros_local(
            DbuBox(0, 0, 160, 80), macro_grid=(2, 1), core_size_dbu=40,
            context_dbu=20, pixel_dbu=4, canvas_pixels=256)
        problems = []
        for macro in macros:  # 相邻两宏各自独立 problem（独立 macro 语义）
            batch = RegionBatch(
                {LAYER: kdb.Region(kdb.Box(20, 20, 140, 60))},
                macro.query_box)
            problems.append(prepare_pixel_macro_problem(
                batch, LAYER, "clear", macro, layout_bounds=DbuBox(0, 0, 160, 80)))
        grid_a = self._state0_raster(problems[0])
        grid_b = self._state0_raster(problems[1])
        pixel = 4
        beta = float(_DEFAULTS["sigmoid_steepness"])
        box = problems[1].macro.ownership_box  # B 的 ownership
        b_query = problems[1].macro.query_box
        b_r0 = (box.bottom - b_query.bottom) // pixel
        b_c0 = (box.left - b_query.left) // pixel
        # A 的 query 只覆盖 B 靠 seam 的一侧：比较 B∩A-query 的重叠带
        a_query = problems[0].macro.query_box
        a_r0 = max((box.bottom - a_query.bottom) // pixel, 0)
        a_c0 = max((box.left - a_query.left) // pixel, 0)
        rows = min((box.top - a_query.bottom) // pixel, grid_a.shape[0]) - a_r0
        cols = min((box.right - a_query.left) // pixel, grid_a.shape[1]) - a_c0
        assert rows > 0 and cols > 0  # 重叠带非空（context 覆盖 seam）
        seen_a = grid_a[a_r0:a_r0 + rows, a_c0:a_c0 + cols]
        seen_b = grid_b[b_r0:b_r0 + rows, b_c0:b_c0 + cols]
        # 核心断言：A 的固定 context 值 == B 自身 trainable 的 state0 值
        assert np.array_equal(seen_a, seen_b)
        # 判别性：一致值是 σ(β(2T−1)) 而非原始 T（T=1 格为 ≈0.982 而非 1.0）
        target = (problems[1].target_u8[b_r0:b_r0 + rows, b_c0:b_c0 + cols]
                  .astype(np.float64) / 255.0)
        expected = 1.0 / (1.0 + np.exp(-beta * (2.0 * target - 1.0)))
        assert np.allclose(seen_b, expected.astype(np.float32), atol=1e-6)
        assert float(seen_b.max()) < 1.0


class TestCanvasPaddingSemantics:
    """数值 padding 与物理 context 的三值画布语义（P1-1 Rev 1.2）。"""

    def test_padding_strictly_zero_and_context_soft(self):
        """window 显著小于 256：padding 严格 0；物理 T=0 context 格为 σ(−β)。"""
        # core 40 + context 4 → window (40+8)/4=12px << 256，padding 占绝大多数
        macro = _macro(context_dbu=4)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 40)), macro=macro)
        model = _MaskCaptureModel()
        optimize_simple_macro(
            problem, model,
            _config(iterations=1, batch_size=problem.macro.core_count))
        beta = float(_DEFAULTS["sigmoid_steepness"])
        soft_of = 1.0 / (1.0 + np.exp(-beta * (2.0 * 0.0 - 1.0)))  # σ(−β)
        for core_index in range(problem.macro.core_count):
            canvas = model.captured[0][core_index].numpy()
            valid = problem.context_valid_canvas(core_index)
            trainable = problem.trainable_index_canvas(core_index)
            # 数值 padding：必须严格 0（曾被误 sigmoid 成 σ(−β)≈0.018）
            assert float(np.abs(canvas[~valid]).max()) == 0.0
            # 真实 context 中物理 T=0 格：初始 soft σ(−β)（不是 0）
            physical_zero = valid & (trainable < 0)
            target = problem.target_canvas(core_index).astype(
                np.float64) / 255.0
            cells = canvas[physical_zero & (target == 0.0)]
            assert cells.size > 0  # 判别性：该 core 确有此类格
            assert np.allclose(cells, soft_of, atol=1e-6)


class TestRealModel:
    """真实 ICCAD13 的 CPU 更新与 CUDA parity（TEST-009）。"""

    @staticmethod
    def _small_problem():
        """单 core 小问题：非对齐边保留分数覆盖格（历史：logit 初始化
        时期它是唯一梯度入口；OpenILT 2T−1 初始化后不再必要，保留作
        分数格路径的回归几何）。
        """
        macro = _macro(core_size_dbu=80)
        return _problem(kdb.Region(kdb.Box(21, 20, 61, 60)), macro=macro)

    @staticmethod
    def _aligned_problem():
        """单 core 纯对齐问题：无分数覆盖格（P1-1 判别几何）。"""
        macro = _macro(core_size_dbu=80)
        return _problem(kdb.Region(kdb.Box(20, 20, 60, 60)), macro=macro)

    def test_real_cpu_update_finite_and_effective(self, cpu_model):
        """CPU 一次更新：全部有限且至少一个像素参数改变。"""
        problem = self._small_problem()
        result = optimize_simple_macro(
            problem, cpu_model, _config(iterations=1, batch_size=1))
        assert len(result.records) == 2
        for record in result.records:
            assert np.isfinite(record.total_loss)
        assert np.all(np.isfinite(result.best_parameters))
        assert result.records[1].total_loss != result.records[0].total_loss

    def test_real_cpu_updates_aligned_geometry(self, cpu_model):
        """P1-1 回归：纯对齐几何（无分数格）一轮更新即可观测。

        旧 logit+eps 初始化下 0/1 像素 sigmoid 斜率 ≈ β·eps（饱和），
        该几何的 state1 损失与 state0 完全相等；OpenILT 2T−1 初始化
        斜率 ≈ β·σ(β)σ(−β)（β=4 时 0.0707），更新必须可见。
        """
        problem = self._aligned_problem()
        config = _config(iterations=1, batch_size=1)
        result = optimize_simple_macro(problem, cpu_model, config)
        assert result.records[1].total_loss != result.records[0].total_loss
        # best 参数相对初值 2T−1 的最大位移须显著高于饱和区：实测新方案
        # ≈3.9e-4（真实光刻链路局部灵敏度低于线性 stub），旧 logit+eps
        # 初始化下 <1e-6——阈值取 1e-5 同时远离两者。
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        hm, wm = problem.ownership_shape
        r0 = (box.bottom - query.bottom) // pixel
        c0 = (box.left - query.left) // pixel
        initial = (2.0 * problem.target_u8[r0:r0 + hm, c0:c0 + wm]
                   .astype(np.float32) / 255.0 - 1.0)
        assert float(np.abs(result.best_parameters - initial).max()) > 1e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_real_cuda_matches_cpu(self, cpu_model):
        """CUDA 与 CPU 同输入 loss 一致（1e-4 容差）。"""
        problem = self._small_problem()
        cuda_model = ICCAD13Lithography(device="cuda")
        cpu_result = optimize_simple_macro(
            problem, cpu_model, _config(iterations=1, batch_size=1))
        cuda_result = optimize_simple_macro(
            problem, cuda_model, _config(iterations=1, batch_size=1))
        for cpu_record, cuda_record in zip(cpu_result.records,
                                           cuda_result.records):
            assert cuda_record.total_loss == pytest.approx(
                cpu_record.total_loss, rel=1e-4, abs=1e-6)


class TestCallCountsAndProgress:
    """调用计数、curvature 开关与进度回调（TEST-010）。"""

    def test_mixed_batch_progress_counts(self):
        """尾批不足 B 时按真实批大小回调；总数恒 core×(N+1)。"""
        macro = _macro(core_size_dbu=20)  # 4×4=16 core
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)), macro=macro)
        received = []
        model = _DoseModel()
        optimize_simple_macro(
            problem, model, _config(iterations=2, batch_size=6),
            on_tiles_completed=received.append)
        assert received == [6, 6, 4] * 3  # 每 state [6,6,4] × 3 状态
        assert model.calls == 3 * 3  # 每状态 3 批

    def test_no_conv_when_curvature_zero(self, monkeypatch):
        """curvature_weight=0 时不构建卷积图；>0 时才调用 conv2d。"""
        def _forbidden(*args, **kwargs):
            """curvature 关闭时不得出现卷积。"""
            raise AssertionError("curvature_weight=0 不应执行 conv2d")

        monkeypatch.setattr(functional, "conv2d", _forbidden)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        optimize_simple_macro(problem, _DoseModel(), _config())
        monkeypatch.undo()
        optimize_simple_macro(
            problem, _DoseModel(), _config(curvature_weight=1.0))  # 正常执行

    def test_nonfinite_loss_raises(self):
        """NaN 模型触发 FloatingPointError，不发布结果。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        with pytest.raises(FloatingPointError):
            optimize_simple_macro(problem, _NaNModel(), _config())

    def test_canvas_mismatch_rejected(self):
        """模型画布与 problem 不一致在入口失败。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        model = _DoseModel()
        model.config = _StubConfig(canvas=128)
        with pytest.raises(ValueError, match="画布"):
            optimize_simple_macro(problem, model, _config())

    def test_curvature_requires_at_least_one_context_pixel(self):
        """P2-1：curvature>0 且 context<1 像素时入口拒绝；关曲率则合法。"""
        macro = _macro(context_dbu=0)  # 窗口恰等于 core（10px）
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 40)), macro=macro)
        with pytest.raises(ValueError, match="context"):
            optimize_simple_macro(problem, _DoseModel(),
                                  _config(curvature_weight=1.0))
        # 曲率关闭时 context=0 完全合法（不构造卷积）
        result = optimize_simple_macro(problem, _DoseModel(), _config())
        assert len(result.records) == 2


class TestFinalContextHelper:
    """Simple 终评 fixed-context helper 与训练 context 公式等价（REQ-012）。"""

    def test_helper_equals_solver_context_formula(self):
        """helper 输出 = 训练热路径同款 σ(β(2T−1))｜valid，padding 恒 0。

        numpy 公式逐位相等（同一实现路径）；torch sigmoid 允许 float32
        实现差异（REQ-012 容差），训练与终评宏观边界仍一致。
        """
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))  # 含分数覆盖格
        config = _config()
        for core_index in range(problem.macro.core_count):
            actual = build_simple_final_context_canvas(
                problem, core_index, config)
            valid = problem.context_valid_canvas(core_index)
            target = problem.target_canvas(core_index)
            # numpy 同式复算：helper 与 _soft0 是同一实现路径，必须逐位相等
            expected_np = np.where(
                valid, _soft0(target.astype(np.float32) / 255.0), 0.0)
            np.testing.assert_array_equal(actual, expected_np)
            # torch 训练公式（solver 内联同式）：跨实现只要求浮点容差
            target_t = torch.from_numpy(target).to(torch.float32) / 255.0
            expected_t = torch.sigmoid(
                4.0 * (target_t * 2.0 - 1.0)).numpy()
            np.testing.assert_allclose(
                actual, np.where(valid, expected_t, 0.0),
                rtol=0.0, atol=1e-6)
            assert actual.dtype == np.float32
            # 三值语义：window 外数值 padding 严格为 0
            assert not actual[~valid].any()
