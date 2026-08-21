"""CurvMulti ILT 的多尺度语义、wafer 曲率、warm-start 与真实模型测试。"""

import klayout.db as kdb
import numpy as np
import pytest
import torch

from lithography import ICCAD13Lithography
from opc.iteration.ilt import (
    CurvMultiConfig,
    build_curvmulti_final_context_canvas,
    build_simple_final_context_canvas,
    optimize_curvmulti_macro,
)
from opc.iteration.ilt import curvmulti as curvmulti_module
from opc.iteration.ilt._common import (
    curvature_loss,
    resize_image,
    smooth_sigmoid_mask,
)

# 复用 Simple 套件已验证的生成式基建（跨测试文件导入先例：test_levelset_ilt）。
from tests.opc.iteration.test_simple_ilt import (
    _ConstantModel,
    _DoseModel,
    _IdentityModel,
    _macro,
    _problem,
    _StubConfig,
)

_DEFAULTS = {"scales": (2, 1), "iterations_per_stage": 1, "step_size": 0.5,
             "smoothing_kernel": 3, "sigmoid_steepness": 4.0,
             "sigmoid_offset": 0.5, "weight_process_l2": 1.0,
             "weight_pvband": 0.5, "curvature_weight": 0.0,
             "mask_threshold": 0.5, "batch_size": 2}


def _config(**overrides):
    """按默认值组装 CurvMulti 配置（80² 版图 → 宏 20×20 像素）。"""
    return CurvMultiConfig(**{**_DEFAULTS, **overrides})


class _LowpassCaptureModel(_IdentityModel):
    """5×5 低通 + 画布捕获的假光刻：wafer 曲率与 mask 曲率可判别。"""

    def __init__(self):
        super().__init__()
        self.masks = []
        self.printed = []

    def forward_many(self, mask, conditions):
        """记录输入 mask 与低通后输出（printed 形状同 mask、数值更平滑）。"""
        self.calls += 1
        self.masks.append(mask.detach().clone())
        smoothed = torch.nn.functional.avg_pool2d(
            mask[:, None], 5, stride=1, padding=2)[:, 0]
        output = {condition.name: smoothed.clone() for condition in conditions}
        self.printed.append({name: value.detach().clone()
                             for name, value in output.items()})
        return output


class _MaskCaptureModel(_IdentityModel):
    """记录每次 forward 输入/输出画布的恒等模型（state0 链路捕获）。"""

    def __init__(self):
        super().__init__()
        self.masks = []
        self.printed = []

    def forward_many(self, mask, conditions):
        """克隆记录输入画布与各条件输出后透传恒等前向。"""
        self.masks.append(mask.detach().clone())
        output = super().forward_many(mask, conditions)
        self.printed.append({name: value.detach().clone()
                             for name, value in output.items()})
        return output


class TestHelpers:
    """_common 新增 helper 的 numpy 参照逐位测试。"""

    def test_resize_nearest_is_block_repeat(self):
        """整数倍 nearest 上采样 = 块复制（warm-start 不引入新灰度）。"""
        grid = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4) / 16.0
        up = resize_image(grid, (8, 8), "nearest")
        reference = np.repeat(np.repeat(grid[0].numpy(), 2, axis=0), 2, axis=1)
        assert np.array_equal(up[0].numpy(), reference)

    def test_resize_area_is_block_mean(self):
        """整数倍 area 下采样 = 块均值（stage 参考保覆盖率）。"""
        grid = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
        down = resize_image(grid, (2, 2), "area")
        reference = grid[0].numpy().reshape(2, 2, 2, 2).mean(axis=(1, 3))
        assert np.allclose(down[0].numpy(), reference, atol=1e-12)

    def test_smooth_sigmoid_mask_manual(self):
        """k=3 零补边均值池化 + σ(β(x−offset)) 的手算逐位对照。"""
        beta, offset = 4.0, 0.5
        values = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        padded = np.pad(values, 1)  # avg_pool 边缘补零
        pooled = np.zeros_like(values)
        for row in range(3):
            for column in range(3):
                pooled[row, column] = padded[row:row + 3,
                                             column:column + 3].mean()
        expected = 1.0 / (1.0 + np.exp(-beta * (pooled - offset)))
        output = smooth_sigmoid_mask(
            torch.from_numpy(values).unsqueeze(0), 3, beta, offset)
        assert np.allclose(output[0].numpy(), expected, atol=1e-12)


class TestConfigValidation:
    """REQ-010 构造即校验的合法/非法边界。"""

    @pytest.mark.parametrize("overrides, match", [
        ({"scales": ()}, "scales"),                      # 空尺度
        ({"scales": (1, 2)}, "递减"),                    # 非递减
        ({"scales": (2,)}, "结尾"),                      # 未以 1 结尾
        ({"scales": [2, 1]}, "元组"),                    # list 不是 tuple
        ({"scales": (True, 1)}, "正整数"),               # bool 冒充 int
        ({"iterations_per_stage": 0}, "为正"),
        ({"batch_size": 0}, "为正"),
        ({"smoothing_kernel": 2}, "奇数"),               # 偶核
        ({"smoothing_kernel": 0}, "奇数"),
        ({"step_size": 0.0}, "范围"),
        ({"sigmoid_steepness": -1.0}, "范围"),
        ({"sigmoid_offset": 1.5}, "范围"),
        ({"weight_pvband": -0.1}, "范围"),
        ({"mask_threshold": 0.0}, "范围"),
        ({"step_size": float("nan")}, "有限"),
    ])
    def test_invalid_rejected(self, overrides, match):
        """非法配置在构造期失败，不进入张量分配。"""
        with pytest.raises(ValueError, match=match):
            _config(**overrides)

    def test_valid_multi_scale_accepted(self):
        """合法粗到细尺度与奇核通过构造。"""
        assert _config(scales=(4, 2, 1), smoothing_kernel=7).scales == (4, 2, 1)


class TestSolverSemantics:
    """多尺度状态语义、warm-start、优化器独立与曲率对象。"""

    def test_single_scale_degenerates(self):
        """scales=[1]：单尺度退化，records 全 stage 0/scale 1。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = optimize_curvmulti_macro(
            problem, _DoseModel(), _config(scales=(1,), batch_size=4))
        assert len(result.records) == 2  # N+1
        for index, record in enumerate(result.records):
            assert record.state_index == index
            assert record.stage_index == 0
            assert record.stage_state_index == index
            assert record.scale == 1
        assert result.best_parameters.shape == problem.ownership_shape

    def test_records_stage_coordinates(self):
        """scales=(2,1)、N=1：4 条记录的 stage/scale/编号逐条正确。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = optimize_curvmulti_macro(
            problem, _DoseModel(), _config(scales=(2, 1)))
        expected = [(0, 0, 0, 2), (1, 0, 1, 2), (2, 1, 0, 1), (3, 1, 1, 1)]
        actual = [(record.state_index, record.stage_index,
                   record.stage_state_index, record.scale)
                  for record in result.records]
        assert actual == expected

    def test_float64_mirror_single_scale(self):
        """单 core 逐 state float64 镜像：整条可微链与 SGD 更新逐值一致。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)),
                           macro=_macro(core_size_dbu=80))
        config = _config(scales=(1,), batch_size=4, step_size=0.25,
                         weight_process_l2=1.0, weight_pvband=0.5)
        beta, offset, kernel = (config.sigmoid_steepness,
                                config.sigmoid_offset, config.smoothing_kernel)
        doses = {"nominal": 1.0, "dose_max": 1.4, "defocus_min": 1.2}
        hm, wm = problem.ownership_shape
        trainable_flat = problem.trainable_index_canvas(0).reshape(-1)
        owned = trainable_flat >= 0
        safe = torch.from_numpy(np.where(owned, trainable_flat, 0))
        index3 = torch.from_numpy(problem.trainable_index_canvas(0))
        target = torch.from_numpy(
            problem.target_canvas(0).astype(np.float64) / 255.0)
        valid = torch.from_numpy(problem.context_valid_canvas(0))
        ownership = torch.from_numpy(problem.ownership_canvas(0))
        context = torch.where(valid, torch.sigmoid(beta * (2.0 * target - 1.0)),
                              torch.zeros_like(target))

        def state_total(control):
            """复刻求解器链路并返回 (total, autograd 可回传的 total)。"""
            mask_control = smooth_sigmoid_mask(control, kernel, beta, offset)
            full = resize_image(mask_control, (hm, wm), "nearest").view(-1)
            soft = full[safe].view(256, 256)
            mask = torch.where(index3 >= 0, soft, context)
            nominal = mask * doses["nominal"]
            dose_max = mask * doses["dose_max"]
            defocus = mask * doses["defocus_min"]
            nominal_l2 = ((nominal - target) ** 2 * ownership).sum()
            process_l2 = (((dose_max - target) ** 2
                           + (defocus - target) ** 2) * ownership).sum()
            pvband = ((dose_max - defocus) ** 2 * ownership).sum()
            total = (nominal_l2 + config.weight_process_l2 * process_l2
                     + config.weight_pvband * pvband)
            return total

        # 初值 = 宏 ownership 的 raw T（OpenILT 直用 [0,1] target）
        mrow0 = (problem.macro.ownership_box.bottom
                 - problem.macro.query_box.bottom) // 4
        mcol0 = (problem.macro.ownership_box.left
                 - problem.macro.query_box.left) // 4
        initial = torch.from_numpy(problem.target_u8[
            mrow0:mrow0 + hm, mcol0:mcol0 + wm].astype(np.float64) / 255.0
        ).unsqueeze(0).requires_grad_(True)
        mirror_losses = []
        control = initial
        for _ in range(2):  # state0 评价 + 一步 SGD + state1 评价
            total = state_total(control)
            mirror_losses.append(float(total.detach()))
            if len(mirror_losses) == 1:
                total.backward()
                control = (control.detach()
                           - config.step_size * control.grad)
                control.requires_grad_(True)
        result = optimize_curvmulti_macro(problem, _DoseModel(), config)
        for record, mirror in zip(result.records, mirror_losses):
            assert record.total_loss == pytest.approx(mirror, rel=1e-5)

    def test_batch_split_invariant(self):
        """batch=1 与 batch=4：records 与 best 逐值一致（宏同步屏障）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))  # 2×2 core
        results = {}
        for batch_size in (1, 4):
            results[batch_size] = optimize_curvmulti_macro(
                problem, _DoseModel(), _config(batch_size=batch_size))
        first, second = results[1], results[4]
        assert first.best_state_index == second.best_state_index
        assert np.allclose(first.best_parameters, second.best_parameters,
                           atol=1e-6)
        for left, right in zip(first.records, second.records):
            assert left.total_loss == pytest.approx(right.total_loss, rel=1e-6)

    def test_stage_optimizers_independent(self, monkeypatch):
        """每 stage 恰一个新 SGD 实例，参数张量不跨 stage 复用。"""
        real_sgd = torch.optim.SGD
        created = []

        class _SpySGD(real_sgd):
            """记录构造参数后透传真 SGD。"""

            def __init__(self, params, lr):
                super().__init__(params, lr=lr)
                created.append(list(self.param_groups[0]["params"]))

        monkeypatch.setattr(torch.optim, "SGD", _SpySGD)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        optimize_curvmulti_macro(problem, _DoseModel(), _config(scales=(2, 1)))
        assert len(created) == 2  # 每 stage 独立构造
        assert created[0][0] is not created[1][0]  # 控制张量不共享

    def test_warm_start_uses_stage_best_nearest(self, monkeypatch):
        """warm-start 调用存在且输入恒为 stage0 控制网格（ConstantModel 下 ==
        area 参考值），与批内上采样按调用计数区分。"""
        calls = []
        real_resize = curvmulti_module.resize_image

        def spy_resize(image, shape, mode):
            """记录 (输入形状, 输出形状, 模式, 输入快照) 后透传。"""
            calls.append((tuple(image.shape), shape, mode,
                          image.detach().cpu().clone()))
            return real_resize(image, shape, mode)

        monkeypatch.setattr(curvmulti_module, "resize_image", spy_resize)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        # 常数模型梯度恒零：控制网格全程不动，warm-start 输入必等于
        # stage0 initial（= area(target)），与批内上采样输入同值。
        optimize_curvmulti_macro(problem, _ConstantModel(),
                                 _config(scales=(2, 1)))
        area_calls = [call for call in calls if call[2] == "area"]
        assert len(area_calls) == 1 and area_calls[0][1] == (10, 10)
        coarse = [call for call in calls
                  if call[2] == "nearest" and call[0] == (1, 10, 10)]
        fine = [call for call in calls
                if call[2] == "nearest" and call[0] == (1, 20, 20)]
        # 计数解剖：粗网格 nearest = stage0 批内上采样 4（2 状态×2 批）
        # + warm-start 1 + best 物化 2（参数与软掩膜）= 7；细网格 identity
        # 上采样 = stage1 批内 4。warm-start 作为独立调用存在。
        assert len(coarse) == 7 and len(fine) == 4
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        mrow0 = (box.bottom - query.bottom) // 4
        mcol0 = (box.left - query.left) // 4
        area_reference = real_resize(torch.from_numpy(
            problem.target_u8[mrow0:mrow0 + 20, mcol0:mcol0 + 20]
            .astype(np.float32) / 255.0).unsqueeze(0), (10, 10), "area")
        # 粗网格 nearest 的输入有两类：批内上采样/软掩膜物化喂 sigmoid 值，
        # warm-start 与 best_parameters 物化喂原始控制网格（ConstantModel 下
        # 恒等于 area 参考）。恰 2 个原始值调用 = warm-start 与参数物化。
        raw_inputs = [call for call in coarse
                      if torch.equal(call[3], area_reference)]
        assert len(raw_inputs) == 2

    def test_best_ties_keep_earliest_across_stages(self):
        """常数模型：全部状态 loss 相同 → 全局 best 恒为 state 0。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = optimize_curvmulti_macro(
            problem, _ConstantModel(), _config(scales=(2, 1)))
        assert len(result.records) == 4
        assert result.best_state_index == 0

    def test_curvature_acts_on_printed_wafer(self):
        """曲率记录值 = curvature(printed nominal)，不等于 curvature(mask)。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        model = _LowpassCaptureModel()
        config = _config(scales=(1,), curvature_weight=10.0, batch_size=4)
        result = optimize_curvmulti_macro(problem, model, config)
        record = result.records[0]  # state0：mask 由初值决定，可独立复算
        mask = model.masks[0]  # [1,256,256]
        wafer = model.printed[0]["nominal"]
        ownership = torch.from_numpy(
            problem.ownership_canvas(0)).unsqueeze(0)
        wafer_value = float(curvature_loss(wafer.double(), ownership.double()))
        mask_value = float(curvature_loss(mask.double(), ownership.double()))
        assert wafer_value != mask_value  # 低通模型保证两者可判别
        assert record.curvature_loss == pytest.approx(wafer_value, rel=1e-4)

    def test_curvature_zero_weight_skips_conv(self, monkeypatch):
        """curvature_weight=0：不构建曲率卷积。"""
        calls = {"count": 0}
        # 曲率调用已上提到公共骨架（P3），spy 宿主随之迁移
        import opc.iteration.ilt._skeleton as ilt_skeleton
        real = ilt_skeleton.curvature_loss

        def spy(*args, **kwargs):
            """计数后透传。"""
            calls["count"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(ilt_skeleton, "curvature_loss", spy)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        result = optimize_curvmulti_macro(problem, _DoseModel(), _config())
        assert calls["count"] == 0
        assert all(record.curvature_loss == 0.0 for record in result.records)

    def test_entry_rejections(self):
        """求解器入口：整除/最粗核/画布一致/曲率 context 契约。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        with pytest.raises(ValueError, match="整除"):
            optimize_curvmulti_macro(
                problem, _DoseModel(), _config(scales=(3, 1)))
        with pytest.raises(ValueError, match="smoothing_kernel"):
            optimize_curvmulti_macro(
                problem, _DoseModel(), _config(scales=(4, 1),
                                               smoothing_kernel=7))
        mismatch = _DoseModel()
        mismatch.config = _StubConfig(canvas=128)
        with pytest.raises(ValueError, match="画布"):
            optimize_curvmulti_macro(problem, mismatch, _config())
        tight = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)),
                         macro=_macro(context_dbu=0))
        with pytest.raises(ValueError, match="context"):
            optimize_curvmulti_macro(
                tight, _DoseModel(), _config(curvature_weight=1.0))


class TestContextHelpers:
    """终评固定 context 与 Simple 逐值一致 + 三值语义。"""

    def test_context_helper_matches_simple_bitwise(self):
        """同 β 下 CurvMulti 与 Simple 的终评 context 画布逐位一致。"""
        problem = _problem(kdb.Region(kdb.Box(4, 4, 76, 76)))
        simple_config = type("SimpleView", (), {
            "sigmoid_steepness": _DEFAULTS["sigmoid_steepness"]})()
        for core_index in range(problem.macro.core_count):
            left = build_curvmulti_final_context_canvas(
                problem, core_index, _config())
            right = build_simple_final_context_canvas(
                problem, core_index, simple_config)
            assert np.array_equal(left, right)

    def test_state0_canvas_three_value(self):
        """state0 画布：trainable=上采样平滑 sigmoid、context=σ(β(2T−1))、padding=0。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)),
                           macro=_macro(core_size_dbu=80))
        model = _MaskCaptureModel()
        config = _config(scales=(1,), batch_size=4)
        optimize_curvmulti_macro(problem, model, config)
        mask = model.masks[0][0]  # 单 core：取批次唯一画布
        valid = torch.from_numpy(problem.context_valid_canvas(0))
        index3 = torch.from_numpy(problem.trainable_index_canvas(0))
        target = problem.target_canvas(0).astype(np.float64) / 255.0
        # 数值 padding 恒 0
        assert not mask[~valid].any()
        # context = 初始 soft（与 mask 上采样链无关）
        context_expected = torch.sigmoid(
            config.sigmoid_steepness * (2.0 * torch.from_numpy(target) - 1.0))
        assert torch.allclose(mask[valid & (index3 < 0)].double(),
                              context_expected[valid & (index3 < 0)], atol=1e-6)
        # trainable = 平滑 sigmoid 最近邻上采样（scale=1 时即控制网格本身）
        mrow0 = (problem.macro.ownership_box.bottom
                 - problem.macro.query_box.bottom) // 4
        mcol0 = (problem.macro.ownership_box.left
                 - problem.macro.query_box.left) // 4
        hm, wm = problem.ownership_shape
        initial = torch.from_numpy(problem.target_u8[
            mrow0:mrow0 + hm, mcol0:mcol0 + wm].astype(np.float64) / 255.0
        ).unsqueeze(0)
        control_mask = smooth_sigmoid_mask(
            initial, config.smoothing_kernel, config.sigmoid_steepness,
            config.sigmoid_offset)[0]
        trainable_slots = valid & (index3 >= 0)
        rows = (index3[trainable_slots] // wm)
        columns = (index3[trainable_slots] % wm)
        assert torch.allclose(
            mask[trainable_slots].double(), control_mask[rows, columns],
            atol=1e-6)


@pytest.fixture(scope="session")
def cpu_model():
    """共享一个真实 CPU ICCAD13 模型（资产加载昂贵）。"""
    return ICCAD13Lithography(device="cpu")


class TestRealModel:
    """真实 ICCAD13 的 CPU 有限性与 CUDA parity。"""

    @staticmethod
    def _small_problem():
        """单 core 小问题（同 Simple 真模型几何）。"""
        return _problem(kdb.Region(kdb.Box(21, 20, 61, 60)),
                        macro=_macro(core_size_dbu=80))

    def test_real_cpu_finite(self, cpu_model):
        """CPU 两尺度两状态：全部记录有限、物化契约成立。"""
        problem = self._small_problem()
        result = optimize_curvmulti_macro(
            problem, cpu_model, _config(scales=(2, 1), batch_size=4))
        assert len(result.records) == 4
        for record in result.records:
            assert np.isfinite(record.total_loss)
            assert np.isfinite(record.curvature_loss)
        assert result.best_parameters.shape == problem.ownership_shape
        assert (result.soft_mask >= 0.0).all() and (result.soft_mask <= 1.0).all()
        assert np.array_equal(result.binary_mask,
                              result.soft_mask >= 0.5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_real_cuda_matches_cpu(self, cpu_model):
        """CUDA 与 CPU 同输入 loss 一致（1e-4 容差）。"""
        problem = self._small_problem()
        cuda_model = ICCAD13Lithography(device="cuda")
        cpu_result = optimize_curvmulti_macro(
            problem, cpu_model, _config(scales=(2, 1), batch_size=4))
        cuda_result = optimize_curvmulti_macro(
            problem, cuda_model, _config(scales=(2, 1), batch_size=4))
        for cpu_record, cuda_record in zip(cpu_result.records,
                                           cuda_result.records):
            assert cuda_record.total_loss == pytest.approx(
                cpu_record.total_loss, rel=1e-4, abs=1e-6)
