"""LevelSet ILT：SDF 定义、halo 梯度、STE、跨 core 求和、宏 Adam 与真实模型测试。"""

import klayout.db as kdb
import numpy as np
import pytest
import torch

import opc.iteration.ilt.levelset as levelset_module
from lithography import ICCAD13Lithography
from opc.input.raster import _center_padding
from opc.iteration.ilt import (
    LevelSetILTConfig,
    build_levelset_final_context_canvas,
    macro_gradient_magnitude,
    optimize_levelset_macro,
    signed_distance_initialization,
)
from opc.iteration.ilt._common import owned_continuous_losses
from opc.iteration.ilt.levelset import _LevelSetBinarize

# 复用 Simple 测试的生成式 problem 构造与假光刻（单一事实源，避免复制漂移）
from tests.opc.iteration.test_simple_ilt import (
    _DoseModel,
    _IdentityModel,
    _LocalAverageModel,
    _macro,
    _MaskCaptureModel,
    _NaNModel,
    _problem,
    _StubConfig,
)

_DEFAULTS = {"iterations": 1, "step_size": 0.5, "weight_process_l2": 1.0,
             "weight_pvband": 0.5, "curvature_weight": 0.0, "batch_size": 2}


def _ls_config(**overrides):
    """按默认值组装 LevelSet 配置。"""
    return LevelSetILTConfig(**{**_DEFAULTS, **overrides})


@pytest.fixture()
def cpu_model():
    """真实 ICCAD13 CPU 模型（256² 画布）。"""
    return ICCAD13Lithography(device="cpu")


def _ownership_crop(problem, field):
    """取 query 域数组在 macro ownership 的切片（与实现同式换算）。"""
    pixel = problem.macro.pixel_dbu
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    r0 = (box.bottom - query.bottom) // pixel
    c0 = (box.left - query.left) // pixel
    return field[r0:r0 + hm, c0:c0 + wm]


class TestLevelSetConfigValidation:
    """LevelSetILTConfig 的数值契约（规格 §8.1：bool 拒当 int、权重非负）。"""

    @pytest.mark.parametrize("field,value", [
        ("iterations", 0), ("iterations", True), ("iterations", 1.5),
        ("batch_size", 0), ("batch_size", True),
        ("step_size", 0.0), ("step_size", float("inf")),
        ("step_size", float("nan")),
        ("weight_process_l2", -1.0), ("weight_pvband", -0.1),
        ("curvature_weight", -2.0)])
    def test_invalid_rejected(self, field, value):
        """非法字段在构造期失败（分配优化张量前）。"""
        with pytest.raises(ValueError):
            _ls_config(**{field: value})

    def test_minimal_valid_config(self):
        """最小合法配置可构造（权重全 0 合法）。"""
        config = LevelSetILTConfig(
            iterations=1, step_size=0.2, weight_process_l2=0.0,
            weight_pvband=0.0, curvature_weight=0.0, batch_size=1)
        assert config.iterations == 1


class TestSignedDistanceInitialization:
    """SDF 定义：精确像素中心距离 + 阈值语义 + 退化场 + once/macro（TEST-001）。"""

    @staticmethod
    def _brute_force_sdf(target_u8):
        """逐像素暴力最近异类距离 oracle（测试专用，禁入生产路径）。"""
        binary = target_u8.astype(np.float64) / 255.0 >= 0.5
        height, width = binary.shape
        phi = np.zeros((height, width), np.float64)
        for row in range(height):
            for col in range(width):
                best = None
                for r2 in range(height):
                    for c2 in range(width):
                        if binary[r2, c2] != binary[row, col]:
                            d = float(np.hypot(r2 - row, c2 - col))
                            if best is None or d < best:
                                best = d
                phi[row, col] = -best if binary[row, col] else best
        return phi

    @pytest.mark.parametrize("target_u8", [
        np.zeros((5, 7), np.uint8),  # 全背景（单类退化由常量场用例覆盖）
        (np.arange(35).reshape(5, 7) % 3 == 0).astype(np.uint8) * 255,  # 散点
        np.pad(np.full((3, 3), 255, np.uint8), 2),  # 居中矩形
    ])
    def test_matches_brute_force_reference(self, target_u8):
        """混合类 target：SDF 与暴力 oracle 逐值一致（float32 舍入容差）。"""
        binary = target_u8.astype(np.float64) / 255.0 >= 0.5
        if binary.all() or not binary.any():
            pytest.skip("单类退化由常量场用例覆盖")
        actual = signed_distance_initialization(target_u8)
        expected = self._brute_force_sdf(target_u8)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-6)

    def test_hole_pattern_matches_reference(self):
        """带洞环形 target（内外双向距离都非平凡）。"""
        target = np.zeros((6, 6), np.uint8)
        target[1:5, 1:5] = 255
        target[2:4, 2:4] = 0  # 中心洞
        actual = signed_distance_initialization(target)
        np.testing.assert_allclose(
            actual, self._brute_force_sdf(target), rtol=1e-6, atol=1e-6)

    def test_threshold_semantics_127_128(self):
        """阈值事实源 target/255>=0.5：127 背景、128 前景（phi 符号逐格一致）。"""
        target = np.array([[127, 128, 255, 0],
                           [128, 127, 64, 200],
                           [255, 128, 127, 128]], np.uint8)
        phi = signed_distance_initialization(target)
        binary = target.astype(np.float32) / 255.0 >= 0.5
        assert np.array_equal(phi < 0, binary)  # INV-004：前景 phi<0
        assert np.all(np.isfinite(phi))

    def test_all_foreground_constant_field(self):
        """全前景：有限常量场 -max(H,W)。"""
        phi = signed_distance_initialization(np.full((4, 9), 255, np.uint8))
        np.testing.assert_array_equal(phi, np.full((4, 9), -9.0, np.float32))

    def test_all_background_constant_field(self):
        """全背景：有限常量场 +max(H,W)。"""
        phi = signed_distance_initialization(np.zeros((4, 9), np.uint8))
        np.testing.assert_array_equal(phi, np.full((4, 9), 9.0, np.float32))

    def test_production_calls_scipy_edt(self, monkeypatch):
        """生产实现走 SciPy EDT：替换成计数 spy 后混合 target 必须命中。"""
        calls = {"n": 0}
        real = levelset_module.distance_transform_edt

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(
            levelset_module, "distance_transform_edt", spy)
        target = np.zeros((5, 5), np.uint8)
        target[2, 2] = 255
        signed_distance_initialization(target)
        assert calls["n"] == 2  # outside + inside 各一次（Python 逐像素路径为 0）

    def test_sdf_computed_once_per_macro(self, monkeypatch):
        """SDF once/macro：iterations=2 全程恰一次初始化。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        calls = {"n": 0}
        real = levelset_module.signed_distance_initialization

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(
            levelset_module, "signed_distance_initialization", spy)
        optimize_levelset_macro(
            problem, _DoseModel(), _ls_config(iterations=2))
        assert calls["n"] == 1


class TestMacroGradientMagnitude:
    """macro 域唯一 |grad(phi)|：中心差分精确性与调用次数（TEST-002）。"""

    def test_hand_computed_central_difference(self):
        """独立标量循环复算 halo 中心差分（含边缘参数用真实 query context）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        initial_phi = signed_distance_initialization(problem.target_u8)
        hm, wm = problem.ownership_shape
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        mrow0 = (box.bottom - query.bottom) // pixel
        mcol0 = (box.left - query.left) // pixel
        crop = initial_phi[mrow0:mrow0 + hm, mcol0:mcol0 + wm].copy()
        crop[::2, ::3] += np.float32(0.75)  # 打破 SDF 局部线性，差分值非平凡
        actual = macro_gradient_magnitude(problem, initial_phi, crop)
        # 参考实现：标量循环 + float64 halo（中心为快照，外围为 initial ring）
        halo = np.zeros((hm + 2, wm + 2), np.float64)
        halo[1:-1, 1:-1] = crop
        for i in range(hm + 2):
            for j in range(wm + 2):
                if not (1 <= i <= hm and 1 <= j <= wm):
                    halo[i, j] = initial_phi[mrow0 - 1 + i, mcol0 - 1 + j]
        expected = np.zeros((hm, wm), np.float64)
        for i in range(hm):
            for j in range(wm):
                dx = (halo[i + 1, j + 2] - halo[i + 1, j]) / 2.0
                dy = (halo[i + 2, j + 1] - halo[i, j + 1]) / 2.0
                expected[i, j] = (dx * dx + dy * dy) ** 0.5
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_edge_pixel_uses_query_context_not_replicate(self):
        """常数 phi + 非 0 initial ring：边缘系数非零（replicate 实现会全 0）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        initial_phi = signed_distance_initialization(problem.target_u8)
        hm, wm = problem.ownership_shape
        constant = np.full((hm, wm), 3.0, np.float32)
        magnitude = macro_gradient_magnitude(
            problem, initial_phi, constant)
        # 内部参数四周同为常数 → 系数恰 0；边缘行与 initial ring 不同 → 非零
        np.testing.assert_array_equal(magnitude[1:-1, 1:-1], 0.0)
        assert float(magnitude[0, :].max()) > 0.0
        assert float(magnitude[-1, :].max()) > 0.0

    def test_shape_mismatch_rejected(self):
        """query/ownership 形状不符在入口失败（规格 §12 契约错误）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        initial_phi = signed_distance_initialization(problem.target_u8)
        hm, wm = problem.ownership_shape
        with pytest.raises(ValueError, match="形状"):
            macro_gradient_magnitude(
                problem, initial_phi, np.zeros((hm + 1, wm), np.float32))

    def test_called_once_per_backward_state(self, monkeypatch):
        """调用数恰等于 iterations：末纯评价 state 不计算系数。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        calls = {"n": 0}
        real = levelset_module.macro_gradient_magnitude

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(
            levelset_module, "macro_gradient_magnitude", spy)
        optimize_levelset_macro(
            problem, _DoseModel(), _ls_config(iterations=2))
        assert calls["n"] == 2


class TestLevelSetBinarize:
    """external-gradient STE：前向 hard、反向 -mag·上游（TEST-004）。"""

    def test_forward_backward_exact(self):
        """forward == (phi<0)（phi==0 不透光）；backward == -mag·grad_output。"""
        phi = torch.tensor(
            [-2.0, -0.5, 0.0, 1.5, 3.0], requires_grad=True)
        magnitude = torch.tensor([1.0, 0.25, 0.5, 2.0, 0.0])
        out = _LevelSetBinarize.apply(phi, magnitude)
        torch.testing.assert_close(
            out, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
        upstream = torch.tensor([0.3, -1.0, 2.0, 0.5, -4.0])
        out.backward(upstream)
        torch.testing.assert_close(phi.grad, -magnitude * upstream)
        assert magnitude.grad is None  # 系数只读，不参与 autograd

    def test_backward_needs_no_spatial_difference(self):
        """一维张量可反传：内部无 pad/差分（旧实现 replicate 路径需 4-D）。"""
        phi = torch.tensor([1.0, -1.0], requires_grad=True)
        magnitude = torch.tensor([0.5, 2.0])
        out = _LevelSetBinarize.apply(phi, magnitude)
        out.sum().backward()
        torch.testing.assert_close(phi.grad, -magnitude)


class _AdamRecorder:
    """monkeypatch torch.optim.Adam 期间捕获每次 step 的宏梯度与参数。"""

    def __init__(self):
        self.real_cls = torch.optim.Adam  # patch 前捕获的真类（参考优化器用）
        self.grads = []  # 每次 step 前的 param.grad（float32 扁平副本）
        self.phis = []   # 每次 step 后的参数（float32 扁平副本）
        self.events = []  # 全局事件序（"f"=forward、"s"=step）

    def patch(self, monkeypatch):
        """把全局 Adam 替换为记录子类（测试结束由 monkeypatch 复原）。"""
        recorder = self

        class _RecordingAdam(recorder.real_cls):
            def step(self, closure=None):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            recorder.grads.append(
                                param.grad.detach().clone().reshape(-1).numpy())
                result = super().step(closure)
                for group in self.param_groups:
                    for param in group["params"]:
                        recorder.phis.append(
                            param.detach().clone().reshape(-1).numpy())
                recorder.events.append("s")
                return result

        monkeypatch.setattr(torch.optim, "Adam", _RecordingAdam)


def _float64_mirror(problem, config, model):
    """float64 单图镜像：无组批、独立 halo 差分、真实 Adam 更新。

    与被测实现完全不同的路径复算逐 state 损失/梯度/参数——raw-sum 与
    状态语义的独立 oracle（STE 复用生产 Function，其正确性另测；仅支持
    curvature_weight=0 的镜像，曲率正则由专测覆盖）。
    """
    initial = signed_distance_initialization(
        problem.target_u8).astype(np.float64)
    hm, wm = problem.ownership_shape
    pixel = problem.macro.pixel_dbu
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    mrow0 = (box.bottom - query.bottom) // pixel
    mcol0 = (box.left - query.left) // pixel
    ring = initial[mrow0 - 1:mrow0 + hm + 1, mcol0 - 1:mcol0 + wm + 1].copy()
    canvases = []
    for core_index in range(problem.macro.core_count):
        # 统一 [1,C,C]：假光刻的批维即 core 数，2-D 输入会被 avg_pool2d
        # 当作 [N,C,L] 逐行池化（静默错误语义）。
        target = (torch.from_numpy(
            problem.target_canvas(core_index)).to(torch.float64)
            / 255.0).unsqueeze(0)
        ownership = torch.from_numpy(
            problem.ownership_canvas(core_index)).unsqueeze(0)
        trainable = torch.from_numpy(
            problem.trainable_index_canvas(core_index)).unsqueeze(0)
        valid = torch.from_numpy(
            problem.context_valid_canvas(core_index)).unsqueeze(0)
        canvases.append((target, ownership, trainable, valid))
    conditions = (model.condition("nominal"), model.condition("dose_max"),
                  model.condition("defocus_min"))
    phi = torch.from_numpy(
        initial[mrow0:mrow0 + hm, mcol0:mcol0 + wm].copy())  # float64 参数
    optimizer = torch.optim.Adam([phi], lr=config.step_size)
    grads = []
    losses = []
    for state_index in range(config.iterations + 1):
        build = state_index < config.iterations
        if build:
            phi.requires_grad_(True)
            # 镜像独立重算 halo 中心差分（float64，与生产同定义不同代码）
            halo = ring.copy()
            halo[1:-1, 1:-1] = phi.detach().numpy()
            dx = (halo[1:-1, 2:] - halo[1:-1, :-2]) * 0.5
            dy = (halo[2:, 1:-1] - halo[:-2, 1:-1]) * 0.5
            magnitude = torch.from_numpy(
                np.sqrt(dx * dx + dy * dy).reshape(-1))
        total = torch.zeros((), dtype=torch.float64)
        for target, ownership, trainable, valid in canvases:
            flat = phi.reshape(-1)
            safe = trainable.reshape(-1).clamp_min(0)
            local = flat[safe]
            hard = _LevelSetBinarize.apply(local, magnitude[safe])
            context_hard = (target >= 0.5).to(torch.float64)
            context = torch.where(
                valid, context_hard, torch.zeros_like(context_hard))
            mask = torch.where(trainable >= 0, hard.view(target.shape),
                               context)
            printed = model.forward_many(mask, conditions)
            nominal_l2, process_l2, pvband_loss = owned_continuous_losses(
                printed["nominal"], printed["dose_max"],
                printed["defocus_min"], target, ownership)
            total = total + weighted_mirror(
                nominal_l2, process_l2, pvband_loss, config)
        losses.append(float(total.detach()))
        if not build:
            break
        gradient, = torch.autograd.grad(total, phi)
        grads.append(gradient.reshape(-1).numpy().copy())
        phi.grad = gradient.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        phi.requires_grad_(False)
    return {"grads": grads, "losses": losses, "phi": phi.numpy()}


def weighted_mirror(nominal_l2, process_l2, pvband_loss, config):
    """镜像内的加权总损失（curvature=0 场景）。"""
    return (nominal_l2 + config.weight_process_l2 * process_l2
            + config.weight_pvband * pvband_loss)


class TestCrossCoreIdentity:
    """同一参数跨 core 的 phi/系数同一性（TEST-003）。"""

    def test_same_state_mask_agrees_across_cores(self):
        """同一 state 内，各 core 画布中对应同一物理像素的 mask 逐位相等。

        trainable 位由同一 macro_phi 决定、context 位由同一 hard target
        决定——按 core 复制参数或 context 的实现会在重叠带失配。
        """
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        model = _MaskCaptureModel()
        optimize_levelset_macro(
            problem, model,
            _ls_config(iterations=1, batch_size=problem.macro.core_count))
        batch = model.captured[0].numpy()  # state0 单批全 core
        windows = []
        for core_index in range(problem.macro.core_count):
            _, r0, r1, c0, c1 = problem._context_window(core_index)
            low_y, _, low_x, _ = _center_padding(r1 - r0, c1 - c0, 256)
            windows.append((r0, r1, c0, c1, low_y, low_x))
        checked = 0
        for a in range(problem.macro.core_count):
            r0a, r1a, c0a, c1a, lya, lxa = windows[a]
            for b in range(a + 1, problem.macro.core_count):
                r0b, r1b, c0b, c1b, lyb, lxb = windows[b]
                rr0, rr1 = max(r0a, r0b), min(r1a, r1b)  # query 栅格交集
                cc0, cc1 = max(c0a, c0b), min(c1a, c1b)
                if rr0 >= rr1 or cc0 >= cc1:
                    continue
                va = batch[a][lya + rr0 - r0a:lya + rr1 - r0a,
                              lxa + cc0 - c0a:lxa + cc1 - c0a]
                vb = batch[b][lyb + rr0 - r0b:lyb + rr1 - r0b,
                              lxb + cc0 - c0b:lxb + cc1 - c0b]
                assert np.array_equal(va, vb)
                checked += va.size
        assert checked > 0  # 判别性：core 窗口确有重叠（context 覆盖邻核）


class TestGradientSumAndAdam:
    """跨 core raw sum、batch 不变性、宏屏障与 Adam 契约（TEST-005/006）。"""

    def test_macro_gradient_is_raw_sum(self, monkeypatch):
        """solver 捕获梯度 == float64 单图镜像（求和而非平均）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        model = _LocalAverageModel()  # 3×3 耦合：邻核 loss 依赖本核参数
        # 步长 0.2：SDF 初值 |phi|>=1，两步内 |Δ|<=0.2×2 不会跨越 0 等值
        # 线——更大步长会让个别像素恰落在符号翻转阈值上，f32/f64 的
        # Adam 舍入差异被 mask 阈值放大成损失跳变（非语义属性）。
        config = _ls_config(iterations=2, batch_size=2, step_size=0.2)
        # 镜像必须在 patch 前运行：patch 是全局 Adam 替换，镜像的 step
        # 也会被记录器捕获（否则 grads 数量翻倍）。
        mirror = _float64_mirror(problem, config, model)
        recorder = _AdamRecorder()
        recorder.patch(monkeypatch)
        result = optimize_levelset_macro(problem, model, config)
        assert len(recorder.grads) == config.iterations == len(mirror["grads"])
        for solver_grad, mirror_grad in zip(recorder.grads, mirror["grads"]):
            # 平均化实现会把梯度整体缩小 → 此处失配
            np.testing.assert_allclose(
                solver_grad, mirror_grad.astype(np.float32),
                rtol=1e-5, atol=1e-5)
        for record, mirror_loss in zip(result.records, mirror["losses"]):
            assert record.total_loss == pytest.approx(
                mirror_loss, rel=1e-5)
        assert float(np.abs(mirror["grads"][0]).max()) > 0  # 判别性

    def test_batch_size_invariance(self, monkeypatch):
        """batch_size=1 与全 core 单批的逐 state 梯度一致（仅累加误差）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        model = _LocalAverageModel()
        recorder = _AdamRecorder()
        recorder.patch(monkeypatch)  # 单记录器：二次 patch 会叠加记录链
        optimize_levelset_macro(
            problem, model, _ls_config(iterations=2, batch_size=1))
        solo = list(recorder.grads)  # 第 1 次运行的 2 个状态梯度快照
        optimize_levelset_macro(
            problem, model,
            _ls_config(iterations=2, batch_size=problem.macro.core_count))
        batched = recorder.grads[len(solo):]  # 第 2 次运行新增的梯度
        assert len(solo) == len(batched) == 2
        for grad_a, grad_b in zip(solo, batched):
            np.testing.assert_allclose(grad_a, grad_b, rtol=1e-6, atol=1e-7)

    def test_adam_matches_torch_reference_exactly(self, monkeypatch):
        """捕获梯度喂独立 torch Adam：逐位复现 solver 最终 phi。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        config = _ls_config(iterations=2, batch_size=2)
        recorder = _AdamRecorder()
        recorder.patch(monkeypatch)
        optimize_levelset_macro(problem, _DoseModel(), config)
        gradients = list(recorder.grads)  # 快照：参考优化器不得再被记录
        initial = signed_distance_initialization(problem.target_u8)
        reference = torch.from_numpy(_ownership_crop(problem, initial).copy())
        optimizer = recorder.real_cls(
            [reference], lr=config.step_size, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False)
        for gradient in gradients:
            reference.grad = torch.from_numpy(
                gradient.copy()).reshape(reference.shape)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        np.testing.assert_array_equal(
            reference.detach().numpy().reshape(-1), recorder.phis[-1])

    def test_step_only_after_all_cores_of_state(self, monkeypatch):
        """屏障：每 state 全部 forward 完成后才恰一次 step。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        recorder = _AdamRecorder()
        recorder.patch(monkeypatch)
        model = _DoseModel()
        real_forward = model.forward_many

        def logged_forward(mask, conditions):
            """记录 forward 事件后透传。"""
            recorder.events.append("f")
            return real_forward(mask, conditions)

        model.forward_many = logged_forward
        optimize_levelset_macro(
            problem, model, _ls_config(iterations=2, batch_size=1))
        # 4 core：每 backward state 4 次 forward 后 1 次 step，末纯评价
        # state 再 4 次 forward 且不 step；batch 内提前 step 的实现会失配
        assert recorder.events == (["f"] * 4 + ["s"]) * 2 + ["f"] * 4

    def test_two_updates_three_evaluated_states(self):
        """N 次 Adam 更新 + N+1 个完整已评价状态（REQ-009）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        result = optimize_levelset_macro(
            problem, _DoseModel(), _ls_config(iterations=2))
        assert [record.state_index for record in result.records] == [0, 1, 2]
        assert result.best_state_index in (0, 1, 2)


class TestContextAndPadding:
    """context 宽度前置与三值画布语义（TEST-007）。"""

    def test_context_below_pixel_rejected(self):
        """context < 1 像素无条件拒绝（与 curvature_weight 无关）。"""
        problem = _problem(
            kdb.Region(kdb.Box(8, 8, 40, 40)), macro=_macro(context_dbu=0))
        with pytest.raises(ValueError, match="context"):
            optimize_levelset_macro(problem, _DoseModel(), _ls_config())

    def test_hard_context_and_zero_padding(self):
        """真实 context 取 hard target；数值 padding 严格 0。"""
        # context=4（1 像素）→ 12px window << 256，padding 占绝大多数
        problem = _problem(
            kdb.Region(kdb.Box(8, 8, 40, 40)), macro=_macro(context_dbu=4))
        model = _MaskCaptureModel()
        optimize_levelset_macro(
            problem, model,
            _ls_config(iterations=1, batch_size=problem.macro.core_count))
        for core_index in range(problem.macro.core_count):
            canvas = model.captured[0][core_index].numpy()
            valid = problem.context_valid_canvas(core_index)
            trainable = problem.trainable_index_canvas(core_index)
            target = (problem.target_canvas(core_index).astype(
                np.float64) / 255.0)
            assert float(np.abs(canvas[~valid]).max()) == 0.0  # padding 恒 0
            outside = valid & (trainable < 0)  # macro 外真实 context
            assert outside.any()
            assert np.array_equal(
                canvas[outside], (target >= 0.5)[outside])

    def test_canvas_mismatch_rejected(self):
        """模型画布与 problem 不一致在入口失败。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 40, 48)))
        model = _DoseModel()
        model.config = _StubConfig(canvas=128)
        with pytest.raises(ValueError, match="画布"):
            optimize_levelset_macro(problem, model, _ls_config())

    def test_nonfinite_loss_raises(self):
        """NaN 前向在 state 汇总处终止（FloatingPointError）。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        with pytest.raises(FloatingPointError):
            optimize_levelset_macro(problem, _NaNModel(), _ls_config())


class TestLossesCurvatureAndRealModel:
    """共享损失、hard 曲率与真实模型（TEST-008）。"""

    def test_state0_losses_hand_computed_identity(self):
        """恒等模型：state0 hard ≡ target 二值 → 三损失可 numpy 复算。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))  # 含分数覆盖格
        result = optimize_levelset_macro(
            problem, _IdentityModel(),
            _ls_config(weight_process_l2=1.0, weight_pvband=0.5))
        expected_nom = 0.0
        for core_index in range(problem.macro.core_count):
            target = (problem.target_canvas(core_index).astype(
                np.float64) / 255.0)
            own = problem.ownership_canvas(core_index)
            valid = problem.context_valid_canvas(core_index)
            # INV-004：phi<0 ⟺ T>=0.5，state0 全画布 hard 即 target 二值
            hard = np.where(valid, target >= 0.5, 0.0)
            expected_nom += float(((hard - target) ** 2 * own).sum())
        record = result.records[0]
        assert record.nominal_l2 == pytest.approx(expected_nom, rel=1e-5)
        assert record.process_l2 == pytest.approx(
            2.0 * expected_nom, rel=1e-5)  # 恒等三条件同值
        assert record.pvband_loss == pytest.approx(0.0, abs=1e-6)
        assert record.total_loss == pytest.approx(
            3.0 * expected_nom, rel=1e-5)

    def test_curvature_on_hard_mask_and_zero_weight_skips(self,
                                                          monkeypatch):
        """weight=0 不构建卷积；weight>0 曲率作用于 hard mask 可复算。"""
        calls = {"n": 0}
        # 曲率调用已上提到公共骨架（P3），spy 宿主随之迁移
        import opc.iteration.ilt._skeleton as ilt_skeleton
        real = ilt_skeleton.curvature_loss

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(ilt_skeleton, "curvature_loss", spy)
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        optimize_levelset_macro(problem, _IdentityModel(), _ls_config())
        assert calls["n"] == 0  # weight=0：一次都不调用
        config = _ls_config(curvature_weight=2.0)
        result = optimize_levelset_macro(problem, _IdentityModel(), config)
        # 曲率每状态每批恰一次：状态数 × 批数（含末纯评价状态）
        batches = -(-problem.macro.core_count // config.batch_size)
        assert calls["n"] == (config.iterations + 1) * batches
        kernel = np.array([[-1, 5, -1], [5, -16, 5], [-1, 5, -1]],
                          np.float64) / 16.0
        expected = 0.0
        for core_index in range(problem.macro.core_count):
            target = (problem.target_canvas(core_index).astype(
                np.float64) / 255.0)
            own = problem.ownership_canvas(core_index)
            valid = problem.context_valid_canvas(core_index)
            hard = np.where(valid, target >= 0.5, 0.0)
            # 与实现同式：3×3 valid 卷积（输出小 2）∩ ownership 内圈
            convo = np.zeros((254, 254))
            for dy in range(3):
                for dx in range(3):
                    convo += (kernel[dy, dx]
                              * hard[dy:dy + 254, dx:dx + 254])
            expected += float((convo ** 2 * own[1:-1, 1:-1]).sum())
        assert result.records[0].curvature_loss == pytest.approx(
            expected, rel=1e-4, abs=1e-6)

    def test_real_cpu_update_finite(self, cpu_model):
        """真实 ICCAD13 CPU 一轮更新：全有限、N+1 状态、INV-004 成立。"""
        problem = _problem(
            kdb.Region(kdb.Box(21, 20, 61, 60)),
            macro=_macro(core_size_dbu=80))
        result = optimize_levelset_macro(
            problem, cpu_model, _ls_config(iterations=1, batch_size=1))
        assert len(result.records) == 2
        for record in result.records:
            assert np.isfinite(record.total_loss)
        assert np.all(np.isfinite(result.best_parameters))
        assert result.binary_mask.dtype == np.bool_
        # INV-004：二值掩膜恰为 best phi<0
        assert np.array_equal(
            result.binary_mask, result.best_parameters < 0.0)
        # state0 二值与 target 二值逐格一致（SDF 符号契约）
        initial = signed_distance_initialization(problem.target_u8)
        assert np.array_equal(
            _ownership_crop(problem, initial) < 0,
            _ownership_crop(problem, problem.target_u8).astype(
                np.float32) / 255.0 >= 0.5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_real_cuda_matches_cpu(self, cpu_model):
        """CUDA 与 CPU 同输入 loss 一致（1e-4 容差）。"""
        problem = _problem(
            kdb.Region(kdb.Box(21, 20, 61, 60)),
            macro=_macro(core_size_dbu=80))
        cuda_model = ICCAD13Lithography(device="cuda")
        config = _ls_config(iterations=1, batch_size=1)
        cpu_result = optimize_levelset_macro(problem, cpu_model, config)
        cuda_result = optimize_levelset_macro(problem, cuda_model, config)
        for cpu_record, cuda_record in zip(cpu_result.records,
                                           cuda_result.records):
            assert cuda_record.total_loss == pytest.approx(
                cpu_record.total_loss, rel=1e-4, abs=1e-6)


class TestFinalContextHelper:
    """LevelSet 终评 fixed-context helper（TEST-009 的模块部分）。"""

    def test_hard_context_without_sdf(self, monkeypatch):
        """真实 context 取 hard target、padding 0，且不运行 SDF。"""
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        calls = {"n": 0}
        real = levelset_module.signed_distance_initialization

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(
            levelset_module, "signed_distance_initialization", spy)
        canvas = build_levelset_final_context_canvas(
            problem, 0, _ls_config())
        valid = problem.context_valid_canvas(0)
        target = problem.target_canvas(0).astype(np.float32) / 255.0
        expected = np.where(valid, (target >= 0.5).astype(np.float32),
                            np.float32(0.0)).astype(np.float32)
        assert canvas.dtype == np.float32
        np.testing.assert_array_equal(canvas, expected)
        assert calls["n"] == 0
