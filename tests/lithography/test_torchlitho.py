"""TorchLitho 物理光刻模型单元测试（批 A：核心契约、源形状、前向与梯度）。"""

import pytest
import torch

from lithography import ICCAD13Lithography
from lithography.contracts import LithographyModel
from lithography.torchlitho import TorchLithoCondition, TorchLithoConfig, TorchLithoLithography
from lithography.torchlitho.source import frequency_grid

# 统一小画布参数：64 网格 × 8nm 像素（视场 512nm），与原库 example/rect.py 同参数，
# TCC 直通路径（size=64）构造约 1 秒。
CANVAS, PIXEL_NM = 64, 8.0
NA, WAVELENGTH = 1.35, 193.0


def _make_model(**config_kwargs) -> TorchLithoLithography:
    """按默认参数构造 CPU 模型，供各类测试复用。"""
    return TorchLithoLithography(TorchLithoConfig(**config_kwargs), CANVAS, PIXEL_NM, device="cpu")


def _rect_mask() -> torch.Tensor:
    """构造与原库 rect.py 同款的 32×32 居中矩形掩模。"""
    mask = torch.zeros(CANVAS, CANVAS)
    mask[16:48, 16:48] = 1.0
    return mask


class TestTorchLithoCondition:
    """条件令牌的三命名映射与非法输入拒绝。"""

    def test_named_conditions_map_defocus_and_dose(self):
        """nominal/dose_max 共享 defocus=0，defocus_min 取配置值并配最小剂量。"""
        model = _make_model()
        nominal = model.condition("nominal")
        dose_max = model.condition("dose_max")
        defocus_min = model.condition("defocus_min")
        assert nominal.defocus_nm == 0.0 and nominal.dose == pytest.approx(1.0)
        assert dose_max.defocus_nm == 0.0 and dose_max.dose == pytest.approx(1.02)
        assert defocus_min.defocus_nm == pytest.approx(40.0) and defocus_min.dose == pytest.approx(0.98)

    def test_unknown_name_raises(self):
        """未知名必须在构造点失败，不允许静默默认。"""
        with pytest.raises(ValueError, match="未知默认工艺条件"):
            _make_model().condition("nominal_max")

    def test_invalid_fields_rejected(self):
        """空名称、非有限 defocus 与非正剂量都拒绝。"""
        with pytest.raises(ValueError, match="名称"):
            TorchLithoCondition("", 0.0, 1.0)
        with pytest.raises(ValueError, match="defocus"):
            TorchLithoCondition("x", float("nan"), 1.0)
        with pytest.raises(ValueError, match="剂量"):
            TorchLithoCondition("x", 0.0, 0.0)

    def test_condition_is_frozen(self):
        """冻结 dataclass 不可写字段。"""
        condition = TorchLithoCondition("x", 0.0, 1.0)
        with pytest.raises(AttributeError):
            condition.dose = 2.0


class TestTorchLithoConfigValidation:
    """配置的物理与剂量契约校验。"""

    def test_defaults(self):
        """默认配置等于原库典型参数与 ICCAD13 胶模型参数。"""
        config = TorchLithoConfig()
        assert config.method == "abbe"
        assert config.source_shape == "point"
        assert config.sigma == pytest.approx(0.05)
        assert config.print_steepness == pytest.approx(50.0)
        assert config.target_density == pytest.approx(0.225)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"method": "both"}, "method"),
            ({"source_shape": "annular"}, "source_shape"),
            ({"sigma": 0.0}, "sigma"),
            ({"sigma": 1.5}, "sigma"),
            ({"source_shape": "dipole"}, "pole_center"),
            ({"na": 1.5}, "NA"),
            ({"dose_max": 0.9}, "剂量"),
            ({"print_threshold": 1.0}, "print_threshold"),
        ],
    )
    def test_invalid_configs_rejected(self, kwargs, message):
        """非法枚举、区间与依赖关系都在构造期失败。"""
        with pytest.raises(ValueError, match=message):
            TorchLithoConfig(**kwargs)


class TestPaddingParity:
    """居中 padding 与 ICCAD13 逐位一致（同一几何同一 canvas 布局）。"""

    @pytest.mark.parametrize("height,width", [(256, 256), (255, 255), (192, 128), (17, 100)])
    def test_prepare_mask_matches_iccad13(self, height, width):
        """四种奇偶尺寸下 padded 张量与裁剪元组逐位一致（两模型同为 256 画布）。"""
        iccad13 = ICCAD13Lithography(device="cpu")
        torchlitho = TorchLithoLithography(TorchLithoConfig(), 256, PIXEL_NM, device="cpu")
        for generator in (torch.zeros, torch.ones, torch.rand):
            if generator is torch.rand:
                torch.manual_seed(7)
            mask = generator(height, width)
            prepared_a, pad_a, single_a = iccad13._prepare_mask(mask)
            prepared_b, pad_b, single_b = torchlitho._prepare_mask(mask)
            assert torch.equal(prepared_a, prepared_b)
            assert pad_a == pad_b
            assert single_a == single_b

    def test_oversized_mask_rejected(self):
        """超过 canvas 的输入在频域分配前拒绝。"""
        with pytest.raises(ValueError, match="超过 canvas"):
            _make_model()._prepare_mask(torch.zeros(CANVAS + 1, CANVAS))


class TestContractCompliance:
    """LithographyModel 协议的结构化满足。"""

    def test_runtime_protocol_check(self):
        """isinstance 结构检查通过（方法与属性齐全）。"""
        assert isinstance(_make_model(), LithographyModel)

    def test_config_view(self):
        """config 视图暴露 canvas 与 print_threshold。"""
        model = _make_model()
        assert model.config.canvas == CANVAS
        assert model.config.print_threshold == pytest.approx(0.5)

    def test_device_property(self):
        """device 属性跟随 buffer。"""
        assert _make_model().device == torch.device("cpu")


class TestSigmoidExit:
    """forward_many 出口的胶模型语义与剂量平方缩放。"""

    def _model_with_fixed_aerial(self, value: float) -> TorchLithoLithography:
        """用固定 aerial 替换两方法成像，隔离 sigmoid 出口验证。"""
        model = _make_model()
        fixed = value * torch.ones(1, CANVAS, CANVAS)

        def fake_aerial(prepared, defocus_nm):
            """返回常数 aerial 供出口断言。"""
            return fixed.expand(prepared.shape[0], CANVAS, CANVAS)

        model._abbe_aerial = fake_aerial
        model._hopkins_aerial = fake_aerial
        return model

    def test_printed_matches_closed_form(self):
        """printed 逐点等于 sigmoid(steepness·(I·dose²−target))。"""
        model = self._model_with_fixed_aerial(0.0045)
        result = model.forward_many(
            torch.ones(CANVAS, CANVAS), (TorchLithoCondition("a", 0.0, 1.0), TorchLithoCondition("b", 0.0, 2.0))
        )
        for name, dose in (("a", 1.0), ("b", 2.0)):
            expected = torch.full(
                (CANVAS, CANVAS), torch.sigmoid(torch.tensor(50.0 * (0.0045 * dose**2 - 0.225))).item()
            )
            assert torch.allclose(result[name], expected)

    def test_crop_restores_input_shape(self):
        """非满幅输入的输出恢复输入 H×W，单张输入压回 [H,W]。"""
        model = self._model_with_fixed_aerial(0.0045)
        printed = model(torch.ones(48, 32), TorchLithoCondition("a", 0.0, 1.0))
        assert printed.shape == (48, 32)


class TestAbbeCoherentAnalytic:
    """点源无离焦下 Abbe 前向与独立手写相干成像公式逐点一致。"""

    @pytest.mark.parametrize("defocus_nm", [0.0, 40.0])
    def test_matches_manual_formula(self, defocus_nm):
        """手写「居中谱 × 瞳 → ifftshift → ifft2 → 模平方」对照模型输出。"""
        model = _make_model()
        mask = _rect_mask()
        prepared = model._prepare_mask(mask)[0]
        # 手写链不复用模型内部方法：坐标、瞳、FFT 全部在测试内展开。
        _, _, freq = frequency_grid(CANVAS, PIXEL_NM, torch.device("cpu"))
        limit = NA / WAVELENGTH
        pupil = (freq < limit).to(torch.complex64)
        if defocus_nm != 0.0:
            squared = 1.44**2 - (WAVELENGTH**2) * freq * freq
            opd = defocus_nm * (1.44 - torch.sqrt(squared.clamp(min=0.0)))
            pupil = pupil * torch.exp(1j * (2 * torch.pi / WAVELENGTH) * opd).to(torch.complex64)
        spectrum = torch.fft.fftshift(torch.fft.fft2(prepared.to(torch.complex64)), dim=(-2, -1))
        field = torch.fft.ifft2(torch.fft.ifftshift(spectrum * pupil, dim=(-2, -1)), dim=(-2, -1))
        reference = field.real.square() + field.imag.square()
        aerial = model._abbe_aerial(prepared, defocus_nm)
        assert torch.allclose(aerial, reference, rtol=1e-5, atol=1e-8)


class TestPupilShiftRegression:
    """R2 修正回归：瞳必须按源点向量平移（原库为范数标量同心放大）。"""

    def _model_with_points(self, points) -> TorchLithoLithography:
        """把源点集合替换为指定坐标（仅测试用）。"""
        model = _make_model()
        model.source_xy = torch.tensor(points, dtype=torch.float32)
        return model

    def test_single_offaxis_source_matches_shifted_pupil(self):
        """单离轴源点 (c,0) 的成像与手写平移瞳公式逐点一致。"""
        offset = 2.0e-3  # cycles/nm，瞳半径 6.99e-3 内的明显离轴
        model = self._model_with_points([(offset, 0.0)])
        mask = _rect_mask()
        prepared = model._prepare_mask(mask)[0]
        fx, fy, _ = frequency_grid(CANVAS, PIXEL_NM, torch.device("cpu"))
        dist = torch.sqrt((fx - offset) ** 2 + fy * fy)
        pupil = (dist < NA / WAVELENGTH).to(torch.complex64)
        spectrum = torch.fft.fftshift(torch.fft.fft2(prepared.to(torch.complex64)), dim=(-2, -1))
        field = torch.fft.ifft2(torch.fft.ifftshift(spectrum * pupil, dim=(-2, -1)), dim=(-2, -1))
        reference = field.real.square() + field.imag.square()
        assert torch.allclose(model._abbe_aerial(prepared, 0.0), reference, rtol=1e-5, atol=1e-8)

    def test_symmetric_pair_gives_mirrored_aerial(self):
        """对称双源点 (±c,0) 的 aerial 必须左右镜像对称（同心放大实现同样对称，
        该断言防退化；方向区分由单离轴用例保证）。"""
        offset = 2.0e-3
        model = self._model_with_points([(offset, 0.0), (-offset, 0.0)])
        prepared = model._prepare_mask(_rect_mask())[0]
        aerial = model._abbe_aerial(prepared, 0.0)
        assert torch.allclose(aerial, aerial.flip(-1), rtol=1e-5, atol=1e-8)


class TestSourceShapes:
    """四种源形状的离散格点语义。"""

    def test_point_source_is_dc_only(self):
        """点源恰为 DC 单点，掩膜仅中心格。"""
        model = _make_model(source_shape="point")
        assert model.source_xy.shape == (1, 2)
        assert torch.equal(model.source_xy, torch.zeros(1, 2))

    def test_disk_point_count_matches_mask(self):
        """disk 格点数等于掩膜非零数，且坐标与掩膜格点一致。"""
        model = _make_model(source_shape="disk", sigma=0.3)
        _, _, freq = frequency_grid(CANVAS, PIXEL_NM, torch.device("cpu"))
        expected = (freq <= 0.3 * NA / WAVELENGTH).sum().item()
        assert model.source_xy.shape[0] == expected
        assert expected > 1  # 512nm 视场下 σ=0.3 采样出多个源点

    def test_dipole_and_quadrupole_partition_symmetrically(self):
        """dipole/quadrupole 的源点集合关于原点中心对称（成对出现）。"""
        for shape in ("dipole", "quadrupole"):
            model = _make_model(source_shape=shape, sigma=0.2, pole_center=0.6)
            points = model.source_xy
            flipped = -points
            # 每个源点的镜像也必须在集合内（允许顺序不同）。
            matched = {tuple(p.tolist()) for p in points} == {tuple(p.tolist()) for p in flipped}
            assert matched, shape
            assert points.shape[0] >= 2, shape

    def test_empty_pole_union_rejected(self):
        """极盘完全落出频率网格时构造期失败。"""
        with pytest.raises(ValueError, match="没有任何格点"):
            _make_model(source_shape="dipole", sigma=0.01, pole_center=0.95)


class TestForwardManyBatching:
    """多条件批量语义与 defocus 去重。"""

    def test_three_conditions_batch(self):
        """三命名条件一次前向、键为条件名、单张与批量输入一致。"""
        for method in ("abbe", "hopkins"):
            model = _make_model(method=method)
            conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
            results = model.forward_many(_rect_mask(), conditions)
            assert set(results) == {"nominal", "dose_max", "defocus_min"}
            single = model.forward_many(torch.stack([_rect_mask(), _rect_mask()]), conditions)
            assert torch.allclose(single["nominal"][0], results["nominal"], rtol=1e-6)
            assert torch.allclose(single["nominal"][1], results["nominal"], rtol=1e-6)

    def test_duplicate_names_rejected(self):
        """同批条件重名拒绝。"""
        model = _make_model()
        condition = TorchLithoCondition("same", 0.0, 1.0)
        with pytest.raises(ValueError, match="不能重复"):
            model.forward_many(_rect_mask(), (condition, condition))

    def test_tcc_built_once_per_defocus(self, monkeypatch):
        """Hopkins 的 TCC 按不同 defocus 值恰构造一次并缓存复用。"""
        import lithography.torchlitho.model as model_module

        calls = []
        original = model_module.build_tcc_kernels

        def counting(*args, **kwargs):
            """记录每次 TCC 构造的 defocus 实参。"""
            calls.append(args[-1])
            return original(*args, **kwargs)

        monkeypatch.setattr(model_module, "build_tcc_kernels", counting)
        model = _make_model(method="hopkins")
        conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
        model.forward_many(_rect_mask(), conditions)
        model.forward_many(_rect_mask(), conditions)
        assert sorted(calls) == [0.0, 40.0]  # 三条件两 defocus、两次前向不重建


class TestAutograd:
    """两方法的 autograd 梯度正确性。"""

    def test_abbe_gradient_matches_manual_chain(self):
        """模型梯度与手写相干链的 autograd 梯度一致（同精度同容差）。"""
        model = _make_model()
        mask = _rect_mask().requires_grad_(True)
        printed = model(mask, TorchLithoCondition("nominal", 0.0, 1.0))
        printed.sum().backward()
        # 手写链（TestAbbeCoherentAnalytic 同式）重算梯度。
        _, _, freq = frequency_grid(CANVAS, PIXEL_NM, torch.device("cpu"))
        pupil = (freq < NA / WAVELENGTH).to(torch.complex64)
        manual = _rect_mask().requires_grad_(True)
        spectrum = torch.fft.fftshift(torch.fft.fft2(manual.to(torch.complex64)), dim=(-2, -1))
        field = torch.fft.ifft2(torch.fft.ifftshift(spectrum * pupil, dim=(-2, -1)), dim=(-2, -1))
        aerial = field.real.square() + field.imag.square()
        torch.sigmoid(50.0 * (aerial - 0.225)).sum().backward()
        assert torch.allclose(mask.grad, manual.grad, rtol=1e-4, atol=1e-7)

    def test_hopkins_point_source_gradient_matches_abbe(self):
        """点源下两方法梯度一致（rank-1 等价性的梯度面印证）。"""
        mask_a = _rect_mask().requires_grad_(True)
        mask_h = _rect_mask().requires_grad_(True)
        condition = TorchLithoCondition("x", 0.0, 1.0)
        _make_model(method="abbe")(mask_a, condition).sum().backward()
        _make_model(method="hopkins")(mask_h, condition).sum().backward()
        rel = (mask_a.grad - mask_h.grad).abs().max() / mask_a.grad.abs().max()
        assert rel < 1e-4
