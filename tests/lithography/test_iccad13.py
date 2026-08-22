"""ICCAD13 配置解析、Hopkins 资产契约、前向数值与 backward 的生成式测试。"""

import hashlib
import json
import subprocess
import sys
import time

import pytest
import torch

from lithography import ICCAD13Config, ICCAD13Lithography, ProcessCondition

# 四个资产的 SHA-256 是模型身份（迁移设计文档 §3.2）：哈希漂移说明资产被替换，
# 旧数值参考全部失效，因此作为硬断言而不是软警告。
_ASSET_SHA256 = {
    "focus.pt": "204bee39d8225c6d3cda52ea2d13b7c6f6cf4e4244de2ce960576d1bc741438f",
    "defocus.pt": "df624de9e17485d819e488ccada7edff133690cfa01370bfabec8f9e7cb8d532",
    "focus_scale.pt": "4e6f6136d419bdf0b56e9b461471c72d97ab3ba582fa19fe65ffaf25d188dab6",
    "defocus_scale.pt": "4ce70debf23593594c2fda1bd0cadf427abb0c132b16a047f6589f51148c8dc8",
}


def _config_lines(**overrides: str) -> list[str]:
    """按标准九字段生成“字段 值”配置行，允许按字段名覆盖值或删除字段。"""
    values = {  # 与 lithography/config/iccad13.txt 逐字段一致的合法基准
        "KernelNum": "24", "TargetDensity": "0.225", "PrintThresh": "0.5",
        "PrintSteepness": "50.0", "DoseMax": "1.02", "DoseMin": "0.98",
        "DoseNom": "1.00", "Canvas": "256", "Resolution": "256",
    }
    for key in list(overrides):  # None 表示从配置中删除该字段
        if overrides[key] is None:
            del values[key]
            del overrides[key]
    values.update(overrides)  # 其余覆盖按字符串写入
    return [f"{name} {value}" for name, value in values.items()]


def _write_config(tmp_path, lines: list[str]):
    """把配置行写入临时文件并返回路径。"""
    path = tmp_path / "iccad13.txt"  # 临时配置路径
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写盘
    return path  # 返回路径


def _model_mask(height: int = 200, width: int = 150) -> torch.Tensor:
    """构造同时包含实心矩形和孔洞的确定性 batch mask（旧测试移植）。"""
    mask = torch.zeros((2, height, width), dtype=torch.float32)  # 两张同尺寸
    mask[0, 40:160, 40:110] = 1.0  # 第一张：居中实心矩形
    mask[1, 20:180, 20:130] = 1.0  # 第二张：更大的外框
    mask[1, 70:130, 60:90] = 0.0  # 第二张：中心孔洞
    return mask


def _default_conditions(model: ICCAD13Lithography) -> list[ProcessCondition]:
    """返回三个默认工艺条件（nominal / dose_max / defocus_min）。"""
    return [model.condition(name) for name in
            ("nominal", "dose_max", "defocus_min")]


class TestConfigParsing:
    """ICCAD13Config.from_file 的解析与校验（设计文档 §11.1）。"""

    def test_standard_config_parses(self, tmp_path):
        """随包分发的标准配置解析为 24 核、256 画布、256 分辨率。"""
        config = ICCAD13Config.from_file(_write_config(tmp_path, _config_lines()))
        assert config.kernel_count == 24  # 全部 24 个核
        assert config.canvas == 256  # 冻结画布
        assert config.resolution == 256  # 冻结分辨率
        assert config.target_density == pytest.approx(0.225)  # 胶阈值
        assert config.print_threshold == pytest.approx(0.5)  # 二值化阈值
        assert config.print_steepness == pytest.approx(50.0)  # sigmoid 陡度
        assert config.dose_max == pytest.approx(1.02)  # 剂量上界
        assert config.dose_min == pytest.approx(0.98)  # 剂量下界
        assert config.dose_nominal == pytest.approx(1.00)  # 标称剂量

    def test_missing_field_fails(self, tmp_path):
        """缺少任一必需字段时解析失败并列出字段名。"""
        path = _write_config(tmp_path, _config_lines(Resolution=None))
        with pytest.raises(ValueError, match="缺少字段.*Resolution"):
            ICCAD13Config.from_file(path)

    def test_malformed_line_fails(self, tmp_path):
        """行内出现三段（非“名称 值”结构）时失败。"""
        lines = _config_lines() + ["DoseNom 1.00 extra"]  # 三段注入
        with pytest.raises(ValueError, match="名称和值"):
            ICCAD13Config.from_file(_write_config(tmp_path, lines))

    @pytest.mark.parametrize("field", ["KernelNum", "TargetDensity"])
    def test_non_numeric_value_fails(self, tmp_path, field):
        """字段值无法转换为数字时失败并报字段名。"""
        path = _write_config(tmp_path, _config_lines(**{field: "abc"}))
        with pytest.raises(ValueError, match=field):
            ICCAD13Config.from_file(path)

    def test_unknown_field_fails(self, tmp_path):
        """出现九字段之外的未知字段时解析失败。"""
        lines = _config_lines() + ["DoseTypo 1.0"]  # 注入拼错的字段
        with pytest.raises(ValueError, match="未知字段.*DoseTypo"):
            ICCAD13Config.from_file(_write_config(tmp_path, lines))

    def test_duplicate_field_fails(self, tmp_path):
        """同一字段重复出现时失败，不做静默覆盖。"""
        lines = _config_lines() + ["KernelNum 12"]  # 二次出现
        with pytest.raises(ValueError, match="重复出现.*KernelNum"):
            ICCAD13Config.from_file(_write_config(tmp_path, lines))

    @pytest.mark.parametrize("value", ["0", "-3", "25", "48"])
    def test_kernel_count_out_of_range_fails(self, tmp_path, value):
        """核数量为 0、负数或超过资产上限 24 时失败。"""
        path = _write_config(tmp_path, _config_lines(KernelNum=value))
        with pytest.raises(ValueError, match="KernelNum"):
            ICCAD13Config.from_file(path)

    @pytest.mark.parametrize("field", ["Canvas", "Resolution"])
    @pytest.mark.parametrize("value", ["128", "512"])
    def test_frozen_canvas_fails(self, tmp_path, field, value):
        """画布或分辨率不是 256 时失败（尺寸冻结，无 resize 分支）。"""
        path = _write_config(tmp_path, _config_lines(**{field: value}))
        with pytest.raises(ValueError, match="固定为 256"):
            ICCAD13Config.from_file(path)

    @pytest.mark.parametrize("field", ["TargetDensity", "PrintThresh",
                                       "PrintSteepness", "DoseMax"])
    def test_nonfinite_parameter_fails(self, tmp_path, field):
        """浮点字段取 nan 或 inf 时失败。"""
        for bad in ("nan", "inf", "-inf"):  # 三种非有限形态
            path = _write_config(tmp_path, _config_lines(**{field: bad}))
            with pytest.raises(ValueError, match=f"{field} 必须是有限数"):
                ICCAD13Config.from_file(path)

    @pytest.mark.parametrize(
        "overrides",
        [{"PrintThresh": "0"}, {"PrintThresh": "1.5"},
         {"PrintSteepness": "0"}, {"PrintSteepness": "-1"},
         {"TargetDensity": "0"}, {"TargetDensity": "-0.1"},
         {"DoseMin": "1.03"}, {"DoseNom": "0.97"}, {"DoseMax": "0.9"}],
        ids=["thresh=0", "thresh=1.5", "steep=0", "steep<0",
             "density=0", "density<0", "min>max", "nom<min", "max<min"])
    def test_threshold_steepness_dose_contract_fails(self, tmp_path, overrides):
        """阈值区间、陡峭度为正、剂量单调顺序任一破坏时失败。"""
        path = _write_config(tmp_path, _config_lines(**overrides))
        with pytest.raises(ValueError):
            ICCAD13Config.from_file(path)


class TestAssets:
    """资产哈希、加载布局与 buffer 契约（设计文档 §11.2）。"""

    @pytest.mark.parametrize("filename,expected", sorted(_ASSET_SHA256.items()))
    def test_asset_sha256_matches_design(self, project_root, filename, expected):
        """四个资产 SHA-256 与迁移设计文档 §3.2 声明一致。"""
        asset = project_root / "lithography" / "assets" / "iccad13" / filename
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()  # 全文件哈希
        assert digest == expected  # 身份断言

    def test_kernel_banks_are_contiguous_layout(self):
        """加载后 kernel 是连续 complex64[24,35,35]，scale 是连续 float32[24]。"""
        model = ICCAD13Lithography(device="cpu")  # 显式 CPU，测试可复现
        for kernels in (model.focus_kernels, model.defocus_kernels):
            assert kernels.shape == (24, 35, 35)  # [K,H,W] 计算布局
            assert kernels.dtype == torch.complex64  # 复数振幅
            assert kernels.is_contiguous()  # 连续存储
        for scales in (model.focus_scales, model.defocus_scales):
            assert scales.shape == (24,)  # 每核一个权重
            assert scales.dtype == torch.float32  # 实数权重
            assert scales.is_contiguous()  # 连续存储

    def test_kernels_are_buffers_not_parameters(self):
        """kernel/scale 注册为 buffer，模型没有任何可训练 parameter。"""
        model = ICCAD13Lithography(device="cpu")
        assert set(dict(model.named_buffers())) == {  # 恰好四个 buffer
            "focus_kernels", "defocus_kernels", "focus_scales", "defocus_scales"}
        assert list(model.named_parameters()) == []  # 无 parameter

    def test_missing_asset_fails(self, tmp_path):
        """资产目录缺少 .pt 文件时加载失败。"""
        with pytest.raises(FileNotFoundError):
            ICCAD13Lithography(asset_dir=tmp_path, device="cpu")

    def test_default_device_is_auto(self):
        """device 省略时自动选择（有 CUDA 用 CUDA，否则 CPU）。"""
        model = ICCAD13Lithography()  # 缺省 device=None
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert model.device.type == expected  # 与显式 auto 同义

    def test_insufficient_asset_kernels_fails(self, tmp_path):
        """资产核数少于配置 kernel_count 时失败。"""
        config = tmp_path / "iccad13.txt"  # 请求 24 核的合法配置
        config.write_text("\n".join(_config_lines()) + "\n", encoding="utf-8")
        for name in ("focus", "defocus"):  # 只提供 4 核资产
            torch.save(torch.zeros(35, 35, 4, dtype=torch.complex64),
                       tmp_path / f"{name}.pt")
            torch.save(torch.zeros(4), tmp_path / f"{name}_scale.pt")
        with pytest.raises(ValueError, match="少于配置要求"):
            ICCAD13Lithography(
                config_path=config, asset_dir=tmp_path, device="cpu")

    def test_missing_scale_fails(self, tmp_path):
        """kernel 存在而 scale 缺失时失败。"""
        torch.save(torch.zeros(35, 35, 24, dtype=torch.complex64),
                   tmp_path / "focus.pt")  # 只给 kernel
        with pytest.raises(FileNotFoundError, match="focus_scale"):
            ICCAD13Lithography(asset_dir=tmp_path, device="cpu")

    def test_non_vector_scale_fails(self, tmp_path):
        """scale 是二维张量时拒绝。"""
        torch.save(torch.zeros(35, 35, 24, dtype=torch.complex64),
                   tmp_path / "focus.pt")
        torch.save(torch.zeros(24, 2), tmp_path / "focus_scale.pt")  # 二维
        with pytest.raises(ValueError, match="一维"):
            ICCAD13Lithography(asset_dir=tmp_path, device="cpu")

    def test_non_square_kernel_layout_fails(self, tmp_path):
        """kernel 不是方阵（如已是 [K,H,W] 计算布局）时拒绝。"""
        torch.save(torch.zeros(24, 35, 35, dtype=torch.complex64),
                   tmp_path / "focus.pt")  # 24≠35，非方阵
        torch.save(torch.zeros(24), tmp_path / "focus_scale.pt")  # 合法 scale
        with pytest.raises(ValueError, match="方阵"):
            ICCAD13Lithography(asset_dir=tmp_path, device="cpu")

    def test_kernel_scale_count_mismatch_fails(self, tmp_path):
        """kernel 核数与 scale 权重数不一致时拒绝。"""
        torch.save(torch.zeros(35, 35, 30, dtype=torch.complex64),
                   tmp_path / "focus.pt")  # 30 个核
        torch.save(torch.zeros(24), tmp_path / "focus_scale.pt")  # 24 个权重
        with pytest.raises(ValueError, match="数量不符"):
            ICCAD13Lithography(asset_dir=tmp_path, device="cpu")

    def test_to_moves_buffers_and_device(self):
        """Module.to() 同时移动四个 buffer，device 属性随之一致。"""
        model = ICCAD13Lithography(device="cpu")
        assert model.device == torch.device("cpu")  # 构造后即 CPU
        assert model.to("cpu") is model  # nn.Module.to 原地返回自身
        if torch.cuda.is_available():  # 有 GPU 时验证跨设备移动
            model.to("cuda")  # 移动到 CUDA
            assert model.device.type == "cuda"  # 属性跟随
            for buffer in model.buffers():  # 全部 buffer 同步
                assert buffer.device.type == "cuda"
            model.to("cpu")  # 还原，避免影响后续测试


class TestProcessCondition:
    """默认工艺条件与 ProcessCondition 自身校验。"""

    def test_default_conditions(self):
        """三个默认条件分别绑定正确的 kernel bank 与剂量。"""
        model = ICCAD13Lithography(device="cpu")
        nominal = model.condition("nominal")  # 标称角
        assert (nominal.kernel, nominal.dose) == ("focus", 1.00)
        dose_max = model.condition("dose_max")  # 最大剂量角
        assert (dose_max.kernel, dose_max.dose) == ("focus", 1.02)
        defocus_min = model.condition("defocus_min")  # 离焦最小剂量角
        assert (defocus_min.kernel, defocus_min.dose) == ("defocus", 0.98)

    def test_unknown_default_condition_fails(self):
        """未知默认条件名失败。"""
        model = ICCAD13Lithography(device="cpu")
        with pytest.raises(ValueError, match="未知默认工艺条件"):
            model.condition("nominal_max")

    @pytest.mark.parametrize(
        "name, kernel, dose",
        [("", "focus", 1.0), ("  ", "focus", 1.0),
         ("ok", "middle", 1.0), ("ok", "focus", 0.0),
         ("ok", "focus", -1.0), ("ok", "focus", float("nan"))],
        ids=["空名", "空白名", "未知bank", "零剂量", "负剂量", "nan剂量"])
    def test_invalid_condition_fails(self, name, kernel, dose):
        """空名称、未知 kernel bank、非正或非有限剂量都失败。"""
        with pytest.raises(ValueError):
            ProcessCondition(name, kernel, dose)


class TestShapeAndPadding:
    """输入形态、居中补零与方向契约（设计文档 §11.3）。"""

    def test_single_image_returns_same_shape(self):
        """[H,W] 输入返回 [H,W] float32 输出。"""
        model = ICCAD13Lithography(device="cpu")
        output = model(torch.zeros((64, 48)), model.condition("nominal"))
        assert output.shape == (64, 48)  # 形状还原
        assert output.dtype == torch.float32  # 计算精度

    def test_batch_returns_same_shape(self):
        """[B,H,W] 输入返回 [B,H,W] 输出。"""
        model = ICCAD13Lithography(device="cpu")
        output = model(torch.zeros((3, 70, 90)), model.condition("defocus_min"))
        assert output.shape == (3, 70, 90)  # 批量形状还原

    def test_center_padding_layout(self):
        """200×150 输入的低/高 padding 为 (28,28,53,53)，内容零移动。"""
        model = ICCAD13Lithography(device="cpu")
        mask = torch.ones((200, 150))  # 全透光便于验证内容位置
        padded, (top, bottom, left, right), was_single = model._prepare_mask(mask)
        assert (top, bottom, left, right) == (28, 28, 53, 53)  # 差值均分
        assert padded.shape == (1, 256, 256)  # 补齐到画布
        assert was_single  # 单张输入标记
        assert torch.count_nonzero(padded[0, :top]) == 0  # 上侧全零
        assert torch.count_nonzero(padded[0, -bottom:]) == 0  # 下侧全零
        assert torch.count_nonzero(padded[0, :, :left]) == 0  # 左侧全零
        assert torch.count_nonzero(padded[0, :, -right:]) == 0  # 右侧全零
        assert torch.all(padded[0, top:top + 200, left:left + 150] == 1.0)  # 原位

    def test_odd_remainder_goes_to_high_side(self):
        """奇数余量归高坐标侧：201 高 → 低 27 高 28。"""
        model = ICCAD13Lithography(device="cpu")
        _, (top, bottom, left, right), _ = model._prepare_mask(
            torch.zeros((201, 150)))
        assert (top, bottom) == (27, 28)  # 高度差 55，低侧取半
        assert (left, right) == (53, 53)  # 宽度差 106 均分

    def test_full_canvas_not_shifted_or_cropped(self):
        """满 256×256 输入 padding 全零，输出同形状不被二次移动。"""
        model = ICCAD13Lithography(device="cpu")
        mask = torch.zeros((256, 256))  # 满 canvas
        mask[100:150, 120:140] = 1.0  # 局部透光块
        _, padding, _ = model._prepare_mask(mask)
        assert padding == (0, 0, 0, 0)  # 四边零补
        output = model(mask, model.condition("nominal"))
        assert output.shape == (256, 256)  # 不裁剪

    def test_oversized_mask_fails(self):
        """超过 canvas 的输入在频域数组分配前失败。"""
        model = ICCAD13Lithography(device="cpu")
        with pytest.raises(ValueError, match="超过 canvas"):
            model(torch.zeros((257, 256)), model.condition("nominal"))

    def test_four_dimensional_input_fails(self):
        """四维输入拒绝。"""
        model = ICCAD13Lithography(device="cpu")
        with pytest.raises(ValueError, match="形状"):
            model(torch.zeros((2, 64, 64, 64)), model.condition("nominal"))

    def test_raster_canvas_passes_through_directly(self):
        """opc.input 的 256 raster canvas 可直接作为输入，不被二次移动。"""
        import klayout.db as kdb  # 仅集成测试引入版图依赖

        from layout import DbuBox
        from opc.input.raster import rasterize_mask_canvas
        region = (kdb.Region(kdb.Box(200, 200, 1400, 1300)) -  # 非对称实心块
                  kdb.Region(kdb.Box(500, 500, 900, 700)))  # 中心孔洞
        canvas = rasterize_mask_canvas(  # 与 main 演示相同参数
            region, DbuBox(0, 0, 1824, 1824), 8, 256, polarity="clear")
        model = ICCAD13Lithography(device="cpu")
        output = model(torch.from_numpy(canvas), model.condition("nominal"))
        assert output.shape == (256, 256)  # 满 canvas 直通

    def test_output_orientation_matches_input(self):
        """输出坐标方向与输入一致：低 Y 亮半区的输出仍在低 Y。"""
        model = ICCAD13Lithography(device="cpu")
        mask = torch.zeros((128, 128))  # 行 0 = 最低 Y（左下原点）
        mask[:64] = 1.0  # 仅低 Y 半区透光
        with torch.no_grad():
            output = model(mask, model.condition("nominal"))
        # 若模型错误地翻转 Y，亮暗半区会对调
        assert output[:64].mean() > output[64:].mean()  # 方向一致


class TestCpuNumerics:
    """CPU 固定数值与批量一致性（设计文档 §11.4）。"""

    def test_reference_sums_match_openilt_baseline(self):
        """三工艺角 sums 与 OpenILT 同资产基线一致（atol 0.05）。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask()  # 旧测试的确定性 [2,200,150]
        with torch.no_grad():
            batch = model.forward_many(mask, _default_conditions(model))
        expected = torch.tensor([  # OpenILT 基线（实测逐位复现）
            25802.533203125, 26009.16796875, 25675.23828125])
        actual = torch.stack((  # 按基线顺序堆叠
            batch["nominal"].sum(), batch["dose_max"].sum(),
            batch["defocus_min"].sum()))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.05)

    def test_batch_matches_single_images(self):
        """batch 输出与逐张运行逐像素一致（atol 1e-6）。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)  # 小尺寸降低耗时
        with torch.no_grad():
            batch = model.forward_many(mask, _default_conditions(model))
            first = model.forward_many(mask[:1], _default_conditions(model))
            second = model.forward_many(mask[1:], _default_conditions(model))
        for name in ("nominal", "dose_max", "defocus_min"):  # 逐条件逐像素
            torch.testing.assert_close(
                batch[name][0], first[name][0], rtol=0.0, atol=1e-6)
            torch.testing.assert_close(
                batch[name][1], second[name][0], rtol=0.0, atol=1e-6)

    def test_full_canvas_output_is_continuous_not_input(self):
        """满 canvas 输出不是原 mask，且含 (0,1) 开区间连续值。"""
        model = ICCAD13Lithography(device="cpu")
        generator = torch.Generator().manual_seed(7)  # 确定性随机
        mask = (torch.rand((256, 256), generator=generator) > 0.5).float()
        with torch.no_grad():
            output = model(mask, model.condition("nominal"))
        assert not torch.equal(output, mask)  # 经过真实光刻传播
        assert torch.any((output > 0.0) & (output < 1.0))  # 连续过渡
        assert 0.0 <= output.min().item()  # 范围下界
        assert output.max().item() <= 1.0  # 范围上界

    def test_custom_condition_takes_effect(self):
        """自定义条件的名称、kernel、剂量都生效。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)  # [2,64,64]
        conditions = (  # 自定义 + 两个对照
            ProcessCondition("focus_101", "focus", 1.01),
            model.condition("nominal"),
            ProcessCondition("defocus_101", "defocus", 1.01))
        with torch.no_grad():
            result = model.forward_many(mask, conditions)
        assert set(result) == {"focus_101", "nominal", "defocus_101"}  # 名称生效
        assert result["focus_101"].shape == mask.shape  # 形状还原
        assert not torch.equal(  # dose 生效：1.01 与 1.00 不同
            result["focus_101"], result["nominal"])
        assert not torch.equal(  # kernel 生效：同 dose 下 focus 与 defocus 不同
            result["focus_101"], result["defocus_101"])

    def test_duplicate_condition_names_fail(self):
        """同一次 forward_many 内条件名称重复时失败。"""
        model = ICCAD13Lithography(device="cpu")
        conditions = [model.condition("nominal"),
                      ProcessCondition("nominal", "focus", 1.5)]  # 重名不同义
        with pytest.raises(ValueError, match="不能重复"):
            model.forward_many(torch.zeros((32, 32)), conditions)

    def test_empty_conditions_fail(self):
        """空条件序列失败。"""
        model = ICCAD13Lithography(device="cpu")
        with pytest.raises(ValueError, match="至少需要一个"):
            model.forward_many(torch.zeros((32, 32)), [])

    def test_non_condition_entries_fail(self):
        """forward_many / forward 混入非 ProcessCondition 时类型失败。"""
        model = ICCAD13Lithography(device="cpu")
        with pytest.raises(TypeError, match="ProcessCondition"):
            model.forward_many(torch.zeros((32, 32)), ["nominal"])  # 字符串混入
        with pytest.raises(TypeError, match="ProcessCondition"):
            model(torch.zeros((32, 32)), "nominal")  # 单条件同样拦截


class TestSharedComputation:
    """一次 FFT 与每 bank 一次传播的性能不变量（设计文档 §11.5）。"""

    def test_single_fft_and_per_bank_propagation(self, monkeypatch):
        """默认三条件调用恰一次 mask fft2、focus/defocus 各传播一次。"""
        model = ICCAD13Lithography(device="cpu")
        fft_calls: list[int] = []  # fft2 调用计数
        original_fft2 = torch.fft.fft2  # 保存原函数
        monkeypatch.setattr(torch.fft, "fft2", lambda *a, **kw: (
            fft_calls.append(1), original_fft2(*a, **kw))[1])
        propagate_calls: list[bool] = []  # 每次传播是否 focus bank
        original_propagate = ICCAD13Lithography._propagate  # 保存原方法
        monkeypatch.setattr(ICCAD13Lithography, "_propagate", lambda self_, s, k, sc: (
            propagate_calls.append(k is self_.focus_kernels),
            original_propagate(self_, s, k, sc))[1])
        with torch.no_grad():
            model.forward_many(_model_mask(64, 64), _default_conditions(model))
        assert len(fft_calls) == 1  # mask FFT 只算一次
        assert propagate_calls == [True, False]  # focus 一次、defocus 一次

    def test_forward_many_matches_independent_calls(self):
        """共享频谱的三条件结果与三次独立 forward 数值一致。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)
        names = ("nominal", "dose_max", "defocus_min")
        with torch.no_grad():
            shared = model.forward_many(mask, _default_conditions(model))
            independent = {name: model(mask, model.condition(name))
                           for name in names}
        for name in names:  # 共享路径不改变数值
            torch.testing.assert_close(
                shared[name], independent[name], rtol=0.0, atol=1e-6)


class TestBackward:
    """原生 autograd 的梯度正确性（设计文档 §11.6）。"""

    def test_single_condition_mask_gradient(self):
        """nominal 单条件对 mask 产生有限非零梯度。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)[0].clone().requires_grad_()  # 单张
        model(mask, model.condition("nominal")).sum().backward()
        assert mask.grad is not None  # 梯度存在
        assert torch.all(torch.isfinite(mask.grad))  # 全部有限
        assert torch.count_nonzero(mask.grad).item() > 0  # 非全零

    def test_joint_condition_loss_gradient(self):
        """nominal+dose_max−defocus_min 联合损失仍向 mask 传播梯度。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)[:1].clone().requires_grad_()  # [1,64,64]
        result = model.forward_many(mask, _default_conditions(model))
        loss = (result["nominal"].mean() + result["dose_max"].mean()
                - result["defocus_min"].mean())  # 联合损失
        loss.backward()
        assert torch.all(torch.isfinite(mask.grad))  # 有限
        assert torch.count_nonzero(mask.grad).item() > 0  # 非零

    def test_batch_every_image_receives_gradient(self):
        """batch 中每张 mask 都能获得非零梯度。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64).requires_grad_()  # [2,64,64]
        result = model.forward_many(mask, [model.condition("nominal")])
        result["nominal"].sum().backward()
        for image in range(2):  # 逐张检查
            assert torch.count_nonzero(mask.grad[image]).item() > 0

    def test_autograd_matches_nonuniform_finite_difference(self):
        """非均匀上游权重下 autograd 与中心有限差分一致（旧测试移植）。"""
        model = ICCAD13Lithography(device="cpu")
        generator = torch.Generator().manual_seed(20260811)  # 固定随机源
        mask = torch.rand((20, 18), generator=generator,
                          dtype=torch.float32).requires_grad_()  # 连续 mask
        # 非均匀权重避免对称损失掩盖梯度错误（均匀权重下对称扰动可能抵消）
        weights = torch.linspace(-0.7, 1.3, mask.numel(),
                                 dtype=torch.float32).reshape_as(mask)
        condition = ProcessCondition("focus_101", "focus", 1.01)
        torch.sum(model(mask, condition) * weights).backward()
        assert mask.grad is not None  # autograd 梯度已就位
        y, x, epsilon = 9, 8, 1e-3  # 检查点与扰动步长
        with torch.no_grad():
            plus, minus = mask.detach().clone(), mask.detach().clone()
            plus[y, x] += epsilon  # 正扰动
            minus[y, x] -= epsilon  # 负扰动
            numerical = ((  # 中心差分：dL/dmask[y,x]
                torch.sum(model(plus, condition) * weights)
                - torch.sum(model(minus, condition) * weights))
                / (2.0 * epsilon))
        torch.testing.assert_close(
            mask.grad[y, x], numerical, rtol=2e-2, atol=2e-2)

    def test_buffers_have_no_grad_after_backward(self):
        """backward 后 kernel/scale buffer 仍无 .grad（非可训练对象）。"""
        model = ICCAD13Lithography(device="cpu")
        mask = torch.zeros((32, 32), requires_grad=True)
        model(mask, model.condition("nominal")).sum().backward()
        for buffer in model.buffers():  # buffer.grad 恒 None
            assert buffer.grad is None

    def test_no_grad_output_has_no_graph(self):
        """no_grad 推理输出不保留 autograd 图。"""
        model = ICCAD13Lithography(device="cpu")
        with torch.no_grad():
            output = model(torch.zeros((32, 32)), model.condition("nominal"))
        assert not output.requires_grad  # 纯推理无图
        requires_input = torch.zeros((32, 32), requires_grad=True)
        with torch.no_grad():  # 即使输入带 grad，no_grad 下也不建图
            detached = model(requires_input, model.condition("nominal"))
        assert not detached.requires_grad


class TestCuda:
    """CUDA parity 与直接环境运行（设计文档 §11.7，无 GPU 时跳过）。"""

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="当前环境没有 CUDA")
    def test_cuda_matches_cpu(self):
        """CPU/GPU 同输入同条件输出在容差内一致。"""
        model = ICCAD13Lithography(device="cpu")
        mask = _model_mask(64, 64)
        conditions = _default_conditions(model)
        with torch.no_grad():
            cpu = model.forward_many(mask, conditions)
            gpu_model = ICCAD13Lithography(device="cuda")
            gpu = gpu_model.forward_many(mask.cuda(), conditions)
        for name in ("nominal", "dose_max", "defocus_min"):
            torch.testing.assert_close(  # FFT 实现差异下的浮点容差
                cpu[name], gpu[name].cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="当前环境没有 CUDA")
    def test_cuda_forward_backward_complete(self):
        """CUDA forward/backward 均完成且梯度有限。"""
        model = ICCAD13Lithography(device="cuda")
        mask = _model_mask(64, 64).cuda().requires_grad_()
        result = model.forward_many(mask, _default_conditions(model))
        result["nominal"].sum().backward()
        torch.cuda.synchronize()  # 显式同步确保异步错误暴露
        assert mask.grad is not None
        assert torch.all(torch.isfinite(mask.grad))

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="当前环境没有 CUDA")
    def test_direct_environment_python_loads_cuda_runtime(self, project_root):
        """环境 python.exe 子进程直跑必须能加载 CUDA 运行时。"""
        code = (  # 与旧测试同构：不依赖 conda run 与项目安装
            "import json,torch; from lithography import ICCAD13Lithography; "
            "m=ICCAD13Lithography(device='cuda'); "
            "x=torch.zeros((1,64,64),device='cuda'); "
            "y=m(x,m.condition('nominal')); torch.cuda.synchronize(); "
            "print(json.dumps({'shape':list(y.shape),'device':str(y.device)}))")
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=project_root, capture_output=True,
            text=True, timeout=120, check=False)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {
            "shape": [1, 64, 64], "device": "cuda:0"}

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="当前环境没有 CUDA")
    def test_cuda_reports_peak_memory(self):
        """记录 CUDA elapsed 与 peak allocated（不设绝对阈值）。"""
        model = ICCAD13Lithography(device="cuda")
        torch.cuda.reset_peak_memory_stats()  # 从模型加载后开始计量
        mask = _model_mask(200, 150).cuda()  # 基线用同尺寸
        started = time.perf_counter()  # 计时起点
        with torch.no_grad():
            model.forward_many(mask, _default_conditions(model))
        torch.cuda.synchronize()  # 等待全部核函数完成
        elapsed = time.perf_counter() - started  # 实测耗时
        peak = torch.cuda.max_memory_allocated()  # 实测峰值
        assert elapsed > 0  # 只记录不设阈值
        assert peak > 0  # 确实发生了 GPU 分配


class TestMainEntry:
    """main/main_test_lithography.py 子进程直跑验证（GDS→光刻留档 CLI）。"""

    @staticmethod
    def _write_gds(path):
        """生成上下覆盖率不同的单层小版图（dbu=1nm，光照结果可判方向）。"""
        import klayout.db as kdb  # 原生版图对象
        layout = kdb.Layout()  # 独立版图
        layout.dbu = 0.001  # 1 nm/DBU
        top = layout.create_cell("TOP")  # 唯一顶层
        top.shapes(layout.layer(1, 0)).insert(kdb.Box(20, 20, 120, 40))  # 下块（全宽）
        top.shapes(layout.layer(1, 0)).insert(kdb.Box(20, 60, 80, 80))  # 上块（半宽）
        layout.write(str(path))  # 写盘
        return path  # 返回路径

    def _run_entry(self, cwd, tmp) -> subprocess.CompletedProcess:
        """以小参数直跑 GDS→光刻入口（CPU、产物显式落在 tmp）。"""
        from pathlib import Path  # 局部导入脚本路径
        gds = self._write_gds(tmp / "reticle.gds")  # 生成式输入版图
        script = (Path(__file__).resolve().parents[2]
                  / "main" / "main_test_lithography.py")  # 入口脚本
        # 小窗口参数：core 128 + 2×64 = 256 ≤ 画布 256×8，单 tile 跑得快
        return subprocess.run(  # 与用户手工直跑同构
            [sys.executable, str(script), str(gds), "--layer", "1/0",
             "--core-nm", "128", "--context-nm", "64", "--pixel-nm", "8",
             "--batch", "2", "--device", "cpu", "--out", str(tmp / "litho_out")],
            cwd=cwd, capture_output=True, text=True, timeout=180, check=False)

    def test_entry_runs_from_repository_root(self, project_root, tmp_path):
        """从仓库根直跑退出码 0，打印关键标记且产物落盘。"""
        completed = self._run_entry(project_root, tmp_path)  # 仓库内直跑
        assert completed.returncode == 0, completed.stderr  # 正常退出
        for marker in ("device=", "tile 数：", "manifest：", "已保存"):  # 输出标记
            assert marker in completed.stdout, marker  # 缺一即失败
        out = tmp_path / "litho_out"  # 留档目录
        manifest = json.loads(  # 读清单
            (out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tile_count"] > 0  # 至少一个 tile
        for tile in manifest["tiles"]:  # 逐 tile 检查
            assert (out / tile["nominal_png"]).is_file()  # 连续 PNG
            assert (out / tile["binary_png"]).is_file()  # 二值 PNG

    def test_entry_runs_outside_repository(self, project_root, tmp_path):
        """从仓库外工作目录直跑同样成功（脚本自做 sys.path 引导）。"""
        completed = self._run_entry(tmp_path, tmp_path)  # cwd=仓库外目录
        assert completed.returncode == 0, completed.stderr  # 不依赖 cwd
        assert "device=" in completed.stdout  # 完整跑通

    def test_entry_leaves_worktree_unchanged(self, project_root, tmp_path):
        """入口不生成仓库内临时产物（显式 --out 到 tmp，git status 前后一致）。"""
        status = ["git", "status", "--porcelain"]  # 只读查询
        before = subprocess.run(  # 运行前快照
            status, cwd=project_root, capture_output=True,
            text=True, check=True).stdout
        self._run_entry(project_root, tmp_path)  # 完整执行一次
        after = subprocess.run(  # 运行后快照
            status, cwd=project_root, capture_output=True,
            text=True, check=True).stdout
        assert after == before  # 零新增产物


class TestEntryValidation:
    """main(argv) 进程内校验直测（不起子进程、错误发生在模型加载之前）。"""

    def _gds(self, tmp_path):
        """复用入口测试的生成式版图。"""
        return TestMainEntry._write_gds(tmp_path / "reticle.gds")

    def test_off_grid_nm_reports_flag_name(self, tmp_path):
        """--core-nm 落不了格点时报错含 flag 名。"""
        import main.main_test_lithography as entry  # 入口模块
        with pytest.raises(ValueError, match="--core-nm"):
            entry.main([str(self._gds(tmp_path)), "--layer", "1/0",
                        "--core-nm", "128.5", "--context-nm", "64",
                        "--pixel-nm", "8", "--out", str(tmp_path / "o")])

    def test_empty_layer_reports_layer_numbers(self, tmp_path):
        """目标层无图形时报错含层号与 datatype。"""
        import main.main_test_lithography as entry  # 入口模块
        with pytest.raises(ValueError, match="目标层 5/0"):
            entry.main([str(self._gds(tmp_path)), "--layer", "5/0",
                        "--core-nm", "128", "--context-nm", "64",
                        "--pixel-nm", "8", "--out", str(tmp_path / "o")])

    def test_bad_layer_format_exits_two(self, tmp_path):
        """--layer 非 N/D 格式时 argparse 以退出码 2 终止。"""
        import main.main_test_lithography as entry  # 入口模块
        with pytest.raises(SystemExit) as caught:
            entry.main([str(self._gds(tmp_path)), "--layer", "11"])
        assert caught.value.code == 2  # 用法错误退出码

    def test_nonpositive_batch_exits_two(self, tmp_path):
        """--batch 0 在解析层被拒（退出码 2），不做脏崩溃。"""
        import main.main_test_lithography as entry  # 入口模块
        with pytest.raises(SystemExit) as caught:
            entry.main([str(self._gds(tmp_path)), "--batch", "0"])
        assert caught.value.code == 2  # 用法错误退出码
