"""ICCAD13 配置解析、Hopkins 资产契约与工艺条件的生成式测试。"""

import hashlib

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
