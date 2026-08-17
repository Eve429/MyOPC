"""统一配置体系 load_config 的行为测试（规格 §24.1–24.9）。"""

from decimal import Decimal

import pytest

from main import configuration
from main.configuration import (
    GradientConfig,
    LayoutConfig,
    LithographyConfig,
    MBOPCConfig,
    OutputConfig,
    PartitionConfig,
    SinglePassConfig,
    load_config,
)

# 完整合法 TOML：含全部七个注册段，作为多数用例的基底（请求谁由用例决定）。
_FULL_TOML = """
[layout]
layout = "reticle.gds"
top_cell = "TOP"
layer = 11
datatype = 0
polarity = "clear"

[partition]
macro_grid = [2, 2]
core_size_nm = 1024
context_nm = 400

[lithography]
pixel_nm = 8
canvas_pixels = 256
device = "auto"

[mbopc]
iterations = 8
initial_step_nm = 8
decay_every = 4
epe_distance_nm = 16
batch_size = 8
target_cache_mb = 512

[gradient]
iterations = 1
learning_rate_nm = 1.0
weight_nominal_l2 = 1.0
weight_process_l2 = 0.5
weight_pvband = 0.1
epe_distance_nm = 16
batch_size = 8
target_cache_mb = 512

[single_pass]
displacement_nm = 5

[output]
work_dir = "work"
final_layout = "final.gds"
final_cell_mode = "single_cell"
"""


def _write(tmp_path, text=_FULL_TOML, name="config.toml"):
    """写入 TOML 文本并返回配置路径；同步放置空版图占位不需要（不打开）。"""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestSingleAndMultiConfig:
    """单/多 Config 加载（§24.1/24.2）。"""

    def test_single_config(self, tmp_path):
        """只请求 MBOPCConfig：类型与字段正确。"""
        cfg, = load_config(_write(tmp_path), MBOPCConfig)  # 尾随单元素解包
        assert isinstance(cfg, MBOPCConfig)  # 类型
        assert cfg.iterations == 8  # 整数原样
        assert cfg.initial_step_nm == Decimal(8)  # Decimal 精确
        assert cfg.batch_size == 8  # 批大小
        assert cfg.target_cache_mb == 512  # 缓存上限

    def test_multi_config_order_and_types(self, tmp_path):
        """多 Config：返回顺序与请求一致、类型正确。"""
        layout, partition, litho, mbopc, output = load_config(
            _write(tmp_path), LayoutConfig, PartitionConfig,
            LithographyConfig, MBOPCConfig, OutputConfig)
        assert isinstance(layout, LayoutConfig)  # 顺序 1
        assert isinstance(partition, PartitionConfig)  # 顺序 2
        assert isinstance(litho, LithographyConfig)  # 顺序 3
        assert isinstance(mbopc, MBOPCConfig)  # 顺序 4
        assert isinstance(output, OutputConfig)  # 顺序 5
        assert layout.layer == 11 and layout.datatype == 0  # 版图字段
        assert partition.macro_grid == (2, 2)  # 元组归一
        assert litho.pixel_nm == Decimal(8)  # Decimal 字段
        assert output.final_cell_mode == "single_cell"  # Literal 字段

    def test_repeated_request_returns_independent_instances(self, tmp_path):
        """同一 Config 请求两次：两次独立解析（无共享可变状态）。"""
        first, second = load_config(_write(tmp_path), MBOPCConfig, MBOPCConfig)
        assert first == second  # 值相等（frozen）
        assert first is not second  # 实例独立


class TestSingleRead:
    """TOML 只读一次（§24.3）。"""

    def test_toml_read_exactly_once(self, tmp_path, monkeypatch):
        """无论请求多少 Config，读盘只发生一次。"""
        calls = []  # 读盘计数
        real = configuration.toml_loads  # 原函数

        def counting(text):
            """计数并透传。"""
            calls.append(1)
            return real(text)

        monkeypatch.setattr(configuration, "toml_loads", counting)
        load_config(_write(tmp_path), LayoutConfig, PartitionConfig,
                    LithographyConfig, MBOPCConfig, GradientConfig,
                    SinglePassConfig, OutputConfig)  # 请求全部七类
        assert calls == [1]  # 恰一次读盘


class TestUnrequestedSections:
    """请求之外的合法 section（§24.4/24.5）。"""

    def test_unrequested_sections_allowed(self, tmp_path):
        """TOML 含全部段、只请求 MBOPCConfig：成功。"""
        cfg, = load_config(_write(tmp_path), MBOPCConfig)
        assert cfg.iterations == 8  # 解析不受影响

    def test_unknown_field_in_unrequested_section_still_fails(self, tmp_path):
        """未请求 [gradient] 内的拼错键仍必须报错。"""
        text = _FULL_TOML.replace("weight_pvband = 0.1",  # 注入拼错
                                  "weight_pvband = 0.1\nweigth_nominal = 1.0")  # 未知键
        with pytest.raises(ValueError, match=r"\[gradient\] 含未知键.*weigth_nominal"):
            load_config(_write(tmp_path, text), MBOPCConfig)  # 只请求 simple


class TestStrictness:
    """未知 section / required / default / 类型严格性（§24.6–24.8 + §14）。"""

    def test_unknown_section_fails_with_path(self, tmp_path):
        """拼错段名报错且含配置路径。"""
        text = _FULL_TOML + "\n[gradinet]\nfoo = 1\n"  # 拼错段
        path = _write(tmp_path, text)
        with pytest.raises(ValueError, match="未知配置段.*gradinet") as excinfo:
            load_config(path, MBOPCConfig)
        assert str(path) in str(excinfo.value)  # 错误含路径

    def test_missing_required_fails(self, tmp_path):
        """删掉请求 Config 的必填键必须报错。"""
        text = _FULL_TOML.replace("iterations = 8\n", "")  # 删必填
        with pytest.raises(ValueError,
                           match=r"\[mbopc\] 缺少必填键：\['iterations'\]"):
            load_config(_write(tmp_path, text), MBOPCConfig)

    def test_default_applied_when_field_absent(self, tmp_path):
        """有默认值的字段缺省时使用 dataclass 默认。"""
        text = _FULL_TOML.replace('top_cell = "TOP"\n', "")  # 删可选 top_cell
        layout, output = load_config(_write(tmp_path, text),
                                     LayoutConfig, OutputConfig)
        assert layout.top_cell is None  # 默认 None
        assert output.save_final_lithography is False  # 默认 False
        assert output.show_progress is False  # 默认 False

    @pytest.mark.parametrize(
        ("inject", "field"),
        [("layer = 11.5", "layer"), ("layer = true", "layer"),
         ('layer = "11"', "layer"), ("datatype = 0.5", "datatype"),
         ("iterations = 1.5", "iterations"), ("iterations = true", "iterations")],
        ids=["layer=1.5", "layer=true", "layer=str", "dt=0.5",
             "iter=1.5", "iter=true"])
    def test_strict_types_reject_bad_values(self, tmp_path, inject, field):
        """整数字段拒绝 float/bool/string（沿用 P1-3 严格语义）。"""
        original = {"layer": "layer = 11", "datatype": "datatype = 0",
                    "iterations": "iterations = 8"}[field]  # 原行
        text = _FULL_TOML.replace(original, inject)  # 注入非法值
        requested = (LayoutConfig, MBOPCConfig)[field == "iterations"]  # 请求类
        with pytest.raises(ValueError, match=field):
            load_config(_write(tmp_path, text), requested)

    def test_decimal_rejects_string(self, tmp_path):
        """Decimal 字段拒绝字符串（\"8\" 不当数值）。"""
        text = _FULL_TOML.replace("initial_step_nm = 8", 'initial_step_nm = "8"')
        with pytest.raises(ValueError, match="initial_step_nm"):
            load_config(_write(tmp_path, text), MBOPCConfig)

    def test_partition_mutex_in_post_init(self, tmp_path):
        """macro_grid/macro_size_nm 同现或同缺在构造期失败。"""
        both = _FULL_TOML.replace("macro_grid = [2, 2]",
                                  "macro_grid = [2, 2]\nmacro_size_nm = 4096")
        with pytest.raises(ValueError, match="恰好填写一个"):
            load_config(_write(tmp_path, both), PartitionConfig)
        neither = _FULL_TOML.replace("macro_grid = [2, 2]\n", "")
        with pytest.raises(ValueError, match="恰好填写一个"):
            load_config(_write(tmp_path, neither), PartitionConfig)


class TestPaths:
    """路径三态（§24.9）。"""

    def test_relative_path_resolved_against_toml_dir(self, tmp_path):
        """相对路径相对 TOML 所在目录解析。"""
        (tmp_path / "sub").mkdir()  # 子目录放配置
        path = _write(tmp_path / "sub")  # 配置在 sub/ 下
        layout, output = load_config(path, LayoutConfig, OutputConfig)
        assert layout.layout == (tmp_path / "sub" / "reticle.gds").resolve()  # 相对 sub
        assert output.final_layout == (tmp_path / "sub" / "final.gds").resolve()

    def test_absolute_path_kept(self, tmp_path):
        """绝对路径原样保留。"""
        target = tmp_path / "elsewhere" / "reticle.gds"  # 绝对路径
        text = _FULL_TOML.replace('layout = "reticle.gds"',
                                  f'layout = "{target.as_posix()}"')  # 注入
        layout, = load_config(_write(tmp_path, text), LayoutConfig)
        assert layout.layout == target.resolve()  # 绝对语义保持

    def test_tilde_expanded(self, tmp_path, monkeypatch):
        """~ 前缀按用户主目录展开（用假 HOME 隔离）。"""
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows expanduser 锚
        monkeypatch.setenv("HOME", str(tmp_path))  # 跨平台保险
        text = _FULL_TOML.replace('layout = "reticle.gds"',
                                  'layout = "~/reticle.gds"')  # ~ 路径
        layout, = load_config(_write(tmp_path, text), LayoutConfig)
        assert layout.layout == (tmp_path / "reticle.gds").resolve()  # 展开正确
