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

# 完整合法 TOML：含全部八个注册段，作为多数用例的基底（请求谁由用例决定）。
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

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24
miter_limit = 4.0

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


class TestLevelSetSection:
    """[levelset_ilt] 段注册与严格解析（CHG-20260818-levelset IF-003）。"""

    _SECTION = """
[levelset_ilt]
iterations = 2
step_size = 0.2
weight_process_l2 = 1.0
weight_pvband = 0.5
curvature_weight = 0.0
batch_size = 4
"""

    def test_section_parses_to_config(self, tmp_path):
        """合法段解析为 LevelSetILTConfig（全部字段无默认）。"""
        from opc.iteration.ilt import LevelSetILTConfig
        path = _write(tmp_path, _FULL_TOML + self._SECTION)
        config, = load_config(path, LevelSetILTConfig)
        assert config.iterations == 2
        assert config.step_size == pytest.approx(0.2)
        assert config.batch_size == 4
        assert configuration.CONFIG_SECTIONS[LevelSetILTConfig] == "levelset_ilt"

    def test_missing_required_key_fails(self, tmp_path):
        """缺 step_size 必填键在加载期失败。"""
        from opc.iteration.ilt import LevelSetILTConfig
        text = _FULL_TOML + self._SECTION.replace("step_size = 0.2\n", "")
        with pytest.raises(ValueError,
                           match=r"\[levelset_ilt\] 缺少必填键：\['step_size'\]"):
            load_config(_write(tmp_path, text), LevelSetILTConfig)

    def test_bool_rejected_for_iterations(self, tmp_path):
        """iterations 传 bool 冒充 int 在加载期失败。"""
        from opc.iteration.ilt import LevelSetILTConfig
        text = _FULL_TOML + self._SECTION.replace(
            "iterations = 2", "iterations = true")
        with pytest.raises(ValueError, match="iterations"):
            load_config(_write(tmp_path, text), LevelSetILTConfig)


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



class TestFieldBoxSection:
    """[layout] 处理框字段：双写法解析与互斥（field_box/field_size）。"""

    def test_field_box_parses_four_decimals(self, tmp_path):
        """field_box_nm 四元 Decimal 定长元组解析。"""
        text = _FULL_TOML.replace(
            'polarity = "clear"',
            'polarity = "clear"\nfield_box_nm = [-512.0, -512.0, 1536.0, 1536.0]')
        layout, = load_config(_write(tmp_path, text), LayoutConfig)
        assert layout.field_box_nm == (Decimal("-512.0"), Decimal("-512.0"),
                                       Decimal("1536.0"), Decimal("1536.0"))

    def test_field_size_parses_two_decimals(self, tmp_path):
        """field_size_nm 二元 Decimal 定长元组解析。"""
        text = _FULL_TOML.replace(
            'polarity = "clear"',
            'polarity = "clear"\nfield_size_nm = [2048.0, 2048.0]')
        layout, = load_config(_write(tmp_path, text), LayoutConfig)
        assert layout.field_size_nm == (Decimal("2048.0"), Decimal("2048.0"))

    def test_both_fields_rejected(self, tmp_path):
        """双填即意图不明，构造期拒绝。"""
        text = _FULL_TOML.replace(
            'polarity = "clear"',
            'polarity = "clear"\nfield_box_nm = [0.0, 0.0, 1.0, 1.0]\n'
            'field_size_nm = [2048.0, 2048.0]')
        with pytest.raises(ValueError, match="至多填写一个"):
            load_config(_write(tmp_path, text), LayoutConfig)

    def test_neither_field_keeps_defaults(self, tmp_path):
        """双空 = 现行 layer bbox 行为，字段为 None。"""
        layout, = load_config(_write(tmp_path), LayoutConfig)
        assert layout.field_box_nm is None and layout.field_size_nm is None

    def test_field_box_length_strict(self, tmp_path):
        """三元组按定长形状拒绝（防漏写坐标）。"""
        text = _FULL_TOML.replace(
            'polarity = "clear"',
            'polarity = "clear"\nfield_box_nm = [0.0, 0.0, 1.0]')
        with pytest.raises(ValueError, match="列表"):
            load_config(_write(tmp_path, text), LayoutConfig)


class TestGridRuntime:
    """IF-001：算法无关网格解析与 PrepareRuntime 组合结构。"""

    @staticmethod
    def _partition(**overrides):
        """按数量模式默认值组装划分配置。"""
        values = {"core_size_nm": Decimal(1024), "context_nm": Decimal(400),
                  "macro_grid": (2, 2)}
        values.update(overrides)
        return configuration.PartitionConfig(**values)

    def test_count_mode_grid_values(self):
        """nm→DBU 精确换算；数量模式 macro_size_dbu 保持 None。"""
        grid = configuration.resolve_grid_config(
            self._partition(), LithographyConfig(pixel_nm=Decimal(8)),
            Decimal(1))
        assert isinstance(grid, configuration.GridRuntime)
        assert (grid.core_dbu, grid.context_dbu, grid.pixel_dbu) == (1024, 400, 8)
        assert grid.macro_size_dbu is None

    def test_size_mode_converts_macro_size(self):
        """尺寸模式换算 macro_size_dbu（0.1nm DBU 台阶放大十倍）。"""
        partition = self._partition(macro_grid=None, macro_size_nm=Decimal(4096))
        grid = configuration.resolve_grid_config(
            partition, LithographyConfig(pixel_nm=Decimal(8)), Decimal("0.1"))
        assert grid.macro_size_dbu == 40960

    def test_nonexact_pixel_fails(self):
        """不能整除的 pixel_nm 在解析期失败，不四舍五入。"""
        with pytest.raises(ValueError):
            configuration.resolve_grid_config(
                self._partition(), LithographyConfig(pixel_nm=Decimal("2.5")),
                Decimal(1))

    def test_prepare_runtime_composes_grid(self):
        """resolve_prepare_config 组合 GridRuntime 与边段配置，数值不变。"""
        edge = configuration.EdgeConfig(
            corner_nm=Decimal(16), segment_nm=Decimal(32),
            max_displacement_nm=Decimal(24), miter_limit=4.0)
        runtime = configuration.resolve_prepare_config(
            self._partition(), LithographyConfig(pixel_nm=Decimal(8)),
            edge, Decimal(1))
        assert runtime.grid.core_dbu == 1024  # 旧平铺字段值经 grid 到达
        assert runtime.grid.context_dbu == 400
        assert runtime.grid.pixel_dbu == 8
        assert runtime.grid.macro_size_dbu is None
        assert runtime.fragmentation.corner_length_dbu == 16.0
        assert runtime.fragmentation.max_segment_length_dbu == 32.0
        assert runtime.fragmentation.max_displacement_dbu == 24.0

    def test_prepare_runtime_keeps_context_contract(self):
        """max_displacement 超过 context 的既有契约在组合结构下不变。"""
        edge = configuration.EdgeConfig(
            corner_nm=Decimal(16), segment_nm=Decimal(32),
            max_displacement_nm=Decimal(800), miter_limit=4.0)
        with pytest.raises(ValueError, match="context_nm"):
            configuration.resolve_prepare_config(
                self._partition(), LithographyConfig(pixel_nm=Decimal(8)),
                edge, Decimal(1))
