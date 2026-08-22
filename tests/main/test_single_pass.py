"""单遍偏置扩张管线的环双向扩张与产物契约生成式测试。"""

import klayout.db as kdb
import pytest

import main.run_single_pass as single_pass
from layout import LayerSpec, LayoutDB
from main.configuration import LayoutConfig, load_config

# 测试版图：DBU=1nm。正向位移用「锚框 + 完全内部 donut」——贴着层 bbox 的
# 图形外扩会全部落在 macro ownership 之外被正确裁掉，无法证明外扩生效；
# 负向位移向内收缩永不越界，用独立 donut 即可。2×2 macro、core 30、
# context 10、pixel 1 满足全部契约。
# 几何退化避开：macro 切线为 x=80 与 y=50；图形边恰好与内部 macro 切线
# 重合时，边整条归一侧 macro、另一侧以 context 原位参与拐角重建，拼合处
# 会出现位移宽度的台阶——所有 donut 边都刻意偏离切线（bbox 外沿例外：
# 外沿处的邻侧 context 副本被裁剪成零宽，不会产生台阶）。


def _insert_donut(layout, top, layer_index, outer, hole):
    """以 Region 减法构造带孔 polygon 并写入指定层。"""
    donut = kdb.Region(kdb.Box(*outer)) - kdb.Region(kdb.Box(*hole))  # 外减孔
    for polygon in donut.each():  # 展开为带孔 polygon（恰一个）
        top.shapes(layer_index).insert(polygon)  # 写入


def _write_anchored_gds(tmp_path):
    """生成锚框 + 内部 donut + 2/0 对照层的 GDS 并返回路径（正向位移用）。"""
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    one = layout.layer(1, 0)  # 目标层
    top.shapes(one).insert(kdb.Box(20, 20, 100, 26))  # 下锚框（撑住层 bbox）
    top.shapes(one).insert(kdb.Box(20, 74, 100, 80))  # 上锚框
    _insert_donut(layout, top, one, (35, 37, 85, 63), (52, 42, 68, 58))  # 内部 donut（孔 16×16，双向收缩后仍留 6×6）
    top.shapes(layout.layer(2, 0)).insert(kdb.Box(30, 30, 90, 70))  # 非目标对照层
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_plain_donut_gds(tmp_path):
    """生成独立 donut + 2/0 对照层的 GDS 并返回路径（负向位移用）。"""
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    one = layout.layer(1, 0)  # 目标层
    _insert_donut(layout, top, one, (20, 20, 100, 80), (42, 42, 72, 58))  # donut（孔边避开切线 x=80/y=50）
    top.shapes(layout.layer(2, 0)).insert(kdb.Box(30, 30, 90, 70))  # 非目标对照层
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _donut(outer, hole):
    """按外框与孔框构造期望的 donut Region。"""
    return kdb.Region(kdb.Box(*outer)) - kdb.Region(kdb.Box(*hole))  # 外减孔


def _write_config(tmp_path, layout_path, **overrides):
    """按默认契约生成单遍 TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值全部满足网格与位移契约
        "macro_grid": "[2, 2]",
        "core_size_nm": 30,
        "context_nm": 10,
        "pixel_nm": 1,
        "corner_nm": 4,
        "segment_nm": 10,
        "max_displacement_nm": 8,
        "miter_limit": 4.0,
        "displacement_nm": 5,
        "final_cell_mode": "single_cell",
    }
    values.update(overrides)  # 应用覆盖
    text = f"""  # 组装 TOML 文本
[layout]
layout = "{layout_path.as_posix()}"
top_cell = "TOP"
layer = 1
datatype = 0
polarity = "clear"

[partition]
macro_grid = {values["macro_grid"]}
core_size_nm = {values["core_size_nm"]}
context_nm = {values["context_nm"]}

[lithography]
pixel_nm = {values["pixel_nm"]}
canvas_pixels = 256

[edge]
corner_nm = {values["corner_nm"]}
segment_nm = {values["segment_nm"]}
max_displacement_nm = {values["max_displacement_nm"]}
miter_limit = {values["miter_limit"]}

[single_pass]
displacement_nm = {values["displacement_nm"]}

[output]
final_layout = "{(tmp_path / "out" / "final.gds").as_posix()}"
final_cell_mode = "{values["final_cell_mode"]}"
"""
    config_path = tmp_path / "single_pass.toml"  # 配置路径
    config_path.write_text(text, encoding="utf-8")  # 写盘
    return config_path  # 返回路径


def _load(path):
    """经统一 load_config 加载单遍六 Config 并按序返回元组。"""
    from main.configuration import (
        EdgeConfig,
        LayoutConfig,
        LithographyConfig,
        OutputConfig,
        PartitionConfig,
        SinglePassConfig,
        load_config,
    )

    return load_config(
        path,
        LayoutConfig,
        PartitionConfig,  # 六 Config 元组
        LithographyConfig,
        EdgeConfig,
        SinglePassConfig,
        OutputConfig,
    )


def _final_region(path, layer=None):
    """回读最终版图并物化目标层覆盖。"""
    if layer is None:  # 默认目标层在函数体内构造，避免可变默认值风格问题
        layer = LayerSpec(1, 0)  # 目标层
    with LayoutDB.open(path) as db:  # 打开最终版图
        box = db.bbox()  # 版图包围盒
        assert box is not None  # 非空
        return db.query([layer], box).materialize().region(layer)  # 物化目标层


class TestRingExpansion:
    """环双向扩张语义（设计文档 §2.2）。"""

    def test_positive_displacement_expands_ring_both_ways(self, tmp_path):
        """+5 nm：donut 外环外扩 5、孔壁内收 5，环带宽 20→30。"""
        gds = _write_anchored_gds(tmp_path)  # 生成锚框版图
        configs = _load(_write_config(tmp_path, gds))  # 加载
        final = single_pass.run_single_pass(*configs)  # 执行
        # donut 外扩 5、孔缩 5；锚框外扩被 bbox 裁剪，仅内边各进 5。
        expected = (
            kdb.Region(kdb.Box(20, 20, 100, 31))  # 下锚框（顶边 26→31）
            + kdb.Region(kdb.Box(20, 69, 100, 80))  # 上锚框（底边 74→69）
            + _donut((30, 32, 90, 68), (57, 47, 63, 53))
        )  # donut 双向扩张（孔 16×16→6×6）
        actual = _final_region(final)  # 实际覆盖
        assert (actual ^ expected).area() == 0  # XOR 零
        assert actual.count() == 3  # 两个锚框 + 一个带孔 polygon，无 seam

    def test_negative_displacement_shrinks_ring_both_ways(self, tmp_path):
        """-5 nm：外环内收 5、孔壁外扩 5，环带宽 20→10。"""
        gds = _write_plain_donut_gds(tmp_path)  # 生成独立 donut 版图
        configs = _load(  # 加载（负位移）
            _write_config(tmp_path, gds, displacement_nm=-5)
        )  # 覆盖位移
        final = single_pass.run_single_pass(*configs)  # 执行
        expected = _donut((25, 25, 95, 75), (37, 37, 77, 63))  # 手算期望（孔 30×16 扩 5）
        actual = _final_region(final)  # 实际覆盖
        assert (actual ^ expected).area() == 0  # XOR 零
        assert actual.count() == 1  # 单 polygon 带 hole


class TestArtifactContract:
    """产物唯一性与未处理层契约（设计文档 §5/§8）。"""

    def test_final_layout_is_the_only_artifact(self, tmp_path):
        """执行后输出目录树中仅存在 final_layout 一个文件。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        configs = _load(_write_config(tmp_path, gds))  # 加载
        single_pass.run_single_pass(*configs)  # 执行
        out_dir = tmp_path / "out"  # 输出目录
        files = [p for p in out_dir.rglob("*") if p.is_file()]  # 全部文件
        assert files == [out_dir / "final.gds"]  # 唯一产物

    def test_unprocessed_layers_are_not_copied(self, tmp_path):
        """源含 1/0 与 2/0 两层时，最终版图只保留目标层 1/0。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS（含 2/0 对照层）
        with LayoutDB.open(gds) as db:  # 回读源
            assert db.layers() == (LayerSpec(1, 0), LayerSpec(2, 0))  # 源确有两层
        configs = _load(_write_config(tmp_path, gds))  # 加载
        final = single_pass.run_single_pass(*configs)  # 执行
        with LayoutDB.open(final) as db:  # 回读最终版图
            assert db.layers() == (LayerSpec(1, 0),)  # 只有目标层被复制

    def test_macro_cells_mode_matches_single_cell_coverage(self, tmp_path):
        """macro_cells 与 single_cell 的顶层物理覆盖 XOR 为零。"""
        shared = tmp_path / "shared"  # 共用源版图目录
        shared.mkdir()  # 创建
        gds = _write_anchored_gds(shared)  # 生成一份源版图
        coverage = {}  # 模式 → 覆盖 Region
        for mode in ("single_cell", "macro_cells"):  # 两种模式
            base = tmp_path / mode  # 独立目录
            base.mkdir()  # 创建
            coverage[mode] = _final_region(  # 覆盖
                single_pass.run_single_pass(
                    *_load(  # 逐模式加载执行
                        _write_config(base, gds, final_cell_mode=mode)
                    )
                )
            )
        assert (coverage["single_cell"] ^ coverage["macro_cells"]).area() == 0  # XOR 零


class TestConfigValidation:
    """单遍配置的显式校验（设计文档 §4）。"""

    def test_displacement_off_grid_fails_with_parameter_name(self, tmp_path):
        """位移无法精确落 DBU 格点时失败并写明参数名。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        configs = _load(  # 0.5nm 无法落 1nm 格点
            _write_config(tmp_path, gds, displacement_nm=5.5)
        )  # 覆盖位移
        with pytest.raises(ValueError, match="displacement_nm"):  # 报错含参数名
            single_pass.run_single_pass(*configs)  # 执行

    def test_displacement_above_limit_fails(self, tmp_path):
        """位移绝对值超过 max_displacement 时阶段 0 校验失败。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        configs = _load(  # 9 > max_disp=8
            _write_config(tmp_path, gds, displacement_nm=9)
        )  # 覆盖位移
        with pytest.raises(ValueError, match="max_displacement"):  # 必须报错
            single_pass.run_single_pass(*configs)  # 执行

    def test_macro_entry_exclusivity(self, tmp_path):
        """macro_grid 与 macro_size_nm 同现或同缺都失败。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        both = _write_config(tmp_path, gds)  # 基础配置
        text = both.read_text(encoding="utf-8").replace(  # 注入尺寸模式
            "macro_grid = [2, 2]", "macro_grid = [2, 2]\nmacro_size_nm = 60"
        )  # 双入口
        both.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="恰好填写一个"):  # 必须报错
            _load(both)  # 统一加载（解析即校验）

    @pytest.mark.parametrize(
        ("original", "injected", "field"),
        [
            ("layer = 1", "layer = 1.5", "layer"),
            ("layer = 1", "layer = true", "layer"),
            ("layer = 1", 'layer = "1"', "layer"),
            ("datatype = 0", "datatype = 0.5", "datatype"),
            ("datatype = 0", "datatype = true", "datatype"),
            ("datatype = 0", 'datatype = "0"', "datatype"),
            ("canvas_pixels = 256", "canvas_pixels = 256.0", "canvas_pixels"),
            ("canvas_pixels = 256", "canvas_pixels = true", "canvas_pixels"),
            ("canvas_pixels = 256", 'canvas_pixels = "256"', "canvas_pixels"),
        ],
        ids=[
            "layer=1.5",
            "layer=true",
            "layer=str",
            "dt=0.5",
            "dt=true",
            "dt=str",
            "canvas=256.0",
            "canvas=true",
            "canvas=str",
        ],
    )
    def test_common_integer_fields_strictly_typed(self, tmp_path, original, injected, field):
        """公共整数字段拒绝 float/bool/string（审查 P1-3 回归）。

        修复前 single-pass 复制的解析层用裸 int()：1.5 截断为 1、true 当 1、
        字符串 "1" 被接受；公共段收敛到共享层后与权威实现同语义。
        """
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds)  # 基础配置
        text = path.read_text(encoding="utf-8").replace(original, injected)  # 注入
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match=field):  # 统一 ValueError 含字段名
            _load(path)  # 统一加载（解析即校验）

    def test_work_dir_optional_and_unused(self, tmp_path):
        """不填 work_dir 可加载（None 默认），单遍不产生工作目录产物。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        configs = _load(_write_config(tmp_path, gds))  # 模板无 work_dir
        assert configs[-1].work_dir is None  # 可选字段默认 None
        final = single_pass.run_single_pass(*configs)  # 执行
        assert final.is_file()  # 唯一产物照常

    def test_shared_rejection_single_loader(self, tmp_path):
        """同一非法公共输入在只请求 LayoutConfig 与全量请求下同语义拒绝。"""
        gds = _write_anchored_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds)  # 基础配置
        text = path.read_text(encoding="utf-8").replace(  # 注入浮点层号
            "layer = 1", "layer = 1.5"
        )  # 同款非法值
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="layer"):  # 单 Config 请求拒绝
            load_config(path, LayoutConfig)  # 只请求版图
        with pytest.raises(ValueError, match="layer"):  # 全量请求同样拒绝
            _load(path)  # 六 Config 全量
