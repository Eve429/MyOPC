"""Simple ILT 入口端到端：配置解析、产物、合并与异常收尾的生成式测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

import main._ilt_workflow as ilt_workflow
import main._simple_ilt_workflow as simple_workflow
from layout import DbuBox, LayerSpec, LayoutDB
from main import configuration
from main.configuration import load_config

_TARGET_LAYER = LayerSpec(1, 0)  # 生成式版图唯一目标层


def _write_gds(tmp_path):
    """生成矩形 + hole + 斜边三角形的混合 GDS 并返回路径。

    目标层 bbox 恰为 (8,8)-(152,88)：144×80 均为 pixel 4 的整数倍
    （像素 ILT 的实际 box 整像素契约）；斜边保证存在分数覆盖格。
    """
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU，配置数值即 DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    shapes = top.shapes(layout.layer(1, 0))  # 目标层
    shapes.insert(kdb.Box(8, 8, 88, 88))  # 左侧实心矩形
    hole_region = (kdb.Region(kdb.Box(96, 8, 152, 88)) -  # 右侧外框
                   kdb.Region(kdb.Box(112, 24, 136, 72)))  # 中心 hole
    hole_region.insert_into(layout, top.cell_index(),
                            layout.layer(1, 0))  # 插入带孔图形
    shapes.insert(kdb.Polygon([(16, 92), (64, 92), (16, 96)]))  # 斜边三角
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_config(tmp_path, layout_path, macro_grid="[2, 2]", **overrides):
    """按默认契约生成 Simple ILT TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值满足全部网格与迭代契约
        "macro_grid": macro_grid, "core_size_nm": 40, "context_nm": 20,
        "pixel_nm": 4, "iterations": 1, "step_size": 10.0,
        "sigmoid_steepness": 4.0, "weight_process_l2": 1.0,
        "weight_pvband": 0.5, "curvature_weight": 0.0,
        "mask_threshold": 0.5, "batch_size": 4, "device": "cpu",
        "save_final_lithography": "false", "show_progress": "false",
        "final_cell_mode": "single_cell"}
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
device = "{values["device"]}"

[simple_ilt]
iterations = {values["iterations"]}
step_size = {values["step_size"]}
sigmoid_steepness = {values["sigmoid_steepness"]}
weight_process_l2 = {values["weight_process_l2"]}
weight_pvband = {values["weight_pvband"]}
curvature_weight = {values["curvature_weight"]}
mask_threshold = {values["mask_threshold"]}
batch_size = {values["batch_size"]}

[output]
save_final_lithography = {values["save_final_lithography"]}
show_progress = {values["show_progress"]}
work_dir = "{(tmp_path / "work").as_posix()}"
final_layout = "{(tmp_path / "final.gds").as_posix()}"
final_cell_mode = "{values["final_cell_mode"]}"
"""
    path = tmp_path / "config.toml"  # 配置路径
    path.write_text(text, encoding="utf-8")  # 写盘
    return path  # 返回路径


@pytest.fixture()
def prepared(tmp_path):
    """生成 GDS + 配置并执行一次完整 Simple ILT，返回 (路径组, summary)。"""
    layout_path = _write_gds(tmp_path)
    config_path = _write_config(tmp_path, layout_path)
    summary = simple_workflow.run_simple_ilt(config_path)
    return tmp_path, summary


class TestConfigAndEntry:
    """配置契约与直接入口（TEST-011）。"""

    def test_valid_run_completes(self, prepared):
        """合法配置完成全流程，summary method=simple_ilt。"""
        _, summary = prepared
        assert summary["method"] == "simple_ilt"
        assert summary["macro_count"] == 4

    def test_missing_required_fails_before_prepare(self, tmp_path):
        """缺必填键在准备前失败：不创建工作目录。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "step_size = 10.0\n", "")  # 抠除一个必填键
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="缺少必填键"):
            simple_workflow.run_simple_ilt(config_path)
        assert not (tmp_path / "work").exists()

    def test_unknown_key_fails(self, tmp_path):
        """段内未知键（拼写错误）在加载期失败。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "[simple_ilt]", "[simple_ilt]\nunknown_key = 1")
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知键"):
            simple_workflow.run_simple_ilt(config_path)

    def test_unknown_section_fails(self, tmp_path):
        """未知 section 在加载期失败。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8") + "\n[nosuch]\nx = 1\n"
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知配置段"):
            simple_workflow.run_simple_ilt(config_path)

    def test_float_rejected_for_int(self, tmp_path):
        """iterations 传 float 冒充 int 被拒绝。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path, iterations="1.5")
        with pytest.raises(ValueError, match="整数"):
            simple_workflow.run_simple_ilt(config_path)

    def test_edge_section_not_required(self, tmp_path):
        """像素 ILT 不要求 [edge]：无该段的配置正常加载运行。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8")
        assert "[edge]" not in text  # 生成器本就不含 edge
        summary = simple_workflow.run_simple_ilt(config_path)
        assert summary["method"] == "simple_ilt"

    def test_get_type_hints_supports_postponed_annotations(self, tmp_path):
        """启用 postponed annotations 的外部 Config 可直接注册解析。"""
        # 本测试模块启用 from __future__ import annotations，类内注解
        # 均为字符串——正是 IF-003 要支持的场景。
        @dataclass(frozen=True, slots=True)
        class _PostponedConfig:
            """临时探针 Config：覆盖 int/float/Path/tuple 四类注解。"""
            count: int
            ratio: float
            base: Path
            pair: tuple[int, int]

        saved_sections = dict(configuration.CONFIG_SECTIONS)
        try:
            configuration.CONFIG_SECTIONS[_PostponedConfig] = "probe"
            configuration._SECTION_TO_TYPE.clear()
            configuration._SECTION_TO_TYPE.update(
                {v: k for k, v in configuration.CONFIG_SECTIONS.items()})
            path = tmp_path / "probe.toml"
            path.write_text(
                "[probe]\ncount = 3\nratio = 0.5\nbase = \"x\"\npair = [1, 2]\n",
                encoding="utf-8")
            parsed, = load_config(path, _PostponedConfig)
            assert parsed.count == 3
            assert parsed.ratio == 0.5
            assert parsed.base == (tmp_path / "x").resolve()
            assert parsed.pair == (1, 2)
        finally:  # 测试后复原注册表
            configuration.CONFIG_SECTIONS.clear()
            configuration.CONFIG_SECTIONS.update(saved_sections)
            configuration._SECTION_TO_TYPE.clear()
            configuration._SECTION_TO_TYPE.update(
                {v: k for k, v in configuration.CONFIG_SECTIONS.items()})

    def test_out_of_repo_direct_run(self, prepared, tmp_path):
        """仓库外 cwd 直接运行入口脚本（免安装）。"""
        (tmp_path / "sub").mkdir()  # 先建目录再写 GDS
        layout_path = _write_gds(tmp_path / "sub")
        config_path = _write_config(tmp_path / "sub", layout_path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[2]
                                 / "main" / "run_simple_ilt.py"),
             str(config_path)],
            cwd=str(tmp_path), capture_output=True, text=True,
            timeout=300, check=False)
        assert result.returncode == 0, result.stderr


class TestArtifacts:
    """端到端产物字段与 dtype（TEST-012）。"""

    def test_full_artifact_set_and_dtypes(self, prepared):
        """plan/problem/result/metrics/best/final/summary 字段完整。"""
        tmp_path, summary = prepared
        work = tmp_path / "work"
        plan = json.loads((work / "ilt_plan.json").read_text(encoding="utf-8"))
        assert plan["format_version"] == 1
        assert {key in plan for key in (
            "layer", "dbu_um", "polarity", "pixel_dbu", "canvas_pixels",
            "core_size_dbu", "context_dbu", "macros")} == {True}
        assert (tmp_path / "final.gds").is_file()
        assert (work / "summary.json").is_file()
        assert summary["seam_strategy"] == "macro_independent_fixed_context"
        for entry in plan["macros"]:
            macro_dir = work / "macros" / entry["macro_id"]
            result_path = macro_dir / "simple_ilt_result.npz"
            metrics_path = macro_dir / "metrics.json"
            assert result_path.is_file() and metrics_path.is_file()
            assert (macro_dir / "best.gds").is_file()
            with np.load(result_path, allow_pickle=False) as data:
                assert int(data["format_version"][0]) == 1
                assert data["best_parameters"].dtype == np.float32
                assert data["soft_mask"].dtype == np.float32
                assert data["binary_mask"].dtype == np.uint8
                assert data["best_parameters"].ndim == 2
                assert data["binary_mask"].shape == data["best_parameters"].shape
                assert int(data["best_state_index"][0]) >= 0
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert len(metrics["records"]) == 2  # iterations=1 → N+1
            first = metrics["records"][0]
            assert (first["state_index"] == 0 and first["stage_index"] == 0
                    and first["scale"] == 1)
            assert isinstance(metrics["binary_l2"], int)


class TestMergeAndSeam:
    """多 macro 合并与 seam ownership（TEST-013）。"""

    def test_merge_called_exactly_once(self, prepared, monkeypatch):
        """merge_macro_results 全流程恰调用一次。"""
        count = {"n": 0}
        real = ilt_workflow.merge_macro_results

        def spy(*args, **kwargs):
            """计数并透传。"""
            count["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(ilt_workflow, "merge_macro_results", spy)
        tmp_path = _write_gds(prepared[0].parent)
        config_path = _write_config(prepared[0].parent, tmp_path)
        simple_workflow.run_simple_ilt(config_path)
        assert count["n"] == 1

    def test_final_raster_equals_owned_binary_union(self, prepared):
        """最终版图栅格等于全部 macro ownership 二值输出精确拼接。"""
        tmp_path, _ = prepared
        work = tmp_path / "work"
        plan = json.loads((work / "ilt_plan.json").read_text(encoding="utf-8"))
        pixel = int(plan["pixel_dbu"])
        left = min(e["ownership_box"][0] for e in plan["macros"])
        bottom = min(e["ownership_box"][1] for e in plan["macros"])
        right = max(e["ownership_box"][2] for e in plan["macros"])
        top = max(e["ownership_box"][3] for e in plan["macros"])
        expected = np.zeros(((top - bottom) // pixel, (right - left) // pixel),
                            dtype=np.bool_)
        covered = np.zeros_like(expected)  # ownership 覆盖标记（无遗漏检查）
        for entry in plan["macros"]:
            macro_id = entry["macro_id"]
            with np.load(work / "macros" / macro_id / "simple_ilt_result.npz",
                         allow_pickle=False) as data:
                binary = data["binary_mask"].astype(np.bool_)
            box = entry["ownership_box"]
            r0 = (box[1] - bottom) // pixel
            c0 = (box[0] - left) // pixel
            block = expected[r0:r0 + binary.shape[0], c0:c0 + binary.shape[1]]
            block |= binary  # macro 间 ownership 不重叠，直接或入
            covered[r0:r0 + binary.shape[0],
                    c0:c0 + binary.shape[1]] = True
        assert covered.all()  # 全部像素恰属于一个 macro
        with LayoutDB.open(tmp_path / "final.gds") as database:
            layer = LayerSpec(plan["layer"][0], plan["layer"][1])
            bounds = database.layer_bbox(layer)
            region = (database.query([layer], bounds)
                      .materialize_intersecting().region(layer))
        from opc.input import rasterize_region_window
        coverage = rasterize_region_window(
            region, DbuBox(left, bottom, right, top), pixel)
        assert np.array_equal(coverage >= 0.5, expected)


class TestExceptionCleanup:
    """进度与异常收尾（TEST-014）。"""

    def test_second_macro_failure_keeps_no_summary(self, tmp_path,
                                                   monkeypatch):
        """第二 macro 中途异常：原样传播，无 summary，首 macro 产物留诊断。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        real = ilt_workflow.reconstruct_pixel_region
        seen = {"n": 0}

        def flaky(problem, binary):
            """第二次调用起失败（第二个 macro）。"""
            seen["n"] += 1
            if seen["n"] >= 2:
                raise RuntimeError("boom")
            return real(problem, binary)

        monkeypatch.setattr(ilt_workflow, "reconstruct_pixel_region", flaky)
        with pytest.raises(RuntimeError, match="boom"):
            simple_workflow.run_simple_ilt(config_path)
        work = tmp_path / "work"
        assert not (work / "summary.json").exists()  # 不发布完成摘要
        assert not (tmp_path / "final.gds").exists()  # 不合并最终版图
        assert (work / "macros" / "mr0c0" / "best.gds").is_file()  # 留诊断
