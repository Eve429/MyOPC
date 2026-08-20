"""LevelSet ILT 入口端到端：配置解析、产物、final context 与异常收尾测试。"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import main._ilt_workflow as ilt_workflow
import main._levelset_ilt_workflow as levelset_workflow
import opc.iteration.ilt.levelset as levelset_module
from main import configuration

# 复用 Simple runner 的生成式版图构造（同一 workload 便于方法间对照）
from tests.main.test_simple_ilt_runner import _write_gds


def _write_config(tmp_path, layout_path, macro_grid="[2, 2]", **overrides):
    """按默认契约生成 LevelSet ILT TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值满足全部网格与迭代契约（step 取旧 LevelSet 默认）
        "macro_grid": macro_grid, "core_size_nm": 40, "context_nm": 20,
        "pixel_nm": 4, "iterations": 1, "step_size": 0.2,
        "weight_process_l2": 1.0, "weight_pvband": 0.5,
        "curvature_weight": 0.0, "batch_size": 4, "device": "cpu",
        "save_final_lithography": "false", "show_progress": "false",
        "final_cell_mode": "single_cell"}
    values.update(overrides)
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

[levelset_ilt]
iterations = {values["iterations"]}
step_size = {values["step_size"]}
weight_process_l2 = {values["weight_process_l2"]}
weight_pvband = {values["weight_pvband"]}
curvature_weight = {values["curvature_weight"]}
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
    return path


@pytest.fixture()
def prepared(tmp_path):
    """生成 GDS + 配置并执行一次完整 LevelSet ILT，返回 (路径组, summary)。"""
    layout_path = _write_gds(tmp_path)
    config_path = _write_config(tmp_path, layout_path)
    summary = levelset_workflow.run_levelset_ilt(config_path)
    return tmp_path, summary


class TestConfigAndEntry:
    """配置契约与直接入口（TEST-010 配置部分）。"""

    def test_valid_run_completes(self, prepared):
        """合法配置完成全流程，summary method=levelset_ilt。"""
        _, summary = prepared
        assert summary["method"] == "levelset_ilt"
        assert summary["macro_count"] == 4
        assert summary["iterations"] == 1
        assert summary["seam_strategy"] == "macro_independent_fixed_context"

    def test_missing_required_fails_before_prepare(self, tmp_path):
        """缺必填键在准备前失败：不创建工作目录。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "step_size = 0.2\n", "")  # 删除一个必填键
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="缺少必填键"):
            levelset_workflow.run_levelset_ilt(config_path)
        assert not (tmp_path / "work").exists()

    def test_unknown_key_fails(self, tmp_path):
        """段内未知键（含 Simple 专属字段）在加载期失败。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "[levelset_ilt]", "[levelset_ilt]\nsigmoid_steepness = 4.0")
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知键"):
            levelset_workflow.run_levelset_ilt(config_path)

    def test_unknown_section_fails(self, tmp_path):
        """未知 section 在加载期失败。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8") + "\n[nosuch]\nx = 1\n"
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知配置段"):
            levelset_workflow.run_levelset_ilt(config_path)

    def test_bool_rejected_for_int(self, tmp_path):
        """iterations 传 bool 冒充 int 被拒绝。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path, iterations="true")
        with pytest.raises(ValueError):
            levelset_workflow.run_levelset_ilt(config_path)

    def test_section_registered_in_configuration(self):
        """LevelSetILTConfig 注册到 [levelset_ilt] 段。"""
        assert (configuration.CONFIG_SECTIONS
                [levelset_workflow.LEVELSET_ILT_METHOD.config_type]
                == "levelset_ilt")

    def test_context_below_pixel_rejected_by_workflow(self, tmp_path):
        """context < 1 像素：solver 前置失败传播，无 summary。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path, context_nm="0")
        with pytest.raises(ValueError, match="context"):
            levelset_workflow.run_levelset_ilt(config_path)
        assert not (tmp_path / "work" / "summary.json").exists()

    def test_out_of_repo_direct_run(self, prepared, tmp_path):
        """仓库外 cwd 直接运行入口脚本（免安装）。"""
        (tmp_path / "sub").mkdir()
        layout_path = _write_gds(tmp_path / "sub")
        config_path = _write_config(tmp_path / "sub", layout_path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[2]
                                 / "main" / "run_levelset_ilt.py"),
             str(config_path)],
            cwd=str(tmp_path), capture_output=True, text=True,
            timeout=300, check=False)
        assert result.returncode == 0, result.stderr


class TestArtifacts:
    """端到端产物 schema 与公共格式一致性（TEST-010 产物部分）。"""

    def test_full_artifact_set_and_dtypes(self, prepared):
        """plan/result/metrics/best/final/summary 字段完整且与公共格式一致。"""
        tmp_path, _ = prepared
        work = tmp_path / "work"
        plan = json.loads((work / "ilt_plan.json").read_text(encoding="utf-8"))
        assert plan["format_version"] == 1
        assert (tmp_path / "final.gds").is_file()
        assert (work / "summary.json").is_file()
        for entry in plan["macros"]:
            macro_dir = work / "macros" / entry["macro_id"]
            result_path = macro_dir / "levelset_ilt_result.npz"
            metrics_path = macro_dir / "metrics.json"
            assert result_path.is_file() and metrics_path.is_file()
            assert (macro_dir / "best.gds").is_file()
            with np.load(result_path, allow_pickle=False) as data:
                assert int(data["format_version"][0]) == 1
                # LevelSet 语义：best_parameters 是 phi（正负距离场）
                assert data["best_parameters"].dtype == np.float32
                assert data["soft_mask"].dtype == np.float32
                assert data["binary_mask"].dtype == np.uint8
                phi = data["best_parameters"]
                # REQ-010：binary == (phi < 0)，soft == sigmoid(-phi)
                np.testing.assert_array_equal(
                    data["binary_mask"].astype(bool), phi < 0.0)
                np.testing.assert_allclose(
                    data["soft_mask"], 1.0 / (1.0 + np.exp(phi)),
                    rtol=1e-5, atol=1e-6)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert len(metrics["records"]) == 2  # iterations=1 → N+1
            first = metrics["records"][0]
            assert (first["state_index"] == 0 and first["stage_index"] == 0
                    and first["scale"] == 1)
            assert isinstance(metrics["binary_l2"], int)
            assert metrics["core_count"] == entry["core_count"]

    def test_state0_binary_equals_target_binary(self, prepared):
        """INV-004 端到端：state0 二值掩膜与 target 二值逐格一致。"""
        tmp_path, _ = prepared
        work = tmp_path / "work"
        plan = json.loads((work / "ilt_plan.json").read_text(encoding="utf-8"))
        from opc.input.pixel import PixelMacroProblem
        for entry in plan["macros"]:
            problem = PixelMacroProblem.load(Path(entry["problem_file"]))
            with np.load(work / "macros" / entry["macro_id"]
                         / "levelset_ilt_result.npz",
                         allow_pickle=False) as data:
                binary = data["binary_mask"].astype(bool)
                best_state = int(data["best_state_index"][0])
            pixel = problem.macro.pixel_dbu
            query = problem.macro.query_box
            box = problem.macro.ownership_box
            hm, wm = problem.ownership_shape
            r0 = (box.bottom - query.bottom) // pixel
            c0 = (box.left - query.left) // pixel
            target_binary = (
                problem.target_u8[r0:r0 + hm, c0:c0 + wm].astype(
                    np.float32) / 255.0 >= 0.5)
            if best_state == 0:  # best 是 state0 时可直接对靶断言
                assert np.array_equal(binary, target_binary)


class TestFinalContext:
    """终评 fixed-context 策略与 SDF 不重复（TEST-009）。"""

    def test_final_context_does_not_rerun_sdf(self, tmp_path, monkeypatch):
        """全流程 SDF 调用恰 macro 次：终评不触发额外初始化。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        calls = {"n": 0}
        real = levelset_module.signed_distance_initialization

        def spy(*args, **kwargs):
            """计数透传。"""
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(
            levelset_module, "signed_distance_initialization", spy)
        summary = levelset_workflow.run_levelset_ilt(config_path)
        assert calls["n"] == summary["macro_count"]  # 仅 solver 初始化

    def test_workflow_has_no_method_math_fields(self):
        """ILTMethod 的公共 workflow 消费面不含方法数学字段。"""
        method = levelset_workflow.LEVELSET_ILT_METHOD
        field_names = {field.name for field in dataclasses.fields(method)}
        assert field_names == {
            "method_name", "config_type", "optimize_macro",
            "evaluated_states", "build_fixed_context_canvas"}
        workflow_fields = {
            field.name for field in dataclasses.fields(
                ilt_workflow.ILTMethod)}
        assert workflow_fields == field_names
