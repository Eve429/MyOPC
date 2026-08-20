"""CurvMulti ILT 入口的配置契约、直跑与产物 schema 测试。"""

import json
import subprocess
import sys
from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

import main._ilt_workflow as ilt_workflow
import main.run_curvmulti_ilt as curvmulti_workflow


def _write_gds(tmp_path):
    """生成矩形 + hole + 斜边三角形的混合 GDS 并返回路径。

    目标层 bbox 恰为 (8,8)-(168,88)：160×80 均为 pixel 4 的整数倍；
    2×2 macro 网格下每宏 ownership 20×10 像素，整除 scales=[2,1] 且
    最粗控制网格（10×5）不小于 smoothing_kernel=3。
    """
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU，配置数值即 DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    shapes = top.shapes(layout.layer(1, 0))  # 目标层
    shapes.insert(kdb.Box(8, 8, 88, 88))  # 左侧实心矩形
    hole_region = (kdb.Region(kdb.Box(96, 8, 168, 88)) -  # 右侧外框
                   kdb.Region(kdb.Box(112, 24, 136, 72)))  # 中心 hole
    hole_region.insert_into(layout, top.cell_index(),
                            layout.layer(1, 0))  # 插入带孔图形
    shapes.insert(kdb.Polygon([(16, 80), (64, 80), (16, 86)]))  # 斜边三角
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_config(tmp_path, layout_path, macro_grid="[2, 2]", **overrides):
    """按默认契约生成 CurvMulti ILT TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值满足全部网格/尺度/迭代契约
        "macro_grid": macro_grid, "core_size_nm": 40, "context_nm": 20,
        "pixel_nm": 4, "scales": "[2, 1]", "iterations_per_stage": 1,
        "step_size": 0.5, "smoothing_kernel": 3, "sigmoid_steepness": 4.0,
        "sigmoid_offset": 0.5, "weight_process_l2": 1.0,
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

[curvmulti_ilt]
scales = {values["scales"]}
iterations_per_stage = {values["iterations_per_stage"]}
step_size = {values["step_size"]}
smoothing_kernel = {values["smoothing_kernel"]}
sigmoid_steepness = {values["sigmoid_steepness"]}
sigmoid_offset = {values["sigmoid_offset"]}
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
    """生成 GDS + 配置并执行一次完整 CurvMulti ILT，返回 (路径组, summary)。"""
    layout_path = _write_gds(tmp_path)
    config_path = _write_config(tmp_path, layout_path)
    summary = curvmulti_workflow.run_curvmulti_ilt(config_path)
    return tmp_path, summary


class TestConfigAndEntry:
    """配置契约与直接入口。"""

    def test_valid_run_completes(self, prepared):
        """合法配置完成全流程：method/macro 数/逐宏状态数正确。"""
        _, summary = prepared
        assert summary["method"] == "curvmulti_ilt"
        assert summary["macro_count"] == 4
        for macro in summary["macros"]:
            assert macro["state_count"] == 4  # 2 stage × (N+1)

    def test_missing_required_fails_before_prepare(self, tmp_path):
        """缺必填键在准备前失败：不创建工作目录。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "step_size = 0.5\n", "")  # 删除一个必填键
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="缺少必填键"):
            curvmulti_workflow.run_curvmulti_ilt(config_path)
        assert not (tmp_path / "work").exists()

    def test_unknown_key_fails(self, tmp_path):
        """[curvmulti_ilt] 段未知键在加载期暴露。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "[curvmulti_ilt]\n", "[curvmulti_ilt]\nunknown_key = 1\n")
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知键"):
            curvmulti_workflow.run_curvmulti_ilt(config_path)

    def test_unknown_section_fails(self, tmp_path):
        """未知段（拼写错误的算法段）被严格拒绝。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path)
        text = config_path.read_text(encoding="utf-8").replace(
            "[curvmulti_ilt]", "[curvmult_ilt]")
        config_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="未知配置段"):
            curvmulti_workflow.run_curvmulti_ilt(config_path)

    def test_scales_float_rejected(self, tmp_path):
        """scales 含浮点（[2.0, 1]）按整数注解拒绝。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path, scales="[2.0, 1]")
        with pytest.raises(ValueError, match="整数"):
            curvmulti_workflow.run_curvmulti_ilt(config_path)

    def test_scales_not_dividing_ownership_rejected(self, tmp_path):
        """scale 不整除宏 ownership：进入求解前失败。"""
        layout_path = _write_gds(tmp_path)
        config_path = _write_config(tmp_path, layout_path, scales="[3, 1]")
        with pytest.raises(ValueError, match="整除"):
            curvmulti_workflow.run_curvmulti_ilt(config_path)

    def test_out_of_repo_direct_run(self, prepared, tmp_path):
        """仓库外 cwd 直接运行入口脚本（免安装）。"""
        (tmp_path / "sub").mkdir()  # 先建目录再写 GDS
        layout_path = _write_gds(tmp_path / "sub")
        config_path = _write_config(tmp_path / "sub", layout_path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[2]
                                 / "main" / "run_curvmulti_ilt.py"),
             str(config_path)],
            cwd=str(tmp_path), capture_output=True, text=True,
            timeout=300, check=False)
        assert result.returncode == 0, result.stderr


class TestArtifacts:
    """产物 schema 与公共格式一致。"""

    def test_result_npz_and_metrics_schema(self, prepared):
        """curvmulti_ilt_result.npz 键集/dtype 与 metrics 的 stage 坐标。"""
        _, summary = prepared
        macro = summary["macros"][0]
        with np.load(macro["result_npz"]) as data:
            assert int(data["format_version"][0]) >= 1  # 公共结果格式
            assert data["best_parameters"].dtype == np.float32
            assert data["best_parameters"].shape == (10, 20)  # 40×80nm/4
            assert data["soft_mask"].shape == (10, 20)
            assert data["binary_mask"].dtype == np.uint8  # 持久化为 uint8
            assert int(data["best_state_index"][0]) == macro["best_state_index"]
        metrics = json.loads(Path(macro["metrics_json"]).read_text("utf-8"))
        stages = [(record["stage_index"], record["scale"])
                  for record in metrics["records"]]
        assert stages == [(0, 2), (0, 2), (1, 1), (1, 1)]
        assert [record["state_index"] for record in metrics["records"]] == [
            0, 1, 2, 3]

    def test_merge_called_exactly_once(self, prepared, monkeypatch):
        """全部 macro 完成后恰一次 merge（显式 macro_id→GDS 映射）。"""
        counts = {"merge": 0}
        real_merge = ilt_workflow.merge_macro_results

        def spy(*args, **kwargs):
            """计数后透传。"""
            counts["merge"] += 1
            return real_merge(*args, **kwargs)

        monkeypatch.setattr(ilt_workflow, "merge_macro_results", spy)
        tmp_path, _ = prepared  # prepared 已跑一次；再跑一次计 merge 次数
        (tmp_path / "second").mkdir()  # 先建目录再写 GDS
        layout_path = _write_gds(tmp_path / "second")
        config_path = _write_config(tmp_path / "second", layout_path)
        curvmulti_workflow.run_curvmulti_ilt(config_path)
        assert counts["merge"] == 1

    def test_evaluated_states_total(self):
        """evaluated_states = Σ(iterations_per_stage+1)。"""
        config = curvmulti_workflow.CurvMultiConfig(
            scales=(4, 2, 1), iterations_per_stage=2, step_size=0.5,
            smoothing_kernel=3, sigmoid_steepness=4.0, sigmoid_offset=0.5,
            weight_process_l2=1.0, weight_pvband=0.5, curvature_weight=0.0,
            mask_threshold=0.5, batch_size=4)
        assert curvmulti_workflow._evaluated_states(config) == 9
