"""MultilevelILT 的级别监督、物理尺度、窗口、反向和入口专项测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

from layout import LayerSpec
from lithography import ICCAD13Lithography, ProcessCondition
from main.run_ilt import run_ilt
from opc.iteration.ilt import MultilevelConfig, optimize_multilevel
from tests.opc import FlatWaferLithography, RecordingLithography, write_ilt_gds


def test_multilevel_rejects_invalid_config_and_inputs() -> None:
    """错配级别配置、非法数值、不可整除画布和重复条件必须提前拒绝。"""
    invalid_configs = (
        {"scales": ()}, {"scales": (1, 2)}, {"scales": (2,)},
        {"stage_iterations": (1,)}, {"stage_step_sizes": (0.1,)},
        {"smoothing_kernel": 4}, {"stage_step_sizes": (0.2, float("nan"))},
    )
    for kwargs in invalid_configs:
        with pytest.raises(ValueError):
            MultilevelConfig(**kwargs)
    with pytest.raises(ValueError, match="整除"):
        optimize_multilevel(
            torch.zeros((15, 16)), RecordingLithography(),
            MultilevelConfig((2, 1), (1, 1), (0.1, 0.1), 3))
    invalid = torch.zeros((16, 16)); invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="有限数"):
        optimize_multilevel(
            invalid, RecordingLithography(),
            MultilevelConfig((1,), (1,), (0.1,), 3))
    duplicate = ProcessCondition("same", "focus", 1.0)
    with pytest.raises(ValueError, match="名称不能重复"):
        optimize_multilevel(
            torch.zeros((16, 16)), RecordingLithography(),
            MultilevelConfig((1,), (1,), (0.1,), 3),
            nominal_condition=duplicate, process_conditions=(duplicate,))


def test_multilevel_coarse_loss_uses_level_supervision_grid() -> None:
    """粗级首轮损失必须在降采样监督网格计算，而不是完整网格损失换名。"""
    target = torch.zeros((16, 16)); target[3:13, 5:11] = 1.0
    config = MultilevelConfig(
        (2, 1), (1, 1), (0.1, 0.1), 3, 5.0, 0.4,
        weight_pvband=0.0)
    result = optimize_multilevel(
        target, RecordingLithography(), config, process_conditions=())
    stage_target = functional.interpolate(
        target[None, None], size=(8, 8), mode="area")[0, 0]
    stage_mask = torch.sigmoid(5.0 * (
        functional.avg_pool2d(stage_target[None, None], 3, 1, 1)[0, 0] - 0.4))
    expected = float(torch.sum((stage_mask - stage_target).square()))
    assert result.records[0].nominal_l2 == pytest.approx(expected)


def test_multilevel_uses_independent_adam_levels_and_reduces_loss() -> None:
    """两级应产生各自记录、完成参数 warm-start，并在恒等光刻下继续优化。"""
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_multilevel(
        target, RecordingLithography(),
        MultilevelConfig((2, 1), (2, 3), (0.2, 0.1), 3,
                         weight_pvband=0.0), process_conditions=())
    assert [record.iteration for record in result.records] == list(range(5))
    assert result.records[1].total_loss < result.records[0].total_loss
    assert result.records[-1].total_loss < result.records[2].total_loss
    assert result.best_parameters.shape == target.shape
    assert not torch.equal(result.best_parameters, target)


@pytest.mark.parametrize("pattern", ("hole", "diagonal", "cross", "multiple"))
def test_multilevel_handles_diverse_target_geometry(pattern: str) -> None:
    """孔洞、斜边、十字和多组件目标必须通过全部级别且保持有限同形。"""
    target = torch.zeros((16, 16))
    if pattern == "hole":
        target[2:14, 2:14] = 1.0; target[6:10, 6:10] = 0.0
    elif pattern == "diagonal":
        target = torch.tril(torch.ones((16, 16)), diagonal=-1)
    elif pattern == "cross":
        target[6:10, 2:14] = 1.0; target[2:14, 6:10] = 1.0
    else:
        target[2:6, 2:7] = 1.0; target[9:14, 10:14] = 1.0
    result = optimize_multilevel(
        target, RecordingLithography(),
        MultilevelConfig((2, 1), (1, 1), (0.1, 0.1), 3))
    assert result.soft_mask.shape == target.shape
    assert torch.all(torch.isfinite(result.soft_mask))


def test_multilevel_all_levels_keep_full_physical_optical_grid() -> None:
    """所有级别的 Hopkins 调用都必须使用完整物理网格，不能把粗图居中补零。"""
    model = RecordingLithography()
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_multilevel(
        target, model,
        MultilevelConfig((4, 2, 1), (1, 2, 1), (0.1, 0.1, 0.1), 3),
        process_conditions=())
    assert model.shapes == [(1, 16, 16)] * 4
    assert len(result.records) == 4


def test_multilevel_fixed_area_and_empty_process_window() -> None:
    """空工艺窗口只算 nominal，窗口外输出必须保持完整初始软 mask。"""
    model = RecordingLithography()
    target = torch.zeros((16, 16)); target[3:13, 3:13] = 1.0
    movable = torch.zeros_like(target); movable[5:11, 5:11] = 1.0
    config = MultilevelConfig((2, 1), (1, 1), (0.1, 0.1), 3)
    result = optimize_multilevel(
        target, model, config, optimization_mask=movable,
        process_conditions=())
    initial_soft = torch.sigmoid(4.0 * (
        functional.avg_pool2d(target[None, None], 3, 1, 1)[0, 0] - 0.5))
    assert torch.equal(result.soft_mask[movable == 0], initial_soft[movable == 0])
    assert all(names == ("nominal",) for names in model.names)
    assert all(record.process_l2 == 0.0 and record.pvband_loss == 0.0
               for record in result.records)


def test_multilevel_curvature_is_computed_on_level_wafer() -> None:
    """空间恒定 wafer 的级别曲率必须为零，即使输入软 mask 含明显边界。"""
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_multilevel(
        target, FlatWaferLithography(),
        MultilevelConfig((2, 1), (1, 1), (0.1, 0.1), 3,
                         curvature_weight=10.0), process_conditions=())
    assert all(record.curvature_loss == 0.0 for record in result.records)


def test_multilevel_supports_batch_and_real_hopkins_backward() -> None:
    """批量 target 应保留 batch 维，真实 Hopkins 应完成两级反向。"""
    target = torch.zeros((2, 16, 16)); target[0, 4:12, 4:12] = 1.0
    target[1, 5:11, 3:13] = 1.0
    batch_result = optimize_multilevel(
        target, RecordingLithography(),
        MultilevelConfig((2, 1), (1, 1), (0.1, 0.1), 3),
        process_conditions=())
    assert batch_result.binary_mask.shape == target.shape
    real_result = optimize_multilevel(
        target[0], ICCAD13Lithography(device="cpu"),
        MultilevelConfig((2, 1), (1, 1), (1e-3, 1e-3), 3,
                         curvature_weight=0.1))
    assert len(real_result.records) == 2
    assert torch.all(torch.isfinite(real_result.best_parameters))


def test_multilevel_runner_accepts_gds_and_saves_complete_artifacts(tmp_path: Path) -> None:
    """统一入口应直接读取 GDS，并保存逐级配置、评价、光刻和资源记录。"""
    source = tmp_path / "multilevel.gds"; write_ilt_gds(source)
    output = tmp_path / "output"
    summary = run_ilt(
        source, output, method="multilevel", scales=(2, 1),
        stage_iterations=(1, 2), stage_step_sizes=(1e-3, 2e-3),
        smoothing_kernel=3, device="cpu", save_png=True,
        layer=LayerSpec(1, 0), box=(0, 0, 64, 64), pixel_nm=1.0, canvas=256)
    assert summary["method"] == "multilevel"
    assert summary["config"]["stage_iterations"] == (1, 2)
    assert len(summary["records"]) == 3
    assert [record["stage_scale"] for record in summary["records"]] == [2, 1, 1]
    assert [record["stage_iteration"] for record in summary["records"]] == [0, 0, 1]
    assert {"start", "input", "model", "optimization", "evaluation", "output"} <= set(
        summary["memory_checkpoints"])
    assert Path(summary["artifacts"]["result_npz"]).is_file()
    assert (output / "final_lithography.npz").is_file()
    with np.load(output / "ilt_result.npz", allow_pickle=False) as data:
        assert str(data["method"].item()) == "multilevel"
        assert data["soft_mask"].shape == (256, 256)


def test_multilevel_runner_resolves_method_defaults_and_custom_scale_contract(
        tmp_path: Path) -> None:
    """默认两级配置必须可追溯，自定义尺度不得静默猜测各级迭代数。"""
    source = tmp_path / "defaults.gds"; write_ilt_gds(source)
    # 用显式零轮之外的最小真实调用会执行默认 120 轮，故这里只从默认配置类和入口
    # 的自定义错误路径共同锁定契约；默认类正是入口无覆盖时构造的数值来源。
    assert MultilevelConfig().scales == (2, 1)
    assert MultilevelConfig().stage_iterations == (20, 100)
    assert MultilevelConfig().stage_step_sizes == (0.2, 0.2)
    with pytest.raises(ValueError, match="自定义 Multilevel scales"):
        run_ilt(
            source, tmp_path / "invalid", method="multilevel", scales=(4, 2, 1),
            smoothing_kernel=3, device="cpu", save_png=False,
            layer=LayerSpec(1, 0), box=(0, 0, 64, 64),
            pixel_nm=1.0, canvas=256)


def test_multilevel_cli_runs_from_outside_repository(tmp_path: Path) -> None:
    """用户应能从仓库外直接运行 Multilevel GDS 流程并读取 JSON 产物。"""
    source = tmp_path / "cli.gds"; write_ilt_gds(source)
    output = tmp_path / "cli_output"
    script = Path(__file__).resolve().parents[2] / "main" / "run_ilt.py"
    command = [sys.executable, str(script), str(source), "--output-dir", str(output),
               "--method", "multilevel", "--scales", "2", "1",
               "--stage-iterations", "1", "1", "--stage-step-sizes", "0.001", "0.001",
               "--smoothing-kernel", "3", "--device", "cpu", "--layer", "1/0",
               "--box", "0", "0", "64", "64", "--pixel-nm", "1", "--canvas", "256",
               "--no-png", "--json"]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert summary["config"]["stage_iterations"] == [1, 1]
    assert Path(summary["artifacts"]["final_lithography"]["npz"]).is_file()
