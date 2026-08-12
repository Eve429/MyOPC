"""CurvMultiILT 的参数化、尺度、窗口、光刻反向和入口专项测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import klayout.db as kdb
import numpy as np
import pytest
import torch
from torch.nn import functional

from layout import LayerSpec
from lithography import ICCAD13Lithography, ProcessCondition
from main.run_ilt import run_ilt
from opc.iteration.ilt import CurvMultiConfig, optimize_curvmulti


class _RecordingLithography:
    """以廉价可微映射记录 CurvMulti 送入模型的 shape 和条件。"""

    def __init__(self, canvas: int = 64) -> None:
        """建立 CPU 设备、固定 canvas 和调用记录。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas)
        self.shapes: list[tuple[int, ...]] = []
        self.names: list[tuple[str, ...]] = []

    def condition(self, name: str) -> ProcessCondition:
        """按默认名称返回独立伪工艺条件。"""
        return ProcessCondition(
            name, "defocus" if name == "defocus_min" else "focus", 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """记录完整网格并返回保持梯度的条件偏移图。"""
        self.shapes.append(tuple(mask.shape))
        self.names.append(tuple(condition.name for condition in conditions))
        offsets = {"dose_max": 0.04, "defocus_min": -0.04}
        return {condition.name: torch.clamp(
            mask + offsets.get(condition.name, 0.0), 0.0, 1.0)
                for condition in conditions}


class _FlatWaferLithography(_RecordingLithography):
    """返回与 mask 保持计算图关系但空间上恒定的 wafer。"""

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """用零倍 mask 证明曲率作用对象是 wafer 而非 mask。"""
        self.shapes.append(tuple(mask.shape))
        self.names.append(tuple(condition.name for condition in conditions))
        return {condition.name: mask * 0.0 for condition in conditions}


def test_curvmulti_rejects_invalid_config_and_inputs() -> None:
    """无序尺度、偶数平滑核、非法权重和不可整除画布必须提前拒绝。"""
    for kwargs in ({"scales": ()}, {"scales": (1, 2)}, {"scales": (2,)},
                   {"smoothing_kernel": 4}, {"step_size": 0.0},
                   {"curvature_weight": float("nan")}):
        with pytest.raises(ValueError):
            CurvMultiConfig(**kwargs)
    with pytest.raises(ValueError, match="整除"):
        optimize_curvmulti(
            torch.zeros((15, 16)), _RecordingLithography(),
            CurvMultiConfig((2, 1), iterations_per_stage=1, curvature_weight=0.0))
    invalid = torch.zeros((16, 16)); invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="有限数"):
        optimize_curvmulti(
            invalid, _RecordingLithography(),
            CurvMultiConfig((1,), iterations_per_stage=1, curvature_weight=0.0))
    invalid_mask = torch.ones((16, 16)); invalid_mask[0, 0] = -1.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        optimize_curvmulti(
            torch.zeros((16, 16)), _RecordingLithography(),
            CurvMultiConfig((1,), iterations_per_stage=1, curvature_weight=0.0),
            optimization_mask=invalid_mask)
    duplicate = ProcessCondition("same", "focus", 1.0)
    with pytest.raises(ValueError, match="名称不能重复"):
        optimize_curvmulti(
            torch.zeros((16, 16)), _RecordingLithography(),
            CurvMultiConfig((1,), iterations_per_stage=1, curvature_weight=0.0),
            nominal_condition=duplicate, process_conditions=(duplicate,))


def test_curvmulti_single_stage_matches_smooth_sigmoid_formula() -> None:
    """单阶段首轮软 mask 必须等于平均池化加带偏移 sigmoid 的定义式。"""
    target = torch.zeros((16, 16)); target[4:12, 5:11] = 1.0
    config = CurvMultiConfig(
        (1,), 1, 0.1, 3, 5.0, 0.4, 0.0, 0.0, 0.0, 0.5)
    result = optimize_curvmulti(
        target, _RecordingLithography(), config, process_conditions=())
    pooled = functional.avg_pool2d(target[None, None], 3, 1, 1)[0, 0]
    expected = torch.sigmoid(5.0 * (pooled - 0.4))
    assert torch.allclose(result.soft_mask, expected)
    assert result.records[0].nominal_l2 == pytest.approx(
        float(torch.sum((expected - target).square())))


def test_curvmulti_optimization_reduces_identity_loss() -> None:
    """廉价恒等光刻下连续参数应实际更新，并在多轮内降低复合损失。"""
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_curvmulti(
        target, _RecordingLithography(),
        CurvMultiConfig((1,), 3, 0.5, 3, curvature_weight=0.0),
        process_conditions=())
    losses = [record.total_loss for record in result.records]
    assert losses[-1] < losses[0]
    assert not torch.equal(result.best_parameters, target)


@pytest.mark.parametrize("pattern", ("hole", "diagonal", "cross", "multiple"))
def test_curvmulti_handles_diverse_target_geometry(pattern: str) -> None:
    """孔洞、斜边、十字和多组件目标在粗到细路径中都必须保持有限且同形。"""
    target = torch.zeros((16, 16))
    if pattern == "hole":
        target[2:14, 2:14] = 1.0; target[6:10, 6:10] = 0.0
    elif pattern == "diagonal":
        target = torch.tril(torch.ones((16, 16)), diagonal=-1)
    elif pattern == "cross":
        target[6:10, 2:14] = 1.0; target[2:14, 6:10] = 1.0
    else:
        target[2:6, 2:7] = 1.0; target[9:14, 10:14] = 1.0
    result = optimize_curvmulti(
        target, _RecordingLithography(),
        CurvMultiConfig((2, 1), 1, 0.1, 3, curvature_weight=0.0),
        process_conditions=())
    assert result.soft_mask.shape == target.shape
    assert torch.all(torch.isfinite(result.soft_mask))


def test_curvmulti_coarse_control_grid_keeps_full_optical_grid() -> None:
    """粗尺度只能减少控制变量，所有光刻调用仍须使用完整物理像素网格。"""
    model = _RecordingLithography()
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_curvmulti(
        target, model,
        CurvMultiConfig((2, 1), 1, 0.1, 3, curvature_weight=0.0),
        process_conditions=())
    assert model.shapes == [(1, 16, 16), (1, 16, 16)]
    assert len(result.records) == 2
    assert result.best_parameters.shape == target.shape


def test_curvmulti_fixed_area_and_empty_process_window() -> None:
    """显式空工艺窗口只算 nominal，窗口外软 mask 必须保持初始参考值。"""
    model = _RecordingLithography()
    target = torch.zeros((16, 16)); target[3:13, 3:13] = 1.0
    movable = torch.zeros_like(target); movable[5:11, 5:11] = 1.0
    config = CurvMultiConfig((2, 1), 1, 0.1, 3, curvature_weight=0.0)
    result = optimize_curvmulti(
        target, model, config, optimization_mask=movable,
        process_conditions=())
    initial_soft = torch.sigmoid(
        config.sigmoid_steepness *
        (functional.avg_pool2d(target[None, None], 3, 1, 1)[0, 0] -
         config.sigmoid_offset))
    assert torch.equal(result.soft_mask[movable == 0], initial_soft[movable == 0])
    assert all(names == ("nominal",) for names in model.names)
    assert all(record.process_l2 == 0.0 and record.pvband_loss == 0.0
               for record in result.records)


def test_curvmulti_curvature_is_computed_on_nominal_wafer() -> None:
    """空间恒定 wafer 的曲率必须为零，即使输入软 mask 含有明显边界。"""
    target = torch.zeros((16, 16)); target[4:12, 4:12] = 1.0
    result = optimize_curvmulti(
        target, _FlatWaferLithography(),
        CurvMultiConfig((1,), 1, 0.1, 3, curvature_weight=10.0),
        process_conditions=())
    assert result.records[0].curvature_loss == 0.0


def test_curvmulti_supports_batch_and_real_hopkins_backward() -> None:
    """批量 target 应保持 batch 维，真实 Hopkins 模型应完成全部尺度反向。"""
    target = torch.zeros((2, 16, 16)); target[0, 4:12, 4:12] = 1.0
    target[1, 5:11, 3:13] = 1.0
    batch_result = optimize_curvmulti(
        target, _RecordingLithography(),
        CurvMultiConfig((2, 1), 1, 0.1, 3, curvature_weight=0.0),
        process_conditions=())
    assert batch_result.binary_mask.shape == target.shape
    real_result = optimize_curvmulti(
        target[0], ICCAD13Lithography(device="cpu"),
        CurvMultiConfig((2, 1), 1, 1e-3, 3, curvature_weight=0.1))
    assert len(real_result.records) == 2
    assert torch.all(torch.isfinite(real_result.best_parameters))


def _write_curvmulti_gds(path: Path) -> None:
    """写入供函数入口和仓库外 CLI 共用的确定性矩形版图。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer_index = layout.layer(1, 0); top = layout.create_cell("TOP")
    top.shapes(layer_index).insert(kdb.Box(8, 8, 40, 40))
    layout.write(str(path))


def test_curvmulti_runner_accepts_gds_and_saves_complete_artifacts(tmp_path: Path) -> None:
    """统一入口应直接读取 GDS，并保存配置、评价、最终光刻和阶段记录。"""
    source = tmp_path / "curvmulti.gds"; _write_curvmulti_gds(source)
    output = tmp_path / "output"
    summary = run_ilt(
        source, output, method="curvmulti", iterations=1, step_size=1e-3,
        scales=(4, 2, 1), smoothing_kernel=3, curvature_weight=0.1,
        device="cpu", save_png=True, layer=LayerSpec(1, 0),
        box=(0, 0, 64, 64), pixel_nm=1.0, canvas=256)
    assert summary["method"] == "curvmulti"
    # Python API 保留 dataclass 的 tuple 不变量；写入 JSON 后才自然序列化为 list。
    assert summary["config"]["scales"] == (4, 2, 1)
    assert len(summary["records"]) == 3
    assert [record["stage_scale"] for record in summary["records"]] == [4, 2, 1]
    assert {"start", "input", "model", "optimization", "evaluation", "output"} <= set(
        summary["memory_checkpoints"])
    assert all(checkpoint["rss_bytes"] >= 0
               for checkpoint in summary["memory_checkpoints"].values())
    assert Path(summary["artifacts"]["result_npz"]).is_file()
    assert (output / "final_lithography.npz").is_file()
    with np.load(output / "ilt_result.npz", allow_pickle=False) as data:
        assert str(data["method"].item()) == "curvmulti"
        assert data["soft_mask"].shape == (256, 256)


def test_curvmulti_cli_runs_from_outside_repository(tmp_path: Path) -> None:
    """用户应能在仓库外直接运行 CurvMulti 并读取 JSON 产物索引。"""
    source = tmp_path / "cli.gds"; _write_curvmulti_gds(source)
    output = tmp_path / "cli_output"
    script = Path(__file__).resolve().parents[2] / "main" / "run_ilt.py"
    command = [sys.executable, str(script), str(source), "--output-dir", str(output),
               "--method", "curvmulti", "--iterations", "1", "--step-size", "0.001",
               "--scales", "2", "1", "--smoothing-kernel", "3",
               "--curvature-weight", "0", "--device", "cpu", "--layer", "1/0",
               "--box", "0", "0", "64", "64", "--pixel-nm", "1", "--canvas", "256",
               "--no-png", "--json"]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert summary["config"]["scales"] == [2, 1]
    assert Path(summary["artifacts"]["final_lithography"]["npz"]).is_file()
