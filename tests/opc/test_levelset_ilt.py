"""LevelSetILT 的 SDF、代理梯度、契约和真实光刻回归测试。"""

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

from layout import LayerSpec
from lithography import ICCAD13Lithography, ProcessCondition
from main.run_ilt import run_ilt
from opc.iteration.ilt import LevelSetConfig, optimize_levelset
from opc.iteration.ilt.levelset import _LevelSetBinarize, signed_distance_initialization


class _IdentityLithography:
    """用廉价恒等曝光隔离检查 LevelSet 优化器。"""

    def __init__(self, canvas: int = 64) -> None:
        """建立 CPU 设备、最小配置和条件调用记录。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas)
        self.requested: list[tuple[str, ...]] = []

    def condition(self, name: str) -> ProcessCondition:
        """返回默认名称对应的独立伪工艺条件。"""
        return ProcessCondition(name, "defocus" if name == "defocus_min" else "focus", 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """记录条件并返回保持梯度的连续曝光图。"""
        self.requested.append(tuple(condition.name for condition in conditions))
        offsets = {"dose_max": 0.04, "defocus_min": -0.04}
        return {condition.name: torch.clamp(
            mask + offsets.get(condition.name, 0.0), 0.0, 1.0)
                for condition in conditions}


def test_signed_distance_has_expected_sign_and_distance() -> None:
    """SDF 必须前景为负、背景为正，并保持像素中心欧氏距离。"""
    target = torch.zeros((5, 5))
    target[2, 2] = 1.0
    field = signed_distance_initialization(target)
    assert field[2, 2] == pytest.approx(-1.0)
    assert field[2, 3] == pytest.approx(1.0)
    assert field[0, 0] == pytest.approx(8.0 ** 0.5)


def test_levelset_surrogate_gradient_uses_spatial_change_and_direction() -> None:
    """代理反向应由 phi 空间变化调制，并按负号传播上游梯度。"""
    phi = torch.tensor([[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0],
                         [-1.0, -1.0, -1.0]]], requires_grad=True)
    _LevelSetBinarize.apply(phi).sum().backward()
    assert torch.all(phi.grad[:, :2] < 0.0)
    assert torch.equal(phi.grad[:, 2], torch.zeros_like(phi.grad[:, 2]))


def test_levelset_rejects_invalid_config_and_inputs() -> None:
    """无效迭代、权重、窗口和重复条件必须在首轮仿真前拒绝。"""
    for kwargs in ({"iterations": 0}, {"step_size": 0.0},
                   {"weight_pvband": -1.0}, {"curvature_weight": float("nan")}):
        with pytest.raises(ValueError):
            LevelSetConfig(**kwargs)
    model = _IdentityLithography()
    target = torch.zeros((8, 8))
    invalid = torch.ones_like(target); invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        optimize_levelset(target, model, LevelSetConfig(iterations=1),
                          optimization_mask=invalid)
    invalid_target = target.clone(); invalid_target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="有限数"):
        optimize_levelset(invalid_target, model, LevelSetConfig(iterations=1))
    with pytest.raises(ValueError, match="高度和宽度"):
        signed_distance_initialization(torch.empty((0, 8)))
    duplicate = ProcessCondition("same", "focus", 1.0)
    with pytest.raises(ValueError, match="名称不能重复"):
        optimize_levelset(target, model, LevelSetConfig(iterations=1),
                          nominal_condition=duplicate, process_conditions=(duplicate,))


def test_levelset_supports_empty_process_window_fixed_area_and_curvature() -> None:
    """空工艺窗口应只算 nominal，固定区不动，曲率记录保持有限。"""
    model = _IdentityLithography()
    target = torch.zeros((12, 12)); target[3:9, 3:9] = 1.0
    initial = signed_distance_initialization(target)
    movable = torch.zeros_like(target); movable[4:8, 4:8] = 1.0
    result = optimize_levelset(
        target, model, LevelSetConfig(iterations=2, curvature_weight=0.1),
        initial_levelset=initial, optimization_mask=movable,
        process_conditions=())
    assert all(names == ("nominal",) for names in model.requested)
    assert torch.equal(result.best_parameters[movable == 0], initial[movable == 0])
    assert all(record.process_l2 == 0.0 and record.pvband_loss == 0.0
               for record in result.records)
    assert all(torch.isfinite(torch.tensor(record.curvature_loss))
               for record in result.records)


def test_levelset_binary_uses_strict_zero_boundary() -> None:
    """最优 phi 恰为零时硬结果必须与前向的 phi<0 判定保持一致。"""
    target = torch.zeros((8, 8))
    result = optimize_levelset(
        target, _IdentityLithography(), LevelSetConfig(iterations=1),
        initial_levelset=torch.zeros_like(target), process_conditions=())
    # 最佳状态记录在参数更新前；全零 phi 的硬前向全部关闭，输出不能因
    # sigmoid(-phi)==0.5 再使用 >=0.5 而被错误反转成全开。
    assert not torch.any(result.binary_mask)


def test_real_model_completes_levelset_backward() -> None:
    """真实 Hopkins 模型应完成一轮 LevelSet backward 并返回有限参数。"""
    target = torch.zeros((24, 24)); target[7:17, 8:16] = 1.0
    result = optimize_levelset(
        target, ICCAD13Lithography(device="cpu"),
        LevelSetConfig(iterations=1, step_size=1e-3, weight_process_l2=0.1))
    assert len(result.records) == 1
    assert torch.all(torch.isfinite(result.best_parameters))
    assert result.binary_mask.dtype == torch.bool


def test_levelset_runner_accepts_gds_and_saves_complete_artifacts(tmp_path: Path) -> None:
    """统一入口应直接读取指定 Layer/ROI，并保存结果、评价和最终光刻图。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer_index = layout.layer(1, 0); top = layout.create_cell("TOP")
    top.shapes(layer_index).insert(kdb.Box(8, 8, 40, 40))
    source = tmp_path / "levelset.gds"; layout.write(str(source))
    output = tmp_path / "output"
    summary = run_ilt(
        source, output, method="levelset", iterations=1, step_size=1e-3,
        device="cpu", save_png=True, layer=LayerSpec(1, 0),
        box=(0, 0, 64, 64), pixel_nm=1.0, canvas=256)
    assert summary["method"] == "levelset"
    assert summary["evaluation"]["binary_l2"] >= 0
    assert summary["timing_seconds"]["optimization"] >= 0.0
    assert {"input", "optimization", "evaluation", "output", "total"} <= set(
        summary["timing_seconds"])
    assert Path(summary["artifacts"]["result_npz"]).is_file()
    assert (output / "final_lithography.npz").is_file()
    assert (output / "summary.json").is_file()
    with np.load(output / "ilt_result.npz", allow_pickle=False) as data:
        assert str(data["method"].item()) == "levelset"
        assert data["best_parameters"].shape == (256, 256)


def test_ilt_layout_input_does_not_call_edge_macro_preparation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ILT 仍应使用精确像素 ROI，不得依赖 MB-OPC 的边段 macro 准备。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer_index = layout.layer(1, 0); top = layout.create_cell("TOP")
    top.shapes(layer_index).insert(kdb.Box(-8, 8, 40, 40))
    source = tmp_path / "pixel-only.gds"; layout.write(str(source))

    def forbidden_prepare(*args: object, **kwargs: object) -> None:
        """若像素 ILT 错误进入边段准备就立即失败。"""
        raise AssertionError("ILT 不应调用 prepare_macro")

    monkeypatch.setattr("opc.input.edge.prepare_macro", forbidden_prepare)
    summary = run_ilt(
        source, tmp_path / "pixel-output", method="levelset", iterations=1,
        device="cpu", save_png=False, layer=LayerSpec(1, 0),
        box=(0, 0, 64, 64), pixel_nm=1.0, canvas=256)
    assert summary["status"] == "completed"


def test_levelset_cli_runs_from_outside_repository(tmp_path: Path) -> None:
    """用户应能在仓库外直接运行主文件，并从 JSON 取得完整产物位置。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer_index = layout.layer(1, 0); top = layout.create_cell("TOP")
    top.shapes(layer_index).insert(kdb.Box(8, 8, 40, 40))
    source = tmp_path / "cli.gds"; layout.write(str(source))
    output = tmp_path / "cli_output"
    script = Path(__file__).resolve().parents[2] / "main" / "run_ilt.py"
    command = [sys.executable, str(script), str(source), "--output-dir", str(output),
               "--method", "levelset", "--iterations", "1", "--step-size", "0.001",
               "--device", "cpu", "--layer", "1/0", "--box", "0", "0", "64", "64",
               "--pixel-nm", "1", "--canvas", "256", "--no-png", "--json"]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert Path(summary["artifacts"]["final_lithography"]["npz"]).is_file()
