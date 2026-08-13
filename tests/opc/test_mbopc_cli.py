"""simple MB-OPC 根入口的直接运行、物理网格和产物测试。"""

import json
import subprocess
import sys
from pathlib import Path

import klayout.db as kdb
import pytest

from main.run_mbopc import build_parser, run


def _write_single_layer_layout(path: Path) -> Path:
    """写出恰好覆盖 256×256 DBU 画布的单层矩形版图。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    layer = layout.layer(1, 0)
    top = layout.create_cell("TOP")
    top.shapes(layer).insert(kdb.Box(80, 80, 176, 176))
    # 外围框只用于把默认 bbox 固定为模型画布；与中心图形连通后仍是一个物理
    # Polygon 会改变目标，因此这里通过显式 --box 选择 0..256，无需增加假图形。
    layout.write(str(path))
    return path


def test_direct_mbopc_runner_works_outside_repository_and_writes_results(
        project_root: Path, tmp_path: Path) -> None:
    """从仓库外直接执行根脚本，应完成真实模型一轮并写出 JSON 和 GDS。"""
    source = _write_single_layer_layout(tmp_path / "single.gds")
    output = tmp_path / "result"
    command = [
        sys.executable, str(project_root / "main" / "run_mbopc.py"), str(source),
        "--box", "0", "0", "256", "256", "--tile-size-nm", "128",
        "--tile-halo-nm", "48", "--pixel-nm", "1", "--iterations", "1",
        "--batch-size", "1", "--device", "cpu", "--output-dir", str(output), "--json",
    ]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True,
        encoding="utf-8", timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["counts"]["cores"] == 4
    assert len(result["optimization"]["records"]) == 1
    assert result["verification"]["reconstructed_valid"] is True
    assert Path(result["artifacts"]["summary"]).is_file()
    assert "npz" not in result["artifacts"]
    gds_path = Path(result["artifacts"]["gds"])
    assert gds_path.is_file()
    layout = kdb.Layout()
    layout.read(str(gds_path))
    assert sorted(cell.name for cell in layout.top_cells()) == ["RECONSTRUCTED", "REFERENCE"]


def test_runner_rejects_tile_not_aligned_to_pixel_before_model_load(tmp_path: Path) -> None:
    """tile 与像素晶格不对齐时应在光刻资产加载前拒绝，避免跨 core 采样错位。"""
    source = _write_single_layer_layout(tmp_path / "single.gds")
    args = build_parser().parse_args([
        str(source), "--box", "0", "0", "256", "256",
        "--tile-size-nm", "255", "--tile-halo-nm", "24", "--pixel-nm", "2",
        "--output-dir", str(tmp_path / "unused"),
    ])
    with pytest.raises(ValueError, match="整数倍"):
        run(args)


def test_solver_preflight_only_does_not_load_lithography_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """实际求解入口只预检时不得加载光刻资产或构造 GPU 模型。"""
    source = _write_single_layer_layout(tmp_path / "preflight.gds")

    def forbidden_model(*args: object, **kwargs: object) -> None:
        """若预检路径错误进入光刻模型构造，就让回归明确失败。"""
        raise AssertionError("preflight-only 不应加载光刻模型")

    monkeypatch.setattr("main.run_mbopc.ICCAD13Lithography", forbidden_model)
    args = build_parser().parse_args([
        str(source), "--box", "0", "0", "256", "256",
        "--tile-size-nm", "256", "--tile-halo-nm", "24", "--pixel-nm", "1",
        "--preflight-only", "--output-dir", str(tmp_path / "result"),
    ])
    result = run(args)
    assert result["status"] == "preflight_only"
    assert result["preflight"]["accepted"] is True
    assert Path(result["artifacts"]["summary"]).is_file()
