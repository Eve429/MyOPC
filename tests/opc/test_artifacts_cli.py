"""MB-OPC 产物序列化、标注图片和直接主程序测试。"""

import json
import subprocess
from pathlib import Path

import klayout.db as kdb
import numpy as np

from run_mbopc_frontend import build_parser, run
from tests.fixtures.layout_factory import write_advanced_layout


def test_direct_runner_writes_all_artifacts_and_validates_round_trips(tmp_path: Path) -> None:
    """无输入合成模式应生成 JSON、NPZ、PNG、GDS 并通过内部几何检查。"""
    args = build_parser().parse_args(["--output-dir", str(tmp_path)])
    result = run(args)
    assert result["verification"] == {
        "zero_displacement_xor_area": 0, "stitch_xor_area": 0,
        "reconstructed_valid": True,
        "geometry_suite_case_count": 5,
    }
    for path in result["artifacts"].values():
        assert Path(path).is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["updated_segments"] == 2
    with np.load(tmp_path / "segments.npz", allow_pickle=False) as arrays:
        assert len(arrays["segment_keys"]) == summary["counts"]["segments"]
        assert arrays["format_version"].tolist() == [1]
    layout = kdb.Layout()
    layout.read(str(tmp_path / "reconstruction.gds"))
    assert sorted(cell.name for cell in layout.top_cells()) == ["RECONSTRUCTED", "REFERENCE"]


def test_runner_executes_from_external_working_directory_without_install(tmp_path: Path) -> None:
    """根入口应依靠自身目录导入项目，而不是当前工作目录或 editable install。"""
    script = Path(__file__).resolve().parents[2] / "run_mbopc_frontend.py"
    output = tmp_path / "external"
    completed = subprocess.run(
        [str(Path(__import__("sys").executable)), str(script), "--output-dir", str(output), "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["verification"]["stitch_xor_area"] == 0


def test_runner_prepares_real_hierarchical_input_before_database_close(tmp_path: Path) -> None:
    """真实文件分支必须在 LayoutDB 关闭前完成物理 mask 和紧凑问题准备。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    output = tmp_path / "real-output"
    args = build_parser().parse_args([
        str(source), "--layer", "1/0", "--output-dir", str(output),
        "--skip-geometry-suite",
    ])
    result = run(args)
    assert result["source"] == str(source.resolve())
    assert result["counts"]["polygons"] > 1
    assert result["verification"]["zero_displacement_xor_area"] == 0
