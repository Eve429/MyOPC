"""MB-OPC 产物序列化、标注图片和直接主程序测试。"""

import json
import subprocess
from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

import run_mbopc_frontend as frontend
from opc.input.grid import axis_cuts_by_size
from run_mbopc_frontend import build_parser, run
from tests.fixtures.layout_factory import write_advanced_layout


def test_direct_runner_writes_all_artifacts_and_validates_round_trips(tmp_path: Path) -> None:
    """无输入合成模式应生成 JSON、NPZ、PNG、GDS 并通过内部几何检查。"""
    args = build_parser().parse_args(["--output-dir", str(tmp_path)])
    result = run(args)
    assert result["verification"] == {
        "zero_displacement_xor_area": 0, "core_coverage_xor_area": 0,
        "core_overlap_area": 0,
        "reconstructed_valid": True,
        "geometry_suite_case_count": 5,
    }
    for path in result["artifacts"].values():
        assert Path(path).is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["updated_segments"] == 2
    with np.load(tmp_path / "segments.npz", allow_pickle=False) as arrays:
        assert len(arrays["segment_edge_ids"]) == summary["counts"]["segments"]
        assert "segment_keys" not in arrays.files
        assert arrays["format_version"].tolist() == [3]
    layout = kdb.Layout()
    layout.read(str(tmp_path / "reconstruction.gds"))
    assert sorted(cell.name for cell in layout.top_cells()) == ["RECONSTRUCTED", "REFERENCE"]


def test_frontend_probe_distance_does_not_follow_corner_length(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """诊断探针必须使用显式 probe 距离，不能随分段 corner 长度改变。"""
    captured: list[float] = []
    original = frontend.edge_probe_points

    def record_distance(starts: object, ends: object, normals: object,
                        distance_dbu: float) -> tuple[np.ndarray, np.ndarray]:
        """记录前端传给公共探针函数的 DBU 距离并保持原行为。"""
        captured.append(distance_dbu)
        return original(starts, ends, normals, distance_dbu)

    monkeypatch.setattr(frontend, "edge_probe_points", record_distance)
    args = build_parser().parse_args([
        "--corner-nm", "7", "--probe-distance-nm", "3",
        "--output-dir", str(tmp_path), "--skip-geometry-suite",
    ])
    run(args)
    assert captured == [3.0]


def test_runner_executes_from_external_working_directory_without_install(tmp_path: Path) -> None:
    """根入口应依靠自身目录导入项目，而不是当前工作目录或 editable install。"""
    script = Path(__file__).resolve().parents[2] / "run_mbopc_frontend.py"
    output = tmp_path / "external"
    completed = subprocess.run(
        [str(Path(__import__("sys").executable)), str(script), "--output-dir", str(output), "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["verification"]["core_coverage_xor_area"] == 0
    assert result["verification"]["core_overlap_area"] == 0


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


def test_physical_tile_size_uses_fixed_step_and_clips_last_tiles(tmp_path: Path) -> None:
    """固定 nm 边长应从处理框起点切分，且末列末行不得越过范围。"""
    np.testing.assert_array_equal(axis_cuts_by_size(11, 36, 10), [11, 21, 31, 36])
    args = build_parser().parse_args([
        "--tile-size-nm", "100", "--output-dir", str(tmp_path),
        "--skip-geometry-suite",
    ])
    result = run(args)
    assert result["tiling"] == {
        "mode": "physical_size", "columns": 3, "rows": 3,
        "requested_tile_size_nm": 100.0,
    }
    assert result["counts"]["cores"] == 9
    assert result["verification"]["core_coverage_xor_area"] == 0
    assert result["verification"]["core_overlap_area"] == 0


@pytest.mark.parametrize("value", ["0", "nan", "0.4"])
def test_physical_tile_size_rejects_invalid_or_sub_dbu_values(tmp_path: Path,
                                                               value: str) -> None:
    """非正、非有限及小于一个版图 DBU 的物理 tile 尺寸必须立即拒绝。"""
    args = build_parser().parse_args([
        "--tile-size-nm", value, "--output-dir", str(tmp_path),
        "--skip-geometry-suite",
    ])
    with pytest.raises(ValueError, match="tile-size-nm"):
        run(args)


def test_grid_count_and_physical_tile_size_are_mutually_exclusive() -> None:
    """一次运行只能选择按数量或按物理尺寸切分，避免参数优先级不明确。"""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--grid", "2", "1", "--tile-size-nm", "100"])


def test_skip_artifacts_keeps_verification_and_detailed_resource_stats(tmp_path: Path) -> None:
    """跳过大型产物时仍应完成重建验证并记录各阶段时间和内存。"""
    args = build_parser().parse_args([
        "--output-dir", str(tmp_path), "--skip-artifacts",
    ])
    result = run(args)
    assert result["status"] == "completed"
    assert result["verification"]["zero_displacement_xor_area"] == 0
    assert result["artifacts"] == {
        "json": str((tmp_path / "summary.json").resolve()),
        "npz": None, "png": None, "gds": None,
    }
    expected = {
        "layout_open", "preflight", "roi_materialize", "problem_prepare",
        "demo_update", "segment_materialize_and_probes", "reconstruct",
        "verification", "npz", "gds", "png", "geometry_suite", "total",
    }
    assert expected <= result["timing_seconds"].keys()
    assert expected <= result["memory_checkpoints"].keys()
    assert all(result["timing_seconds"][name] >= 0.0 for name in expected)


def test_real_frontend_preflight_only_and_low_budget_rejection(
        tmp_path: Path) -> None:
    """真实版图应支持只预检，并在低预算下于 Region 物化前稳定拒绝。"""
    source = write_advanced_layout(tmp_path / "preflight.gds")
    accepted = run(build_parser().parse_args([
        str(source), "--layer", "1/0", "--preflight-only",
        "--output-dir", str(tmp_path / "accepted"),
    ]))
    assert accepted["status"] == "preflight_only"
    assert accepted["preflight"]["accepted"] is True
    rejected = run(build_parser().parse_args([
        str(source), "--layer", "1/0", "--memory-budget-gib", "0.000000001",
        "--output-dir", str(tmp_path / "rejected"),
    ]))
    assert rejected["status"] == "rejected"
    assert rejected["preflight"]["memory_budget_ok"] is False
    assert rejected["preflight"]["recommended_mode"] == "sharded_required"
