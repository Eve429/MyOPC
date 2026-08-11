"""无需安装项目包的根目录 Python 入口端到端测试。"""

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.layout_factory import write_advanced_layout


def test_direct_python_entry_reads_and_exports(project_root: Path, tmp_path: Path) -> None:
    """从仓库外工作目录直接运行脚本，并完成查询、数组转换和 Patch 输出。"""
    script = project_root / "main" / "run_layout_geometry.py"
    source = write_advanced_layout(tmp_path / "advanced.gds")
    output = tmp_path / "cli_patch.gds"
    image = tmp_path / "cli_roi.png"
    command = [
        sys.executable, str(script), str(source),
        "--layer", "1/0", "--box", "-100", "-50", "100", "50",
        "--arrays", "--diagnostics", "--output", str(output),
        "--png", str(image), "--pixel-size-nm", "5", "--json",
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, encoding="utf-8", check=True)
    result = json.loads(completed.stdout)
    assert result["top_cell"] == "TOP"
    assert result["layers"]["1/0"]["polygon_count"] == 1
    assert result["layers"]["1/0"]["edge_count"] > 0
    assert result["layers"]["1/0"]["diagnostics"]["text"] == 0
    assert output.is_file()
    assert image.is_file()
    assert result["image"] == {
        "path": str(image.resolve()), "shown": False, "width": 40, "height": 20,
        "pixel_size_nm": 5.0, "layer": "1/0",
    }


def test_png_rejects_ambiguous_multiple_layers(project_root: Path, tmp_path: Path) -> None:
    """未明确单 Layer 时，PNG 模式应返回简洁错误而不是猜测混色规则。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    command = [
        sys.executable, str(project_root / "main" / "run_layout_geometry.py"), str(source),
        "--png", str(tmp_path / "ambiguous.png"),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, encoding="utf-8", check=False)
    assert completed.returncode == 2
    assert "必须且只能选择一个 Layer" in completed.stderr


def test_mbopc_frontend_benchmark_uses_current_problem_contract(
        project_root: Path, tmp_path: Path) -> None:
    """小规模严格基准应直接运行，防止再次引用已删除的数据结构字段。"""
    command = [
        sys.executable, str(project_root / "benchmarks" / "benchmark_mbopc_frontend.py"),
        "--shapes", "100", "--strict",
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, encoding="utf-8", check=True, timeout=60)
    result = json.loads(completed.stdout)
    assert result["counts"]["input_shapes"] == 100
    assert result["counts"]["segments"] > 0
    assert result["strict_failures"] == []
