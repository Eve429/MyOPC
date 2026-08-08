"""无需安装项目包的根目录 Python 入口端到端测试。"""

import json
import subprocess
import sys
from pathlib import Path


def test_direct_python_entry_reads_and_exports(project_root: Path, reticle_dir: Path,
                                                tmp_path: Path) -> None:
    """从仓库外工作目录直接运行脚本，并完成查询、数组转换和 Patch 输出。"""
    script = project_root / "run_layout_geometry.py"
    output = tmp_path / "cli_patch.gds"
    command = [
        sys.executable, str(script), str(reticle_dir / "simple.gds"),
        "--layer", "1/0", "--box", "-2500", "-600", "500", "1600",
        "--arrays", "--diagnostics", "--output", str(output), "--json",
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, encoding="utf-8", check=True)
    result = json.loads(completed.stdout)
    assert result["top_cell"] == "TOP"
    assert result["layers"]["1/0"]["polygon_count"] == 10
    assert result["layers"]["1/0"]["edge_count"] > 0
    assert result["layers"]["1/0"]["diagnostics"]["text"] == 1
    assert output.is_file()
