"""从离线 segment NPZ 运行独立 DiffOPC 梯度边段优化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lithography import ICCAD13Lithography
from main.offline_inputs import _atomic_npz, load_segment_input
from opc.input.edge import reconstruct_region
from opc.iteration.diffopc import DiffOPCConfig, optimize


def run_diffopc(input_path: str | Path, output_dir: str | Path, *, iterations: int = 8,
                learning_rate: float = 1.0, device: str = "auto") -> dict[str, object]:
    """加载离线问题、执行 DiffOPC 并保存位移与精确 Region。"""
    problem, metadata = load_segment_input(input_path)
    model = ICCAD13Lithography(device=device)
    result = optimize(problem, model, DiffOPCConfig(iterations=iterations, learning_rate=learning_rate))
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    result_path = _atomic_npz(output / "diffopc_result.npz", {
        "format_name": np.array("myopc.diffopc-result"),
        "format_version": np.array(1, dtype=np.int32),
        "best_displacements": result.best_displacements,
    })
    region = reconstruct_region(problem, result.best_displacements)
    return {"metadata": metadata, "best_iteration": result.best_iteration,
            "segments": problem.segments.segment_count, "valid": bool(region.has_valid_polygons()),
            "result_npz": str(result_path), "records": len(result.records)}


def main(argv: list[str] | None = None) -> int:
    """解析 DiffOPC 命令行并返回标准退出码。"""
    parser = argparse.ArgumentParser(description="运行 DiffOPC 梯度边段优化")
    parser.add_argument("input", type=Path); parser.add_argument("--output-dir", type=Path, default=Path("output/diffopc"))
    parser.add_argument("--iterations", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        print(run_diffopc(args.input, args.output_dir, iterations=args.iterations,
                          learning_rate=args.learning_rate, device=args.device))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
