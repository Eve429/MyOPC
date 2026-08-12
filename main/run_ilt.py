"""统一运行 Simple、LevelSet、CurvMulti 和 Multilevel ILT。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lithography import ICCAD13Lithography
from main.offline_inputs import _atomic_npz, resolve_raster_input
from opc.iteration.ilt import (
    LevelSetConfig,
    MultiScaleILTConfig,
    SimpleILTConfig,
    optimize,
    optimize_levelset,
    optimize_multiscale,
)


def run_ilt(input_path: str | Path, output_dir: str | Path, *, method: str = "simple",
            iterations: int = 10, step_size: float = 0.2,
            device: str = "auto") -> dict[str, object]:
    """读取版图或 raster NPZ，运行指定 ILT 方法并保存结果。"""
    target_array, metadata = resolve_raster_input(input_path)
    model = ICCAD13Lithography(device=device)
    target = torch.as_tensor(target_array, device=model.device)
    if method == "simple":
        result = optimize(target, model, SimpleILTConfig(iterations, step_size))
    elif method == "levelset":
        result = optimize_levelset(target, model, LevelSetConfig(iterations, step_size))
    elif method in ("curvmulti", "multilevel"):
        result = optimize_multiscale(target, model, MultiScaleILTConfig((2, 1), iterations, step_size))
    else:
        raise ValueError(f"未知 ILT 方法：{method}")
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    result_path = _atomic_npz(output / "ilt_result.npz", {
        "format_name": np.array("myopc.ilt-result"),
        "format_version": np.array(1, dtype=np.int32),
        "best_parameters": result.best_parameters.detach().cpu().numpy(),
        "soft_mask": result.soft_mask.detach().cpu().numpy(),
        "binary_mask": result.binary_mask.detach().cpu().numpy(),
    })
    return {"method": method, "input": str(Path(input_path).resolve()),
            "metadata": metadata, "best_iteration": result.best_iteration,
            "result_npz": str(result_path), "records": len(result.records)}


def main(argv: list[str] | None = None) -> int:
    """解析统一 ILT 命令行并返回标准退出码。"""
    parser = argparse.ArgumentParser(description="运行统一 ILT 方法")
    parser.add_argument("input", type=Path); parser.add_argument("--output-dir", type=Path, default=Path("output/ilt"))
    parser.add_argument("--method", choices=("simple", "levelset", "curvmulti", "multilevel"), default="simple")
    parser.add_argument("--iterations", type=int, default=10); parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        print(run_ilt(args.input, args.output_dir, method=args.method,
                      iterations=args.iterations, step_size=args.step_size, device=args.device))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
