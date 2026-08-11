"""从离线像素目标直接运行可微 SimpleILT 并保存数值与诊断结果。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import estimate_rectangular_shots, evaluate_binary_l2, evaluate_pvband
from lithography import ICCAD13Lithography
from main.offline_inputs import (
    _atomic_json,
    _atomic_npz,
    _atomic_png,
    load_raster_input,
)
from opc.iteration.ilt import SimpleILTConfig, SimpleILTResult, optimize


def run_simpleilt(
        input_path: str | Path, output_dir: str | Path, *, iterations: int = 20,
        step_size: float = 0.5, sigmoid_steepness: float = 4.0,
        weight_pvband: float = 0.0, weight_process_l2: float = 1.0,
        curvature_weight: float = 0.0, device: str = "auto",
        save_png: bool = True) -> tuple[SimpleILTResult, dict[str, Any]]:
    """加载一次像素目标，运行 ILT，并保存结果、评价与性能统计。"""
    target_array, metadata = load_raster_input(input_path)
    model = ICCAD13Lithography(device=device)
    if target_array.shape[0] > model.config.canvas or target_array.shape[1] > model.config.canvas:
        raise ValueError("离线像素目标超过当前光刻模型 canvas")
    config = SimpleILTConfig(
        iterations, step_size, sigmoid_steepness, weight_pvband,
        weight_process_l2, curvature_weight)
    target = torch.as_tensor(target_array, device=model.device)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = perf_counter()
    result = optimize(target, model, config)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    optimized = perf_counter()
    conditions = tuple(model.condition(name) for name in (
        "nominal", "dose_max", "defocus_min"))
    with torch.no_grad():
        printed = model.forward_many(result.binary_mask.to(torch.float32), conditions)
        l2 = evaluate_binary_l2(
            target, printed["nominal"], model.config.print_threshold)
        pvband = evaluate_pvband(
            printed["dose_max"], printed["defocus_min"],
            model.config.print_threshold)
        shots = estimate_rectangular_shots(result.binary_mask)
    evaluated = perf_counter()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = result.best_parameters.detach().cpu().numpy().astype(np.float32, copy=False)
    soft_mask = result.soft_mask.detach().cpu().numpy().astype(np.float32, copy=False)
    binary_mask = result.binary_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    result_path = _atomic_npz(output / "simpleilt_result.npz", {
        "format_name": np.array("myopc.simpleilt-result"),
        "format_version": np.array(1, dtype=np.int32),
        "best_parameters": parameters, "soft_mask": soft_mask,
        "binary_mask": binary_mask,
        "best_iteration": np.array(result.best_iteration, dtype=np.int32),
    }, compressed=False)
    images: dict[str, str] = {}
    if save_png:
        for name, values in (("target", target_array), ("soft_mask", soft_mask),
                             ("binary_mask", binary_mask)):
            images[name] = str(_atomic_png(output / f"{name}.png", values))
    summary: dict[str, Any] = {
        "status": "completed", "input": str(Path(input_path).expanduser().resolve()),
        "source_layout": metadata.get("source"), "device": str(model.device),
        "shape": list(target_array.shape), "config": asdict(config),
        "best_iteration": result.best_iteration,
        "records": [asdict(record) for record in result.records],
        "evaluation": {"binary_l2": l2, "pvband": pvband,
                       "rectangular_shot_estimate": shots,
                       "shot_shape": [512, 512]},
        "timing_seconds": {"optimization": optimized - started,
                           "evaluation": evaluated - optimized,
                           "total": evaluated - started},
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda" else 0),
        "artifacts": {"result_npz": str(result_path), "images": images,
                      "summary": str(output / "summary.json")},
    }
    _atomic_json(output / "summary.json", summary)
    return result, summary


def build_parser() -> argparse.ArgumentParser:
    """构造离线 SimpleILT 命令行参数。"""
    parser = argparse.ArgumentParser(description="从离线像素目标运行 SimpleILT。")
    parser.add_argument("input", type=Path, help="offline_inputs raster 生成的 NPZ")
    parser.add_argument("--output-dir", type=Path, default=Path("output/simpleilt"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--sigmoid-steepness", type=float, default=4.0)
    parser.add_argument("--weight-pvband", type=float, default=0.0)
    parser.add_argument("--weight-process-l2", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--no-png", action="store_true", help="不保存诊断 PNG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并把输入、配置和运行错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        _, summary = run_simpleilt(
            args.input, args.output_dir, iterations=args.iterations,
            step_size=args.step_size, sigmoid_steepness=args.sigmoid_steepness,
            weight_pvband=args.weight_pvband,
            weight_process_l2=args.weight_process_l2,
            curvature_weight=args.curvature_weight, device=args.device,
            save_png=not args.no_png)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"SimpleILT 完成：best={summary['best_iteration']}，"
          f"L2={summary['evaluation']['binary_l2']}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
