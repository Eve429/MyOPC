"""从 GDS/OASIS 或像素 NPZ 运行可微 SimpleILT 并保存结果。"""

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
from layout import DbuBox, LayerSpec
from lithography import ICCAD13Lithography
from main.offline_inputs import (
    _atomic_json,
    _atomic_npz,
    _atomic_png,
    add_layout_source_arguments,
    resolve_raster_input,
    save_final_lithography_result,
)
from main.configuration import ConfiguredArgumentParser, glp_layer_map
from opc.iteration.ilt import SimpleILTConfig, SimpleILTResult, optimize


def run_simpleilt(
        input_path: str | Path, output_dir: str | Path, *, iterations: int = 20,
        step_size: float = 0.5, sigmoid_steepness: float = 4.0,
        weight_pvband: float = 0.0, weight_process_l2: float = 1.0,
        curvature_weight: float = 0.0, device: str = "auto",
        save_png: bool = True, layer: LayerSpec | None = None,
        top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0, polarity: str = "clear",
        glp_layers: dict[str, LayerSpec] | None = None,
        run_configuration: dict[str, object] | None = None
        ) -> tuple[SimpleILTResult, dict[str, Any]]:
    """加载一次版图或像素目标，运行 ILT，并保存结果、评价与性能统计。"""
    # GDS/OASIS 在此处按 ROI 直接生成 CPU mask，NPZ 则直接加载；优化器只看到
    # 同一个连续 float32 目标，因此输入方式不会分叉梯度、评价或输出逻辑。
    target_array, metadata = resolve_raster_input(
        input_path, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib, polarity=polarity,
        glp_layers=glp_layers)
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
    final_lithography = save_final_lithography_result(
        output, binary_mask, printed, save_png=save_png)
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
        "run_configuration": run_configuration,
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
                      "final_lithography": final_lithography,
                      "summary": str(output / "summary.json")},
    }
    _atomic_json(output / "summary.json", summary)
    return result, summary


def build_parser() -> argparse.ArgumentParser:
    """构造支持版图和离线像素输入的 SimpleILT 命令行参数。"""
    parser = ConfiguredArgumentParser(
        workflow="ilt", entry="simpleilt", valid_entries=("ilt", "simpleilt"),
        description="从 GDS/OASIS ROI 或离线像素目标运行 SimpleILT。")
    parser.add_argument("input", type=Path, help="输入 GDS/OASIS 或 raster NPZ")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--sigmoid-steepness", type=float)
    parser.add_argument("--weight-pvband", type=float)
    parser.add_argument("--weight-process-l2", type=float)
    parser.add_argument("--curvature-weight", type=float)
    parser.add_argument("--device", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--no-png", action="store_true", help="不保存诊断 PNG")
    add_layout_source_arguments(parser)
    parser.add_argument("--pixel-nm", type=float,
                        help="直接版图输入的像素尺寸")
    parser.add_argument("--canvas", type=int,
                        help="直接版图输入的固定方形画布")
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
            save_png=not args.no_png, layer=args.layer,
            top_cell=args.top_cell,
            box=None if args.box is None else tuple(args.box),
            pixel_nm=args.pixel_nm, canvas=args.canvas,
            max_file_gib=args.max_file_gib,
            max_shape_occurrences=args.max_shapes,
            max_source_vertices=args.max_vertices,
            max_estimated_gib=args.max_estimated_gib,
            polarity=args.polarity, glp_layers=glp_layer_map(args.glp_layers),
            run_configuration=args._configuration)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"SimpleILT 完成：best={summary['best_iteration']}，"
          f"L2={summary['evaluation']['binary_l2']}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
