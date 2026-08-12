"""统一运行 Simple、LevelSet、CurvMulti 和 Multilevel ILT。"""

from __future__ import annotations

import argparse
import json
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
from opc.iteration.ilt import (
    LevelSetConfig,
    MultiScaleILTConfig,
    SimpleILTConfig,
    optimize,
    optimize_levelset,
    optimize_multiscale,
)


def run_ilt(
        input_path: str | Path, output_dir: str | Path, *, method: str = "simple",
        iterations: int = 10, step_size: float = 0.2,
        weight_pvband: float = 0.0, weight_process_l2: float = 1.0,
        curvature_weight: float = 0.0, device: str = "auto",
        save_png: bool = True, layer: LayerSpec | None = None,
        top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0) -> dict[str, Any]:
    """读取版图或 raster NPZ，运行指定 ILT 并保存评价、光刻图和资源统计。"""
    loaded_started = perf_counter()
    # GDS/OASIS 分支在 Region 物化前执行文件、层级图形、顶点、预计内存和
    # canvas 上限检查；NPZ 分支验证版本及解压上限。两条路径只在内存中汇合。
    target_array, metadata = resolve_raster_input(
        input_path, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib)
    loaded = perf_counter()
    model = ICCAD13Lithography(device=device)
    if target_array.shape[0] > model.config.canvas or target_array.shape[1] > model.config.canvas:
        raise ValueError("ILT target 超过当前光刻模型 canvas")
    target = torch.as_tensor(target_array, device=model.device)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    optimized_started = perf_counter()
    # 方法分派只构造各求解器已有配置，不建立注册器或包装基类。第一阶段只对
    # levelset 做完整质量承诺，另外两个选项保留原有后续阶段实验入口。
    if method == "simple":
        config: object = SimpleILTConfig(
            iterations, step_size, weight_pvband=weight_pvband,
            weight_process_l2=weight_process_l2,
            curvature_weight=curvature_weight)
        result = optimize(target, model, config)
    elif method == "levelset":
        config = LevelSetConfig(
            iterations, step_size, weight_process_l2,
            weight_pvband, curvature_weight)
        result = optimize_levelset(target, model, config)
    elif method in ("curvmulti", "multilevel"):
        config = MultiScaleILTConfig(
            (2, 1), iterations, step_size, weight_process_l2,
            weight_pvband, curvature_weight)
        result = optimize_multiscale(target, model, config)
    else:
        raise ValueError(f"未知 ILT 方法：{method}")
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    optimized = perf_counter()
    conditions = tuple(model.condition(name) for name in (
        "nominal", "dose_max", "defocus_min"))
    # 最佳硬 mask 只在优化结束后追加一次统一评价；no_grad 避免为纯报告保存
    # autograd 图。forward_many 让三个条件共享 mask FFT，不重复占用输入频谱。
    with torch.no_grad():
        printed = model.forward_many(result.binary_mask.to(torch.float32), conditions)
        l2 = evaluate_binary_l2(target, printed["nominal"], model.config.print_threshold)
        pvband = evaluate_pvband(
            printed["dose_max"], printed["defocus_min"], model.config.print_threshold)
        shots = estimate_rectangular_shots(result.binary_mask)
    evaluated = perf_counter()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # GPU Tensor 只在这里各拷回一次；后续 NPZ/PNG 全部复用 CPU 数组，防止
    # 每个产物重复触发设备同步和临时主机内存分配。
    parameters = result.best_parameters.detach().cpu().numpy().astype(np.float32, copy=False)
    soft_mask = result.soft_mask.detach().cpu().numpy().astype(np.float32, copy=False)
    binary_mask = result.binary_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    result_path = _atomic_npz(output / "ilt_result.npz", {
        "format_name": np.array("myopc.ilt-result"),
        "format_version": np.array(1, dtype=np.int32),
        "method": np.array(method), "best_parameters": parameters,
        "soft_mask": soft_mask, "binary_mask": binary_mask,
        "best_iteration": np.array(result.best_iteration, dtype=np.int32),
    })
    final_lithography = save_final_lithography_result(
        output, binary_mask, printed, save_png=save_png)
    images: dict[str, str] = {}
    if save_png:
        for name, values in (("target", target_array), ("soft_mask", soft_mask),
                             ("binary_mask", binary_mask)):
            images[name] = str(_atomic_png(output / f"{name}.png", values))
    finished = perf_counter()
    summary: dict[str, Any] = {
        "status": "completed", "method": method,
        "input": str(Path(input_path).expanduser().resolve()),
        "source_layout": metadata.get("source"), "device": str(model.device),
        "shape": list(target_array.shape), "config": asdict(config),
        "best_iteration": result.best_iteration,
        "records": [asdict(record) for record in result.records],
        "evaluation": {"binary_l2": l2, "pvband": pvband,
                       "rectangular_shot_estimate": shots},
        "timing_seconds": {"input": loaded - loaded_started,
                           "optimization": optimized - optimized_started,
                           "evaluation": evaluated - optimized,
                           "output": finished - evaluated,
                           "total": finished - loaded_started},
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda" else 0),
        "artifacts": {"result_npz": str(result_path), "images": images,
                      "final_lithography": final_lithography,
                      "summary": str(output / "summary.json")},
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """解析统一 ILT 命令行并返回标准退出码。"""
    parser = argparse.ArgumentParser(description="运行统一 ILT 方法")
    parser.add_argument("input", type=Path); parser.add_argument("--output-dir", type=Path, default=Path("output/ilt"))
    parser.add_argument("--method", choices=("simple", "levelset", "curvmulti", "multilevel"), default="simple")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--weight-pvband", type=float, default=0.0)
    parser.add_argument("--weight-process-l2", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--json", action="store_true", help="终端输出完整 JSON 汇总")
    add_layout_source_arguments(parser)
    parser.add_argument("--pixel-nm", type=float, default=8.0)
    parser.add_argument("--canvas", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        summary = run_ilt(
            args.input, args.output_dir, method=args.method,
            iterations=args.iterations, step_size=args.step_size,
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
            max_estimated_gib=args.max_estimated_gib)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"ILT 完成：{summary['method']}，输出：{summary['artifacts']['summary']}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
