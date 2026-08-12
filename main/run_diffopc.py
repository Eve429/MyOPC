"""从 GDS/OASIS 或离线边段 NPZ 运行完整 DiffOPC 工作流。"""

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

from evaluation import estimate_rectangular_shots  # noqa: E402
from lithography import ICCAD13Lithography  # noqa: E402
from main.offline_inputs import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _exact_dbu,
    add_layout_source_arguments,
    load_segment_input,
    materialize_segment_input,
    save_final_lithography_tiles,
)
from opc.diagnostics import render_boundary_overlay, write_debug_gds  # noqa: E402
from opc.input import process_memory_snapshot  # noqa: E402
from opc.input.edge import edge_probe_points, reconstruct_region  # noqa: E402
from opc.input.raster import rasterize_region_canvas  # noqa: E402
from opc.iteration.diffopc import DiffOPCConfig, optimize  # noqa: E402


def _load_problem(
        input_path: str | Path, *, layer: object = None,
        top_cell: str | None = None, box: tuple[int, int, int, int] | None = None,
        tile_size_nm: float = 1024.0, halo_nm: float = 512.0,
        corner_nm: float = 16.0, segment_nm: float = 32.0,
        max_displacement_nm: float = 24.0, max_file_gib: float = 4.0,
        max_shapes: int = 5_000_000, max_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0) -> tuple[object, dict[str, object]]:
    """自动分派边段归档或经安全预检的直接版图输入。"""
    source = Path(input_path).expanduser().resolve()
    if source.suffix.lower() == ".npz":
        return load_segment_input(source, max_archive_gib=max_estimated_gib)
    # 直接版图复用唯一正式前端的内存层，不复制 Layer/ROI/层级和 owner 构造，
    # 也不先写再读大型临时 NPZ；离线工作台仍通过 prepare_segment_input 显式归档。
    return materialize_segment_input(
        source, layer=layer,
        top_cell=top_cell, box=box, tile_size_nm=tile_size_nm,
        halo_nm=halo_nm, corner_nm=corner_nm, segment_nm=segment_nm,
        max_displacement_nm=max_displacement_nm, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shapes, max_source_vertices=max_vertices,
        max_estimated_gib=max_estimated_gib)


def run_diffopc(
        input_path: str | Path, output_dir: str | Path, *, iterations: int = 8,
        learning_rate_nm: float = 1.0, soft_temperature_nm: float = 4.0,
        weight_l2: float = 1.0, weight_pvband: float = 0.0,
        weight_epe: float = 1.0, epe_distance_nm: float = 16.0,
        pixel_nm: float = 8.0, batch_size: int = 8,
        raster_chunk_size: int = 32, target_cache_mb: int = 512,
        device: str = "auto", save_preview: bool = True,
        save_final_lithography_png: bool = True, layer: object = None,
        top_cell: str | None = None, box: tuple[int, int, int, int] | None = None,
        tile_size_nm: float = 1024.0, halo_nm: float = 512.0,
        corner_nm: float = 16.0, segment_nm: float = 32.0,
        max_displacement_nm: float | None = None, max_file_gib: float = 4.0,
        max_shapes: int = 5_000_000, max_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0) -> dict[str, object]:
    """执行输入准备、梯度迭代、精确重建和全部最终产物保存。"""
    started = perf_counter()
    memory = {"start": process_memory_snapshot()}
    source = Path(input_path).expanduser().resolve()
    direct_limit_nm = 24.0 if max_displacement_nm is None else max_displacement_nm
    problem, metadata = _load_problem(
        source, layer=layer, top_cell=top_cell, box=box,
        tile_size_nm=tile_size_nm, halo_nm=halo_nm, corner_nm=corner_nm,
        segment_nm=segment_nm, max_displacement_nm=direct_limit_nm,
        max_file_gib=max_file_gib, max_shapes=max_shapes,
        max_vertices=max_vertices, max_estimated_gib=max_estimated_gib)
    input_ready = perf_counter()
    memory["input"] = process_memory_snapshot()
    try:
        dbu_um = float(metadata["dbu_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DiffOPC 输入缺少有效 dbu_um") from exc
    dbu_nm = dbu_um * 1000.0
    pixel_dbu = _exact_dbu(pixel_nm, dbu_nm, "pixel_nm")
    if not isinstance(target_cache_mb, int) or target_cache_mb < 0:
        raise ValueError("target_cache_mb 必须是非负整数")
    if max_displacement_nm is None:
        maximum_dbu = float(problem.config.max_displacement_dbu)
    else:
        maximum_dbu = max_displacement_nm / dbu_nm
    model = ICCAD13Lithography(device=device)
    config = DiffOPCConfig(
        iterations=iterations, learning_rate=learning_rate_nm / dbu_nm,
        soft_temperature=soft_temperature_nm / dbu_nm,
        weight_l2=weight_l2, weight_pvband=weight_pvband,
        weight_epe=weight_epe, max_displacement_dbu=maximum_dbu,
        epe_distance_dbu=epe_distance_nm / dbu_nm,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=batch_size, raster_chunk_size=raster_chunk_size,
        target_cache_bytes=target_cache_mb * 1024 * 1024)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    model_ready = perf_counter()
    memory["model"] = process_memory_snapshot()
    optimized = optimize(problem, model, config)
    iterated = perf_counter()
    memory["optimization"] = process_memory_snapshot()
    reconstructed = reconstruct_region(problem, optimized.best_displacements)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = _atomic_npz(output / "diffopc_result.npz", {
        "format_name": np.array("myopc.diffopc-result"),
        "format_version": np.array(2, dtype=np.int32),
        "best_displacements": optimized.best_displacements,
        "best_iteration": np.array(optimized.best_iteration, dtype=np.int32),
        "stop_reason": np.array(optimized.stop_reason),
    }, compressed=False)
    layer_spec = problem.physical_mask.layer
    gds_path = write_debug_gds(
        problem.physical_mask.region, reconstructed, output / "diffopc_result.gds",
        dbu_um, layer_spec.layer, layer_spec.datatype)
    final_lithography = save_final_lithography_tiles(
        output / "final_lithography", reconstructed, problem.grid, model,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=batch_size, save_png=save_final_lithography_png)
    preview_path: Path | None = None
    if save_preview:
        reference = problem.segments.materialize()
        moved = problem.segments.materialize(optimized.best_displacements)
        inner, outer = edge_probe_points(
            reference.starts, reference.ends, reference.normals,
            config.epe_distance_dbu)
        preview_path = render_boundary_overlay(
            reconstructed, layer_spec, problem.physical_mask.query_box, dbu_um,
            moved.starts, moved.ends, moved.normals,
            output / "diffopc_result.png", problem.owner_indices, inner, outer,
            problem.grid.cores())
    # Shot 只在最佳矢量结果上以固定 512² 画布估算一次，既不进入梯度，也不让
    # 整张 reticle 高分辨率像素图常驻 CPU/GPU。
    shot_canvas = 512
    query_box = problem.physical_mask.query_box
    shot_pixel_dbu = max(
        1, (max(query_box.width, query_box.height) + shot_canvas - 1) // shot_canvas)
    shot_mask = rasterize_region_canvas(
        reconstructed, query_box, shot_pixel_dbu, shot_canvas)
    shot_estimate = estimate_rectangular_shots(
        torch.as_tensor(shot_mask), shape=(shot_canvas, shot_canvas))
    finished = perf_counter()
    memory["output"] = process_memory_snapshot()
    summary: dict[str, Any] = {
        "status": "completed", "input": str(source),
        "source_layout": metadata.get("source"), "device": str(model.device),
        "dbu_um": dbu_um, "box_dbu": metadata.get("box_dbu"),
        "counts": metadata.get("counts"),
        "config": asdict(config),
        "optimization": {
            "best_iteration": optimized.best_iteration,
            "stop_reason": optimized.stop_reason,
            "shot_estimate": shot_estimate,
            "records": [asdict(record) for record in optimized.records],
        },
        "memory": {
            "problem_persistent_bytes": problem.persistent_nbytes,
            "target_cache_limit_bytes": config.target_cache_bytes,
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(model.device))
                if model.device.type == "cuda" else 0),
            "checkpoints": memory,
        },
        "timing_seconds": {
            "input": input_ready - started,
            "model": model_ready - input_ready,
            "optimization": iterated - model_ready,
            "reconstruct_and_output": finished - iterated,
            "total": finished - started,
        },
        "verification": {
            "reconstructed_valid": bool(reconstructed.has_valid_polygons())},
        "artifacts": {
            "summary": str(output / "summary.json"),
            "result_npz": str(result_path), "gds": str(gds_path),
            "preview": None if preview_path is None else str(preview_path),
            "final_lithography": final_lithography,
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造支持直接版图和离线边段归档的 DiffOPC 参数。"""
    parser = argparse.ArgumentParser(description="运行可微边段 DiffOPC")
    parser.add_argument("input", type=Path, help="输入 GDS/OASIS 或 segment NPZ")
    parser.add_argument("--output-dir", type=Path, default=Path("output/diffopc"))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--learning-rate-nm", type=float, default=1.0)
    parser.add_argument("--soft-temperature-nm", type=float, default=4.0)
    parser.add_argument("--weight-l2", type=float, default=1.0)
    parser.add_argument("--weight-pvband", type=float, default=0.0)
    parser.add_argument("--weight-epe", type=float, default=1.0)
    parser.add_argument("--epe-distance-nm", type=float, default=16.0)
    parser.add_argument("--pixel-nm", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--raster-chunk-size", type=int, default=32)
    parser.add_argument("--target-cache-mb", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-final-lithography-png", action="store_true")
    parser.add_argument("--tile-size-nm", type=float, default=1024.0)
    parser.add_argument("--halo-nm", type=float, default=512.0)
    parser.add_argument("--corner-nm", type=float, default=16.0)
    parser.add_argument("--segment-nm", type=float, default=32.0)
    parser.add_argument("--max-displacement-nm", type=float)
    add_layout_source_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数，打印 JSON 摘要并把可预期错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        result = run_diffopc(
            args.input, args.output_dir, iterations=args.iterations,
            learning_rate_nm=args.learning_rate_nm,
            soft_temperature_nm=args.soft_temperature_nm,
            weight_l2=args.weight_l2, weight_pvband=args.weight_pvband,
            weight_epe=args.weight_epe, epe_distance_nm=args.epe_distance_nm,
            pixel_nm=args.pixel_nm, batch_size=args.batch_size,
            raster_chunk_size=args.raster_chunk_size,
            target_cache_mb=args.target_cache_mb, device=args.device,
            save_preview=args.preview,
            save_final_lithography_png=not args.no_final_lithography_png,
            layer=args.layer, top_cell=args.top_cell,
            box=None if args.box is None else tuple(args.box),
            tile_size_nm=args.tile_size_nm, halo_nm=args.halo_nm,
            corner_nm=args.corner_nm, segment_nm=args.segment_nm,
            max_displacement_nm=args.max_displacement_nm,
            max_file_gib=args.max_file_gib, max_shapes=args.max_shapes,
            max_vertices=args.max_vertices,
            max_estimated_gib=args.max_estimated_gib)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
