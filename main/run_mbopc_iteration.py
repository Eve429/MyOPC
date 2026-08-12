"""直接恢复离线 MBOPCProblem 并独立运行 simple MB-OPC 迭代。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

# main 入口按文件位置引入仓库根，保证外部工作目录下也不需要 pip install。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lithography import ICCAD13Lithography
from main.artifacts import atomic_json, atomic_npz, save_final_lithography_tiles
from main.offline_inputs import load_segment_input
from main.configuration import ConfiguredArgumentParser, exact_dbu
from opc.diagnostics import render_boundary_overlay, write_debug_gds
from opc.input.edge import edge_probe_points, reconstruct_region
from opc.iteration.mbopc import SimpleMBOPCConfig, SimpleMBOPCResult, optimize


def run_mbopc_iteration_test(
        input_path: str | Path, output_dir: str | Path, *, iterations: int = 8,
        step_nm: float = 8.0, decay_every: int = 4,
        epe_distance_nm: float = 16.0, pixel_nm: float = 8.0,
        batch_size: int = 8, target_cache_mb: int = 512,
        device: str = "auto", save_preview: bool = True,
        save_final_lithography_png: bool = True,
        run_configuration: dict[str, object] | None = None) -> SimpleMBOPCResult:
    """仅从离线边段问题运行同步迭代并保存可继续分析的完整结果。"""
    loaded = perf_counter()
    problem, metadata = load_segment_input(input_path)
    restored = perf_counter()
    try:
        dbu_um = float(metadata["dbu_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("边段输入缺少有效 dbu_um") from exc
    dbu_nm = dbu_um * 1000.0
    pixel_dbu = exact_dbu(pixel_nm, dbu_nm, "pixel_nm")
    step_dbu = float(exact_dbu(step_nm, dbu_nm, "step_nm"))
    epe_dbu = float(exact_dbu(epe_distance_nm, dbu_nm, "epe_distance_nm"))
    if not isinstance(target_cache_mb, int) or target_cache_mb < 0:
        raise ValueError("target_cache_mb 必须是非负整数")
    model = ICCAD13Lithography(device=device)
    config = SimpleMBOPCConfig(
        iterations=iterations, initial_step_dbu=step_dbu,
        decay_every=decay_every, epe_distance_dbu=epe_dbu,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=batch_size, target_cache_bytes=target_cache_mb * 1024 * 1024)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    # optimize 每轮只读同一 current，owner 更新到 next 后等待全局屏障。本入口加载的
    # owner/CSR 与原流程完全相同，因此离线运行不会改变跨 core 的同步可见性。
    optimized = optimize(problem, model, config)
    iterated = perf_counter()
    reconstructed = reconstruct_region(problem, optimized.best_displacements)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_lithography = save_final_lithography_tiles(
        output / "final_lithography", reconstructed, problem.grid, model,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=batch_size, save_png=save_final_lithography_png,
        polarity=problem.physical_mask.polarity,
        field_box=problem.physical_mask.query_box)
    layer, box = problem.physical_mask.layer, problem.physical_mask.query_box
    gds_path = write_debug_gds(
        problem.physical_mask.region, reconstructed, output / "mbopc_result.gds",
        dbu_um, layer.layer, layer.datatype)
    result_path = atomic_npz(output / "mbopc_result.npz", {
        "format_name": np.array("myopc.mbopc-result"),
        "format_version": np.array(1, dtype=np.int32),
        "best_displacements": optimized.best_displacements,
        "best_iteration": np.array(optimized.best_iteration, dtype=np.int32),
        "stop_reason": np.array(optimized.stop_reason),
    }, compressed=False)
    preview_path: Path | None = None
    if save_preview:
        geometry = problem.segments.materialize(optimized.best_displacements)
        reference = problem.segments.materialize()
        inner, outer = edge_probe_points(
            reference.starts, reference.ends, reference.normals, config.epe_distance_dbu)
        preview_path = render_boundary_overlay(
            reconstructed, layer, box, dbu_um, geometry.starts, geometry.ends,
            geometry.normals, output / "mbopc_result.png",
            problem.owner_indices, inner, outer, problem.grid.cores())
    finished = perf_counter()
    summary: dict[str, Any] = {
        "run_configuration": run_configuration,
        "input": str(Path(input_path).expanduser().resolve()),
        "source_layout": metadata.get("source"), "device": str(model.device),
        "dbu_um": dbu_um, "box_dbu": metadata.get("box_dbu"),
        "counts": metadata.get("counts"),
        "iteration_config": {
            "iterations": iterations, "step_nm": step_nm,
            "decay_every": decay_every, "epe_distance_nm": epe_distance_nm,
            "pixel_nm": pixel_nm, "batch_size": batch_size,
            "target_cache_mb": target_cache_mb,
        },
        "optimization": {"best_iteration": optimized.best_iteration,
                         "stop_reason": optimized.stop_reason,
                         "records": [asdict(record) for record in optimized.records]},
        "timing_seconds": {"archive_load": restored - loaded,
                           "iterations": iterated - restored,
                           "reconstruct_and_output": finished - iterated,
                           "total": finished - loaded},
        "gpu_peak_allocated_bytes": (int(torch.cuda.max_memory_allocated(model.device))
                                     if model.device.type == "cuda" else 0),
        "verification": {"reconstructed_valid": bool(reconstructed.has_valid_polygons())},
        "artifacts": {"summary": str(output / "summary.json"),
                      "result_npz": str(result_path), "gds": str(gds_path),
                      "preview": None if preview_path is None else str(preview_path),
                      "final_lithography": final_lithography},
    }
    atomic_json(output / "summary.json", summary)
    return optimized


def build_parser() -> argparse.ArgumentParser:
    """构造离线 simple MB-OPC 迭代测试参数。"""
    parser = ConfiguredArgumentParser(
        description="从离线边段问题独立运行 simple MB-OPC。", workflow="mbopc",
        entry="mbopc_iteration",
        valid_entries=("mbopc", "mbopc_frontend", "mbopc_iteration"))
    parser.add_argument("input", type=Path, help="prepare_segment_input 生成的 NPZ")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--step-nm", type=float)
    parser.add_argument("--decay-every", type=int)
    parser.add_argument("--epe-distance-nm", type=float)
    parser.add_argument("--pixel-nm", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-cache-mb", type=int)
    parser.add_argument("--device", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction,
                        help="保存边段、owner、core 和探针标注图")
    parser.add_argument("--no-final-lithography-png", action="store_true",
                        help="只保存最终光刻 NPZ 和 manifest，不保存 tile PNG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把可预期输入、模型或 OPC 错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        result = run_mbopc_iteration_test(
            args.input, args.output_dir, iterations=args.iterations,
            step_nm=args.step_nm, decay_every=args.decay_every,
            epe_distance_nm=args.epe_distance_nm, pixel_nm=args.pixel_nm,
            batch_size=args.batch_size, target_cache_mb=args.target_cache_mb,
            device=args.device, save_preview=args.preview,
            save_final_lithography_png=not args.no_final_lithography_png,
            run_configuration=args._configuration)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"MB-OPC 完成：best={result.best_iteration}，"
          f"stop={result.stop_reason}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
