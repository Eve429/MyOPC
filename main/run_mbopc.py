"""直接从 GDS/OASIS 运行流式 simple MB-OPC 并保存最佳全局结果。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

# 由脚本位置解析项目根，保证 `python main/run_mbopc.py` 可从任意目录运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluation import estimate_rectangular_shots
from layout import DbuBox, LayerSpec, LayoutDB, LayoutError
from lithography import ICCAD13Lithography
from main.offline_inputs import (
    _atomic_json,
    _exact_dbu,
    parse_layer,
    save_final_lithography_tiles,
)
from main.configuration import ConfiguredArgumentParser, glp_layer_map, parse_glp_layer
from opc import OPCError
from opc.diagnostics import render_boundary_overlay, write_debug_gds
from opc.input import (
    MaskPolarity,
    RectilinearCoreGrid,
    preflight_layout,
    process_memory_snapshot,
    resolve_memory_budget_bytes,
)
from opc.input.edge import (
    FragmentationConfig,
    edge_probe_points,
    prepare_problem,
    reconstruct_region,
)
from opc.input.grid import axis_cuts_by_size
from opc.input.raster import rasterize_mask_canvas
from opc.iteration.mbopc import SimpleMBOPCConfig, optimize


def build_parser() -> argparse.ArgumentParser:
    """构造可直接运行整张版图或指定 ROI 的 simple MB-OPC 参数。"""
    default_layout = PROJECT_ROOT / "TestReticle" / "simple.gds"
    parser = ConfiguredArgumentParser(
        description="流式运行 simple MB-OPC。", workflow="mbopc", entry="mbopc",
        valid_entries=("mbopc", "mbopc_frontend", "mbopc_iteration"))
    parser.add_argument("layout", nargs="?", type=Path, default=default_layout,
                        help="输入 GDS/OASIS，默认 TestReticle/simple.gds")
    parser.add_argument("--top-cell", help="可选顶层 Cell；多顶层版图必须指定")
    parser.add_argument("--glp-layer", dest="glp_layers", action="append", type=parse_glp_layer)
    parser.add_argument("--layer", type=parse_layer, help="目标 layer/datatype；单层时可省略")
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="可选 DBU 处理范围；默认使用顶层完整 bbox")
    parser.add_argument("--polarity", choices=[item.value for item in MaskPolarity],
                        help="源多边形为透光 clear 或不透光 opaque")
    parser.add_argument("--tile-size-nm", type=float,
                        help="core 正方形边长，默认 1024 nm")
    parser.add_argument("--halo-nm", type=float,
                        help="只读光学上下文宽度，默认 512 nm")
    parser.add_argument("--pixel-nm", type=float,
                        help="一个光刻像素的物理尺寸，默认 8 nm")
    parser.add_argument("--corner-nm", type=float,
                        help="控制边角部段长度，默认 16 nm")
    parser.add_argument("--segment-nm", type=float,
                        help="控制边最大长度，默认 32 nm")
    parser.add_argument("--max-displacement-nm", type=float,
                        help="绝对法向位移上限，默认 24 nm")
    parser.add_argument("--iterations", type=int, help="最大评价轮数，默认 8")
    parser.add_argument("--step-nm", type=float, help="初始步长，默认 8 nm")
    parser.add_argument("--decay-every", type=int,
                        help="每多少轮把步长减半，默认 4")
    parser.add_argument("--epe-distance-nm", type=float,
                        help="inner/outer 探针距离，默认 16 nm")
    parser.add_argument("--batch-size", type=int,
                        help="单次 GPU tile 数，默认 8；显存不足时调小")
    parser.add_argument("--target-cache-mb", type=int,
                        help="CPU target LRU 上限，0 表示关闭，默认 512 MiB")
    parser.add_argument("--device", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--output-dir", type=Path,
                        help="结果目录，默认 output/mbopc")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction,
                        help="是否保存带分段、法向、探针和 core 的诊断 PNG")
    parser.add_argument("--no-final-lithography-png", action="store_true",
                        help="只保存最终光刻 NPZ 和 manifest，不保存 tile PNG")
    parser.add_argument("--json", action="store_true", help="终端输出完整 JSON")
    parser.add_argument("--preflight-only", action="store_true",
                        help="只扫描版图容量，不物化 Region、边段或光刻模型")
    parser.add_argument("--memory-budget-gib", type=float,
                        help="CPU 内存预算；默认取启动时系统可用内存的 70%%")
    return parser


def _select_layer(database: LayoutDB, requested: LayerSpec | None) -> LayerSpec:
    """选择显式目标层，或在版图仅有一个层时自动选择。"""
    layers = database.layers()
    if requested is not None:
        if requested not in layers:
            raise ValueError(f"版图中不存在 Layer {requested.layer}/{requested.datatype}")
        return requested
    if len(layers) != 1:
        names = ", ".join(f"{layer.layer}/{layer.datatype}" for layer in layers)
        raise ValueError(f"版图包含多个 Layer，请用 --layer 选择：{names}")
    return layers[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行版图读取、前端准备、流式迭代和最佳结果全局一次重建。"""
    started = perf_counter()
    memory_checkpoints = {"start": process_memory_snapshot()}
    source = args.layout.expanduser().resolve()
    if args.polarity == MaskPolarity.OPAQUE.value and not args.box:
        raise ValueError("opaque 极性必须通过 --box 显式提供处理范围")
    with LayoutDB.open(source, top_cell=args.top_cell,
                       glp_layer_map=glp_layer_map(args.glp_layers)) as database:
        layer = _select_layer(database, args.layer)
        bbox = database.bbox()
        if bbox is None:
            raise ValueError("输入顶层 Cell 为空")
        box = DbuBox(*args.box) if args.box else bbox
        dbu_um = database.dbu_um
        loaded = perf_counter()
        memory_checkpoints["layout_open"] = process_memory_snapshot()
        dbu_nm = dbu_um * 1000.0
        pixel_dbu = _exact_dbu(args.pixel_nm, dbu_nm, "pixel-nm")
        tile_dbu = _exact_dbu(args.tile_size_nm, dbu_nm, "tile-size-nm")
        halo_dbu = _exact_dbu(args.halo_nm, dbu_nm, "halo-nm", allow_zero=True)
        if tile_dbu % pixel_dbu or halo_dbu % pixel_dbu:
            raise ValueError("tile-size-nm 和 halo-nm 必须是 pixel-nm 的整数倍")
        grid = RectilinearCoreGrid(
            axis_cuts_by_size(box.left, box.right, tile_dbu),
            axis_cuts_by_size(box.bottom, box.top, tile_dbu), halo_dbu)
        fragmentation = FragmentationConfig(
            args.corner_nm / dbu_nm, args.segment_nm / dbu_nm,
            args.max_displacement_nm / dbu_nm)
        preflight_started = perf_counter()
        preflight = preflight_layout(
            database, layer=layer, box=box,
            corner_dbu=fragmentation.corner_length_dbu,
            maximum_segment_dbu=fragmentation.max_segment_length_dbu, grid=grid,
            memory_budget_bytes=resolve_memory_budget_bytes(args.memory_budget_gib))
        preflight_finished = perf_counter()
        memory_checkpoints["preflight"] = process_memory_snapshot()
        if not preflight["accepted"] or args.preflight_only:
            output_dir = args.output_dir.expanduser().resolve()
            status = "preflight_only" if preflight["accepted"] else "rejected"
            result = {
                "status": status, "source": str(source),
                "layer": f"{layer.layer}/{layer.datatype}",
                "top_cell": database.top_cell.name, "dbu_um": dbu_um,
                "box_dbu": [box.left, box.bottom, box.right, box.top],
                "preflight": preflight, "memory_checkpoints": memory_checkpoints,
                "timing_seconds": {
                    "layout_load": loaded - started,
                    "preflight": preflight_finished - preflight_started,
                    "total": perf_counter() - started,
                },
                "artifacts": {"summary": str(output_dir / "summary.json")},
                "run_configuration": args._configuration,
            }
            _atomic_json(output_dir / "summary.json", result)
            return result
        materialize_started = perf_counter()
        batch = database.query([layer], box).materialize()
        materialized = perf_counter()
        memory_checkpoints["roi_materialize"] = process_memory_snapshot()
        # 层级遍历只在 materialize 时发生一次；prepare_problem 批量生成全局参考边、
        # 参数化 segment 和 owner/context CSR，之后每轮不会重新读取源版图。
        problem = prepare_problem(
            batch, layer, fragmentation, grid, args.polarity,
            max_memberships=int(preflight["max_memberships"]))
    prepared = perf_counter()
    memory_checkpoints["problem_prepare"] = process_memory_snapshot()

    model = ICCAD13Lithography(device=args.device)
    iteration = SimpleMBOPCConfig(
        iterations=args.iterations, initial_step_dbu=args.step_nm / dbu_nm,
        decay_every=args.decay_every,
        epe_distance_dbu=args.epe_distance_nm / dbu_nm,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=args.batch_size,
        target_cache_bytes=args.target_cache_mb * 1024 * 1024)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    # optimize 对所有 batch 只读同一 current，并把 owner 方向暂存到 next；只有整轮
    # 评价完成且全局轮廓合法才跨越屏障发布，因此 tile 顺序不会提前改变后续边段。
    optimized = optimize(problem, model, iteration)
    iterated = perf_counter()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 最佳状态在所有 tile 完成后只做一次全局矢量重建。core 从不裁最终 Polygon，
    # 所以跨 core 与斜边不会产生两套取整端点；halo 也从不回写。
    reconstructed = reconstruct_region(problem, optimized.best_displacements)
    final_lithography = save_final_lithography_tiles(
        output_dir / "final_lithography", reconstructed, problem.grid, model,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=args.batch_size, save_png=not args.no_final_lithography_png,
        polarity=problem.physical_mask.polarity, field_box=box)
    reference = problem.physical_mask.region
    # Shot 只在最终最佳几何上计算一次。固定 512×512 诊断画布使内存有严格上界；
    # 像素 DBU 向上取整以完整覆盖 ROI，避免为整张 reticle 常驻高分辨率 mask。
    shot_canvas = 512
    shot_pixel_dbu = max(1, (max(box.width, box.height) + shot_canvas - 1) // shot_canvas)
    shot_mask = rasterize_mask_canvas(
        reconstructed, box, shot_pixel_dbu, shot_canvas,
        polarity=problem.physical_mask.polarity, field_box=box)
    shot_estimate = estimate_rectangular_shots(
        torch.as_tensor(shot_mask), shape=(shot_canvas, shot_canvas))
    del shot_mask
    gds_path = write_debug_gds(
        reference, reconstructed, output_dir / "mbopc_result.gds",
        dbu_um, layer.layer, layer.datatype)
    preview_path: Path | None = None
    if args.preview:
        geometry = problem.segments.materialize(optimized.best_displacements)
        reference_geometry = problem.segments.materialize()
        inner, outer = edge_probe_points(
            reference_geometry.starts, reference_geometry.ends,
            reference_geometry.normals, iteration.epe_distance_dbu)
        preview_path = render_boundary_overlay(
            reconstructed, layer, box, dbu_um, geometry.starts, geometry.ends,
            geometry.normals, output_dir / "mbopc_result.png",
            problem.owner_indices, inner, outer, problem.grid.cores())
    finished = perf_counter()
    gpu_peak = (int(torch.cuda.max_memory_allocated(model.device))
                if model.device.type == "cuda" else 0)
    result: dict[str, Any] = {
        "status": "completed", "source": str(source),
        "layer": f"{layer.layer}/{layer.datatype}",
        "top_cell": args.top_cell, "dbu_um": dbu_um,
        "polarity": problem.physical_mask.polarity.value,
        "box_dbu": [box.left, box.bottom, box.right, box.top],
        "device": str(model.device),
        "run_configuration": args._configuration,
        "tiling": {"columns": grid.column_count, "rows": grid.row_count,
                   "tile_size_nm": args.tile_size_nm, "halo_nm": args.halo_nm,
                   "pixel_nm": args.pixel_nm},
        "counts": {"polygons": problem.segments.contours.polygon_count,
                   "rings": problem.segments.contours.ring_count,
                   "edges": len(problem.segments.contours.vertices),
                   "segments": problem.segments.segment_count,
                   "cores": problem.core_count,
                   "memberships": len(problem.member_segment_indices)},
        "optimization": {"best_iteration": optimized.best_iteration,
                         "stop_reason": optimized.stop_reason,
                         "shot_estimate": shot_estimate,
                         "shot_evaluation_shape": [shot_canvas, shot_canvas],
                         "records": [asdict(record) for record in optimized.records]},
        "memory": {"problem_persistent_bytes": problem.persistent_nbytes,
                   "segment_persistent_bytes": problem.segments.persistent_nbytes,
                   "target_cache_limit_bytes": iteration.target_cache_bytes,
                   "gpu_peak_allocated_bytes": gpu_peak},
        "preflight": preflight, "memory_checkpoints": memory_checkpoints,
        "timing_seconds": {"layout_load": loaded - started,
                           "preflight": preflight_finished - preflight_started,
                           "roi_materialize": materialized - materialize_started,
                           "frontend_prepare": prepared - materialized,
                           "iterations": iterated - prepared,
                           "final_reconstruct_and_output": finished - iterated,
                           "total": finished - started},
        "verification": {"reconstructed_valid": bool(reconstructed.has_valid_polygons())},
        "artifacts": {"summary": str(output_dir / "summary.json"),
                      "gds": str(gds_path),
                      "preview": None if preview_path is None else str(preview_path),
                      "final_lithography": final_lithography},
    }
    _atomic_json(output_dir / "summary.json", result)
    return result


def print_text(result: dict[str, Any]) -> None:
    """输出最重要的规模、停止原因、耗时和产物位置。"""
    if result["status"] != "completed":
        print(f"状态：{result['status']}  原因：{result['preflight']['reason']}")
        print(f"SUMMARY：{result['artifacts']['summary']}")
        return
    counts, optimization = result["counts"], result["optimization"]
    print(f"来源：{result['source']}  Layer：{result['layer']}  Device：{result['device']}")
    print(f"Polygon/Edge/Segment/Core：{counts['polygons']}/{counts['edges']}/"
          f"{counts['segments']}/{counts['cores']}")
    print(f"最佳轮次：{optimization['best_iteration']}  停止：{optimization['stop_reason']}")
    print(f"总耗时：{result['timing_seconds']['total']:.3f} s")
    for name, path in result["artifacts"].items():
        if path is not None:
            print(f"{name.upper()}：{path}")


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把可预期输入、版图和 OPC 异常转换为退出码 2。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (LayoutError, OPCError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 2 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
