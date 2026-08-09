"""直接从 GDS/OASIS 运行流式 simple MB-OPC 并保存最佳全局结果。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from layout import DbuBox, LayerSpec, LayoutDB, LayoutError
from lithography import ICCAD13Lithography
from opc import OPCError
from opc.input import RectilinearCoreGrid
from opc.input.edge import (
    FragmentationConfig,
    prepare_problem,
    reconstruct_region,
    render_boundary_overlay,
    sample_lines,
    save_problem_npz,
    write_debug_gds,
)
from opc.iteration.mbopc import SimpleMBOPCConfig, optimize


def parse_layer(value: str) -> LayerSpec:
    """解析命令行中的 `layer` 或 `layer/datatype`。"""
    parts = value.replace(":", "/").split("/")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("Layer 格式应为 layer 或 layer/datatype")
    try:
        return LayerSpec(int(parts[0]), int(parts[1]) if len(parts) == 2 else 0)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"非法 Layer：{value}") from exc


def build_parser() -> argparse.ArgumentParser:
    """构造可直接运行整张版图或指定 ROI 的 simple MB-OPC 参数。"""
    default_layout = Path(__file__).resolve().parent / "TestReticle" / "simple.gds"
    parser = argparse.ArgumentParser(description="流式运行 simple MB-OPC。")
    parser.add_argument("layout", nargs="?", type=Path, default=default_layout,
                        help="输入 GDS/OASIS，默认 TestReticle/simple.gds")
    parser.add_argument("--top-cell", help="可选顶层 Cell；多顶层版图必须指定")
    parser.add_argument("--layer", type=parse_layer, help="目标 layer/datatype；单层时可省略")
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="可选 DBU 处理范围；默认使用顶层完整 bbox")
    parser.add_argument("--tile-size-nm", type=float, default=1024.0,
                        help="core 正方形边长，默认 1024 nm")
    parser.add_argument("--halo-nm", type=float, default=512.0,
                        help="只读光学上下文宽度，默认 512 nm")
    parser.add_argument("--pixel-nm", type=float, default=8.0,
                        help="一个光刻像素的物理尺寸，默认 8 nm")
    parser.add_argument("--corner-nm", type=float, default=16.0,
                        help="控制边角部段长度，默认 16 nm")
    parser.add_argument("--segment-nm", type=float, default=32.0,
                        help="控制边最大长度，默认 32 nm")
    parser.add_argument("--max-displacement-nm", type=float, default=24.0,
                        help="绝对法向位移上限，默认 24 nm")
    parser.add_argument("--iterations", type=int, default=8, help="最大评价轮数，默认 8")
    parser.add_argument("--step-nm", type=float, default=8.0, help="初始步长，默认 8 nm")
    parser.add_argument("--decay-every", type=int, default=4,
                        help="每多少轮把步长减半，默认 4")
    parser.add_argument("--epe-distance-nm", type=float, default=16.0,
                        help="inner/outer 探针距离，默认 16 nm")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="单次 GPU tile 数，默认 8；显存不足时调小")
    parser.add_argument("--target-cache-mb", type=int, default=512,
                        help="CPU target LRU 上限，0 表示关闭，默认 512 MiB")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--output-dir", type=Path, default=Path("output/mbopc"),
                        help="结果目录，默认 output/mbopc")
    parser.add_argument("--preview", action="store_true",
                        help="额外保存带分段、法向、探针和 core 的诊断 PNG")
    parser.add_argument("--json", action="store_true", help="终端输出完整 JSON")
    return parser


def _exact_dbu(value_nm: float, dbu_nm: float, name: str,
               allow_zero: bool = False) -> int:
    """把必须对齐版图格点的物理长度严格换算为整数 DBU。"""
    if (not np.isfinite(value_nm) or value_nm < 0.0 or
            (value_nm == 0.0 and not allow_zero)):
        requirement = "有限非负数" if allow_zero else "有限正数"
        raise ValueError(f"{name} 必须是{requirement}")
    raw = value_nm / dbu_nm
    rounded = round(raw)
    # 像素、tile 和 halo 决定不同 core 的采样原点；静默取整会让相邻 tile 使用
    # 不同物理晶格，因此只接受可由当前 GDS DBU 精确表达的数值。
    if (rounded < 0 or (rounded == 0 and not allow_zero) or
            not np.isclose(raw, rounded, atol=1e-9, rtol=0.0)):
        raise ValueError(f"{name}={value_nm} nm 不能由当前 {dbu_nm} nm/DBU 精确表达")
    return int(rounded)


def _axis_cuts(start: int, end: int, tile_dbu: int) -> np.ndarray:
    """按固定物理边长生成 cuts，并仅把末端不足部分裁短。"""
    if end <= start or tile_dbu <= 0:
        raise ValueError("core 范围和 tile 大小必须为正")
    count = (end - start + tile_dbu - 1) // tile_dbu
    cuts = start + np.arange(count + 1, dtype=np.int64) * tile_dbu
    cuts[-1] = end
    return cuts


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


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    """在目标目录内原子写入 UTF-8 JSON，异常时清理临时文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行版图读取、前端准备、流式迭代和最佳结果全局一次重建。"""
    started = perf_counter()
    source = args.layout.expanduser().resolve()
    with LayoutDB.open(source, top_cell=args.top_cell) as database:
        layer = _select_layer(database, args.layer)
        bbox = database.bbox()
        if bbox is None:
            raise ValueError("输入顶层 Cell 为空")
        box = DbuBox(*args.box) if args.box else bbox
        dbu_um = database.dbu_um
        batch = database.query([layer], box).materialize()
        loaded = perf_counter()
        dbu_nm = dbu_um * 1000.0
        pixel_dbu = _exact_dbu(args.pixel_nm, dbu_nm, "pixel-nm")
        tile_dbu = _exact_dbu(args.tile_size_nm, dbu_nm, "tile-size-nm")
        halo_dbu = _exact_dbu(args.halo_nm, dbu_nm, "halo-nm", allow_zero=True)
        if tile_dbu % pixel_dbu or halo_dbu % pixel_dbu:
            raise ValueError("tile-size-nm 和 halo-nm 必须是 pixel-nm 的整数倍")
        grid = RectilinearCoreGrid(
            _axis_cuts(box.left, box.right, tile_dbu),
            _axis_cuts(box.bottom, box.top, tile_dbu), halo_dbu)
        fragmentation = FragmentationConfig(
            args.corner_nm / dbu_nm, args.segment_nm / dbu_nm,
            args.max_displacement_nm / dbu_nm)
        # 层级遍历只在 materialize 时发生一次；prepare_problem 批量生成全局参考边、
        # 稳定 segment key 和 owner/context CSR，之后每轮不会重新读取源版图。
        problem = prepare_problem(batch, layer, fragmentation, grid)
    prepared = perf_counter()

    model = ICCAD13Lithography(device=args.device)
    iteration = SimpleMBOPCConfig(
        iterations=args.iterations, initial_step_dbu=args.step_nm / dbu_nm,
        decay_every=args.decay_every,
        max_displacement_dbu=args.max_displacement_nm / dbu_nm,
        epe_distance_dbu=args.epe_distance_nm / dbu_nm,
        pixel_dbu=pixel_dbu, canvas=model.config.canvas,
        batch_size=args.batch_size,
        target_cache_bytes=args.target_cache_mb * 1024 * 1024,
        print_threshold=model.config.print_threshold)
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
    reconstructed = reconstruct_region(
        problem.segments, optimized.best_displacements, fragmentation)
    reference = problem.physical_mask.region
    npz_path = save_problem_npz(
        problem, optimized.best_displacements, output_dir / "mbopc_result.npz")
    gds_path = write_debug_gds(
        reference, reconstructed, output_dir / "mbopc_result.gds",
        dbu_um, layer.layer, layer.datatype)
    preview_path: Path | None = None
    if args.preview:
        geometry = problem.segments.materialize(optimized.best_displacements)
        samples = sample_lines(
            geometry.starts, geometry.ends, geometry.normals, problem.sample_template)
        preview_path = render_boundary_overlay(
            reconstructed, layer, box, dbu_um, geometry.starts, geometry.ends,
            geometry.normals, output_dir / "mbopc_result.png",
            problem.ownership.owner_indices, samples, problem.ownership.cores)
    finished = perf_counter()
    gpu_peak = (int(torch.cuda.max_memory_allocated(model.device))
                if model.device.type == "cuda" else 0)
    result: dict[str, Any] = {
        "source": str(source), "layer": f"{layer.layer}/{layer.datatype}",
        "top_cell": args.top_cell, "dbu_um": dbu_um,
        "box_dbu": [box.left, box.bottom, box.right, box.top],
        "device": str(model.device),
        "tiling": {"columns": grid.column_count, "rows": grid.row_count,
                   "tile_size_nm": args.tile_size_nm, "halo_nm": args.halo_nm,
                   "pixel_nm": args.pixel_nm},
        "counts": {"polygons": problem.physical_mask.contours.polygon_count,
                   "rings": problem.physical_mask.contours.ring_count,
                   "edges": problem.segments.edges.edge_count,
                   "segments": problem.segments.segment_count,
                   "cores": len(problem.ownership.cores),
                   "memberships": len(problem.ownership.member_segment_indices)},
        "optimization": {"best_iteration": optimized.best_iteration,
                         "stop_reason": optimized.stop_reason,
                         "records": [asdict(record) for record in optimized.records]},
        "memory": {"segment_persistent_bytes": problem.segments.persistent_nbytes,
                   "target_cache_limit_bytes": iteration.target_cache_bytes,
                   "gpu_peak_allocated_bytes": gpu_peak},
        "timing_seconds": {"layout_load": loaded - started,
                           "frontend_prepare": prepared - loaded,
                           "iterations": iterated - prepared,
                           "final_reconstruct_and_output": finished - iterated,
                           "total": finished - started},
        "verification": {"reconstructed_valid": bool(reconstructed.has_valid_polygons())},
        "artifacts": {"summary": str(output_dir / "summary.json"),
                      "npz": str(npz_path), "gds": str(gds_path),
                      "preview": None if preview_path is None else str(preview_path)},
    }
    _atomic_json(output_dir / "summary.json", result)
    return result


def print_text(result: dict[str, Any]) -> None:
    """输出最重要的规模、停止原因、耗时和产物位置。"""
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
