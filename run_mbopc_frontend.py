"""无需安装项目即可验证 OPC 公共层和 MB-OPC 前端全部功能的主程序。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import klayout.db as kdb
import numpy as np

from geometry import GeometryPatch, PatchSet
from layout import CellRef, DbuBox, LayerSpec, LayoutDB, LayoutError, RegionBatch
from opc import OPCError
from opc.common import RectilinearCoreGrid, render_boundary_overlay, sample_lines
from opc.mbopc import (
    FragmentationConfig,
    MBOPCProblem,
    SegmentUpdateBatch,
    merge_owner_updates,
    prepare_problem,
    reconstruct_region,
    run_geometry_suite,
    save_problem_npz,
    write_debug_gds,
)


def parse_layer(value: str) -> LayerSpec:
    """解析 `layer/datatype` 或单独 layer 参数。"""
    parts = value.replace(":", "/").split("/")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("Layer 格式应为 layer 或 layer/datatype")
    try:
        return LayerSpec(int(parts[0]), int(parts[1]) if len(parts) == 2 else 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"非法 Layer：{value}") from exc


def build_parser() -> argparse.ArgumentParser:
    """构造支持无参数合成验证和真实 GDS 验证的中文命令行。"""
    parser = argparse.ArgumentParser(description="直接验证 OPC 公共层与 MB-OPC 几何前端。")
    parser.add_argument("layout", nargs="?", type=Path, help="可选输入 GDS/OASIS；省略时运行合成测试")
    parser.add_argument("--layer", type=parse_layer, help="真实版图目标 layer/datatype")
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="可选全局 DBU 处理范围；默认使用 top bbox")
    parser.add_argument("--grid", nargs=2, type=int, default=(2, 1), metavar=("COLUMNS", "ROWS"),
                        help="core 网格列数和行数，默认 2 1")
    parser.add_argument("--halo-nm", type=float, default=200.0, help="每个 core 的 halo，默认 200 nm")
    parser.add_argument("--corner-nm", type=float, default=16.0, help="角部段长，默认 16 nm")
    parser.add_argument("--segment-nm", type=float, default=32.0, help="最大段长，默认 32 nm")
    parser.add_argument("--max-displacement-nm", type=float, default=24.0,
                        help="允许的最大法向位移，默认 24 nm")
    parser.add_argument("--demo-displacement-nm", type=float, default=2.0,
                        help="每个 core 示范移动一段的绝对位移，默认 2 nm")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(".benchmarks/mbopc_frontend_demo"), help="验证产物目录")
    parser.add_argument("--json", action="store_true", help="只在终端输出 JSON 汇总")
    parser.add_argument("--skip-geometry-suite", action="store_true",
                        help="跳过多图形标注图集，仅用于快速性能复测")
    return parser


def _demo_batch() -> tuple[RegionBatch, LayerSpec, float]:
    """构造含重叠、孔洞、凹角、斜边和跨 core 长边的合成物理层。"""
    layer, dbu_um = LayerSpec(1, 0), 0.001
    region = kdb.Region()
    region.insert(kdb.Box(0, 0, 180, 60))
    region.insert(kdb.Box(140, 40, 220, 120))
    donut = kdb.Region(kdb.Box(20, 90, 120, 190)) - kdb.Region(kdb.Box(45, 115, 95, 165))
    region += donut
    region.insert(kdb.Polygon([kdb.Point(140, 140), kdb.Point(230, 130),
                              kdb.Point(210, 210), kdb.Point(150, 220)]))
    box = DbuBox(-20, -20, 250, 240)
    return RegionBatch({layer: region}, box, CellRef("SYNTHETIC", 0)), layer, dbu_um


def _axis_cuts(start: int, end: int, count: int) -> np.ndarray:
    """把整数范围均匀切成指定数量且保证严格递增的 core cuts。"""
    if count <= 0 or end - start < count:
        raise ValueError("core grid count is invalid for selected box")
    cuts = start + np.floor(np.arange(count + 1) * (end - start) / count).astype(np.int64)
    cuts[-1] = end
    if np.any(np.diff(cuts) <= 0):
        raise ValueError("core grid produced an empty core")
    return cuts


def _load_database_batch(args: argparse.Namespace,
                         database: LayoutDB) -> tuple[RegionBatch, LayerSpec, float]:
    """在数据库生命周期内选择 Layer、范围并物化局部批次。"""
    layers = database.layers()
    if args.layer is None:
        if len(layers) != 1:
            raise ValueError("多 Layer 版图必须通过 --layer 明确选择")
        layer = layers[0]
    else:
        layer = args.layer
    bbox = database.bbox()
    if bbox is None:
        raise ValueError("输入版图为空")
    box = DbuBox(*args.box) if args.box else bbox
    return database.query([layer], box).materialize(), layer, database.dbu_um


def _prepare_input_problem(args: argparse.Namespace, batch: RegionBatch,
                           layer: LayerSpec, dbu_um: float) -> tuple[
                               MBOPCProblem, RectilinearCoreGrid, FragmentationConfig]:
    """按 CLI 物理尺寸构造 core 网格、分段配置和独立紧凑问题。"""
    dbu_nm = dbu_um * 1000.0
    columns, rows = args.grid
    grid = RectilinearCoreGrid(
        _axis_cuts(batch.query_box.left, batch.query_box.right, columns),
        _axis_cuts(batch.query_box.bottom, batch.query_box.top, rows),
        round(args.halo_nm / dbu_nm))
    config = FragmentationConfig(args.corner_nm / dbu_nm, args.segment_nm / dbu_nm,
                                 args.max_displacement_nm / dbu_nm)
    return prepare_problem(batch, layer, config, grid), grid, config


def _demo_updates(problem: MBOPCProblem,
                  displacement_dbu: float) -> list[SegmentUpdateBatch]:
    """为每个有可拥有边段的 core 选择一段，构造确定性示范更新。"""
    updates: list[SegmentUpdateBatch] = []
    for core_index in range(len(problem.ownership.cores)):
        owned = np.flatnonzero(problem.ownership.owner_indices == core_index)
        if not len(owned):
            continue
        segment_index = int(owned[len(owned) // 2])
        value = displacement_dbu if core_index % 2 == 0 else -displacement_dbu
        updates.append(SegmentUpdateBatch(problem.segments.keys[[segment_index]],
                                          np.array([core_index]), np.array([value])))
    return updates


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行准备、更新、采样、重建、跨 core 拼接和全部产物验证。"""
    if args.layout is None:
        batch, layer, dbu_um = _demo_batch()
        source = "synthetic"
        started = perf_counter()
        problem, grid, config = _prepare_input_problem(args, batch, layer, dbu_um)
    else:
        source = str(args.layout.expanduser().resolve())
        # KLayout 的物化 Region 仍依赖打开的 Layout；因此必须在上下文内
        # 完成物理合并和紧凑数组构建。之后所有迭代与输出均脱离源文件。
        with LayoutDB.open(args.layout, top_cell=None) as database:
            batch, layer, dbu_um = _load_database_batch(args, database)
            started = perf_counter()
            problem, grid, config = _prepare_input_problem(args, batch, layer, dbu_um)
    prepared = perf_counter()
    dbu_nm = dbu_um * 1000.0
    updates = _demo_updates(problem, args.demo_displacement_nm / dbu_nm)
    update_result = merge_owner_updates(problem, updates)
    geometry = problem.segments.materialize(update_result.displacements)
    samples = sample_lines(geometry.starts, geometry.ends, geometry.normals,
                           problem.sample_template)
    reference = reconstruct_region(problem.segments,
                                   np.zeros(problem.segments.segment_count), config)
    reconstructed = reconstruct_region(problem.segments, update_result.displacements, config)
    rebuilt = perf_counter()
    if (reference ^ problem.physical_mask.region).area() != 0:
        raise ValueError("零位移重建与物理参考 mask 不一致")
    patches = PatchSet()
    for core_index, core in enumerate(problem.ownership.cores):
        patches.add(GeometryPatch(f"core-{core_index}", layer, reconstructed,
                                  core.ownership_box))
    clipped_reference = reconstructed & kdb.Region(grid.bounds.to_native())
    stitch_xor = (patches.region(layer) ^ clipped_reference).area()
    if stitch_xor:
        raise ValueError(f"跨 core 拼接 XOR 面积非零：{stitch_xor}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_problem_npz(problem, update_result.displacements, output_dir / "segments.npz")
    gds_path = write_debug_gds(reference, reconstructed, output_dir / "reconstruction.gds",
                               dbu_um, layer.layer, layer.datatype)
    png_path = render_boundary_overlay(
        reconstructed, layer, batch.query_box, dbu_um, geometry.starts, geometry.ends,
        geometry.normals, output_dir / "overview.png", problem.ownership.owner_indices,
        samples, problem.ownership.cores)
    geometry_suite = None if args.skip_geometry_suite else run_geometry_suite(
        output_dir / "geometry_suite")
    finished = perf_counter()
    result: dict[str, Any] = {
        "source": source, "layer": f"{layer.layer}/{layer.datatype}", "dbu_um": dbu_um,
        "box_dbu": [batch.query_box.left, batch.query_box.bottom,
                    batch.query_box.right, batch.query_box.top],
        "counts": {
            "polygons": problem.physical_mask.contours.polygon_count,
            "rings": problem.physical_mask.contours.ring_count,
            "mathematical_edges": problem.segments.edges.edge_count,
            "segments": problem.segments.segment_count,
            "samples": len(samples.points), "cores": len(problem.ownership.cores),
            "memberships": len(problem.ownership.member_segment_indices),
            "updated_segments": len(update_result.changed_segment_indices),
        },
        "memory": {"segment_persistent_bytes": problem.segments.persistent_nbytes},
        "timing_seconds": {
            "prepare": prepared - started, "update_sample_reconstruct": rebuilt - prepared,
            "artifact_output": finished - rebuilt, "total": finished - started,
        },
        "verification": {
            "zero_displacement_xor_area": 0, "stitch_xor_area": int(stitch_xor),
            "reconstructed_valid": bool(reconstructed.has_valid_polygons()),
            "geometry_suite_case_count": 0 if geometry_suite is None else
            geometry_suite["case_count"],
        },
        "artifacts": {"json": str(output_dir / "summary.json"), "npz": str(npz_path),
                      "png": str(png_path), "gds": str(gds_path)},
    }
    summary = output_dir / "summary.json"
    temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, summary)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def print_text(result: dict[str, Any]) -> None:
    """以紧凑中文输出主要计数、性能和产物路径。"""
    counts, timing = result["counts"], result["timing_seconds"]
    print(f"来源：{result['source']}  Layer：{result['layer']}  DBU：{result['dbu_um']} μm")
    print(f"Polygon/Ring/Edge/Segment：{counts['polygons']}/{counts['rings']}/"
          f"{counts['mathematical_edges']}/{counts['segments']}")
    print(f"Core/Context membership/采样点：{counts['cores']}/{counts['memberships']}/"
          f"{counts['samples']}")
    print(f"准备：{timing['prepare'] * 1000:.2f} ms  总计：{timing['total']:.3f} s")
    for name, path in result["artifacts"].items():
        print(f"{name.upper()}：{path}")


def main(argv: list[str] | None = None) -> int:
    """处理 CLI、输出验证结果，并为可预期错误返回稳定退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (LayoutError, OPCError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
