"""无需安装项目即可验证 OPC 公共层和 MB-OPC 前端全部功能的主程序。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

# 所有主入口使用相同的直接运行约定，不修改环境且无需安装当前项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klayout.db as kdb
import numpy as np

from layout import CellRef, DbuBox, LayerSpec, LayoutDB, LayoutError, RegionBatch
from main.configuration import ConfiguredArgumentParser, glp_layer_map, parse_glp_layer
from opc import OPCError
from opc.diagnostics import (
    render_boundary_overlay,
    run_geometry_suite,
    save_problem_npz,
    write_debug_gds,
)
from opc.input import (
    MaskPolarity,
    RectilinearCoreGrid,
    preflight_layout,
    process_memory_snapshot,
    resolve_memory_budget_bytes,
)
from opc.input.edge import (
    FragmentationConfig,
    MBOPCProblem,
    edge_probe_points,
    prepare_problem,
    reconstruct_region,
)
from opc.input.grid import axis_cuts_by_size


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
    parser = ConfiguredArgumentParser(
        description="直接验证 OPC 公共层与 MB-OPC 几何前端。", workflow="mbopc",
        entry="mbopc_frontend",
        valid_entries=("mbopc", "mbopc_frontend", "mbopc_iteration"))
    parser.add_argument("layout", nargs="?", type=Path, help="可选输入 GDS/OASIS；省略时运行合成测试")
    parser.add_argument("--top-cell", help="可选顶层 Cell；多顶层版图必须指定")
    parser.add_argument("--glp-layer", dest="glp_layers", action="append", type=parse_glp_layer)
    parser.add_argument("--polarity", choices=[item.value for item in MaskPolarity])
    parser.add_argument("--layer", type=parse_layer, help="真实版图目标 layer/datatype")
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="可选全局 DBU 处理范围；默认使用 top bbox")
    tiling = parser.add_mutually_exclusive_group()
    tiling.add_argument("--grid", nargs=2, type=int,
                        metavar=("COLUMNS", "ROWS"), help="core 网格列数和行数，默认 2 1")
    tiling.add_argument("--tile-size-nm", type=float, metavar="SIZE",
                        help="按固定正方形边长切分 core，末列和末行自动裁到处理边界")
    parser.add_argument("--halo-nm", type=float, help="每个 core 的 halo，默认 200 nm")
    parser.add_argument("--corner-nm", type=float, help="角部段长，默认 16 nm")
    parser.add_argument("--segment-nm", type=float, help="最大段长，默认 32 nm")
    parser.add_argument("--max-displacement-nm", type=float,
                        help="允许的最大法向位移，默认 24 nm")
    parser.add_argument("--demo-displacement-nm", type=float,
                        help="每个 core 示范移动一段的绝对位移，默认 2 nm")
    parser.add_argument("--probe-distance-nm", type=float,
                        help="诊断图 inner/outer 探针距离，默认 16 nm")
    parser.add_argument("--output-dir", type=Path, help="验证产物目录")
    parser.add_argument("--json", action="store_true", help="只在终端输出 JSON 汇总")
    parser.add_argument("--skip-geometry-suite", action="store_true",
                        help="跳过多图形标注图集，仅用于快速性能复测")
    parser.add_argument("--skip-artifacts", action="store_true",
                        help="跳过 NPZ、GDS、PNG 和图集，仅保留验证摘要")
    parser.add_argument("--preflight-only", action="store_true",
                        help="只扫描真实版图容量，不物化 Region 或边段")
    parser.add_argument("--memory-budget-gib", type=float,
                        help="CPU 内存预算；默认取启动时系统可用内存的 70%%")
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


def _select_layout_scope(args: argparse.Namespace,
                         database: LayoutDB) -> tuple[LayerSpec, DbuBox, float]:
    """在不物化图形的前提下选择真实版图 Layer、范围和 DBU。"""
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
    return layer, box, database.dbu_um


def _problem_configuration(
        args: argparse.Namespace, box: DbuBox, dbu_um: float,
        ) -> tuple[FragmentationConfig, RectilinearCoreGrid]:
    """按 CLI 物理尺寸构造分段配置和规则 core 网格。"""
    dbu_nm = dbu_um * 1000.0
    # 网格数量模式保持原有均分语义；物理尺寸模式只在 CLI 边界做一次 nm→DBU
    # 换算，公共 RectilinearCoreGrid 始终处理整数坐标。两条路径最终都只产生
    # x/y cuts，后续 owner、halo 和重建完全共用原有批量实现，不增加迭代开销。
    if args.tile_size_nm is None:
        columns, rows = args.grid
        x_cuts = _axis_cuts(box.left, box.right, columns)
        y_cuts = _axis_cuts(box.bottom, box.top, rows)
    else:
        if not np.isfinite(args.tile_size_nm) or args.tile_size_nm <= 0.0:
            raise ValueError("tile-size-nm must be finite and positive")
        tile_size_dbu = round(args.tile_size_nm / dbu_nm)
        if tile_size_dbu <= 0:
            raise ValueError("tile-size-nm is smaller than one layout DBU")
        x_cuts = axis_cuts_by_size(box.left, box.right, tile_size_dbu)
        y_cuts = axis_cuts_by_size(box.bottom, box.top, tile_size_dbu)
    grid = RectilinearCoreGrid(
        x_cuts, y_cuts, round(args.halo_nm / dbu_nm))
    config = FragmentationConfig(args.corner_nm / dbu_nm, args.segment_nm / dbu_nm,
                                 args.max_displacement_nm / dbu_nm)
    return config, grid


def _finish_stage(timings: dict[str, float], checkpoints: dict[str, dict[str, int]],
                  name: str, started: float) -> float:
    """结束一个性能阶段，同时记录墙钟耗时和操作系统进程内存。"""
    finished = perf_counter()
    timings[name] = finished - started
    checkpoints[name] = process_memory_snapshot()
    return finished


def _atomic_summary(output_dir: Path, result: dict[str, Any]) -> Path:
    """创建输出目录并原子写入统一 JSON 摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, summary)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def _demo_displacements(problem: MBOPCProblem,
                        displacement_dbu: float) -> tuple[np.ndarray, np.ndarray]:
    """为每个有 owner 边段的 core 选择一段并返回全局对齐位移及变化索引。"""
    values = np.zeros(problem.segments.segment_count, dtype=np.float64)
    changed: list[int] = []
    for core_index in range(problem.core_count):
        members = problem.segments_for_core(core_index)
        owned = members[problem.owner_indices[members] == core_index]
        if not len(owned):
            continue
        segment_index = int(owned[len(owned) // 2])
        values[segment_index] = displacement_dbu if core_index % 2 == 0 else -displacement_dbu
        changed.append(segment_index)
    # 只过滤当前 core 的稀疏 membership，避免每个 core 扫描整条 owner 向量；
    # 仍只选择唯一 owner 数据，索引列表只用于诊断计数，不参与重建。
    return values, np.asarray(changed, dtype=np.int32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行容量预检和完整几何前端，并返回含阶段性能的 JSON 兼容摘要。"""
    total_started = perf_counter()
    timings: dict[str, float] = {}
    checkpoints = {"start": process_memory_snapshot()}
    output_dir = args.output_dir.expanduser().resolve()
    budget = resolve_memory_budget_bytes(args.memory_budget_gib)
    preflight: dict[str, Any]

    if args.layout is None:
        # 合成案例规模固定且很小，不重复写临时 GDS 做层级预检；仍记录构造耗时和
        # 内存检查点。`--preflight-only` 对合成输入只返回明确的无需预检状态。
        stage = perf_counter()
        batch, layer, dbu_um = _demo_batch()
        _finish_stage(timings, checkpoints, "roi_materialize", stage)
        source = "synthetic"
        config, grid = _problem_configuration(args, batch.query_box, dbu_um)
        preflight = {
            "accepted": True, "reason": "synthetic input",
            "recommended_mode": "in_memory", "memory_budget_bytes": budget,
            "scan_complete": True, "counts_are_lower_bounds": False,
        }
        timings["layout_open"] = timings["preflight"] = 0.0
        checkpoints["layout_open"] = checkpoints["preflight"] = process_memory_snapshot()
        if args.preflight_only:
            result = {
                "run_configuration": args._configuration,
                "status": "preflight_only", "source": source,
                "layer": f"{layer.layer}/{layer.datatype}", "dbu_um": dbu_um,
                "box_dbu": [batch.query_box.left, batch.query_box.bottom,
                            batch.query_box.right, batch.query_box.top],
                "preflight": preflight,
                "timing_seconds": timings, "memory_checkpoints": checkpoints,
                "artifacts": {"json": str(output_dir / "summary.json")},
            }
            timings["total"] = perf_counter() - total_started
            _atomic_summary(output_dir, result)
            return result
        stage = perf_counter()
        problem = prepare_problem(batch, layer, config, grid, args.polarity)
        _finish_stage(timings, checkpoints, "problem_prepare", stage)
    else:
        source_path = args.layout.expanduser().resolve()
        source = str(source_path)
        stage = perf_counter()
        # 版图只解析一次并保持数据库打开；严格预检复用原生层级迭代器。只有预检
        # 通过才物化 ROI，避免超限后产生完整 Region 或边段数组。
        if args.polarity == MaskPolarity.OPAQUE.value and not args.box:
            raise ValueError("opaque 极性必须通过 --box 显式提供处理范围")
        with LayoutDB.open(source_path, top_cell=args.top_cell,
                           glp_layer_map=glp_layer_map(args.glp_layers)) as database:
            layer, box, dbu_um = _select_layout_scope(args, database)
            _finish_stage(timings, checkpoints, "layout_open", stage)
            config, grid = _problem_configuration(args, box, dbu_um)
            stage = perf_counter()
            preflight = preflight_layout(
                database, layer=layer, box=box,
                corner_dbu=config.corner_length_dbu,
                maximum_segment_dbu=config.max_segment_length_dbu, grid=grid,
                memory_budget_bytes=budget)
            _finish_stage(timings, checkpoints, "preflight", stage)
            if not preflight["accepted"] or args.preflight_only:
                status = "preflight_only" if preflight["accepted"] else "rejected"
                result = {
                    "run_configuration": args._configuration,
                    "status": status, "source": source,
                    "layer": f"{layer.layer}/{layer.datatype}", "dbu_um": dbu_um,
                    "box_dbu": [box.left, box.bottom, box.right, box.top],
                    "preflight": preflight,
                    "timing_seconds": timings, "memory_checkpoints": checkpoints,
                    "artifacts": {"json": str(output_dir / "summary.json")},
                }
                timings["total"] = perf_counter() - total_started
                checkpoints["total"] = process_memory_snapshot()
                _atomic_summary(output_dir, result)
                return result
            stage = perf_counter()
            batch = database.query([layer], box).materialize()
            _finish_stage(timings, checkpoints, "roi_materialize", stage)
            stage = perf_counter()
            problem = prepare_problem(batch, layer, config, grid, args.polarity)
            _finish_stage(timings, checkpoints, "problem_prepare", stage)

    dbu_nm = dbu_um * 1000.0
    stage = perf_counter()
    displacements, changed = _demo_displacements(
        problem, args.demo_displacement_nm / dbu_nm)
    _finish_stage(timings, checkpoints, "demo_update", stage)

    stage = perf_counter()
    # 当前验证入口明确检查按需端点和 EPE probe；它们不是 problem 常驻字段。超大
    # 输入已在预检阶段拒绝，因此这里不会用诊断功能绕过容量保护。
    geometry = problem.segments.materialize(displacements)
    reference_geometry = problem.segments.materialize()
    inner, outer = edge_probe_points(
        reference_geometry.starts, reference_geometry.ends, reference_geometry.normals,
        args.probe_distance_nm / dbu_nm)
    _finish_stage(timings, checkpoints, "segment_materialize_and_probes", stage)

    stage = perf_counter()
    reference = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
    reconstructed = reconstruct_region(problem, displacements)
    _finish_stage(timings, checkpoints, "reconstruct", stage)

    stage = perf_counter()
    if (reference ^ problem.physical_mask.region).area() != 0:
        raise ValueError("零位移重建与物理参考 mask 不一致")
    # core 只划分责任而不裁最终 Polygon。覆盖与面积和同时检查，分别捕获缺口和
    # 正面积重叠；共享边界允许存在，不会把斜边交点重复量化。
    core_coverage = kdb.Region()
    core_area_sum = 0
    cores = problem.grid.cores()
    for core in cores:
        core_coverage.insert(core.ownership_box.to_native())
        core_area_sum += core.ownership_box.width * core.ownership_box.height
    core_coverage = core_coverage.merged()
    coverage_xor = (core_coverage ^ kdb.Region(grid.bounds.to_native())).area()
    overlap_area = core_area_sum - core_coverage.area()
    if coverage_xor or overlap_area:
        raise ValueError(f"core ownership 覆盖无效：XOR={coverage_xor}，重叠={overlap_area}")
    _finish_stage(timings, checkpoints, "verification", stage)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = gds_path = png_path = None
    geometry_suite = None
    for name in ("npz", "gds", "png", "geometry_suite"):
        timings[name] = 0.0
        checkpoints[name] = process_memory_snapshot()
    if not args.skip_artifacts:
        stage = perf_counter()
        npz_path = save_problem_npz(problem, displacements, output_dir / "segments.npz")
        _finish_stage(timings, checkpoints, "npz", stage)
        stage = perf_counter()
        gds_path = write_debug_gds(
            reference, reconstructed, output_dir / "reconstruction.gds",
            dbu_um, layer.layer, layer.datatype)
        _finish_stage(timings, checkpoints, "gds", stage)
        stage = perf_counter()
        png_path = render_boundary_overlay(
            reconstructed, layer, batch.query_box, dbu_um, geometry.starts, geometry.ends,
            geometry.normals, output_dir / "overview.png", problem.owner_indices,
            inner, outer, cores)
        _finish_stage(timings, checkpoints, "png", stage)
        if not args.skip_geometry_suite:
            stage = perf_counter()
            geometry_suite = run_geometry_suite(output_dir / "geometry_suite")
            _finish_stage(timings, checkpoints, "geometry_suite", stage)

    timings["total"] = perf_counter() - total_started
    checkpoints["total"] = process_memory_snapshot()
    result: dict[str, Any] = {
        "run_configuration": args._configuration,
        "status": "completed", "source": source,
        "layer": f"{layer.layer}/{layer.datatype}", "dbu_um": dbu_um,
        "box_dbu": [batch.query_box.left, batch.query_box.bottom,
                    batch.query_box.right, batch.query_box.top],
        "preflight": preflight,
        "tiling": {
            "mode": "physical_size" if args.tile_size_nm is not None else "count",
            "columns": grid.column_count, "rows": grid.row_count,
            "requested_tile_size_nm": args.tile_size_nm,
        },
        "counts": {
            "polygons": problem.segments.contours.polygon_count,
            "rings": problem.segments.contours.ring_count,
            "mathematical_edges": len(problem.segments.contours.vertices),
            "segments": problem.segments.segment_count, "samples": len(inner) * 2,
            "cores": problem.core_count, "memberships": len(problem.member_segment_indices),
            "updated_segments": len(changed),
        },
        "memory": {
            "problem_persistent_bytes": problem.persistent_nbytes,
            "segment_persistent_bytes": problem.segments.persistent_nbytes,
            "peak_working_set_bytes": max(
                item["peak_working_set_bytes"] for item in checkpoints.values()),
        },
        "timing_seconds": timings, "memory_checkpoints": checkpoints,
        "verification": {
            "zero_displacement_xor_area": 0,
            "core_coverage_xor_area": int(coverage_xor),
            "core_overlap_area": int(overlap_area),
            "reconstructed_valid": bool(reconstructed.has_valid_polygons()),
            "geometry_suite_case_count": 0 if geometry_suite is None else
            geometry_suite["case_count"],
        },
        "artifacts": {
            "json": str(output_dir / "summary.json"),
            "npz": None if npz_path is None else str(npz_path),
            "png": None if png_path is None else str(png_path),
            "gds": None if gds_path is None else str(gds_path),
        },
    }
    _atomic_summary(output_dir, result)
    return result


def print_text(result: dict[str, Any]) -> None:
    """以紧凑中文输出主要计数、性能和产物路径。"""
    if result["status"] != "completed":
        preflight = result["preflight"]
        print(f"状态：{result['status']}  原因：{preflight['reason']}")
        if "estimated_segments" in preflight:
            suffix = "（下界）" if preflight["counts_are_lower_bounds"] else ""
            print(f"估算 Segment：{preflight['estimated_segments']:,}{suffix}  "
                  f"准备峰值：{preflight['estimated_prepare_peak_bytes'] / 1024 ** 3:.3f} GiB")
        print(f"JSON：{result['artifacts']['json']}")
        return
    counts, timing = result["counts"], result["timing_seconds"]
    print(f"来源：{result['source']}  Layer：{result['layer']}  DBU：{result['dbu_um']} μm")
    print(f"Polygon/Ring/Edge/Segment：{counts['polygons']}/{counts['rings']}/"
          f"{counts['mathematical_edges']}/{counts['segments']}")
    print(f"Core/Context membership/采样点：{counts['cores']}/{counts['memberships']}/"
          f"{counts['samples']}")
    print(f"准备：{timing['problem_prepare'] * 1000:.2f} ms  总计：{timing['total']:.3f} s")
    print(f"进程峰值工作集：{result['memory']['peak_working_set_bytes'] / 1024 ** 3:.3f} GiB")
    for name, path in result["artifacts"].items():
        if path is not None:
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
    return 2 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
