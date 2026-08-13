"""无需安装项目即可验证 OPC 公共层和 MB-OPC 前端全部功能的主程序。"""

from __future__ import annotations

import argparse
import json
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
from main.artifacts import atomic_json
from main.configuration import (
    ConfiguredArgumentParser, exact_dbu, glp_layer_map, parse_glp_layer, parse_layer_spec,
)
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
    macro_boxes,
    preflight_layout,
    process_memory_snapshot,
    resolve_memory_budget_bytes,
)
from opc.input.edge import (
    FragmentationConfig,
    MBOPCProblem,
    edge_probe_points,
    prepare_problem,
    prepare_macro,
    reconstruct_region,
)
from opc.input.grid import axis_cuts_by_size
from opc.input.raster import rasterize_mask_canvas


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
    parser.add_argument("--layer", type=parse_layer_spec,
                        help="真实版图目标 layer/datatype")
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="可选全局 DBU 处理范围；默认使用 top bbox")
    tiling = parser.add_mutually_exclusive_group()
    tiling.add_argument("--grid", nargs=2, type=int,
                        metavar=("COLUMNS", "ROWS"), help="core 网格列数和行数，默认 2 1")
    tiling.add_argument("--tile-size-nm", type=float, metavar="SIZE",
                        help="按固定正方形边长切分 core，末列和末行自动裁到处理边界")
    parser.add_argument("--tile-halo-nm", type=float,
                        help="每个 tile 的光刻只读上下文")
    parser.add_argument("--roi-halo-nm", type=float,
                        help="每个 CPU macro 物化完整图形的额外范围")
    parser.add_argument("--macro-size-nm", type=float,
                        help="CPU macro 最大边长；边界自动对齐 tile 切线")
    parser.add_argument("--pixel-nm", type=float,
                        help="macro 栅格逐 tile 对照的物理像素尺寸")
    parser.add_argument("--macro-verify", action="store_true",
                        help="逐 macro 验证未裁剪提边和栅格化时裁剪，不运行全局前端")
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
        x_cuts, y_cuts,
        exact_dbu(args.tile_halo_nm, dbu_nm, "tile-halo-nm", allow_zero=True))
    config = FragmentationConfig(args.corner_nm / dbu_nm, args.segment_nm / dbu_nm,
                                 args.max_displacement_nm / dbu_nm)
    return config, grid


def _run_macro_verification(
        args: argparse.Namespace, database: LayoutDB, layer: LayerSpec,
        box: DbuBox, dbu_um: float, config: FragmentationConfig,
        tile_grid: RectilinearCoreGrid, output_dir: Path,
        started: float, timings: dict[str, float],
        checkpoints: dict[str, dict[str, int]], memory_budget_bytes: int,
        global_preflight: dict[str, Any]) -> dict[str, Any]:
    """逐 macro 准备真实边并逐 tile 对照精确裁剪栅格，返回有界验证摘要。"""
    dbu_nm = dbu_um * 1000.0
    macro_dbu = exact_dbu(args.macro_size_nm, dbu_nm, "macro-size-nm")
    roi_halo_dbu = exact_dbu(
        args.roi_halo_nm, dbu_nm, "roi-halo-nm", allow_zero=True)
    pixel_dbu = exact_dbu(args.pixel_nm, dbu_nm, "pixel-nm")
    required_roi_halo = tile_grid.halo_dbu + int(np.ceil(config.max_displacement_dbu))
    if roi_halo_dbu < required_roi_halo:
        raise ValueError(
            f"roi-halo-nm 过小：至少需要 {required_roi_halo * dbu_nm:.3f} nm，"
            "以覆盖 tile 光学上下文和最大允许边位移")
    if tile_grid.halo_dbu % pixel_dbu:
        raise ValueError("tile-halo-nm 必须是 pixel-nm 的整数倍")
    macros = macro_boxes(tile_grid, macro_dbu)
    mismatch_pixels = duplicate_owner_tiles = 0
    total_owned = total_active = total_memberships = 0
    peak_macro_bytes = 0
    peak_macro_snapshot = checkpoints["preflight"]
    written_cores = np.zeros(tile_grid.core_count, dtype=np.bool_)
    stage = perf_counter()
    for macro_index, ownership_box in enumerate(macros):
        context_box = ownership_box.expanded(roi_halo_dbu)
        # 局部预检只估算当前 macro 所含 tile 的 membership；若仍传全局网格，
        # 一个跨整片的完整 occurrence 会在每个 macro 重复计入所有远端 tile，
        # 从而错误拒绝本可流式处理的输入。正式 owner 随后仍使用全局 tile_grid。
        local_grid = RectilinearCoreGrid(
            tile_grid.x_cuts[(tile_grid.x_cuts >= ownership_box.left) &
                             (tile_grid.x_cuts <= ownership_box.right)],
            tile_grid.y_cuts[(tile_grid.y_cuts >= ownership_box.bottom) &
                             (tile_grid.y_cuts <= ownership_box.top)],
            tile_grid.halo_dbu)
        local_preflight = preflight_layout(
            database, layer=layer, box=context_box,
            corner_dbu=config.corner_length_dbu,
            maximum_segment_dbu=config.max_segment_length_dbu,
            grid=local_grid, memory_budget_bytes=memory_budget_bytes,
            include_layout_load_bytes=False)
        if not local_preflight["accepted"]:
            raise MemoryError(
                f"macro {macro_index} {ownership_box} 容量预检失败："
                f"{local_preflight['reason']}")
        # 这里只筛选与 context 相交的 occurrence 并保留完整 Polygon；查询框不做
        # 布尔相交，因此不会成为边。当前 macro 完成后 batch/prepared 均立即释放。
        batch = database.query([layer], context_box).materialize_intersecting()
        prepared = prepare_macro(
            batch, layer, config, tile_grid, ownership_box, args.polarity,
            max_memberships=int(local_preflight["max_memberships"]))
        total_active += len(prepared.active_segment_indices)
        owned_segments = prepared.owned_segments()
        total_owned += len(owned_segments)
        total_memberships += len(prepared.member_segment_indices)
        peak_macro_bytes = max(
            peak_macro_bytes,
            prepared.segments.persistent_nbytes + prepared.active_segment_indices.nbytes +
            prepared.active_owner_indices.nbytes +
            prepared.core_indices.nbytes + prepared.core_offsets.nbytes +
            prepared.member_segment_indices.nbytes)
        # macro 由完整 tile 组成，segment 唯一写入由它的全局 owner tile 决定。
        # 因此只需验证每个 tile 恰好落入一个 macro；不能保存全局边段签名集合，
        # 否则验证器自身会重新引入 O(整片 segment 数) 的 Python 对象内存。
        duplicate_owner_tiles += int(np.count_nonzero(written_cores[prepared.core_indices]))
        written_cores[prepared.core_indices] = True
        for core_index in prepared.core_indices:
            core = tile_grid.core(int(core_index))
            width = (core.context_box.width + pixel_dbu - 1) // pixel_dbu
            height = (core.context_box.height + pixel_dbu - 1) // pixel_dbu
            canvas = max(width, height)
            actual = rasterize_mask_canvas(
                prepared.physical_mask.region, core.context_box, pixel_dbu, canvas,
                polarity=prepared.physical_mask.polarity, field_box=box)
            # 对照路径使用既有精确裁剪 materialize；它只物化当前 tile context，
            # 不构造整 ROI。两张画布比较后立即释放，峰值不随 macro/tile 数增长。
            expected_batch = database.query([layer], core.context_box).materialize()
            expected = rasterize_mask_canvas(
                expected_batch.region(layer), core.context_box, pixel_dbu, canvas,
                polarity=prepared.physical_mask.polarity, field_box=box)
            mismatch_pixels += int(np.count_nonzero(~np.isclose(
                actual, expected, atol=1e-6, rtol=0.0)))
            del actual, expected, expected_batch
        del prepared, batch
        snapshot = process_memory_snapshot()
        if snapshot["peak_working_set_bytes"] > peak_macro_snapshot["peak_working_set_bytes"]:
            peak_macro_snapshot = snapshot
    timings["macro_prepare_and_raster"] = perf_counter() - stage
    if duplicate_owner_tiles or not np.all(written_cores):
        raise RuntimeError("macro tile 覆盖必须无重复且完整")
    timings["total"] = perf_counter() - started
    checkpoints["macro_peak"] = peak_macro_snapshot
    checkpoints["total"] = process_memory_snapshot()
    result: dict[str, Any] = {
        "run_configuration": args._configuration,
        "status": "macro_verified", "source": str(database.source_path),
        "layer": f"{layer.layer}/{layer.datatype}", "dbu_um": dbu_um,
        "box_dbu": [box.left, box.bottom, box.right, box.top],
        "preflight": global_preflight,
        "tiling": {
            "tile_columns": tile_grid.column_count, "tile_rows": tile_grid.row_count,
            "macro_count": len(macros), "tile_size_nm": args.tile_size_nm,
            "macro_size_nm": args.macro_size_nm,
            "tile_halo_nm": args.tile_halo_nm, "roi_halo_nm": args.roi_halo_nm,
            "pixel_nm": args.pixel_nm,
        },
        "counts": {
            "macro_active_segments": total_active,
            "macro_owned_segments": total_owned,
            "macro_memberships": total_memberships,
        },
        "memory": {"peak_macro_array_bytes": peak_macro_bytes},
        "timing_seconds": timings, "memory_checkpoints": checkpoints,
        "verification": {
            "raster_mismatch_pixels": mismatch_pixels,
            "duplicate_owned_segments": 0,
            "materialization_mode": "complete_intersecting_shapes",
        },
        "artifacts": {"json": str(output_dir / "summary.json")},
    }
    atomic_json(output_dir / "summary.json", result)
    return result


def _finish_stage(timings: dict[str, float], checkpoints: dict[str, dict[str, int]],
                  name: str, started: float) -> float:
    """结束一个性能阶段，同时记录墙钟耗时和操作系统进程内存。"""
    finished = perf_counter()
    timings[name] = finished - started
    checkpoints[name] = process_memory_snapshot()
    return finished


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
    """执行容量预检和完整几何前端，并返回含阶段性能的 JSON 兼容摘要。

    这是基于模型 OPC（MB-OPC）几何前端的验证入口，只跑到 `prepare_problem`
    为止，不接入光刻模型、不做任何迭代求解。它把合并后的物理 mask 转成
    SegmentBatch 与 owner/context CSR——一份可被多轮迭代复用的只读参考几何，
    再用一个 demo 法向位移去检验 materialize（按需端点）、reconstruct_region
    （位移→Region 重建）和 ownership 覆盖是否自洽。管线位置在
    `layout → geometry → opc.input.edge`；它不依赖 lithography/evaluation。
    输入：args —— 已合并 TOML 默认与 CLI 的 Namespace。
    输出：含各阶段耗时、内存检查点、计数与验证的 JSON 兼容 dict。
    """
    total_started = perf_counter()
    timings: dict[str, float] = {}
    checkpoints = {"start": process_memory_snapshot()}
    output_dir = args.output_dir.expanduser().resolve()
    budget = resolve_memory_budget_bytes(args.memory_budget_gib)
    preflight: dict[str, Any]
    if args.macro_verify and args.layout is None:
        raise ValueError("macro-verify 必须提供真实 GDS/OASIS/GLP 输入")

    # 阶段①准备输入。两条分支：合成案例（含重叠、孔洞、凹角、斜边、跨 core 长边，
    # 一次性覆盖前端所有几何路径，无需版图文件）与真实 GDS/OASIS（逐层、按 ROI
    # 物化）。两者之后都汇聚到同一套分段配置与 core 网格。
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
            atomic_json(output_dir / "summary.json", result)
            return result
        stage = perf_counter()
        problem = prepare_problem(batch, layer, config, grid, args.polarity)
        _finish_stage(timings, checkpoints, "problem_prepare", stage)
    else:
        source_path = args.layout.expanduser().resolve()
        source = str(source_path)
        stage = perf_counter()
        # 阶段②容量预检（真实版图）。在物化任何 Region/边段之前，先用原生层级
        # 迭代器估算边段数与准备阶段峰值内存；超预算直接拒绝，避免分配超大数组。
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
            if args.preflight_only or (not preflight["accepted"] and not args.macro_verify):
                status = ("preflight_only" if args.preflight_only and preflight["accepted"]
                          else "rejected")
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
                atomic_json(output_dir / "summary.json", result)
                return result
            if args.macro_verify:
                # Macro 验证在预检通过后直接走未裁剪相交物化，不构造完整处理 ROI
                # 的 Region/SegmentBatch。每个 macro 完成后释放，证明 CPU 工作集可
                # 随 macro 大小受控；此阶段不调用 solver，也不宣称支持多轮 shard。
                return _run_macro_verification(
                    args, database, layer, box, dbu_um, config, grid,
                    output_dir, total_started, timings, checkpoints, budget, preflight)
            stage = perf_counter()
            batch = database.query([layer], box).materialize()
            _finish_stage(timings, checkpoints, "roi_materialize", stage)
            # 阶段③构造可复用问题。prepare_problem 在 with 内完成（它需要原生
            # Region）：产出固定参考几何 + core 网格 + owner/context CSR。此后
            # 多轮迭代只改一个一维位移数组，不再触碰源版图。
            stage = perf_counter()
            problem = prepare_problem(
                batch, layer, config, grid, args.polarity,
                max_memberships=int(preflight["max_memberships"]))
            _finish_stage(timings, checkpoints, "problem_prepare", stage)

    dbu_nm = dbu_um * 1000.0
    stage = perf_counter()
    # 阶段④生成 demo 位移。给每个有 owner 边段的 core 各选一段、按 core 序号
    # 交替取正负法向位移，用来演示「owner 唯一写、halo 只读」的更新语义，并
    # 为后续重建与覆盖率检查提供一组非零位移。
    displacements, changed = _demo_displacements(
        problem, args.demo_displacement_nm / dbu_nm)
    _finish_stage(timings, checkpoints, "demo_update", stage)

    stage = perf_counter()
    # 阶段⑤按需物化端点与探针。端点（materialize）和 EPE（边缘放置误差）
    # 探针点都是诊断时才计算的派生量，不是 problem 的常驻字段；超大输入已
    # 在预检阶段拒绝，因此这里不会用诊断功能绕过容量保护。
    geometry = problem.segments.materialize(displacements)
    reference_geometry = problem.segments.materialize()
    inner, outer = edge_probe_points(
        reference_geometry.starts, reference_geometry.ends, reference_geometry.normals,
        args.probe_distance_nm / dbu_nm)
    _finish_stage(timings, checkpoints, "segment_materialize_and_probes", stage)

    stage = perf_counter()
    # 阶段⑥位移→Region 重建。零位移重建必须与参考 mask 完全一致；非零位移重建
    # 则用于产物 GDS/PNG。重建只做最终/诊断输出，绝不进入未来每轮迭代的热路径。
    reference = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
    reconstructed = reconstruct_region(problem, displacements)
    _finish_stage(timings, checkpoints, "reconstruct", stage)

    stage = perf_counter()
    # 阶段⑦正确性验证。两条核心不变量：零位移重建与参考 mask 异或面积为 0；
    # core ownership 覆盖 query_box 且无重叠、无缺口。core 只划分写出责任，
    # 并不裁最终 Polygon，因此跨 core 共享边界允许存在。
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
        # 阶段⑧落盘产物。NPZ 存问题/位移快照、GDS 存重建前后几何、PNG 存带
        # 分段/法向/探针/core 的标注总图，可选再跑多图形几何套件。任一产物都
        # 可由 --skip-* 跳过，只保留验证摘要。
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
    atomic_json(output_dir / "summary.json", result)
    return result


def print_text(result: dict[str, Any]) -> None:
    """以紧凑中文输出主要计数、性能和产物路径。"""
    if result["status"] == "macro_verified":
        tiling, verification = result["tiling"], result["verification"]
        print(f"Macro/Tile：{tiling['macro_count']}/"
              f"{tiling['tile_columns'] * tiling['tile_rows']}")
        print(f"栅格不一致像素：{verification['raster_mismatch_pixels']}  "
              f"重复 owned 边段：{verification['duplicate_owned_segments']}")
        print(f"JSON：{result['artifacts']['json']}")
        return
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
