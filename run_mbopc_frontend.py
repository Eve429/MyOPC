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
from opc.input import RectilinearCoreGrid
from opc.input.edge import (
    FragmentationConfig,
    MBOPCProblem,
    SegmentUpdateBatch,
    merge_owner_updates,
    prepare_problem,
    reconstruct_region,
    render_boundary_overlay,
    run_geometry_suite,
    sample_lines,
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
    """执行一次完整的 MB-OPC 前端演示，并返回可序列化的运行摘要。

    输入：
        args: ``build_parser()`` 解析得到的命令行参数。它提供版图路径、层、
            查询框、core 网格、halo、边段长度、最大位移、演示位移、
            输出目录以及是否跳过多图形测试等配置。所有几何长度参数在命令行
            中以 nm 表示，进入几何内核前会依据版图 DBU 换算为数据库单位。

    输出：
        一个只包含字符串、数值、布尔值、列表和字典的摘要字典，可直接编码为
        JSON。摘要包括输入来源、层和查询框，图形/边段/采样/core 数量，紧凑
        边段数组占用，分阶段耗时，重建及跨 core 校验结果，以及 NPZ、PNG、
        GDS、JSON 四类产物路径。函数同时将该摘要原子写入 ``summary.json``。

    异常：
        输入版图、参数或几何不合法时，下层函数会抛出 ``LayoutError``、
        ``OPCError`` 或 ``ValueError``；文件创建失败时会抛出 ``OSError``。
        本函数不吞掉这些异常，由 ``main()`` 统一转成命令行错误码和错误信息。
    """
    # 1. 准备输入数据。未传 --layout 时创建内置多图形案例，便于用户直接运行
    #    本文件验证完整流程；batch 是查询框内的物理图形，layer 是目标层，
    #    dbu_um 表示一个数据库单位对应多少微米。source 只用于摘要标识来源。
    if args.layout is None:
        batch, layer, dbu_um = _demo_batch()
        source = "synthetic"
        # 计时从前端问题构建前开始，不包含内置案例的构造时间。prepare 阶段会把
        # RegionBatch 转为稳定轮廓、数学边、紧凑边段、采样模板和 core 归属表；
        # 同时返回 grid（全局网格）和 config（重建约束），供后续校验与重建复用。
        started = perf_counter()
        problem, grid, config = _prepare_input_problem(args, batch, layer, dbu_um)
    else:
        # 传入真实版图时，将路径展开并绝对化，避免摘要依赖启动目录；此字符串
        # 仅用于记录，不参与几何计算。_load_database_batch 输出与演示分支一致。
        source = str(args.layout.expanduser().resolve())
        # KLayout 的物化 Region 仍依赖打开的 Layout；因此必须在上下文内
        # 完成物理合并和紧凑数组构建。离开 with 后文件会关闭，而 problem 中
        # 保留的均为自有 Region/NumPy 数据，之后的迭代与输出不再读取源文件。
        with LayoutDB.open(args.layout, top_cell=None) as database:
            batch, layer, dbu_um = _load_database_batch(args, database)
            # 真实版图读取和查询不计入 prepare，计时专注于可重复执行的前端构建，
            # 便于比较不同分段、采样和 core 配置本身的性能。
            started = perf_counter()
            problem, grid, config = _prepare_input_problem(args, batch, layer, dbu_um)

    # 2. 固化准备阶段结束时间，并完成单位换算。dbu_um 的单位是 μm/DBU，乘
    #    1000 后得到 nm/DBU；演示位移除以该值即可得到几何内核使用的 DBU。
    prepared = perf_counter()
    dbu_nm = dbu_um * 1000.0

    # 3. 构造一次模拟 OPC 更新。_demo_updates 从每个 core 的所有权边段中选取
    #    演示目标，输入为紧凑问题和 DBU 位移量，输出为按 owner 分组的
    #    SegmentUpdateBatch 列表；它只模拟优化器输出，不修改 problem 本身。
    updates = _demo_updates(problem, args.demo_displacement_nm / dbu_nm)

    # 4. 合并各 core 的更新。该步骤校验 owner、边段键、重复写入和最大位移，
    #    输出 update_result：displacements 是与全局边段索引严格对齐的位移数组，
    #    changed_segment_indices 和 dirty_polygon_indices 分别指出改变的边段及多边形，
    #    供后续增量算法定位需要重新计算的局部数据。
    update_result = merge_owner_updates(problem, updates)

    # 5. 将“参考边段 + 法向位移”物化为当前几何。输入位移数组不改变拓扑和
    #    归属；输出 geometry 包含浮点 starts、ends、normals、lengths 等当前值。
    geometry = problem.segments.materialize(update_result.displacements)

    # 6. 依据准备阶段缓存的 sample_template 在当前边段上生成采样点。输入是
    #    移动后的首尾点、法向和不随迭代变化的模板；输出 samples 包含坐标、
    #    对应边段索引、切向比例和法向偏移。这里只做坐标物化，几何合法性由
    #    重建步骤检查；需要轮廓信息时可通过边段索引继续查询 problem.segments。
    samples = sample_lines(geometry.starts, geometry.ends, geometry.normals,
                           problem.sample_template)

    # 7. 执行两次重建。reference 使用全零位移，理论上必须还原输入物理 mask；
    #    reconstructed 使用本轮合并位移，是后续 OPC、拼接与可视化的实际结果。
    #    两者均输入紧凑边段和重建配置，输出 KLayout Region。
    reference = reconstruct_region(problem.segments,
                                   np.zeros(problem.segments.segment_count), config)
    reconstructed = reconstruct_region(problem.segments, update_result.displacements, config)

    # update_sample_reconstruct 阶段到此结束；此时间点之后属于验证和产物输出。
    rebuilt = perf_counter()

    # 8. 验证零位移往返不改变物理图形。异或 Region 的面积为零表示重建前后
    #    覆盖完全一致；非零通常意味着轮廓方向、孔洞归属或分段连接存在错误。
    if (reference ^ problem.physical_mask.region).area() != 0:
        raise ValueError("零位移重建与物理参考 mask 不一致")

    # 9. 验证跨 core 的拼接。每个 GeometryPatch 输入同一份移动后 Region 和
    #    对应 core 的 ownership_box；PatchSet.add 会把 Region 裁剪到该所有权框，
    #    因而跨边界图形会被分别提取，最后由 PatchSet.region 合并为完整目标层。
    patches = PatchSet()
    for core_index, core in enumerate(problem.ownership.cores):
        patches.add(GeometryPatch(f"core-{core_index}", layer, reconstructed,
                                  core.ownership_box))

    # grid.bounds 是本次规划覆盖范围。clipped_reference 这个局部变量表示移动后
    # Region 在网格范围内的直接裁剪结果，它是拼接结果的比较基准，并非上面的
    # 零位移 reference。两者异或面积必须为零，才能证明跨 core 图形无缝且无重叠。
    clipped_reference = reconstructed & kdb.Region(grid.bounds.to_native())
    stitch_xor = (patches.region(layer) ^ clipped_reference).area()
    if stitch_xor:
        raise ValueError(f"跨 core 拼接 XOR 面积非零：{stitch_xor}")

    # 10. 创建产物目录。expanduser 支持用户目录写法，resolve 固定摘要中的绝对
    #     路径，parents=True 允许一次创建缺失的父目录，exist_ok=True 支持复跑。
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 11. 保存紧凑数值数据。输入为 problem 和本轮全局位移数组；输出 NPZ 包含
    #     公共边界、边段拓扑、core 归属及位移，可供其他 OPC 方法加载和分析。
    npz_path = save_problem_npz(problem, update_result.displacements, output_dir / "segments.npz")

    # 12. 保存调试版图。输入零位移参考 Region 与移动后 Region，两者使用相同
    #     layer/datatype、分别写入 REFERENCE 和 RECONSTRUCTED 顶层 Cell；dbu_um
    #     保持物理尺度，输出为可在版图工具中切换两个 Cell 比较的 GDS 文件路径。
    gds_path = write_debug_gds(reference, reconstructed, output_dir / "reconstruction.gds",
                               dbu_um, layer.layer, layer.datatype)

    # 13. 保存边界叠加图。除移动后 Region 和查询框外，还输入边段端点、法向、
    #     owner、采样点及 core 框；输出 PNG 用颜色和标记直观展示分段及归属关系。
    png_path = render_boundary_overlay(
        reconstructed, layer, batch.query_box, dbu_um, geometry.starts, geometry.ends,
        geometry.normals, output_dir / "overview.png", problem.ownership.owner_indices,
        samples, problem.ownership.cores)

    # 14. 默认运行额外的多图形几何回归套件并把图片写入子目录；传入
    #     --skip-geometry-suite 时返回 None，用于只测当前输入的快速运行模式。
    geometry_suite = None if args.skip_geometry_suite else run_geometry_suite(
        output_dir / "geometry_suite")

    # 记录全部校验和文件输出结束时间，供下面拆分阶段耗时。
    finished = perf_counter()

    # 15. 汇总为 JSON 兼容结构：counts 描述问题规模；memory 仅统计紧凑边段
    #     持久数组本身，不等同于进程总内存；timing_seconds 不包含版图读取时间；
    #     verification 给出关键不变量；artifacts 给出四类可直接查看的绝对路径。
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

    # 16. 原子写入 summary.json。先在同目录写入带进程号的临时文件，成功后由
    #     os.replace 一次替换正式文件，避免进程中断留下半份 JSON；finally 确保
    #     写入或替换异常时也清理临时文件。最后返回与磁盘内容一致的 result。
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
