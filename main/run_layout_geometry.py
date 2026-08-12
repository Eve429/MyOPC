"""无需安装项目包即可直接运行的 Layout/Geometry 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 直接运行 main 下脚本时，Python 只加入 main 目录；按文件位置加入仓库根，
# 不要求 pip install，也不依赖用户从哪个工作目录启动。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry import (
    GeometryError,
    GeometryPatch,
    PatchSet,
    PatchWriter,
    extract_contours,
    render_region_batch,
)
from layout import DbuBox, LayoutDB, LayoutError
from main.configuration import (
    ConfiguredArgumentParser, glp_layer_map, parse_glp_layer, parse_layer_spec,
)


def build_parser() -> argparse.ArgumentParser:
    """构造紧凑的中文命令行参数解析器。"""
    parser = ConfiguredArgumentParser(
        workflow="layout", entry="layout_geometry",
        description="直接读取 GDS/OASIS，按 Layer/ROI 查询，并可导出 ownership Patch。")
    parser.add_argument("layout", type=Path, help="输入 GDS/OASIS 文件")
    parser.add_argument("--top", help="多顶层版图必须明确指定的 top Cell")
    parser.add_argument("--glp-layer", dest="glp_layers", action="append",
                        type=parse_glp_layer, help="GLP 符号层映射 NAME=LAYER/DATATYPE")
    parser.add_argument("--layer", action="append", type=parse_layer_spec, dest="layers",
                        help="查询 Layer，可重复传入，例如 --layer 1/0")
    parser.add_argument("--box", nargs=4, type=int,
                        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="查询框的四个整数 DBU 坐标；默认使用 top bbox")
    parser.add_argument("--diagnostics", action="store_true",
                        help="额外统计 ROI 中的 Box/Path/Polygon/Text/Edge；会增加一次遍历")
    parser.add_argument("--arrays", action="store_true",
                        help="额外生成 ContourBatch 并报告 Polygon、Ring 和数学边数量")
    parser.add_argument("--output", type=Path,
                        help="把精确裁剪结果作为单 core Patch 写出到 .gds/.oas")
    parser.add_argument("--png", type=Path, help="把单个 Layer 的 ROI 保存为灰度覆盖率 PNG")
    parser.add_argument("--show-image", action="store_true",
                        help="使用系统图片查看器显示单个 Layer 的 ROI")
    parser.add_argument("--pixel-size-nm", type=float,
                        help="PNG 物理像素尺寸，默认 5 nm/pixel")
    parser.add_argument("--max-image-pixels", type=int,
                        help="PNG 最大像素数，默认 64000000")
    parser.add_argument("--json", action="store_true", help="使用 JSON 输出，便于脚本集成")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次单文件、单 ROI 的完整 Layout 到 Geometry 数据流。

    这是整条 OPC 管线最底层的入口：只覆盖 `layout → geometry` 两个层，
    不做任何 OPC / 光刻 / 迭代。读入一份 GDS/OASIS 版图，按目标层与 ROI
    （感兴趣区，Region of Interest）查询、物化原生 Region，再按需做轮廓
    提取、形状诊断、ownership Patch 导出和覆盖率 PNG 渲染。它定位为「OPC
    之前版图→几何的公共基座」，用来验证层级读取和裁剪语义本身是否正确。
    输入：args —— 已合并 TOML 默认值与 CLI 的 Namespace。
    输出：分层统计与产物路径的 dict（仅当 --json 时随返回值打印）。
    """
    # 阶段①打开版图并圈定查询范围。DBU（database unit，版图最小整数单位）
    # 是本文件所有坐标的共同标尺；查询框默认取顶层 bbox，可由 --box 覆盖。
    # LayoutDB 在 with 生命周期内唯一持有原生版图；后续所有 Region、轮廓和输出
    # 都在关闭前生成，避免惰性查询访问已经释放的 KLayout 对象。
    with LayoutDB.open(args.layout, top_cell=args.top,
                       glp_layer_map=glp_layer_map(args.glp_layers)) as database:
        if (args.layout.suffix.lower() == ".glp" and args.output is not None and
                args.output.suffix.lower() != ".gds"):
            raise ValueError("GLP 输入当前只允许输出 GDS，不支持其他版图格式")
        # 未显式传参时使用现有全部 Layer 和 top bbox；空 top 没有可推导 ROI，
        # 必须在进入几何物化前拒绝，避免生成边界不明确的空 Patch。
        layers = tuple(args.layers) if args.layers else database.layers()
        query_box = DbuBox(*args.box) if args.box else database.bbox()
        if query_box is None:
            raise ValueError("所选 top Cell 为空，无法生成默认查询框")
        if (args.png or args.show_image) and len(layers) != 1:
            raise ValueError("PNG 展示必须且只能选择一个 Layer")
        # 阶段②物化 ROI 内图形。这里不把整片版图展平（flatten），而是借助层级
        # 索引只物化与查询框真正相交的形状；这是 layout 层「读多大量付多少内存」
        # 的核心保证，也是后续 OPC 能在大版图上局部工作的前提。
        # ShapeQuery 在 C++ 侧先用层级索引筛选候选，再以一次 Region 相交精确裁到
        # planner ROI；所有消费者共享相同语义，根入口不再维护第二套裁剪门面。
        batch = database.query(list(layers), query_box).materialize(args.diagnostics)
        result: dict[str, Any] = {
            "source": str(database.source_path),
            "top_cell": database.top_cell.name,
            "dbu_um": database.dbu_um,
            "box_dbu": [query_box.left, query_box.bottom, query_box.right, query_box.top],
            "layers": {},
            "run_configuration": args._configuration,
        }
        contour_batches = extract_contours(batch) if args.arrays else {}
        # 阶段③逐层统计。Polygon 计数、面积与 bbox 直接来自原生 Region；
        # --arrays 时额外提取 ContourBatch（多边形→环→顶点的两级结构），此处
        # 只用它来报顶点/环/边数，诊断几何复杂度，不构造完整边段表。
        for layer in batch.layers:
            key = f"{layer.layer}/{layer.datatype}"
            region = batch.region(layer)
            layer_result: dict[str, Any] = {
                "polygon_count": int(region.count()),
                "area_dbu2": int(region.area()),
                "bbox": None if region.bbox().empty() else region.bbox().to_s(),
            }
            if args.arrays:
                layer_result.update(
                    vertex_count=len(contour_batches[layer].vertices),
                    ring_count=contour_batches[layer].ring_count,
                    # 闭合 ring 的每个顶点恰好对应一条数学边，诊断无需为计数构造
                    # 完整的边起点、终点和归属数组。
                    edge_count=len(contour_batches[layer].vertices),
                )
            if args.diagnostics and batch.stats is not None:
                stats = batch.stats.shapes[layer]
                layer_result["diagnostics"] = {
                    "polygon_like": stats.polygon_like,
                    "text": stats.text,
                    "edge": stats.edge,
                    "other": stats.other,
                }
            result["layers"][key] = layer_result
        if args.output:
            # 阶段④（可选）把精确裁剪结果导出为 ownership Patch。GeometryPatch 是
            # 「单 core 切片」的最小几何交付单元：这里把整张 ROI 当作单个 core，
            # 写出与查询完全一致的几何，便于下游方法在不读原版图时复用。
            patches = PatchSet()
            for layer in batch.layers:
                patches.add(GeometryPatch(
                    f"cli-{layer.layer}-{layer.datatype}", layer,
                    batch.region(layer), query_box))
            result["output"] = str(PatchWriter.write(
                patches, args.output, database.dbu_um, top_name="OPC_PATCHES"))
        if args.png or args.show_image:
            # 阶段⑤（可选）把单个层渲染成覆盖率 PNG。像素值为 0–1 的面积覆盖率，
            # 坐标约定为左下原点；这一步纯诊断，不回流到 OPC 的数值路径。
            layer = layers[0]
            pixels = render_region_batch(
                batch, layer, database.dbu_um, args.pixel_size_nm,
                output_path=args.png, show=args.show_image,
                max_pixels=args.max_image_pixels)
            result["image"] = {
                "path": None if args.png is None else str(args.png.expanduser().resolve()),
                "shown": bool(args.show_image),
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
                "pixel_size_nm": args.pixel_size_nm,
                "layer": f"{layer.layer}/{layer.datatype}",
            }
        return result


def print_text(result: dict[str, Any]) -> None:
    """以便于人工检查的紧凑中文格式打印结果。"""
    print(f"文件：{result['source']}")
    print(f"Top Cell：{result['top_cell']}  DBU：{result['dbu_um']} μm")
    print(f"查询框：{result['box_dbu']} DBU")
    for layer, values in result["layers"].items():
        line = (f"Layer {layer}：Polygon={values['polygon_count']}，"
                f"面积={values['area_dbu2']} DBU²，bbox={values['bbox']}")
        if "edge_count" in values:
            line += (f"，顶点={values['vertex_count']}，环={values['ring_count']}，"
                     f"边={values['edge_count']}")
        print(line)
    if "output" in result:
        print(f"Patch 已写出：{result['output']}")
    if "image" in result:
        image = result["image"]
        print(f"像素图：{image['width']}x{image['height']}，{image['pixel_size_nm']} nm/pixel")
        if image["path"]:
            print(f"PNG 已保存：{image['path']}")


def main(argv: list[str] | None = None) -> int:
    """解析参数、输出结果，并把可预期领域错误转换为简洁退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (LayoutError, GeometryError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
