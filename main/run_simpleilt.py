"""从 GDS/OASIS 或像素 NPZ 运行可微 SimpleILT 并保存结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layout import DbuBox, LayerSpec
from main.offline_inputs import add_layout_source_arguments
from main.configuration import ConfiguredArgumentParser, glp_layer_map
from main.run_ilt import run_ilt
from opc.iteration.ilt import SimpleILTResult


def run_simpleilt(
        input_path: str | Path, output_dir: str | Path, *, iterations: int = 20,
        step_size: float = 0.5, sigmoid_steepness: float = 4.0,
        weight_pvband: float = 0.0, weight_process_l2: float = 1.0,
        curvature_weight: float = 0.0, device: str = "auto",
        save_png: bool = True, layer: LayerSpec | None = None,
        top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0, polarity: str = "clear",
        glp_layers: dict[str, LayerSpec] | None = None,
        run_configuration: dict[str, object] | None = None
        ) -> tuple[SimpleILTResult, dict[str, Any]]:
    """把历史 SimpleILT 参数适配到统一 ILT 执行路径并返回内存结果。

    这是一个兼容垫片入口：SimpleILT 曾是独立入口，现已并入统一 ILT。本函数只把
    历史参数与默认值映射成 `run_ilt(method="simple", return_result=True)` 的调用，
    自身不再实现版图读取、优化、评价与产物写入。这样能避免两个入口各维护一套
    执行与产物逻辑而产生行为、文件格式分叉；return_result=True 使其额外拿回内存
    结果对象，且复用同一次执行，绝不重新优化。
    输入：像素 target 来源、输出目录与 SimpleILT 历史参数。
    输出：(内存结果对象, summary dict)。
    """
    # 兼容入口只映射参数和默认值；版图读取、优化、评价、资源统计与产物写入均
    # 由 run_ilt 完成，防止两个入口修复同一问题后出现行为和文件格式分叉。
    outcome = run_ilt(
        input_path, output_dir, method="simple", iterations=iterations,
        step_size=step_size, sigmoid_steepness=sigmoid_steepness,
        weight_pvband=weight_pvband, weight_process_l2=weight_process_l2,
        curvature_weight=curvature_weight, device=device, save_png=save_png,
        layer=layer, top_cell=top_cell, box=box, pixel_nm=pixel_nm,
        canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib, polarity=polarity,
        glp_layers=glp_layers, run_configuration=run_configuration,
        return_result=True)
    return outcome


def build_parser() -> argparse.ArgumentParser:
    """构造支持版图和离线像素输入的 SimpleILT 命令行参数。"""
    parser = ConfiguredArgumentParser(
        workflow="ilt", entry="simpleilt", valid_entries=("ilt", "simpleilt"),
        description="从 GDS/OASIS ROI 或离线像素目标运行 SimpleILT。")
    parser.add_argument("input", type=Path, help="输入 GDS/OASIS 或 raster NPZ")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--sigmoid-steepness", type=float)
    parser.add_argument("--weight-pvband", type=float)
    parser.add_argument("--weight-process-l2", type=float)
    parser.add_argument("--curvature-weight", type=float)
    parser.add_argument("--device", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--no-png", action="store_true", help="不保存诊断 PNG")
    add_layout_source_arguments(parser)
    parser.add_argument("--pixel-nm", type=float,
                        help="直接版图输入的像素尺寸")
    parser.add_argument("--canvas", type=int,
                        help="直接版图输入的固定方形画布")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并把输入、配置和运行错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        _, summary = run_simpleilt(
            args.input, args.output_dir, iterations=args.iterations,
            step_size=args.step_size, sigmoid_steepness=args.sigmoid_steepness,
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
            max_estimated_gib=args.max_estimated_gib,
            polarity=args.polarity, glp_layers=glp_layer_map(args.glp_layers),
            run_configuration=args._configuration)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"SimpleILT 完成：best={summary['best_iteration']}，"
          f"L2={summary['evaluation']['binary_l2']}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
