"""直接读取 GDS/OASIS 或像素 NPZ 并运行 ICCAD13 光刻模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

# 支持 `python main/run_lithography.py ...`，不要求安装当前项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layout import DbuBox, LayerSpec
from lithography import ICCAD13Lithography
from main.artifacts import atomic_json, atomic_npz, atomic_png
from main.offline_inputs import add_layout_source_arguments, resolve_raster_input
from main.configuration import ConfiguredArgumentParser, glp_layer_map


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    """把模型结果复制成连续 float32 CPU 数组用于归档和图片。"""
    return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def run_lithography_test(
        input_path: str | Path, output_dir: str | Path | None = None, *,
        device: str = "auto", save_png: bool = False,
        layer: LayerSpec | None = None, top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0, polarity: str = "clear",
        glp_layers: dict[str, LayerSpec] | None = None,
        run_configuration: dict[str, object] | None = None) -> dict[str, torch.Tensor]:
    """从版图或像素归档运行光刻模型，并按需保存数值与 PNG。

    这是光刻正向模型的独立入口，不做任何 OPC / 迭代。给定一张 mask，由
    ICCAD13 模型产出三个工艺角（nominal 标称 / dose_max 最大剂量 / defocus_min
    最小焦距）的成像图，用于单独验证光刻模型或诊断成像质量。输入可以是版图 ROI
    （当场栅格化）或离线像素 NPZ（复用已备数据）。管线上只用到
    `layout → geometry + lithography`。
    输入：版图或 NPZ 路径、输出目录、设备与画布参数。
    输出：含三个工艺角成像 Tensor 的 dict（同时按需写 NPZ/PNG/summary）。
    """
    # 阶段①取得模型方向画布。版图分支只在 CPU 内存生成当前 ROI 的固定画布，不
    # 落临时 NPZ；NPZ 分支则复用已准备数据。两者在这里汇合，后续 GPU 传输和
    # 模型计算没有双实现。
    mask, metadata = resolve_raster_input(
        input_path, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib, polarity=polarity,
        glp_layers=glp_layers)
    model = ICCAD13Lithography(device=device)
    if mask.shape[0] > model.config.canvas or mask.shape[1] > model.config.canvas:
        raise ValueError("离线像素输入超过当前光刻模型 canvas")
    if save_png and output_dir is None:
        raise ValueError("save_png=True 时必须提供 output_dir")
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = perf_counter()
    mask_tensor = torch.as_tensor(mask, device=model.device)
    # 阶段②三工艺角正向仿真。nominal 是标称工艺，dose_max 与 defocus_min 是对
    # 剂量/焦距偏移的极端角；三者共享同一 mask 的 FFT，forward_only 一次给出。
    # dose_max 与 defocus_min 的成像差就是 PVBand（工艺变化带），衡量工艺漂移下
    # 边缘位置的不确定性。
    conditions = tuple(model.condition(name) for name in (
        "nominal", "dose_max", "defocus_min"))
    with torch.no_grad():
        result = model.forward_many(mask_tensor, conditions)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = perf_counter() - started
    nominal, maximum, minimum = (
        _cpu_array(result["nominal"]), _cpu_array(result["dose_max"]),
        _cpu_array(result["defocus_min"]))
    if output_dir is not None:
        # 阶段③归档结果。GPU Tensor 各拷回一次连续 float32 CPU 数组，后续 NPZ/PNG
        # 全部复用同一份主机数据，避免每个产物重复触发设备同步。
        output = Path(output_dir).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        result_path = atomic_npz(output / "lithography_result.npz", {
            "format_name": np.array("myopc.lithography-result"),
            "format_version": np.array(1, dtype=np.int32),
            "nominal": nominal, "maximum": maximum, "minimum": minimum,
        }, compressed=False)
        images: dict[str, str] = {}
        if save_png:
            for name, values in (("mask", mask), ("nominal", nominal),
                                 ("maximum", maximum), ("minimum", minimum)):
                images[name] = str(atomic_png(output / f"{name}.png", values))
        summary = {
            "input": str(Path(input_path).expanduser().resolve()),
            "source_layout": metadata.get("source"), "device": str(model.device),
            "shape": list(mask.shape), "elapsed_seconds": elapsed,
            "gpu_peak_allocated_bytes": (int(torch.cuda.max_memory_allocated(model.device))
                                         if model.device.type == "cuda" else 0),
            "ranges": {
                "mask": [float(mask.min()), float(mask.max())],
                "nominal": [float(nominal.min()), float(nominal.max())],
                "maximum": [float(maximum.min()), float(maximum.max())],
                "minimum": [float(minimum.min()), float(minimum.max())],
            },
            "artifacts": {"result_npz": str(result_path), "images": images,
                          "summary": str(output / "summary.json")},
            "run_configuration": run_configuration,
        }
        atomic_json(output / "summary.json", summary)
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造支持版图和离线像素输入的光刻模型命令行。"""
    parser = ConfiguredArgumentParser(
        workflow="lithography", entry="lithography",
        description="从 GDS/OASIS ROI 或离线 mask 运行 ICCAD13 光刻模型。")
    parser.add_argument("input", type=Path, help="输入 GDS/OASIS 或 raster NPZ")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--save-png", action="store_true",
                        help="保存 mask 和三个工艺角 PNG")
    add_layout_source_arguments(parser)
    parser.add_argument("--pixel-nm", type=float,
                        help="直接版图输入的像素尺寸")
    parser.add_argument("--canvas", type=int,
                        help="直接版图输入的固定方形画布")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把可预期输入或运行错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        result = run_lithography_test(
            args.input, args.output_dir, device=args.device, save_png=args.save_png,
            layer=args.layer, top_cell=args.top_cell,
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
    print(f"光刻模型完成：shape={tuple(result['nominal'].shape)}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
