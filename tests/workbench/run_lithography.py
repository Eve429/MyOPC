"""直接读取离线像素输入并独立运行 ICCAD13 光刻模型。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

# 支持 `python tests/workbench/run_lithography.py ...`，不要求安装当前项目。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lithography import ICCAD13Lithography, LithographyResult
from tests.workbench.offline_inputs import (
    _atomic_json,
    _atomic_npz,
    load_raster_input,
)


def _image_array(values: np.ndarray) -> np.ndarray:
    """把模型左下原点浮点数组转换为顶部原点的八位 PNG 数组。"""
    clipped = np.clip(values, 0.0, 1.0)
    return np.ascontiguousarray(np.flipud(np.rint(clipped * 255.0)).astype(np.uint8))


def _atomic_png(path: Path, values: np.ndarray) -> Path:
    """原子保存一个光刻浮点结果，避免异常留下半张图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(_image_array(values), mode="L").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    """把模型结果复制成连续 float32 CPU 数组用于归档和图片。"""
    return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def run_lithography_test(
        input_path: str | Path, output_dir: str | Path | None = None, *,
        device: str = "auto", save_png: bool = False) -> LithographyResult:
    """仅从已保存 mask 运行光刻模型，并按需保存数值与 PNG 结果。"""
    mask, metadata = load_raster_input(input_path)
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
    with torch.no_grad():
        result = model(mask_tensor)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = perf_counter() - started
    nominal, maximum, minimum = (
        _cpu_array(result.nominal), _cpu_array(result.maximum), _cpu_array(result.minimum))
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        result_path = _atomic_npz(output / "lithography_result.npz", {
            "format_name": np.array("myopc.lithography-result"),
            "format_version": np.array(1, dtype=np.int32),
            "nominal": nominal, "maximum": maximum, "minimum": minimum,
        }, compressed=False)
        images: dict[str, str] = {}
        if save_png:
            for name, values in (("mask", mask), ("nominal", nominal),
                                 ("maximum", maximum), ("minimum", minimum)):
                images[name] = str(_atomic_png(output / f"{name}.png", values))
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
        }
        _atomic_json(output / "summary.json", summary)
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造离线光刻模型测试命令行。"""
    parser = argparse.ArgumentParser(description="从离线 mask 独立运行 ICCAD13 光刻模型。")
    parser.add_argument("input", type=Path, help="prepare_raster_input 生成的 NPZ")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/workbench/lithography"))
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda[:序号]")
    parser.add_argument("--save-png", action="store_true",
                        help="保存 mask 和三个工艺角 PNG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把可预期输入或运行错误转换为退出码 2。"""
    args = build_parser().parse_args(argv)
    try:
        result = run_lithography_test(
            args.input, args.output_dir, device=args.device, save_png=args.save_png)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"光刻模型完成：shape={tuple(result.nominal.shape)}，输出={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
