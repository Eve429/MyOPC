"""集中保存运行摘要、数组、诊断图片和最终光刻结果。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import klayout.db as kdb
from PIL import Image

from layout import DbuBox
from lithography import LithographyModel
from opc.input import MaskPolarity, RectilinearCoreGrid
from opc.input.raster import ownership_canvas, rasterize_mask_canvas

_FINAL_LITHOGRAPHY_FORMAT = "myopc.final-lithography"
_FINAL_LITHOGRAPHY_VERSION = 1


def _output_npz(path: str | Path) -> Path:
    """规范化 NPZ 输出路径，并拒绝容易误判格式的其他扩展名。"""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("数组归档必须使用 .npz 扩展名")
    return output


def atomic_npz(path: str | Path, arrays: dict[str, object],
               compressed: bool = False) -> Path:
    """把数组归档原子写入目标文件，异常时删除同目录临时文件。"""
    output = _output_npz(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer = np.savez_compressed if compressed else np.savez
            writer(stream, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def atomic_json(path: str | Path, value: dict[str, Any]) -> Path:
    """以 UTF-8 中文和稳定缩进原子保存运行汇总 JSON。"""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def atomic_png(path: str | Path, values: np.ndarray) -> Path:
    """原子保存左下原点的零到一数组为顶部原点八位灰度 PNG。"""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    # PNG 仅是显式请求的诊断产物；在写盘边界才执行裁剪、量化和纵轴翻转，
    # 不让图片坐标或 uint8 表示回流到模型与优化器的数值路径。
    image = np.flipud(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))
    try:
        Image.fromarray(np.ascontiguousarray(image), mode="L").save(
            temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _final_arrays(mask: object, printed: dict[str, object]) -> dict[str, np.ndarray]:
    """规范化最终 mask 和三种工艺角为连续二维 float32 数组。"""
    arrays: dict[str, np.ndarray] = {}
    values = {"mask": mask, **printed}
    for name in ("mask", "nominal", "dose_max", "defocus_min"):
        if name not in values:
            raise ValueError(f"最终光刻结果缺少 {name} 数组")
        value = values[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"最终光刻结果 {name} 必须是有限二维数组")
        arrays[name] = array
    shape = arrays["mask"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("最终光刻结果的 mask 与工艺角尺寸不一致")
    return arrays


def save_final_lithography_result(
        output_dir: str | Path, mask: object, printed: dict[str, object], *,
        save_png: bool = True) -> dict[str, object]:
    """保存完整二维最终光刻结果，并返回稳定的产物索引。"""
    arrays = _final_arrays(mask, printed)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = atomic_npz(output / "final_lithography.npz", {
        "format_name": np.array(_FINAL_LITHOGRAPHY_FORMAT),
        "format_version": np.array(_FINAL_LITHOGRAPHY_VERSION, dtype=np.int32),
        **arrays,
    })
    images: dict[str, str] = {}
    if save_png:
        for name, values in arrays.items():
            images[name] = str(atomic_png(output / f"final_{name}.png", values))
    return {"npz": str(result_path), "images": images,
            "shape": list(arrays["mask"].shape),
            "format": _FINAL_LITHOGRAPHY_FORMAT,
            "version": _FINAL_LITHOGRAPHY_VERSION}


def _ownership_slice(core: DbuBox, context: DbuBox, pixel_dbu: int,
                     canvas: int) -> tuple[slice, slice, list[int]]:
    """返回 context 画布中 ownership 的矩形切片及其 DBU 原点。"""
    ownership = ownership_canvas(core, context, pixel_dbu, canvas)
    rows, columns = np.where(ownership)
    if len(rows) == 0 or len(columns) == 0:
        raise ValueError("core 在当前 pixel/canvas 下没有可保存的 ownership 像素")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    # 正交 core 的中心采样形成连续矩形；若出现空洞说明坐标配置破坏了
    # ownership 不变量，立即失败而不是保存含 halo 的伪 ownership 图。
    if not np.all(ownership[y0:y1, x0:x1]):
        raise ValueError("ownership 像素不是连续矩形，无法生成稳定 tile")
    return (slice(y0, y1), slice(x0, x1),
            [int(context.left + x0 * pixel_dbu),
             int(context.bottom + y0 * pixel_dbu)])


def save_final_lithography_tiles(
        output_dir: str | Path, region: kdb.Region, grid: RectilinearCoreGrid,
        model: LithographyModel,
        *, pixel_dbu: int, canvas: int, batch_size: int = 1,
        save_png: bool = True, polarity: MaskPolarity | str = MaskPolarity.CLEAR,
        field_box: DbuBox | None = None) -> dict[str, object]:
    """按 core 批量仿真并保存 ownership-only 最终光刻 tile。"""
    if not isinstance(pixel_dbu, int) or pixel_dbu <= 0:
        raise ValueError("pixel_dbu 必须是正整数")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    cores = grid.cores()
    output = Path(output_dir).expanduser().resolve()
    tile_dir = output / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    conditions = tuple(model.condition(name) for name in (
        "nominal", "dose_max", "defocus_min"))
    manifest_tiles: list[dict[str, object]] = []
    for start in range(0, len(cores), batch_size):
        group = cores[start:start + batch_size]
        masks: list[np.ndarray] = []
        slices: list[tuple[slice, slice, list[int]]] = []
        for core in group:
            slices.append(_ownership_slice(core.ownership_box, core.context_box,
                                           pixel_dbu, canvas))
            masks.append(rasterize_mask_canvas(
                region, core.context_box, pixel_dbu, canvas,
                polarity=polarity, field_box=field_box))
        tensor = torch.as_tensor(np.stack(masks), device=model.device)
        with torch.no_grad():
            printed = model.forward_many(tensor, conditions)
        for local, core in enumerate(group):
            y_slice, x_slice, origin = slices[local]
            arrays = {"mask": masks[local][y_slice, x_slice]}
            for name in ("nominal", "dose_max", "defocus_min"):
                arrays[name] = np.ascontiguousarray(
                    printed[name][local].detach().cpu().numpy()[y_slice, x_slice],
                    dtype=np.float32)
            tile_base = tile_dir / core.core_id
            tile_npz = atomic_npz(tile_base.with_suffix(".npz"), {
                "format_name": np.array(_FINAL_LITHOGRAPHY_FORMAT),
                "format_version": np.array(_FINAL_LITHOGRAPHY_VERSION, dtype=np.int32),
                **arrays,
            })
            png_paths: dict[str, str] = {}
            if save_png:
                for name, values in arrays.items():
                    png_paths[name] = str(atomic_png(
                        tile_base.with_name(f"{tile_base.name}_{name}.png"), values))
            manifest_tiles.append({
                "core_id": core.core_id, "core_index": start + local,
                "ownership_box_dbu": [core.ownership_box.left, core.ownership_box.bottom,
                                      core.ownership_box.right, core.ownership_box.top],
                "context_box_dbu": [core.context_box.left, core.context_box.bottom,
                                    core.context_box.right, core.context_box.top],
                "origin_dbu": origin, "shape": list(arrays["mask"].shape),
                "npz": str(tile_npz), "images": png_paths,
            })
        # 一个 batch 的 GPU 输出在写完自身 ownership 后立即释放，避免整张版图
        # tensor 常驻 GPU；CPU 也只保留当前组 mask 和轻量 manifest 元数据。
        del printed, tensor, masks
    manifest = {
        "format": _FINAL_LITHOGRAPHY_FORMAT, "version": _FINAL_LITHOGRAPHY_VERSION,
        "orientation": "bottom_left", "pixel_dbu": pixel_dbu, "canvas": canvas,
        "polarity": MaskPolarity(polarity).value,
        "conditions": [condition.name for condition in conditions],
        "tile_count": len(manifest_tiles), "tiles": manifest_tiles,
    }
    manifest_path = atomic_json(output / "manifest.json", manifest)
    return {"manifest": str(manifest_path), "tile_dir": str(tile_dir),
            "tile_count": len(manifest_tiles), "format": _FINAL_LITHOGRAPHY_FORMAT,
            "version": _FINAL_LITHOGRAPHY_VERSION}
