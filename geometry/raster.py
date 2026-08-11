"""把 planner 局部区域栅格化为可展示和保存的灰度像素图。"""

from __future__ import annotations

import math
import os
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from layout.types import DbuBox, LayerSpec, RegionBatch

from .errors import RasterizationError

if TYPE_CHECKING:
    from layout.database import LayoutDB

_MAX_STRIPE_PIXELS = 1_000_000
_DEFAULT_MAX_PIXELS = 64_000_000


def _pixel_size_dbu(dbu_um: float, pixel_size_nm: float) -> int:
    """把物理像素尺寸精确换算为整数数据库单位。"""
    if (not isinstance(dbu_um, Real) or isinstance(dbu_um, bool) or
            not math.isfinite(float(dbu_um)) or dbu_um <= 0):
        raise RasterizationError("数据库单位必须是正有限数")
    if (not isinstance(pixel_size_nm, Real) or isinstance(pixel_size_nm, bool) or
            not math.isfinite(float(pixel_size_nm)) or pixel_size_nm <= 0):
        raise RasterizationError("像素尺寸必须是正有限数，单位为 nm")
    exact = float(pixel_size_nm) / (float(dbu_um) * 1000.0)
    rounded = round(exact)
    if rounded < 1 or not math.isclose(exact, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise RasterizationError(
            f"{pixel_size_nm:g} nm 不能精确表示为当前版图的整数 DBU 像素尺寸")
    return rounded


def _image_shape(box: DbuBox, pixel_dbu: int, max_pixels: int) -> tuple[int, int]:
    """计算向右上补齐的图片尺寸，并在分配前执行内存保护。"""
    if (not isinstance(max_pixels, Integral) or isinstance(max_pixels, bool) or
            max_pixels <= 0):
        raise RasterizationError("最大像素数必须是正整数")
    width = (box.width + pixel_dbu - 1) // pixel_dbu
    height = (box.height + pixel_dbu - 1) // pixel_dbu
    count = width * height
    if count > int(max_pixels):
        raise RasterizationError(
            f"请求图片为 {width}x{height}={count} 像素，超过上限 {int(max_pixels)}")
    return height, width


def _rasterize(region: kdb.Region, box: DbuBox, pixel_dbu: int,
               shape: tuple[int, int]) -> NDArray[np.uint8]:
    """分块调用原生面积栅格器，并转换为顶部朝上的八位灰度数组。"""
    height, width = shape
    pixels = np.zeros(shape, dtype=np.uint8)
    if region.is_empty():
        return pixels
    # 原生 rasterize 返回从左下角开始的 Python 浮点矩阵。这里按最多约一百万
    # 临时像素切成二维块，避免大 ROI 同时生成数千万个 Python float；最终数组
    # 只保留 uint8 覆盖率，因此稳定内存主要由实际 PNG 尺寸决定。
    tile_width = min(width, _MAX_STRIPE_PIXELS)
    tile_rows = max(1, _MAX_STRIPE_PIXELS // tile_width)
    pixel_area = float(pixel_dbu * pixel_dbu)
    for y0 in range(0, height, tile_rows):
        rows = min(tile_rows, height - y0)
        for x0 in range(0, width, tile_width):
            columns = min(tile_width, width - x0)
            origin = kdb.Point(box.left + x0 * pixel_dbu, box.bottom + y0 * pixel_dbu)
            areas = np.asarray(region.rasterize(
                origin, kdb.Vector(pixel_dbu, pixel_dbu), columns, rows), dtype=np.float64)
            # 面积矩阵是当前块唯一的大型浮点临时量；在原数组上归一化、裁界和取整，
            # 避免表达式链为每个百万像素块再产生两到三份 float64 中间数组。
            areas /= pixel_area
            np.clip(areas, 0.0, 1.0, out=areas)
            areas *= 255.0
            np.rint(areas, out=areas)
            # 图片第 0 行位于顶部，而版图栅格第 0 行位于底部。块内翻转后再映射到
            # 全局反向行区间，保证 planner 坐标系的 +Y 在最终图片中仍然朝上。
            top = height - y0 - rows
            pixels[top:height - y0, x0:x0 + columns] = np.flipud(areas).astype(np.uint8)
    return pixels


def _save_png(pixels: NDArray[np.uint8], output_path: str | Path) -> Path:
    """在目标目录原子保存八位灰度 PNG。"""
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise RasterizationError("像素图输出扩展名必须是 .png")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"输出目录不存在：{output.parent}")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".png",
                                               dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def render_region_batch(batch: RegionBatch, layer: LayerSpec, dbu_um: float,
                        pixel_size_nm: float = 5.0, *, output_path: str | Path | None = None,
                        show: bool = False,
                        max_pixels: int = _DEFAULT_MAX_PIXELS) -> NDArray[np.uint8]:
    """把已提取批次中的单层区域显示或保存为灰度 PNG，并返回像素数组。"""
    if layer not in batch.regions:
        raise RasterizationError(f"批次不包含图层 {layer.layer}/{layer.datatype}")
    pixel_dbu = _pixel_size_dbu(dbu_um, pixel_size_nm)
    shape = _image_shape(batch.query_box, pixel_dbu, max_pixels)
    clip = kdb.Region(batch.query_box.to_native())
    # 原生栅格接口不采用 Region 的 merged semantics。局部裁剪后显式合并，可避免
    # 重叠 Polygon 被重复计算面积，同时把 merge 成本限制在 planner 当前 ROI 内。
    region = (batch.region(layer) & clip).merged()
    pixels = _rasterize(region, batch.query_box, pixel_dbu, shape)
    if output_path is not None:
        _save_png(pixels, output_path)
    if show:
        Image.fromarray(pixels).show(title=f"Layer {layer.layer}/{layer.datatype}")
    return pixels


def render_layout_region(database: LayoutDB, box: DbuBox, layer: LayerSpec,
                         pixel_size_nm: float = 5.0, *, output_path: str | Path | None = None,
                         show: bool = False,
                         max_pixels: int = _DEFAULT_MAX_PIXELS) -> NDArray[np.uint8]:
    """从已打开版图提取单层 planner 区域，并直接显示或保存灰度 PNG。"""
    batch = database.query([layer], box).materialize()
    return render_region_batch(batch, layer, database.dbu_um, pixel_size_nm,
                               output_path=output_path, show=show, max_pixels=max_pixels)
