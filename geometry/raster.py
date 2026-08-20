"""把 planner 局部区域栅格化为可展示和保存的灰度像素图。"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterator
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
_FLOAT64_DTYPE = np.dtype(np.float64)


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


def iter_region_coverage_tiles(
        region: kdb.Region, box: DbuBox, pixel_dbu: int,
        shape: tuple[int, int], *, dtype: np.dtype = _FLOAT64_DTYPE,
        max_tile_pixels: int = _MAX_STRIPE_PIXELS,
        ) -> Iterator[tuple[int, int, NDArray]]:
    """按左下原点分块生成裁剪、合并后的像素覆盖率数组。"""
    height, width = shape
    if (not isinstance(pixel_dbu, Integral) or isinstance(pixel_dbu, bool) or pixel_dbu <= 0 or
            height <= 0 or width <= 0 or max_tile_pixels <= 0):
        raise RasterizationError("像素 DBU、栅格尺寸和分块像素上限必须为正整数")
    # 两个上层调用都需要相同的集合语义：只计算当前 ROI，即进行剪裁，并在原生端合并重叠
    # Polygon，避免面积重复。显示层和 OPC 层共享坐标方向，只在类型、padding
    # 以及是否跨入图片输出边界上分工。
    clipped = (region & kdb.Region(box.to_native())).merged()
    if clipped.is_empty():
        return
    tile_width = min(width, max_tile_pixels)
    tile_rows = max(1, max_tile_pixels // tile_width)
    pixel_area = float(pixel_dbu * pixel_dbu)
    for y0 in range(0, height, tile_rows):
        rows = min(tile_rows, height - y0)
        for x0 in range(0, width, tile_width):
            columns = min(tile_width, width - x0)
            origin = kdb.Point(box.left + x0 * pixel_dbu, box.bottom + y0 * pixel_dbu)
            areas = np.asarray(clipped.rasterize(
                origin, kdb.Vector(pixel_dbu, pixel_dbu), columns, rows), dtype=dtype)
            # 面积矩阵是当前块唯一的大型浮点临时量；在原数组上归一化、裁界和取整，
            # 避免表达式链为每个百万像素块再产生两到三份 float64 中间数组。
            areas /= pixel_area
            np.clip(areas, 0.0, 1.0, out=areas)
            yield y0, x0, areas


def _rasterize(region: kdb.Region, box: DbuBox, pixel_dbu: int,
               shape: tuple[int, int]) -> NDArray[np.uint8]:
    """消费公共覆盖率分块，并转换为左下原点的八位灰度数组。"""
    pixels = np.zeros(shape, dtype=np.uint8)
    for y0, x0, areas in iter_region_coverage_tiles(region, box, pixel_dbu, shape):
        rows, columns = areas.shape
        areas *= 255.0
        np.rint(areas, out=areas)
        # 所有返回给 Python 调用方的版图/模型数组统一保持第 0 行为最低 Y；这样
        # ROI、OPC 探针和光刻画布可以直接共享索引约定。只有写入人眼图片时翻转，
        # 避免同一份覆盖率数组因为调用入口不同而具有相反方向。
        pixels[y0:y0 + rows, x0:x0 + columns] = areas.astype(np.uint8)
    return pixels


def _save_png(pixels: NDArray[np.uint8], output_path: str | Path) -> Path:
    """把左下原点数组翻为图片方向，并在目标目录原子保存灰度 PNG。"""
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
        # 图片文件第 0 行显示在顶部，因此只在 I/O 边界执行一次上下翻转；返回数组
        # 仍保持模型方向，保存动作不会原位修改调用方的数据。
        Image.fromarray(np.flipud(pixels)).save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def render_region_batch(batch: RegionBatch, layer: LayerSpec, dbu_um: float,
                        pixel_size_nm: float = 5.0, *, output_path: str | Path | None = None,
                        show: bool = False,
                        max_pixels: int = _DEFAULT_MAX_PIXELS) -> NDArray[np.uint8]:
    """返回左下原点灰度数组，并可按图片方向显示或保存单层区域。"""
    if layer not in batch.regions:
        raise RasterizationError(f"批次不包含图层 {layer.layer}/{layer.datatype}")
    pixel_dbu = _pixel_size_dbu(dbu_um, pixel_size_nm)
    shape = _image_shape(batch.query_box, pixel_dbu, max_pixels)
    pixels = _rasterize(batch.region(layer), batch.query_box, pixel_dbu, shape)
    if output_path is not None:
        _save_png(pixels, output_path)
    if show:
        Image.fromarray(np.flipud(pixels)).show(title=f"Layer {layer.layer}/{layer.datatype}")
    return pixels


def render_layout_region(database: LayoutDB, box: DbuBox, layer: LayerSpec,
                         pixel_size_nm: float = 5.0, *, output_path: str | Path | None = None,
                         show: bool = False,
                         max_pixels: int = _DEFAULT_MAX_PIXELS) -> NDArray[np.uint8]:
    """从已打开版图提取单层 planner 区域，并直接显示或保存灰度 PNG。"""
    batch = database.query([layer], box).materialize()
    return render_region_batch(batch, layer, database.dbu_um, pixel_size_nm,
                               output_path=output_path, show=show, max_pixels=max_pixels)
