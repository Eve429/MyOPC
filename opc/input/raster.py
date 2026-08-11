"""为 OPC/ILT 的固定画布光刻输入提供原生 Region 栅格化。"""

from __future__ import annotations

from numbers import Integral

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from geometry import iter_region_coverage_tiles
from layout import DbuBox


def rasterize_region_canvas(region: kdb.Region, box: DbuBox, pixel_dbu: int,
                            canvas: int) -> NDArray[np.float32]:
    """把全局 Region 的局部框栅格化到左下对齐的固定方形画布。"""
    if (not isinstance(pixel_dbu, Integral) or pixel_dbu <= 0 or
            not isinstance(canvas, Integral) or canvas <= 0):
        raise ValueError("pixel_dbu 和 canvas 必须是正整数")
    pixel_dbu, canvas = int(pixel_dbu), int(canvas)
    width = (box.width + pixel_dbu - 1) // pixel_dbu
    height = (box.height + pixel_dbu - 1) // pixel_dbu
    if width > canvas or height > canvas:
        raise ValueError(
            f"局部框需要 {width}x{height} 像素，超过 {canvas}x{canvas} 光刻画布")
    result = np.zeros((canvas, canvas), dtype=np.float32)
    # 公共底层分块输出保持左下原点，正好与探针 `(y-bottom)/pixel` 一致；这里
    # 只负责固定 canvas 的 padding，不再维护第二份裁剪、合并和面积归一化逻辑。
    for y0, x0, areas in iter_region_coverage_tiles(
            region, box, pixel_dbu, (height, width), dtype=np.dtype(np.float32)):
        rows, columns = areas.shape
        result[y0:y0 + rows, x0:x0 + columns] = areas
    return result


def ownership_canvas(core: DbuBox, context: DbuBox, pixel_dbu: int,
                     canvas: int) -> NDArray[np.bool_]:
    """按像素中心生成 core 唯一计分区域，halo 像素保持 False。"""
    if (context.left > core.left or context.bottom > core.bottom or
            context.right < core.right or context.top < core.top):
        raise ValueError("context 必须从四个方向完整包含 core")
    xs = context.left + (np.arange(canvas, dtype=np.float64) + 0.5) * pixel_dbu
    ys = context.bottom + (np.arange(canvas, dtype=np.float64) + 0.5) * pixel_dbu
    x_owned = (xs >= core.left) & (xs < core.right)
    y_owned = (ys >= core.bottom) & (ys < core.top)
    return y_owned[:, None] & x_owned[None, :]
