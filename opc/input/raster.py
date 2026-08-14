"""为 OPC/ILT 的固定画布光刻输入提供原生 Region 栅格化。"""

from __future__ import annotations

from numbers import Integral

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from geometry import iter_region_coverage_tiles
from layout import DbuBox

from .mask import MaskPolarity


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
    # 公共底层分块输出保持左下原点；像素 [0,0] 的中心位于 box 原点加半个
    # pixel，因此探针进入数组索引时必须使用 `(xy-origin)/pixel-0.5`。这里仅
    # 负责固定 canvas 的 padding，不再维护第二份裁剪、合并和面积归一化逻辑。
    for y0, x0, areas in iter_region_coverage_tiles(
            region, box, pixel_dbu, (height, width), dtype=np.dtype(np.float32)):
        rows, columns = areas.shape
        result[y0:y0 + rows, x0:x0 + columns] = areas
    return result


def rasterize_mask_canvas(
        region: kdb.Region, box: DbuBox, pixel_dbu: int, canvas: int, *,
        polarity: MaskPolarity | str = MaskPolarity.CLEAR,
        field_box: DbuBox | None = None) -> NDArray[np.float32]:
    """把源多边形转换为统一的透光率画布，其中 1 始终表示透光。"""
    try:
        normalized = polarity if isinstance(polarity, MaskPolarity) else MaskPolarity(polarity)
    except ValueError as exc:
        raise ValueError(f"不支持的 mask 极性：{polarity!r}") from exc
    coverage = rasterize_region_canvas(region, box, pixel_dbu, canvas)
    if normalized is MaskPolarity.CLEAR:
        return coverage
    if field_box is None:
        raise ValueError("opaque 极性必须提供显式 field_box")
    # 只在光学数组边界构造处理框 coverage。处理框绝不进入 ContourBatch，因此其
    # 四条边不会成为虚假 OPC 边；box 跨越处理框时，框外 padding 保持不透光 0。
    field = rasterize_region_canvas(kdb.Region(field_box.to_native()), box, pixel_dbu, canvas)
    np.subtract(field, coverage, out=field)
    np.clip(field, 0.0, 1.0, out=field)
    return field


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
