"""把局部 context 栅格化并居中放入 ICCAD13 固定 256 画布。"""

from __future__ import annotations

from numbers import Integral

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from geometry import iter_region_coverage_tiles
from layout import DbuBox

from .mask import MaskPolarity


def rasterize_region_window(
    region: kdb.Region,
    box: DbuBox,
    pixel_dbu: int,
) -> NDArray[np.float32]:
    """把物理 box 栅格为最小 H×W 覆盖率数组，不添加模型 canvas padding。"""
    if not isinstance(pixel_dbu, Integral) or pixel_dbu <= 0:
        raise ValueError("pixel_dbu 必须是正整数")
    pixel_dbu = int(pixel_dbu)
    # 轴长不是 pixel 整数倍时按向上取整生成最小覆盖数组；边缘像素保留真实
    # 面积覆盖率，不移动几何边界迎合 pixel 网格。
    width = (box.width + pixel_dbu - 1) // pixel_dbu
    height = (box.height + pixel_dbu - 1) // pixel_dbu
    result = np.zeros((height, width), dtype=np.float32)
    # 底层分块输出保持左下原点：行 0 = 最低 Y；像素中心位于 box 原点加
    # 半个 pixel。这里只负责窗口覆盖率本身，canvas 居中由上层负责。
    for y0, x0, areas in iter_region_coverage_tiles(
        region, box, pixel_dbu, (height, width), dtype=np.dtype(np.float32)
    ):
        rows, columns = areas.shape
        result[y0 : y0 + rows, x0 : x0 + columns] = areas
    return result


def _center_padding(local_height: int, local_width: int, canvas_pixels: int) -> tuple[int, int, int, int]:
    """返回低/高 y 和低/高 x 的居中零填充宽度。"""
    if (
        not isinstance(local_height, Integral)
        or local_height <= 0
        or not isinstance(local_width, Integral)
        or local_width <= 0
        or not isinstance(canvas_pixels, Integral)
        or canvas_pixels <= 0
    ):
        raise ValueError("local dims and canvas must be positive integers")
    if local_height > canvas_pixels or local_width > canvas_pixels:
        raise ValueError("local window exceeds the fixed canvas")
    # 差值平均分配到低/高两侧；奇数余量归高坐标侧，与旧模型 _prepare_mask 的
    # 居中补零约定一致，保证同尺寸输入永远得到同一 canvas 布局。
    low_y = (canvas_pixels - local_height) // 2
    low_x = (canvas_pixels - local_width) // 2
    return low_y, canvas_pixels - local_height - low_y, low_x, canvas_pixels - local_width - low_x


def rasterize_mask_canvas(
    region: kdb.Region,
    context_box: DbuBox,
    pixel_dbu: int,
    canvas_pixels: int,
    *,
    polarity: MaskPolarity | str,
) -> NDArray[np.float32]:
    """把 context 透光率居中放入固定 canvas，外围 padding 恒 0。

    窗口外暗场由几何保证（负板 prepare 已补铬、正板包络外天然 coverage=0，
    2026-08-22 起），本函数只做极性变换与居中 padding，无暗界参数。
    """
    try:
        normalized = polarity if isinstance(polarity, MaskPolarity) else MaskPolarity(polarity)
    except ValueError as exc:
        raise ValueError(f"不支持的 mask 极性：{polarity!r}") from exc
    if (
        not isinstance(pixel_dbu, Integral)
        or pixel_dbu <= 0
        or not isinstance(canvas_pixels, Integral)
        or canvas_pixels <= 0
    ):
        raise ValueError("pixel_dbu 和 canvas_pixels 必须是正整数")
    pixel_dbu, canvas_pixels = int(pixel_dbu), int(canvas_pixels)
    # 先用纯算术检查窗口尺寸，超限在分配 canvas 数组之前失败；此时局部
    # 栅格尚未发生，错误不会留下大数组等待回收。
    local_width = (context_box.width + pixel_dbu - 1) // pixel_dbu
    local_height = (context_box.height + pixel_dbu - 1) // pixel_dbu
    if local_width > canvas_pixels or local_height > canvas_pixels:
        raise ValueError(
            f"局部窗口需要 {local_height}x{local_width} 像素，超过 {canvas_pixels}x{canvas_pixels} 固定画布"
        )
    coverage = rasterize_region_window(region, context_box, pixel_dbu)
    low_y, _, low_x, _ = _center_padding(int(coverage.shape[0]), int(coverage.shape[1]), canvas_pixels)
    # 数组值恒为光学定义 1.0=透光、0.0=不透光：clear 时 coverage 即透光，
    # opaque 时 1−coverage（field − 不透光图形）。两种极性外围 padding 恒 0，
    # 极性只改变「源 polygon 如何转换为透光率」。
    canvas = np.zeros((canvas_pixels, canvas_pixels), dtype=np.float32)
    if normalized is MaskPolarity.CLEAR:
        local = coverage
    else:
        local = 1.0 - coverage
    h, w = local.shape
    canvas[low_y : low_y + h, low_x : low_x + w] = local
    return canvas


def ownership_canvas(
    ownership_box: DbuBox,
    context_box: DbuBox,
    pixel_dbu: int,
    canvas_pixels: int,
) -> NDArray[np.bool_]:
    """返回与居中 mask canvas 对齐的唯一计分像素，context/padding 为 False。"""
    if (
        not isinstance(pixel_dbu, Integral)
        or pixel_dbu <= 0
        or not isinstance(canvas_pixels, Integral)
        or canvas_pixels <= 0
    ):
        raise ValueError("pixel_dbu 和 canvas_pixels 必须是正整数")
    pixel_dbu, canvas_pixels = int(pixel_dbu), int(canvas_pixels)
    if (
        context_box.left > ownership_box.left
        or context_box.bottom > ownership_box.bottom
        or context_box.right < ownership_box.right
        or context_box.top < ownership_box.top
    ):
        raise ValueError("context_box 必须从四个方向完整包含 ownership_box")
    # 居中偏移必须与 rasterize_mask_canvas 完全一致，否则计分像素与 mask像素错位；
    # 两者共用 _center_padding 是对齐的数值保证。
    local_width = (context_box.width + pixel_dbu - 1) // pixel_dbu
    local_height = (context_box.height + pixel_dbu - 1) // pixel_dbu
    low_y, _, low_x, _ = _center_padding(int(local_height), int(local_width), canvas_pixels)
    # 全局 DBU 坐标映射到 canvas 像素（后续 EPE/probe 必须复用同一公式，
    # 不能假设 context 位于 canvas 左下角）：
    #   x_canvas = (x_dbu - context.left) / pixel_dbu - 0.5 + low_x
    #   y_canvas = (y_dbu - context.bottom) / pixel_dbu - 0.5 + low_y
    # 像素中心采样：canvas 第 j 列的中心位于 context.left + (j - low_x + 0.5)×pixel。
    columns = np.arange(canvas_pixels, dtype=np.int64)
    rows = np.arange(canvas_pixels, dtype=np.int64)
    x_centers = (context_box.left + (columns - low_x + 0.5) * pixel_dbu).astype(np.float64)
    y_centers = (context_box.bottom + (rows - low_y + 0.5) * pixel_dbu).astype(np.float64)
    # 只有真实存在于局部窗口内、且中心落在 ownership 半开区间内的像素才计分；
    # 窗口外的 padding 列/行中心会落到 context 之外，由有效区间掩码排除。
    x_valid = (
        (columns >= low_x)
        & (columns < low_x + local_width)
        & (x_centers >= ownership_box.left)
        & (x_centers < ownership_box.right)
    )
    y_valid = (
        (rows >= low_y)
        & (rows < low_y + local_height)
        & (y_centers >= ownership_box.bottom)
        & (y_centers < ownership_box.top)
    )
    return y_valid[:, None] & x_valid[None, :]


def points_to_canvas(
    points_dbu: object,
    context_box: DbuBox,
    pixel_dbu: int,
    canvas_pixels: int,
) -> NDArray[np.float64]:
    """把全局 DBU 点转换为居中 canvas 的连续 (x,y) 像素坐标。"""
    if (
        not isinstance(pixel_dbu, Integral)
        or pixel_dbu <= 0
        or not isinstance(canvas_pixels, Integral)
        or canvas_pixels <= 0
    ):
        raise ValueError("pixel_dbu 和 canvas_pixels 必须是正整数")
    pixel_dbu, canvas_pixels = int(pixel_dbu), int(canvas_pixels)
    points = np.asarray(points_dbu, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_dbu 必须是 [N,2] 的 (x,y) 坐标数组")
    # padding 与 rasterize_mask_canvas/ownership_canvas 共用同一 _center_padding，
    # 因此换算出的探针坐标与 mask 像素、ownership 像素天然对齐；本函数只做连续
    # 坐标换算，取整与越界处理留给评价层（evaluate_edge_probes 的 round+in_bounds）。
    local_width = (context_box.width + pixel_dbu - 1) // pixel_dbu
    local_height = (context_box.height + pixel_dbu - 1) // pixel_dbu
    low_y, _, low_x, _ = _center_padding(int(local_height), int(local_width), canvas_pixels)
    canvas = np.empty_like(points)
    # x 进列索引、y 进行索引：canvas 第 j 列中心位于 context.left+(j-low_x+0.5)×pixel，
    # 逆映射即 (x-left)/pixel-0.5+low_x，与 ownership_canvas 的正向公式互为反函数。
    canvas[:, 0] = (points[:, 0] - context_box.left) / pixel_dbu - 0.5 + low_x
    canvas[:, 1] = (points[:, 1] - context_box.bottom) / pixel_dbu - 0.5 + low_y
    return canvas
