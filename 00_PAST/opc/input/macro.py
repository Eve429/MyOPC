"""把全局 tile 切线分组为 CPU 内存有界的 macro ownership 框。"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from layout import DbuBox

from .grid import RectilinearCoreGrid


def _group_cuts(cuts: np.ndarray, maximum_span: int) -> np.ndarray:
    """在不切开 tile 的前提下生成跨度不超过上限的切线子集。"""
    selected = [int(cuts[0])]
    index = 0
    while index < len(cuts) - 1:
        limit = selected[-1] + maximum_span
        next_index = int(np.searchsorted(cuts, limit, side="right") - 1)
        # 单个 tile 可能大于请求 macro；此时必须完整接纳这个 tile，不能为了
        # 严守跨度而在 tile 内制造新的 ownership 边界。
        next_index = max(index + 1, next_index)
        selected.append(int(cuts[next_index]))
        index = next_index
    return np.asarray(selected, dtype=np.int64)


def macro_boxes(tile_grid: RectilinearCoreGrid,
                maximum_span_dbu: int) -> tuple[DbuBox, ...]:
    """按行优先返回与 tile 切线严格对齐的 macro ownership 框。"""
    if (not isinstance(maximum_span_dbu, Integral) or
            isinstance(maximum_span_dbu, bool) or maximum_span_dbu <= 0):
        raise ValueError("maximum_span_dbu 必须是正整数")
    x_cuts = _group_cuts(tile_grid.x_cuts, int(maximum_span_dbu))
    y_cuts = _group_cuts(tile_grid.y_cuts, int(maximum_span_dbu))
    return tuple(
        DbuBox(int(x_cuts[column]), int(y_cuts[row]),
               int(x_cuts[column + 1]), int(y_cuts[row + 1]))
        for row in range(len(y_cuts) - 1)
        for column in range(len(x_cuts) - 1))
