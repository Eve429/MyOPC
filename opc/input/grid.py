"""定义 OPC 方法共享的规则 core 网格及其坐标归属操作。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from layout import DbuBox

from ._arrays import as_points, as_vector

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class CoreSpec:
    """一个 core 的唯一输出范围及其包含光学上下文的查询范围。"""

    core_id: str
    ownership_box: DbuBox
    context_box: DbuBox

    def __post_init__(self) -> None:
        """保证标识非空，且 context 在四个方向完整包住 core。"""
        if not self.core_id.strip():
            raise ValueError("core_id must be non-empty")
        core, context = self.ownership_box, self.context_box
        if (context.left > core.left or context.bottom > core.bottom or
                context.right < core.right or context.top < core.top):
            raise ValueError("context_box must contain ownership_box")


@dataclass(frozen=True, slots=True)
class RectilinearCoreGrid:
    """用有序切线定义可含非等宽边缘 core 的规则正交网格。"""

    x_cuts: IntArray
    y_cuts: IntArray
    halo_dbu: int = 0

    def __post_init__(self) -> None:
        """规范化切线，并拒绝空网格、重复切线和负 halo。"""
        x_cuts = as_vector(self.x_cuts, np.dtype(np.int64), "x_cuts")
        y_cuts = as_vector(self.y_cuts, np.dtype(np.int64), "y_cuts")
        if len(x_cuts) < 2 or len(y_cuts) < 2:
            raise ValueError("core grid needs at least two cuts per axis")
        if np.any(np.diff(x_cuts) <= 0) or np.any(np.diff(y_cuts) <= 0):
            raise ValueError("core grid cuts must be strictly increasing")
        if not isinstance(self.halo_dbu, Integral) or self.halo_dbu < 0:
            raise ValueError("halo_dbu must be a non-negative integer")
        object.__setattr__(self, "x_cuts", x_cuts)
        object.__setattr__(self, "y_cuts", y_cuts)
        object.__setattr__(self, "halo_dbu", int(self.halo_dbu))

    @property
    def column_count(self) -> int:
        """返回水平方向 core 数量。"""
        return len(self.x_cuts) - 1

    @property
    def row_count(self) -> int:
        """返回垂直方向 core 数量。"""
        return len(self.y_cuts) - 1

    @property
    def core_count(self) -> int:
        """返回网格中的 core 总数。"""
        return self.column_count * self.row_count

    @property
    def bounds(self) -> DbuBox:
        """返回所有 ownership core 覆盖的整体范围。"""
        return DbuBox(int(self.x_cuts[0]), int(self.y_cuts[0]),
                      int(self.x_cuts[-1]), int(self.y_cuts[-1]))

    def cores(self) -> tuple[CoreSpec, ...]:
        """按先行后列的稳定顺序生成 core 和 halo 描述。"""
        result: list[CoreSpec] = []
        for row in range(self.row_count):
            for column in range(self.column_count):
                core = DbuBox(int(self.x_cuts[column]), int(self.y_cuts[row]),
                              int(self.x_cuts[column + 1]), int(self.y_cuts[row + 1]))
                result.append(CoreSpec(f"r{row}c{column}", core, core.expanded(self.halo_dbu)))
        return tuple(result)

    def locate_points(self, points: object) -> Int32Array:
        """向量化返回点的 core 索引，范围外点使用 -1。"""
        coords = as_points(points, "points")
        x, y = coords[:, 0], coords[:, 1]
        columns = np.searchsorted(self.x_cuts, x, side="right") - 1
        rows = np.searchsorted(self.y_cuts, y, side="right") - 1
        # 内部共享边界使用半开区间并稳定归右侧/上侧；整体最大边界没有后继
        # core，因此显式归入最后一列/行，保证最外沿边段仍有唯一 owner。
        columns[x == self.x_cuts[-1]] = self.column_count - 1
        rows[y == self.y_cuts[-1]] = self.row_count - 1
        valid = ((columns >= 0) & (columns < self.column_count) &
                 (rows >= 0) & (rows < self.row_count))
        owners = np.full(len(coords), -1, dtype=np.int32)
        owners[valid] = (rows[valid] * self.column_count + columns[valid]).astype(np.int32)
        return owners
