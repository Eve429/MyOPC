"""OPC 方法共享的 core 网格与边界采样批量数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from layout import DbuBox

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]
FloatArray = NDArray[np.float64]


def _vector(value: object, dtype: np.dtype, name: str) -> NDArray:
    """把一维输入转换为连续数组，并统一拒绝隐藏的二维广播。"""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _points(value: object, name: str) -> FloatArray:
    """把坐标输入规范化为连续的 N×2 浮点数组。"""
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


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
        x_cuts = _vector(self.x_cuts, np.dtype(np.int64), "x_cuts")
        y_cuts = _vector(self.y_cuts, np.dtype(np.int64), "y_cuts")
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
        coords = _points(points, "points")
        x, y = coords[:, 0], coords[:, 1]
        columns = np.searchsorted(self.x_cuts, x, side="right") - 1
        rows = np.searchsorted(self.y_cuts, y, side="right") - 1
        # 半开区间可以让内部共享边界稳定归右侧/上侧 core；整体最大边界没有相邻
        # core 接管，因此单独归入最后一列/行，避免版图最外沿边段成为无 owner 数据。
        columns[x == self.x_cuts[-1]] = self.column_count - 1
        rows[y == self.y_cuts[-1]] = self.row_count - 1
        valid = ((columns >= 0) & (columns < self.column_count) &
                 (rows >= 0) & (rows < self.row_count))
        owners = np.full(len(coords), -1, dtype=np.int32)
        owners[valid] = (rows[valid] * self.column_count + columns[valid]).astype(np.int32)
        return owners


@dataclass(frozen=True, slots=True)
class BoundarySampleTemplate:
    """不含实际坐标、可在多轮优化中复用的边界采样模板。"""

    line_indices: Int32Array
    tangent_positions: FloatArray
    normal_offsets: FloatArray

    def __post_init__(self) -> None:
        """统一模板数组并验证切向位置位于闭区间内。"""
        indices = _vector(self.line_indices, np.dtype(np.int32), "line_indices")
        tangents = _vector(self.tangent_positions, np.dtype(np.float64), "tangent_positions")
        offsets = _vector(self.normal_offsets, np.dtype(np.float64), "normal_offsets")
        if len(indices) != len(tangents) or len(indices) != len(offsets):
            raise ValueError("sample template vectors must have equal length")
        if np.any(indices < 0) or np.any((tangents < 0.0) | (tangents > 1.0)):
            raise ValueError("sample indices and tangent positions are invalid")
        if not np.all(np.isfinite(tangents)) or not np.all(np.isfinite(offsets)):
            raise ValueError("sample template values must be finite")
        object.__setattr__(self, "line_indices", indices)
        object.__setattr__(self, "tangent_positions", tangents)
        object.__setattr__(self, "normal_offsets", offsets)


@dataclass(frozen=True, slots=True)
class BoundarySampleBatch:
    """一次物化得到的实际边界采样坐标及其模板元数据。"""

    points: FloatArray
    line_indices: Int32Array
    tangent_positions: FloatArray
    normal_offsets: FloatArray

    def __post_init__(self) -> None:
        """保证采样坐标与三个元数据向量严格对齐。"""
        points = _points(self.points, "points")
        indices = _vector(self.line_indices, np.dtype(np.int32), "line_indices")
        tangents = _vector(self.tangent_positions, np.dtype(np.float64), "tangent_positions")
        offsets = _vector(self.normal_offsets, np.dtype(np.float64), "normal_offsets")
        if any(len(array) != len(points) for array in (indices, tangents, offsets)):
            raise ValueError("sample metadata must match point count")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "line_indices", indices)
        object.__setattr__(self, "tangent_positions", tangents)
        object.__setattr__(self, "normal_offsets", offsets)
