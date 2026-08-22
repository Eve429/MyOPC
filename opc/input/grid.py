"""提供 macro-core 两级不重叠网格的规划、描述与坐标归属操作。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from common.arrays import as_points, as_vector
from layout import DbuBox

IntArray = NDArray[np.int64]
Int32Array = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class CoreSpec:
    """描述一个 core 的唯一写入范围和只读上下文范围。"""

    core_id: str  # macro 内稳定行优先标识，例如 c_r1c2
    ownership_box: DbuBox  # 唯一可更新、可计分、最终可回写的非重叠区域
    context_box: DbuBox  # ownership 四边各扩 context_dbu 后的只读计算范围

    def __post_init__(self) -> None:
        """保证标识非空，且 context 在四个方向完整包住 core。"""
        if not self.core_id.strip():
            raise ValueError("core_id must be non-empty")
        core, context = self.ownership_box, self.context_box
        # ownership 与 context 都是全局 DBU 坐标框；context 允许越出 macro
        # 边界甚至版图 bbox，唯一硬约束是四向完整包含 ownership。
        if (
            context.left > core.left
            or context.bottom > core.bottom
            or context.right < core.right
            or context.top < core.top
        ):
            raise ValueError("context_box must contain ownership_box")


@dataclass(frozen=True, slots=True)
class MacroSpec:
    """保存一个 macro 的唯一写入框和局部 core 切线，不展开 core 对象列表。"""

    macro_id: str  # 全版稳定行优先 ID，例如 mr0c1
    ownership_box: DbuBox  # 当前 macro 对最终版图负责的非重叠范围
    x_cuts: IntArray  # macro 内 core 的全局 x 切线，严格递增
    y_cuts: IntArray  # macro 内 core 的全局 y 切线，严格递增
    context_dbu: int  # 每个 core 四边扩展的只读上下文 DBU
    pixel_dbu: int  # 一个光刻像素对应的整数 DBU
    canvas_pixels: int  # ICCAD13 固定方形 canvas，当前必须为 256

    def __post_init__(self) -> None:
        """规范化切线数组并校验像素、上下文与 canvas 数值契约。"""
        if not self.macro_id.strip():
            raise ValueError("macro_id must be non-empty")
        x_cuts = as_vector(self.x_cuts, np.dtype(np.int64), "x_cuts")
        y_cuts = as_vector(self.y_cuts, np.dtype(np.int64), "y_cuts")
        if len(x_cuts) < 2 or len(y_cuts) < 2:
            raise ValueError("macro core grid needs at least two cuts per axis")
        if np.any(np.diff(x_cuts) <= 0) or np.any(np.diff(y_cuts) <= 0):
            raise ValueError("macro core cuts must be strictly increasing")
        if (
            not isinstance(self.context_dbu, Integral)
            or self.context_dbu < 0
            or not isinstance(self.pixel_dbu, Integral)
            or self.pixel_dbu <= 0
        ):
            raise ValueError("context_dbu/pixel_dbu must be valid integers")
        # ICCAD13 Hopkins 核的频域分辨率冻结为 256；放开该值会在模型侧静默
        # 产生错误缩放，因此在描述层就拒绝而不是等到光刻前向调用。
        if not isinstance(self.canvas_pixels, Integral) or self.canvas_pixels != 256:
            raise ValueError("canvas_pixels is frozen to the ICCAD13 canvas of 256")
        object.__setattr__(self, "x_cuts", x_cuts)
        object.__setattr__(self, "y_cuts", y_cuts)
        object.__setattr__(self, "context_dbu", int(self.context_dbu))
        object.__setattr__(self, "pixel_dbu", int(self.pixel_dbu))
        object.__setattr__(self, "canvas_pixels", int(self.canvas_pixels))

    @property
    def column_count(self) -> int:
        """返回当前 macro 横向 core 数。"""
        return len(self.x_cuts) - 1

    @property
    def row_count(self) -> int:
        """返回当前 macro 纵向 core 数。"""
        return len(self.y_cuts) - 1

    @property
    def core_count(self) -> int:
        """返回当前 macro 的 core 总数。"""
        return self.column_count * self.row_count

    @property
    def query_box(self) -> DbuBox:
        """返回所有 core context_box 的最小包围框，供完整相交物化使用。"""
        # 每个 core 的 context_box 都是 ownership 四边扩 context_dbu，因此全部
        # context 的最小包围框就是 macro ownership 整体再四向扩 context_dbu。
        return self.ownership_box.expanded(self.context_dbu)

    def core(self, core_index: int) -> CoreSpec:
        """按局部行优先索引即时构造 CoreSpec，不缓存 CoreSpec 列表。"""
        if (
            not isinstance(core_index, Integral)
            or isinstance(core_index, bool)
            or core_index < 0
            or core_index >= self.core_count
        ):
            raise IndexError("core index is out of range")
        # 行优先分解到 (row, column)；切线是全局 DBU 坐标，构造的 box 不需要
        # 再做坐标平移，天然与版图全局坐标对齐。
        row, column = divmod(int(core_index), self.column_count)
        ownership = DbuBox(
            int(self.x_cuts[column]), int(self.y_cuts[row]), int(self.x_cuts[column + 1]), int(self.y_cuts[row + 1])
        )
        return CoreSpec(f"c_r{row}c{column}", ownership, ownership.expanded(self.context_dbu))

    def locate_owned_points(self, points: object) -> Int32Array:
        """返回点的局部 core owner；macro ownership 外返回 -1。"""
        coords = as_points(points, "points")
        x, y = coords[:, 0], coords[:, 1]
        columns = np.searchsorted(self.x_cuts, x, side="right") - 1
        rows = np.searchsorted(self.y_cuts, y, side="right") - 1
        # 内部共享切线按半开区间稳定归右/上；macro 整体最大边界归最后一列/行。
        # 与全局网格不同，越出 macro ownership 的点没有可归属的 core，必须保持
        # -1，让调用方把它识别为只读 context 段。
        columns[x == self.x_cuts[-1]] = self.column_count - 1
        rows[y == self.y_cuts[-1]] = self.row_count - 1
        valid = (columns >= 0) & (columns < self.column_count) & (rows >= 0) & (rows < self.row_count)
        owners = np.full(len(coords), -1, dtype=np.int32)
        owners[valid] = (rows[valid] * self.column_count + columns[valid]).astype(np.int32)
        return owners


def _macro_cuts_by_size(start: int, end: int, size: int) -> IntArray:
    """从轴起点按固定尺寸切分，最后一个区间允许缩短。"""
    if end <= start or size <= 0:
        raise ValueError("axis range and cut size must be positive")
    count = (end - start + size - 1) // size
    # 锚点固定为轴起点；最后一项强制等于轴终点，使不足一个尺寸的边缘
    # 仍被最后一个区间覆盖，不会产生越界 ownership 或零宽区间。
    cuts = start + np.arange(count + 1, dtype=np.int64) * size
    cuts[-1] = end
    return cuts


def _macro_cuts_by_count(start: int, end: int, core_size: int, count: int) -> IntArray:
    """按 core 单元数平衡分配指定数量的 macro。"""
    if end <= start or core_size <= 0:
        raise ValueError("axis range and core size must be positive")
    if not isinstance(count, Integral) or isinstance(count, bool) or count <= 0:
        raise ValueError("macro count must be a positive integer")
    unit_count = (end - start + core_size - 1) // core_size
    # 每个 macro 至少包含一个 core 单元；macro 数超过单元数时会出现空 macro，
    # 在规划层直接拒绝，而不是在后续 ownership 面积校验时才暴露。
    if count > unit_count:
        raise ValueError("macro count exceeds core unit count on this axis")
    # 尽量均分 core 单元数，较前的 macro 多分一个，避免几何硬等分产生的
    # 非整数 core 边界；最后一条切线强制等于轴终点收纳余量。
    base, remainder = divmod(unit_count, count)
    units = np.full(count, base, dtype=np.int64)
    units[:remainder] += 1
    cuts = start + np.concatenate(([0], np.cumsum(units))) * core_size
    cuts[-1] = end
    return cuts


def _core_cuts(start: int, end: int, core_size: int) -> IntArray:
    """在一个已确定 macro 内切 core，末端 core 允许缩短。"""
    if end <= start or core_size <= 0:
        raise ValueError("macro range and core size must be positive")
    count = (end - start + core_size - 1) // core_size
    # 与 _macro_cuts_by_size 同一数学形式但语义独立：这里保证的是「macro 内 core 切线」，
    # 末端缩短只会发生在版图最外侧 macro 内，符合名义 core 整数倍的规划约定。
    cuts = start + np.arange(count + 1, dtype=np.int64) * core_size
    cuts[-1] = end
    return cuts


def plan_macros(
    bounds: DbuBox,
    *,
    core_size_dbu: int,
    context_dbu: int,
    pixel_dbu: int,
    canvas_pixels: int,
    macro_size_dbu: int | None = None,
    macro_grid: tuple[int, int] | None = None,
) -> tuple[MacroSpec, ...]:
    """按 size 或 count 二选一先规划 macro，再在每个 macro 内规划 core。"""
    # macro 入口互斥：两种模式同时出现或同时缺失都说明配置意图不明确。
    if (macro_size_dbu is None) == (macro_grid is None):
        raise ValueError("exactly one of macro_size_dbu or macro_grid must be provided")
    # 共有数值契约：core/context 必须落在像素格点上，且 core+2context 的栅格
    # 尺寸不得超过固定 canvas；canvas 已由 MacroSpec 冻结为 256，这里提前检查
    # 可以在构造任何 MacroSpec 前给出更精确的错误位置。
    if (
        not isinstance(core_size_dbu, Integral)
        or core_size_dbu <= 0
        or not isinstance(context_dbu, Integral)
        or context_dbu < 0
        or not isinstance(pixel_dbu, Integral)
        or pixel_dbu <= 0
    ):
        raise ValueError("core/context/pixel sizes must be valid integers")
    if canvas_pixels != 256:
        raise ValueError("canvas_pixels is frozen to the ICCAD13 canvas of 256")
    if core_size_dbu % pixel_dbu or context_dbu % pixel_dbu:
        raise ValueError("core and context sizes must be whole multiples of a pixel")
    canvas_pixels_needed = -(-(core_size_dbu + 2 * context_dbu) // pixel_dbu)
    if canvas_pixels_needed > canvas_pixels:
        raise ValueError("core plus twice context exceeds the fixed canvas")
    # 按 x/y 两轴独立规划 macro 切线；两轴使用同一组私有切分函数，保证
    # size 模式与 count 模式在两个方向上的行为完全对称。
    if macro_size_dbu is not None:
        if not isinstance(macro_size_dbu, Integral) or macro_size_dbu <= 0:
            raise ValueError("macro_size_dbu must be a positive integer")
        # 名义 macro 必须严格大于 core 且为 core 的整数倍（设计文档 §5.3）：
        # 等于 core 会让两级网格退化为纯 core 网格，宏观边界失去意义。
        if macro_size_dbu <= core_size_dbu:
            raise ValueError("macro size must exceed core size")
        if macro_size_dbu % core_size_dbu:
            raise ValueError("macro size must be a whole multiple of core size")
        x_macro = _macro_cuts_by_size(bounds.left, bounds.right, macro_size_dbu)
        y_macro = _macro_cuts_by_size(bounds.bottom, bounds.top, macro_size_dbu)
    elif macro_grid is not None:
        columns, rows = macro_grid
        if (
            not isinstance(columns, Integral)
            or isinstance(columns, bool)
            or not isinstance(rows, Integral)
            or isinstance(rows, bool)
            or columns <= 0
            or rows <= 0
        ):
            raise ValueError("macro_grid entries must be positive integers")
        x_macro = _macro_cuts_by_count(bounds.left, bounds.right, core_size_dbu, columns)
        y_macro = _macro_cuts_by_count(bounds.bottom, bounds.top, core_size_dbu, rows)
    # 行优先展开全部 macro；每个 macro 立即规划自己的 core 切线。所有
    # ownership 由互不相同且首尾相接的切线区间构成，天然无正面积重叠。
    macros: list[MacroSpec] = []
    for row in range(len(y_macro) - 1):
        for column in range(len(x_macro) - 1):
            ownership = DbuBox(int(x_macro[column]), int(y_macro[row]), int(x_macro[column + 1]), int(y_macro[row + 1]))
            macros.append(
                MacroSpec(
                    f"mr{row}c{column}",
                    ownership,
                    _core_cuts(int(x_macro[column]), int(x_macro[column + 1]), core_size_dbu),
                    _core_cuts(int(y_macro[row]), int(y_macro[row + 1]), core_size_dbu),
                    context_dbu,
                    pixel_dbu,
                    canvas_pixels,
                )
            )
    return tuple(macros)
