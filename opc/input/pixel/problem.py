"""像素型宏问题：一次栅格化目标、NPZ 持久化、core 画布映射与像素回写。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from common.io import atomic_write_npz
from layout import DbuBox, LayerSpec, RegionBatch
from opc.input import CoreSpec, MacroSpec, MaskPolarity
from opc.input.mask import normalize_mask
from opc.input.raster import (
    _center_padding,
    ownership_canvas,
    rasterize_region_window,
)

# 持久化格式身份：不兼容的结构变更必须递增版本并在 load 中显式拒绝旧版本。
_FORMAT_NAME = "myopc.pixel-ilt-problem"
_FORMAT_VERSION = 1


def _require_pixel_aligned(box: DbuBox, pixel_dbu: int, what: str) -> None:
    """拒绝宽高非整像素的 box；partial 像素会破坏参数域与回写的一一对应。"""
    if box.width % pixel_dbu or box.height % pixel_dbu:
        raise ValueError(
            f"{what} 宽高必须是 pixel_dbu={pixel_dbu} 的整数倍："
            f"{box.width}x{box.height}")


@dataclass(frozen=True, slots=True)
class PixelMacroProblem:
    """一个像素型 macro 的持久化输入：query box 一张 uint8 transmission 栅格。

    坐标约定与整个输入层一致：数组 [y,x]、行 0 = query box 最低 Y、
    值 1（255）= 透光、中间值 = 面积覆盖率；clear/opaque 只在构造与
    最终 GDS 逆变换两处出现，求解器内部恒为 transmission。
    """

    macro: MacroSpec
    # macro/core 网格、context、pixel 与 canvas 契约（与 edge problem 同源）。

    layer: LayerSpec
    # 最终 GDS 写出的唯一目标 layer/datatype。

    polarity: MaskPolarity
    # 源 polygon 的 mask 极性；target_u8 已统一为 transmission（1=透光）。

    target_u8: NDArray[np.uint8]
    # [Hq,Wq] query box 覆盖率 transmission，量化到 0..255；不存每 core
    # 重复 256 画布，core canvas 由本类按需切片。

    def __post_init__(self) -> None:
        """规范化极性并校验栅格与网格的整像素一致性（构造即校验）。"""
        try:
            polarity = (self.polarity if isinstance(self.polarity, MaskPolarity)
                        else MaskPolarity(self.polarity))
        except ValueError as exc:
            raise ValueError(f"不支持的 mask 极性：{self.polarity!r}") from exc
        target = np.asarray(self.target_u8)
        pixel_dbu = int(self.macro.pixel_dbu)
        query = self.macro.query_box
        # query box（= ownership + 2×context）必须整像素：非整除说明网格
        # 契约被破坏（截断 NPZ 或手工构造），切片区间将无法对齐。
        _require_pixel_aligned(query, pixel_dbu, "macro query box")
        expected = (query.height // pixel_dbu, query.width // pixel_dbu)
        if (target.ndim != 2 or target.dtype != np.dtype(np.uint8)
                or target.shape != expected):
            raise ValueError(
                f"target_u8 必须是 {expected} 的 uint8 数组，"
                f"实际为 {target.shape} / {target.dtype}")
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "target_u8", np.ascontiguousarray(target))

    @property
    def query_shape(self) -> tuple[int, int]:
        """返回 query box 的整像素形状 (Hq, Wq)。"""
        pixel_dbu = int(self.macro.pixel_dbu)
        query = self.macro.query_box
        return query.height // pixel_dbu, query.width // pixel_dbu

    @property
    def ownership_shape(self) -> tuple[int, int]:
        """返回 macro ownership 的整像素形状 (Hm, Wm)，即 macro 参数域形状。"""
        pixel_dbu = int(self.macro.pixel_dbu)
        box = self.macro.ownership_box
        return box.height // pixel_dbu, box.width // pixel_dbu

    def _context_window(self, core_index: int) -> tuple[CoreSpec, int, int, int, int]:
        """返回 core spec 及其 context 在 query 栅格中的 [r0,r1)/[c0,c1) 区间。"""
        spec = self.macro.core(core_index)
        pixel_dbu = int(self.macro.pixel_dbu)
        query = self.macro.query_box
        # 整除由构造期整像素校验保证（context 是 pixel 倍数、core box 已对齐），
        # 因此切片与 query 栅格像素逐一对齐，不存在半个像素的偏移。
        r0 = (spec.context_box.bottom - query.bottom) // pixel_dbu
        c0 = (spec.context_box.left - query.left) // pixel_dbu
        r1 = r0 + spec.context_box.height // pixel_dbu
        c1 = c0 + spec.context_box.width // pixel_dbu
        return spec, r0, r1, c0, c1

    def target_canvas(self, core_index: int) -> NDArray[np.uint8]:
        """切 query 栅格为该 core context 的居中 canvas uint8 画布。"""
        _, r0, r1, c0, c1 = self._context_window(core_index)
        window = self.target_u8[r0:r1, c0:c1]
        # 与 rasterize_mask_canvas 共用同一居中 padding：探针/ownership/mask
        # 画布永远同布局，这是复用模型与指标层的前提。
        low_y, _, low_x, _ = _center_padding(
            r1 - r0, c1 - c0, int(self.macro.canvas_pixels))
        canvas = np.zeros(
            (int(self.macro.canvas_pixels), int(self.macro.canvas_pixels)),
            dtype=np.uint8)
        canvas[low_y:low_y + window.shape[0],
               low_x:low_x + window.shape[1]] = window
        return canvas

    def ownership_canvas(self, core_index: int) -> NDArray[np.bool_]:
        """返回该 core 唯一计分像素画布（复用公共 ownership 对齐规则）。"""
        spec = self.macro.core(core_index)
        return ownership_canvas(
            spec.ownership_box, spec.context_box,
            int(self.macro.pixel_dbu), int(self.macro.canvas_pixels))

    def trainable_index_canvas(self, core_index: int) -> NDArray[np.int64]:
        """返回 macro 参数扁平索引画布：macro 外（含 context/padding）恒 -1。

        非负值是 [Hm,Wm] 行主序扁平下标；同一物理像素在任何 core 的画布
        中都映射到同一索引——索引定义在 macro 网格而非 core 画布上。
        索引域恒 int64：macro 总像素超过 2^31（4nm pixel 下约 185µm² 宏）
        时 int32 在构造期溢出，负值会被误判为 macro 外 context。
        """
        _, r0, r1, c0, c1 = self._context_window(core_index)
        pixel_dbu = int(self.macro.pixel_dbu)
        query = self.macro.query_box
        box = self.macro.ownership_box
        # macro ownership 在 query 栅格中的像素块位置
        mrow0 = (box.bottom - query.bottom) // pixel_dbu
        mcol0 = (box.left - query.left) // pixel_dbu
        hm, wm = self.ownership_shape
        canvas = np.full(
            (int(self.macro.canvas_pixels), int(self.macro.canvas_pixels)),
            -1, dtype=np.int64)
        # context 只覆盖 macro 的一部分时取交集；索引块只按窗口大小直接
        # 构造：block[i,j] = 行基址 + i*wm + j。本函数是每 state × 每 core
        # 的热路径，禁止为单个 core 分配 O(宏像素) 的全宏 arange 索引表。
        row0, row1 = max(r0, mrow0), min(r1, mrow0 + hm)
        col0, col1 = max(c0, mcol0), min(c1, mcol0 + wm)
        if row0 < row1 and col0 < col1:
            row_base = (row0 - mrow0) * wm + (col0 - mcol0)
            rows, cols = row1 - row0, col1 - col0
            block = (row_base
                     + np.arange(rows, dtype=np.int64)[:, None] * wm
                     + np.arange(cols, dtype=np.int64)[None, :])
            low_y, _, low_x, _ = _center_padding(
                r1 - r0, c1 - c0, int(self.macro.canvas_pixels))
            canvas[low_y + row0 - r0:low_y + row1 - r0,
                   low_x + col0 - c0:low_x + col1 - c0] = block
        return canvas

    def context_valid_canvas(self, core_index: int) -> NDArray[np.bool_]:
        """返回真实 context window 掩码：window 内 True，数值 padding False。

        target_canvas 的 window 外恒 0 是填满固定画布的数值 padding，不是
        物理 T=0 mask pixel；消费方对两者必须区别对待（物理 context 用初始
        soft、padding 恒 0），本掩码是区分两者的唯一判据。
        """
        _, r0, r1, c0, c1 = self._context_window(core_index)
        low_y, _, low_x, _ = _center_padding(
            r1 - r0, c1 - c0, int(self.macro.canvas_pixels))
        canvas = np.zeros(
            (int(self.macro.canvas_pixels), int(self.macro.canvas_pixels)),
            dtype=np.bool_)
        canvas[low_y:low_y + (r1 - r0),
               low_x:low_x + (c1 - c0)] = True
        return canvas

    def _arrays(self) -> dict[str, np.ndarray]:
        """按格式版本 1 的键名打包全部待持久化数组。"""
        macro = self.macro
        return {
            "format": np.array([_FORMAT_NAME]),
            "format_version": np.array([_FORMAT_VERSION], dtype=np.int32),
            "macro_id": np.array([macro.macro_id]),
            "macro_ownership_box": np.array(
                [macro.ownership_box.left, macro.ownership_box.bottom,
                 macro.ownership_box.right, macro.ownership_box.top], dtype=np.int64),
            "macro_x_cuts": macro.x_cuts.astype(np.int64, copy=False),
            "macro_y_cuts": macro.y_cuts.astype(np.int64, copy=False),
            "context_dbu": np.array([macro.context_dbu], dtype=np.int64),
            "pixel_dbu": np.array([macro.pixel_dbu], dtype=np.int64),
            "canvas_pixels": np.array([macro.canvas_pixels], dtype=np.int64),
            "layer": np.array([self.layer.layer], dtype=np.int32),
            "datatype": np.array([self.layer.datatype], dtype=np.int32),
            "polarity": np.array([self.polarity.value]),
            "target_u8": self.target_u8.astype(np.uint8, copy=False),
        }

    def save(self, path: str | Path) -> Path:
        """把 problem 以不压缩 NPZ 原子保存，不写重复几何数组。"""
        output = Path(path).expanduser().resolve()
        if not output.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {output.parent}")
        atomic_write_npz(output, **self._arrays())
        return output

    @classmethod
    def load(cls, path: str | Path) -> PixelMacroProblem:
        """使用 allow_pickle=False 读取并通过现有结构不变量校验 problem。"""
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["format"][0]) != _FORMAT_NAME:
                raise ValueError("unsupported pixel problem format name")
            if int(data["format_version"][0]) != _FORMAT_VERSION:
                raise ValueError("unsupported pixel problem format version")
            macro = MacroSpec(
                str(data["macro_id"][0]),
                DbuBox(*[int(v) for v in data["macro_ownership_box"]]),
                data["macro_x_cuts"], data["macro_y_cuts"],
                int(data["context_dbu"][0]), int(data["pixel_dbu"][0]),
                int(data["canvas_pixels"][0]))
            layer = LayerSpec(int(data["layer"][0]), int(data["datatype"][0]))
            # 构造即校验：__post_init__ 复查极性、整像素一致性与栅格形状；
            # 损坏或被篡改的 NPZ 在这里直接失败，不做补零修复。
            return cls(macro, layer, MaskPolarity(str(data["polarity"][0])),
                       data["target_u8"])


def prepare_pixel_macro_problem(
        batch: RegionBatch, layer: LayerSpec, polarity: MaskPolarity | str,
        macro: MacroSpec, *, layout_bounds: DbuBox) -> PixelMacroProblem:
    """从一次完整相交物化构造像素 macro 问题（一次栅格化，不提边）。

    layout_bounds 是 plan_macros 所用的版图层 bbox，即光学场边界（00_PAST
    field_box 契约的迁移等价）：query 超出 bbox 的环带恒不透光。必填无默认，
    缺省会静默保留负板透光缺陷，契约必须显式。
    """
    if batch.query_box != macro.query_box:
        raise ValueError("batch.query_box 必须等于 macro.query_box")
    try:
        normalized = (polarity if isinstance(polarity, MaskPolarity)
                      else MaskPolarity(polarity))
    except ValueError as exc:
        raise ValueError(f"不支持的 mask 极性：{polarity!r}") from exc
    pixel_dbu = int(macro.pixel_dbu)
    query = macro.query_box
    # 实际 box 整像素前置校验：最外侧缩短 core 是网格规划允许的合法形态，
    # 但非整像素缩短会产生 partial ownership pixel——它无法映射到唯一个
    # 参数/计分/回写像素，必须在栅格化之前失败而不是静默取整。
    _require_pixel_aligned(
        macro.ownership_box, pixel_dbu, f"macro {macro.macro_id} ownership")
    for core_index in range(macro.core_count):
        _require_pixel_aligned(
            macro.core(core_index).ownership_box, pixel_dbu,
            f"macro {macro.macro_id} core {core_index} ownership")
    ownership = macro.ownership_box
    if (layout_bounds.left > ownership.left or layout_bounds.bottom > ownership.bottom
            or layout_bounds.right < ownership.right
            or layout_bounds.top < ownership.top):
        raise ValueError(
            "layout_bounds 必须四向包含 macro ownership（应传 plan_macros "
            "所用的版图层 bbox）")
    # bounds 与 query 的交叠边必须落在 query 像素格点上：按网格契约 bounds 边
    # 即 ownership 切线、context 为 pixel 整数倍，天然对齐；余数非零说明调用方
    # 传了与规划网格不一致的 bounds，静默取整会切掉半个像素。
    inside_left = max(layout_bounds.left, query.left)
    inside_right = min(layout_bounds.right, query.right)
    inside_bottom = max(layout_bounds.bottom, query.bottom)
    inside_top = min(layout_bounds.top, query.top)
    row0, rem_y0 = divmod(inside_bottom - query.bottom, pixel_dbu)
    row1, rem_y1 = divmod(inside_top - query.bottom, pixel_dbu)
    col0, rem_x0 = divmod(inside_left - query.left, pixel_dbu)
    col1, rem_x1 = divmod(inside_right - query.left, pixel_dbu)
    if rem_y0 or rem_y1 or rem_x0 or rem_x1:
        raise ValueError(
            f"layout_bounds 与 query 的交叠边必须是 pixel_dbu={pixel_dbu} "
            "的整像素倍")
    # 完整相交物化合并物理覆盖后栅格化一次：查询框不参与布尔相交，
    # 版图真实边界的覆盖率（斜边/半像素）原样进入 transmission。
    region = normalize_mask(batch, layer)
    coverage = rasterize_region_window(region, query, pixel_dbu)
    # 极性只在此边界出现一次：clear 时图形即透光，opaque 时背景透光。
    transmission = (coverage if normalized is MaskPolarity.CLEAR
                    else 1.0 - coverage)
    # 版图 bbox 之外恒不透光：外围 macro 的 query 超出 bbox 的环带没有几何，
    # opaque 的 1−coverage 会把 0 覆盖反成虚假透光环，污染边界 core 的光学
    # 上下文；clear 该处置零是逐位 no-op。场边界只作用于 transmission 数组、
    # 绝不作为图形进入 Region，因此不会产生虚假可动边（旧系统明令）。
    transmission[:row0, :] = 0.0
    transmission[row1:, :] = 0.0
    transmission[:, :col0] = 0.0
    transmission[:, col1:] = 0.0
    target_u8 = np.rint(
        np.clip(transmission, 0.0, 1.0) * 255.0).astype(np.uint8)
    return PixelMacroProblem(
        macro=macro, layer=layer, polarity=normalized, target_u8=target_u8)


def reconstruct_pixel_region(
        problem: PixelMacroProblem,
        binary_ownership: object) -> kdb.Region:
    """把 macro ownership 二值 transmission 按行游程合并为裁剪后的 Region。

    极性逆变换只在此边界出现：clear 输出透光像素，opaque 输出 ownership
    内的不透光像素（field 反转）。行程合并避免逐像素 KLayout 插入。
    """
    expected = problem.ownership_shape
    values = np.asarray(binary_ownership)
    if (values.ndim != 2 or values.dtype != np.dtype(np.bool_)
            or values.shape != expected):
        raise ValueError(
            f"binary_ownership 必须是 {expected} 的布尔数组，"
            f"实际为 {values.shape} / {values.dtype}")
    lit = values if problem.polarity is MaskPolarity.CLEAR else ~values
    pixel_dbu = int(problem.macro.pixel_dbu)
    origin = problem.macro.ownership_box
    region = kdb.Region()
    # 行游程：padded 差分给出 F→T/T→F 边界，偶数为起点奇数为终点；
    # 每段一个 Box，循环次数 = 游程数而不是像素数。
    for row_index in range(expected[0]):
        row = lit[row_index]
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(
            np.concatenate(([False], row, [False])).astype(np.int8)))
        bottom = origin.bottom + row_index * pixel_dbu
        for start, end in zip(edges[0::2], edges[1::2]):
            region.insert(kdb.Box(
                origin.left + int(start) * pixel_dbu, bottom,
                origin.left + int(end) * pixel_dbu, bottom + pixel_dbu))
    # 每个 macro 恰一次合并：垂直相邻游程融合成最小矩形集合，
    # 消除表示层碎片；坐标天然限制在 ownership 像素格内即已裁剪。
    region.merge()
    return region
