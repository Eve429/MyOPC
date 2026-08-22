"""层级 KLayout 数据库的单次加载、只读生命周期管理。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import klayout.db as kdb

from .errors import (
    AmbiguousTopCellError,
    CellNotFoundError,
    ClosedLayoutError,
    LayerNotFoundError,
    LayoutOpenError,
)
from .query import ShapeQuery
from .source import read_glp, read_layout
from .types import DbuBox, LayerSpec


class LayoutDB:
    """在一次 OPC 任务生命周期内唯一持有只读原生版图。"""

    def __init__(self, layout: kdb.Layout, source_path: Path, top_cell: kdb.Cell) -> None:
        """根据已经解析完成的原生版图初始化对象。"""
        self._layout: kdb.Layout | None = layout
        self._source_path = source_path
        self._top_cell = top_cell.name
        self._layer_indexes: dict[LayerSpec, int] = {}
        for index in layout.layer_indexes():
            info = layout.get_info(index)
            self._layer_indexes[LayerSpec(info.layer, info.datatype)] = index

    @classmethod
    def open(
        cls,
        path: str | Path,
        top_cell: str | None = None,
        glp_layer_map: Mapping[str, LayerSpec | tuple[int, int]] | None = None,
    ) -> Self:
        """只解析一次 GDS/OASIS/GLP，并以确定规则选择顶层 Cell。"""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise LayoutOpenError(f"layout file does not exist: {source}")
        normalized = {
            name: value if isinstance(value, LayerSpec) else LayerSpec(*value)
            for name, value in (glp_layer_map or {}).items()
        }
        # 格式分派是 open 的职责：source 层只提供单一职责读取器，GLP 的
        # 符号层映射也只在 GLP 分支消费，非 GLP 输入携带映射即参数误用。
        if source.suffix.lower() == ".glp":
            layout = read_glp(source, normalized)
        else:
            if normalized:
                raise LayoutOpenError("glp_layer_map 只能用于 .glp 输入")
            layout = read_layout(source)
        tops = list(layout.top_cells())
        if top_cell is None:
            if len(tops) != 1:
                names = ", ".join(sorted(cell.name for cell in tops)) or "<none>"
                raise AmbiguousTopCellError(f"select top_cell explicitly; candidates: {names}")
            selected = tops[0]
        else:
            selected = layout.cell(top_cell)
            if selected is None:
                raise CellNotFoundError(f"cell not found: {top_cell}")
        return cls(layout, source, selected)

    @property
    def source_path(self) -> Path:
        """返回规范化后的源文件路径。"""
        return self._source_path

    @property
    def dbu_um(self) -> float:
        """返回每个整数 DBU 对应的微米值，仅供配置换算和报告使用。"""
        return self._native_layout.dbu

    @property
    def top_cell_name(self) -> str:
        """返回已选顶层 Cell 的名称；数据库关闭后调用即失败。"""
        _ = self._native_layout
        return self._top_cell

    @property
    def _native_layout(self) -> kdb.Layout:
        """仅向同级实现模块暴露底层对象，不作为公共算法接口。"""
        layout = self._layout
        if layout is None:
            raise ClosedLayoutError("LayoutDB is closed")
        return layout

    def _native_cell(self, cell: str) -> kdb.Cell:
        """按名称解析原生 Cell；名称查找失败是唯一的失效模式。"""
        native = self._native_layout.cell(cell)
        if native is None:
            raise CellNotFoundError(f"cell not found: {cell}")
        return native

    def _native_layer_index(self, layer: LayerSpec) -> int:
        """解析外部 layer/datatype，且不会因查询而创建新的空层。"""
        try:
            return self._layer_indexes[layer]
        except KeyError as exc:
            raise LayerNotFoundError(f"layer not found: {layer.layer}/{layer.datatype}") from exc

    def layers(self) -> tuple[LayerSpec, ...]:
        """按确定顺序列出所有已有 Layer。"""
        _ = self._native_layout
        return tuple(sorted(self._layer_indexes))

    def bbox(self, cell: str | None = None) -> DbuBox | None:
        """返回 Cell 的层级包围盒；空 Cell 返回 None。"""
        native = self._native_cell(cell or self._top_cell)
        box = native.bbox()
        return None if box.empty() else DbuBox.from_native(box)

    def layer_bbox(self, layer: LayerSpec, cell: str | None = None) -> DbuBox | None:
        """返回 Cell 内指定 Layer 的层级包围盒；该层在此子树无图形时返回 None。"""
        native = self._native_cell(cell or self._top_cell)
        # bbox_per_layer 在 KLayout 原生层完成层级展开与逐层过滤，等价于对
        # recursive_polygon_shapes 全量扫描取极值，但不逐 shape 进入解释器，
        # 也不物化 Region；层存在于版图却不在当前 Cell 子树时返回空框。
        box = native.bbox_per_layer(self._native_layer_index(layer))
        return None if box.empty() else DbuBox.from_native(box)

    def cell_hierarchy(self) -> dict[str, tuple[str, ...]]:
        """返回版图全部 Cell 到其直接子 Cell 名称的只读邻接表。"""
        layout = self._native_layout
        hierarchy: dict[str, tuple[str, ...]] = {}
        # `each_child_cell()` 在 KLayout 原生层完成直接子 Cell 去重：同一父 Cell
        # 中重复的 SREF 与大规模 AREF 都只产生一个关系，不按 occurrence 展开，
        # 因而扫描成本只随 Cell 和直接引用关系增长。叶子 Cell 仍写入空元组，
        # 调用方无需再查询缺失键来区分“叶子”与“未扫描”。
        for cell in layout.each_cell():
            children = (layout.cell(index).name for index in cell.each_child_cell())
            # 子名称排序只作用于当前 Cell 的短邻接表，令测试、日志和人工检查
            # 稳定；不复制图形，也不计算 bbox、实例数量或完整实例路径。
            hierarchy[cell.name] = tuple(sorted(children))
        return hierarchy

    def query(
        self,
        layers: tuple[LayerSpec | tuple[int, int], ...] | list[LayerSpec | tuple[int, int]],
        box: DbuBox,
        cell: str | None = None,
        preserve_properties: bool = False,
    ) -> ShapeQuery:
        """校验少量元数据后创建惰性的 Cell/Layer/ROI 查询。"""
        # Layer 只在查询入口规范化一次；集合去重后排序，使缓存键、诊断和测试输出
        # 与调用顺序无关。空集合在接触 KLayout 前失败，避免产生语义不明的空查询。
        normalized = tuple(sorted({item if isinstance(item, LayerSpec) else LayerSpec(*item) for item in layers}))
        if not normalized:
            raise ValueError("at least one layer must be requested")
        for layer in normalized:
            self._native_layer_index(layer)
        # Cell 与 Layer 同样在入口完成存在性校验，惰性查询此后只携带名称字符串；
        # None 表示沿用 open 时已经唯一确定的顶层 Cell。
        selected = self._top_cell if cell is None else cell
        self._native_cell(selected)
        return ShapeQuery(self, selected, normalized, box, preserve_properties)

    def recursive_polygon_shapes(
        self, layer: LayerSpec, box: DbuBox, cell: str | None = None
    ) -> kdb.RecursiveShapeIterator:
        """创建只读层级 Polygon 类图形迭代器，供物化前容量扫描使用。"""
        # 迭代器仍借用当前 LayoutDB 的原生数据库，调用方必须在数据库关闭前
        # 完成遍历。公共方法统一解析 Cell/Layer 和 shape flags，上层无需访问
        # `_native_*` 私有对象，也不会因查询不存在的 Layer 创建空层。
        selected = cell or self._top_cell
        iterator = kdb.RecursiveShapeIterator(
            self._native_layout, self._native_cell(selected), self._native_layer_index(layer), box.to_native(), True
        )
        iterator.shape_flags = kdb.Shapes.SBoxes | kdb.Shapes.SPaths | kdb.Shapes.SPolygons
        return iterator

    def close(self) -> None:
        """释放原生版图；已有惰性查询此后会安全失败。"""
        self._layout = None
        self._layer_indexes.clear()

    def __enter__(self) -> Self:
        """支持按 OPC 任务生命周期使用上下文管理器。"""
        _ = self._native_layout
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """离开上下文时释放原生版图内存。"""
        self.close()
