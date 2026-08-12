"""层级 KLayout 数据库的单次加载、只读生命周期管理。"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Self

import klayout.db as kdb

from .errors import (
    AmbiguousTopCellError,
    CellNotFoundError,
    ClosedLayoutError,
    LayerNotFoundError,
    LayoutOpenError,
)
from .hierarchy import HierarchySummary, build_hierarchy_summary
from .query import ShapeQuery
from .source import read_layout
from .types import CellRef, DbuBox, LayerSpec


class LayoutDB:
    """在一次 OPC 任务生命周期内唯一持有只读原生版图。"""

    def __init__(self, layout: kdb.Layout, source_path: Path, top_cell: kdb.Cell) -> None:
        """根据已经解析完成的原生版图初始化对象。"""
        self._layout: kdb.Layout | None = layout
        self._source_path = source_path
        self._top_cell = CellRef(top_cell.name, top_cell.cell_index())
        self._layer_indexes: dict[LayerSpec, int] = {}
        for index in layout.layer_indexes():
            info = layout.get_info(index)
            self._layer_indexes[LayerSpec(info.layer, info.datatype)] = index

    @classmethod
    def open(cls, path: str | Path, top_cell: str | None = None,
             glp_layer_map: Mapping[str, LayerSpec | tuple[int, int]] | None = None) -> Self:
        """只解析一次 GDS/OASIS/GLP，并以确定规则选择顶层 Cell。"""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise LayoutOpenError(f"layout file does not exist: {source}")
        normalized = {name: value if isinstance(value, LayerSpec) else LayerSpec(*value)
                      for name, value in (glp_layer_map or {}).items()}
        layout = read_layout(source, normalized)
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
    def top_cell(self) -> CellRef:
        """返回已经选择的顶层 Cell 引用。"""
        _ = self._native_layout
        return self._top_cell

    @property
    def _native_layout(self) -> kdb.Layout:
        """仅向同级实现模块暴露底层对象，不作为公共算法接口。"""
        layout = self._layout
        if layout is None:
            raise ClosedLayoutError("LayoutDB is closed")
        return layout

    def _native_cell(self, cell: CellRef) -> kdb.Cell:
        """把已经校验的 CellRef 解析为原生 Cell 对象。"""
        native = self._native_layout.cell(cell.index)
        if native is None or native.name != cell.name:
            raise CellNotFoundError(f"stale or invalid cell reference: {cell.name}")
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

    def cell(self, name: str) -> CellRef:
        """按照精确名称解析 Cell。"""
        native = self._native_layout.cell(name)
        if native is None:
            raise CellNotFoundError(f"cell not found: {name}")
        return CellRef(native.name, native.cell_index())

    def bbox(self, cell: CellRef | None = None) -> DbuBox | None:
        """返回 Cell 的层级包围盒；空 Cell 返回 None。"""
        native = self._native_cell(cell or self._top_cell)
        box = native.bbox()
        return None if box.empty() else DbuBox.from_native(box)

    def hierarchy_summary(self) -> HierarchySummary:
        """返回只读层级元数据，不物化任何图形。"""
        return build_hierarchy_summary(self)

    def query(self, layers: tuple[LayerSpec | tuple[int, int], ...] | list[LayerSpec | tuple[int, int]],
              box: DbuBox, cell: CellRef | str | None = None,
              preserve_properties: bool = False) -> ShapeQuery:
        """校验少量元数据后创建惰性的 Cell/Layer/ROI 查询。"""
        # Layer 只在查询入口规范化一次；集合去重后排序，使缓存键、诊断和测试输出
        # 与调用顺序无关。空集合在接触 KLayout 前失败，避免产生语义不明的空查询。
        normalized = tuple(sorted({item if isinstance(item, LayerSpec) else LayerSpec(*item)
                                   for item in layers}))
        if not normalized:
            raise ValueError("at least one layer must be requested")
        for layer in normalized:
            self._native_layer_index(layer)
        selected = self._top_cell if cell is None else self.cell(cell) if isinstance(cell, str) else cell
        self._native_cell(selected)
        return ShapeQuery(self, selected, normalized, box, preserve_properties)

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
