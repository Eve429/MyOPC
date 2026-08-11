"""只读层级检查；正式计算路径不会在这里执行版图扁平化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import CellRef, DbuBox

if TYPE_CHECKING:
    from .database import LayoutDB


@dataclass(frozen=True, slots=True)
class CellInfo:
    """供诊断和后续 planner 使用的紧凑 Cell 元数据。"""

    ref: CellRef
    bbox: DbuBox | None
    child_cells: tuple[CellRef, ...]
    instance_records: int
    logical_instances: int


@dataclass(frozen=True, slots=True)
class HierarchySummary:
    """不复制任何图形的版图层级快照。"""

    top_cells: tuple[CellRef, ...]
    cells: tuple[CellInfo, ...]


def build_hierarchy_summary(db: LayoutDB) -> HierarchySummary:
    """读取 Cell 与实例记录，同时保持底层图形存储完全不动。"""
    layout = db._native_layout
    infos: list[CellInfo] = []
    for cell in layout.each_cell():
        children: dict[int, CellRef] = {}
        records = logical = 0
        for inst in cell.each_inst():
            child = layout.cell(inst.cell_index)
            children[child.cell_index()] = CellRef(child.name, child.cell_index())
            records += 1
            logical += max(1, int(inst.na)) * max(1, int(inst.nb))
        bbox = cell.bbox()
        infos.append(CellInfo(
            ref=CellRef(cell.name, cell.cell_index()),
            bbox=None if bbox.empty() else DbuBox.from_native(bbox),
            child_cells=tuple(children[index] for index in sorted(children)),
            instance_records=records,
            logical_instances=logical,
        ))
    tops = tuple(CellRef(cell.name, cell.cell_index()) for cell in layout.top_cells())
    return HierarchySummary(tops, tuple(infos))
