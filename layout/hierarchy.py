"""Read-only hierarchy inspection; production paths never flatten the layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import CellRef, DbuBox

if TYPE_CHECKING:
    from .database import LayoutDB


@dataclass(frozen=True, slots=True)
class CellInfo:
    """Compact cell metadata suitable for diagnostics and planner inspection."""

    ref: CellRef
    bbox: DbuBox | None
    child_cells: tuple[CellRef, ...]
    instance_records: int
    logical_instances: int


@dataclass(frozen=True, slots=True)
class HierarchySummary:
    """Snapshot of layout hierarchy without copying any shapes."""

    top_cells: tuple[CellRef, ...]
    cells: tuple[CellInfo, ...]


def build_hierarchy_summary(db: LayoutDB) -> HierarchySummary:
    """Inspect cells and instance records while leaving shape storage untouched."""
    db._assert_open()
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
