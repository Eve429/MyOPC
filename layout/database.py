"""Immutable, single-load owner of a hierarchical KLayout database."""

from __future__ import annotations

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
from .hierarchy import HierarchySummary, build_hierarchy_summary
from .layer import normalize_layers
from .query import ShapeQuery
from .types import CellRef, DbuBox, LayerSpec


class LayoutDB:
    """Own one read-only native layout for the lifetime of an OPC job."""

    def __init__(self, layout: kdb.Layout, source_path: Path, top_cell: kdb.Cell) -> None:
        """Initialize from an already parsed native layout."""
        self._layout: kdb.Layout | None = layout
        self._source_path = source_path
        self._top_cell = CellRef(top_cell.name, top_cell.cell_index())
        self._layer_indexes = {
            LayerSpec(layout.get_info(index).layer, layout.get_info(index).datatype): index
            for index in layout.layer_indexes()
        }

    @classmethod
    def open(cls, path: str | Path, top_cell: str | None = None) -> Self:
        """Parse GDS/OASIS once and select a deterministic top cell."""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise LayoutOpenError(f"layout file does not exist: {source}")
        layout = kdb.Layout()
        try:
            layout.read(str(source))
        except Exception as exc:
            raise LayoutOpenError(f"failed to read layout {source}: {exc}") from exc
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
        """Return the normalized source path."""
        return self._source_path

    @property
    def dbu_um(self) -> float:
        """Return micrometers per integer DBU for reporting/config conversion only."""
        self._assert_open()
        return self._native_layout.dbu

    @property
    def top_cell(self) -> CellRef:
        """Return the selected top cell reference."""
        self._assert_open()
        return self._top_cell

    @property
    def _native_layout(self) -> kdb.Layout:
        """Expose the backend object only to sibling implementation modules."""
        self._assert_open()
        assert self._layout is not None
        return self._layout

    def _assert_open(self) -> None:
        """Fail fast when a stale database or query is used."""
        if self._layout is None:
            raise ClosedLayoutError("LayoutDB is closed")

    def _native_cell(self, cell: CellRef) -> kdb.Cell:
        """Resolve a checked CellRef to its native object."""
        native = self._native_layout.cell(cell.index)
        if native is None or native.name != cell.name:
            raise CellNotFoundError(f"stale or invalid cell reference: {cell.name}")
        return native

    def _native_layer_index(self, layer: LayerSpec) -> int:
        """Resolve external layer/datatype without creating a new empty layer."""
        try:
            return self._layer_indexes[layer]
        except KeyError as exc:
            raise LayerNotFoundError(f"layer not found: {layer.layer}/{layer.datatype}") from exc

    def layers(self) -> tuple[LayerSpec, ...]:
        """List all existing layers in deterministic order."""
        self._assert_open()
        return tuple(sorted(self._layer_indexes))

    def cell(self, name: str) -> CellRef:
        """Resolve a cell by exact name."""
        native = self._native_layout.cell(name)
        if native is None:
            raise CellNotFoundError(f"cell not found: {name}")
        return CellRef(native.name, native.cell_index())

    def bbox(self, cell: CellRef | None = None) -> DbuBox | None:
        """Return a cell's hierarchical bounding box, or None when it is empty."""
        native = self._native_cell(cell or self._top_cell)
        box = native.bbox()
        return None if box.empty() else DbuBox.from_native(box)

    def hierarchy_summary(self) -> HierarchySummary:
        """Return read-only hierarchy metadata without materializing shapes."""
        return build_hierarchy_summary(self)

    def query(self, layers: tuple[LayerSpec | tuple[int, int], ...] | list[LayerSpec | tuple[int, int]],
              box: DbuBox, cell: CellRef | str | None = None,
              preserve_properties: bool = False) -> ShapeQuery:
        """Create a lazy cell/layer/ROI query after validating small metadata."""
        self._assert_open()
        normalized = normalize_layers(layers)
        for layer in normalized:
            self._native_layer_index(layer)
        selected = self._top_cell if cell is None else self.cell(cell) if isinstance(cell, str) else cell
        self._native_cell(selected)
        return ShapeQuery(self, selected, normalized, box, preserve_properties)

    def close(self) -> None:
        """Release the native layout; existing lazy queries then fail safely."""
        self._layout = None
        self._layer_indexes.clear()

    def __enter__(self) -> Self:
        """Support job-scoped context management."""
        self._assert_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release native memory when leaving a context."""
        self.close()
