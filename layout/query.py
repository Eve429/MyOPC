"""Lazy hierarchical ROI queries and native Region materialization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import klayout.db as kdb

from .types import (
    CellRef,
    DbuBox,
    LayerShapeStats,
    LayerSpec,
    MaterializationStats,
    RegionBatch,
)


@dataclass(frozen=True, slots=True)
class ShapeQuery:
    """A cheap query descriptor that materializes only when explicitly requested."""

    database: object
    cell: CellRef
    layers: tuple[LayerSpec, ...]
    box: DbuBox
    preserve_properties: bool = False

    def materialize(self, diagnostics: bool = False) -> RegionBatch:
        """Materialize polygon-like shapes per layer through KLayout's C++ iterator."""
        db = self.database
        db._assert_open()
        layout, native_cell = db._native_layout, db._native_cell(self.cell)
        native_box = self.box.to_native()
        regions: dict[LayerSpec, kdb.Region] = {}
        diagnostic_map: dict[LayerSpec, LayerShapeStats] = {}
        started = perf_counter()
        for layer in self.layers:
            index = db._native_layer_index(layer)
            # Region consumes the recursive iterator in C++ and applies hierarchy transforms.
            iterator = kdb.RecursiveShapeIterator(layout, native_cell, index, native_box, True)
            # Restrict classes natively: an unfiltered ROI iterator can leak text into
            # Region.count() even though Region polygon iteration ignores the text object.
            iterator.shape_flags = kdb.Shapes.SBoxes | kdb.Shapes.SPaths | kdb.Shapes.SPolygons
            if self.preserve_properties:
                iterator.shape_flags |= kdb.Shapes.SProperties
                iterator.enable_properties()
            regions[layer] = kdb.Region(iterator)
            if diagnostics:
                diagnostic_map[layer] = self._collect_shape_stats(layout, native_cell, index, native_box)
        stats = MaterializationStats(perf_counter() - started, diagnostic_map)
        return RegionBatch(regions, self.box, self.cell, stats)

    @staticmethod
    def _collect_shape_stats(layout: kdb.Layout, cell: kdb.Cell, layer_index: int,
                             box: kdb.Box) -> LayerShapeStats:
        """Run the deliberately optional Python diagnostic pass."""
        polygon_like = text = edge = other = 0
        iterator = kdb.RecursiveShapeIterator(layout, cell, layer_index, box, True)
        for item in iterator:
            shape = item.shape()
            if shape.is_box() or shape.is_polygon() or shape.is_path():
                polygon_like += 1
            elif shape.is_text():
                text += 1
            elif shape.is_edge():
                edge += 1
            else:
                other += 1
        return LayerShapeStats(polygon_like, text, edge, other)
