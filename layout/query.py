"""层级 ROI 惰性查询以及原生 Region 局部物化。"""

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
    """轻量查询描述，仅在调用方明确请求时才物化几何数据。"""

    database: object
    cell: CellRef
    layers: tuple[LayerSpec, ...]
    box: DbuBox
    preserve_properties: bool = False

    def materialize(self, diagnostics: bool = False) -> RegionBatch:
        """通过 KLayout C++ 迭代器按层物化可转为 Polygon 的图形。"""
        db = self.database
        db._assert_open()
        layout, native_cell = db._native_layout, db._native_cell(self.cell)
        native_box = self.box.to_native()
        regions: dict[LayerSpec, kdb.Region] = {}
        diagnostic_map: dict[LayerSpec, LayerShapeStats] = {}
        started = perf_counter()
        for layer in self.layers:
            index = db._native_layer_index(layer)
            # 关键性能路径：递归迭代、实例变换和 Region 构造都在 KLayout C++ 内完成。
            # Python 每层只发起一次批量调用，不逐 Shape 读取坐标，也不展开完整层级。
            iterator = kdb.RecursiveShapeIterator(layout, native_cell, index, native_box, True)
            # 必须在原生迭代器侧限制图形类型。未过滤的 ROI 迭代器可能让 Text 进入
            # Region.count()，但 Region 的 Polygon 遍历又会忽略 Text，造成计数不一致。
            # 在这里过滤既保证语义一致，也避免为过滤类型增加 Python 逐图形循环。
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
        """统计与 ROI 接触的图形，使零面积 Text 点也能出现在诊断结果中。"""
        polygon_like = text = edge = other = 0
        iterator = kdb.RecursiveShapeIterator(layout, cell, layer_index, box, False)
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
