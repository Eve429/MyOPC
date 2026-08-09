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
        clip_region = kdb.Region(native_box)
        regions: dict[LayerSpec, kdb.Region] = {}
        diagnostic_map: dict[LayerSpec, LayerShapeStats] = {}
        started = perf_counter() if diagnostics else 0.0
        for layer in self.layers:
            index = db._native_layer_index(layer)
            # 关键性能路径：递归迭代、实例变换和 Region 构造都在 KLayout C++ 内完成。
            # 解释器每层只发起一次批量调用，不逐个图形读取坐标，也不展开完整层级。
            iterator = kdb.RecursiveShapeIterator(layout, native_cell, index, native_box, True)
            # 必须在原生迭代器侧限制图形类型。未过滤的 ROI 迭代器可能让 Text 进入
            # 原生区域计数，但其多边形遍历又会忽略文本，造成计数不一致。
            # 在这里过滤既保证语义一致，也避免为过滤类型增加 Python 逐图形循环。
            iterator.shape_flags = kdb.Shapes.SBoxes | kdb.Shapes.SPaths | kdb.Shapes.SPolygons
            if self.preserve_properties:
                # 这里只启用属性导入，不叠加 SProperties。后者在 KLayout 中表示
                # “只选择带属性的图形”，会错误丢弃同一 ROI 内没有属性的有效几何。
                iterator.enable_properties()
            # RecursiveShapeIterator 的 ROI 只筛选相交候选，跨边界图形仍保留完整
            # Polygon。这里每层一次原生 Region 相交，统一保证所有消费者拿到的
            # 都是精确 planner ROI；避免 CLI、MB-OPC 等入口各自决定是否再裁剪。
            region = kdb.Region(iterator)
            # KLayout 的普通 `&` 会主动丢弃 Polygon 属性；属性模式必须显式使用
            # NoPropertyConstraint，含义是几何仍与裁剪框求交，但结果继承左侧原图
            # 属性。两条路径都在 C++ 内批量执行，不增加逐 Polygon Python 循环。
            regions[layer] = (region.and_(clip_region, kdb.Region.NoPropertyConstraint)
                              if self.preserve_properties else region & clip_region)
            if diagnostics:
                diagnostic_map[layer] = self._collect_shape_stats(layout, native_cell, index, native_box)
        stats = MaterializationStats(perf_counter() - started, diagnostic_map) if diagnostics else None
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
