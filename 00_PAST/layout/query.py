"""层级 ROI 惰性查询以及原生 Region 局部物化。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import klayout.db as kdb

from .types import (
    CellRef,
    DbuBox,
    LayerShapeStats,
    LayerSpec,
    MaterializationStats,
    RegionBatch,
)

if TYPE_CHECKING:
    from .database import LayoutDB


@dataclass(frozen=True, slots=True)
class ShapeQuery:
    """轻量查询描述，仅在调用方明确请求时才物化几何数据。"""

    database: LayoutDB
    cell: CellRef
    layers: tuple[LayerSpec, ...]
    box: DbuBox
    preserve_properties: bool = False

    def materialize(self, diagnostics: bool = False) -> RegionBatch:
        """物化相交图形并精确裁到查询框，供显示、像素 ROI 和普通查询使用。"""
        return self._materialize(diagnostics, clip=True)

    def materialize_intersecting(self, diagnostics: bool = False) -> RegionBatch:
        """物化与查询框相交的完整图形，供裁剪前提取真实物理边。"""
        return self._materialize(diagnostics, clip=False)

    def _materialize(self, diagnostics: bool, *, clip: bool) -> RegionBatch:
        """在一次原生层级遍历中批量物化，并按调用语义选择是否裁剪。"""
        db = self.database
        layout, native_cell = db._native_layout, db._native_cell(self.cell)
        native_box = self.box.to_native()
        clip_region = kdb.Region(native_box)
        regions: dict[LayerSpec, kdb.Region] = {}
        diagnostic_map: dict[LayerSpec, LayerShapeStats] = {}
        started = perf_counter() if diagnostics else 0.0
        for layer in self.layers:
            index = db._native_layer_index(layer)
            # 关键性能路径：层级筛选、实例变换和 Region 构造均在 KLayout C++
            # 内完成；Python 每层只发起一次批量调用，不逐 occurrence 读取坐标。
            iterator = kdb.RecursiveShapeIterator(layout, native_cell, index, native_box, True)
            # 原生侧只接纳可转换为面积 Polygon 的类型，避免 Text/Edge 进入 Region
            # 计数但不进入轮廓；诊断需要这些类型时另走按需统计，不污染热路径。
            iterator.shape_flags = kdb.Shapes.SBoxes | kdb.Shapes.SPaths | kdb.Shapes.SPolygons
            if self.preserve_properties:
                # enable_properties 只导入属性，不使用 SProperties 过滤器；后者会把
                # 同一 ROI 内无属性的正常几何错误排除。
                iterator.enable_properties()
            # iterator 的 ROI 只筛相交候选，不裁图形；是否精确裁剪由此处唯一参数
            # 决定，普通像素/显示与 macro 真实提边不会维护两套层级遍历逻辑。
            region = kdb.Region(iterator)
            if clip:
                # 普通 `&` 会丢 Polygon 属性；属性模式以 NoPropertyConstraint 求交，
                # 几何仍精确裁到 ROI，结果继承左侧原图属性。
                regions[layer] = (region.and_(clip_region, kdb.Region.NoPropertyConstraint)
                                  if self.preserve_properties else region & clip_region)
            else:
                # 未裁剪 deep Region 借用 LayoutDB，必须在关闭前原生展平。KLayout
                # flatten 会静默丢属性，因此属性模式用 merged 并保持不同属性的
                # shape class；无属性 OPC 热路径继续使用开销更直接的 flatten。
                regions[layer] = (region.merged(False, 0, False)
                                  if self.preserve_properties else region.flatten())
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
