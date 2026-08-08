"""由 KLayout C++ Region 内核实现的批量 Polygon 集合运算。"""

from __future__ import annotations

from collections.abc import Callable

import klayout.db as kdb

from layout.types import DbuBox, LayerSpec, RegionBatch

from .errors import BackendMismatchError, CoordinateSystemError


class GeometryEngine:
    """无状态几何门面，保持原生调用为粗粒度且便于后续替换后端。"""

    backend = "klayout"

    def clip(self, batch: RegionBatch, box: DbuBox) -> RegionBatch:
        """返回 Polygon 与指定 DBU 矩形的精确交集。"""
        self._check_backend(batch)
        clip_region = kdb.Region(box.to_native())
        regions = {layer: region & clip_region for layer, region in batch.regions.items()}
        return RegionBatch(regions, box, batch.cell)

    def combine(self, left: RegionBatch, right: RegionBatch) -> RegionBatch:
        """直接拼接原始 Polygon，不承担布尔 union 的隐式 merge 成本。"""
        return self._binary(left, right, "combine")

    def union(self, left: RegionBatch, right: RegionBatch) -> RegionBatch:
        """按 Layer 返回完成合并的布尔并集。"""
        return self._binary(left, right, "union")

    def intersection(self, left: RegionBatch, right: RegionBatch) -> RegionBatch:
        """对两个批次共同包含的 Layer 计算布尔交集。"""
        return self._binary(left, right, "intersection")

    def difference(self, left: RegionBatch, right: RegionBatch) -> RegionBatch:
        """从左侧各 Layer 中减去右侧同 Layer 的 Polygon。"""
        return self._binary(left, right, "difference")

    def xor(self, left: RegionBatch, right: RegionBatch) -> RegionBatch:
        """按 Layer 返回布尔异或结果。"""
        return self._binary(left, right, "xor")

    def offset(self, batch: RegionBatch, distance_dbu: int, mode: int = 2) -> RegionBatch:
        """使用整数 DBU 距离对各层 Region 进行偏置。"""
        if not isinstance(distance_dbu, int):
            raise TypeError("distance_dbu must be an integer")
        return self._unary(batch, lambda region: region.sized(distance_dbu, mode))

    def merge(self, batch: RegionBatch) -> RegionBatch:
        """显式合并重叠区域；物化过程绝不会隐式调用该操作。"""
        return self._unary(batch, lambda region: region.merged())

    def _unary(self, batch: RegionBatch,
               operation: Callable[[kdb.Region], kdb.Region]) -> RegionBatch:
        """每个 Layer 只调用一次原生操作，禁止逐 Polygon 跨模块调用。"""
        self._check_backend(batch)
        regions = {layer: operation(region) for layer, region in batch.regions.items()}
        return RegionBatch(regions, batch.query_box, batch.cell)

    def _binary(self, left: RegionBatch, right: RegionBatch, operation: str) -> RegionBatch:
        """按确定的 Layer 集合规则执行二元布尔运算。"""
        self._check_compatible(left, right)
        left_layers, right_layers = set(left.regions), set(right.regions)
        if operation == "intersection":
            layers = left_layers & right_layers
        elif operation == "difference":
            layers = left_layers
        else:
            layers = left_layers | right_layers
        regions: dict[LayerSpec, kdb.Region] = {}
        for layer in sorted(layers):
            a, b = left.regions.get(layer), right.regions.get(layer)
            if operation == "intersection":
                regions[layer] = a & b
            elif operation == "difference":
                regions[layer] = a.dup() if b is None else a - b
            elif a is None:
                regions[layer] = b.dup()
            elif b is None:
                regions[layer] = a.dup()
            elif operation == "combine":
                regions[layer] = a + b
            elif operation == "union":
                regions[layer] = a | b
            else:
                regions[layer] = a ^ b
        box = self._result_box(left.query_box, right.query_box, operation)
        return RegionBatch(regions, box, left.cell)

    def _check_backend(self, batch: RegionBatch) -> None:
        """为后续增加后端预留校验，防止批次被意外混用。"""
        if batch.backend != self.backend:
            raise BackendMismatchError(f"expected {self.backend}, got {batch.backend}")

    def _check_compatible(self, left: RegionBatch, right: RegionBatch) -> None:
        """二元运算要求两个批次处于同一 Cell 坐标系。"""
        self._check_backend(left)
        self._check_backend(right)
        if left.cell != right.cell:
            raise CoordinateSystemError("binary batches must use the same cell coordinate system")

    @staticmethod
    def _result_box(left: DbuBox, right: DbuBox, operation: str) -> DbuBox:
        """不遍历 Polygon，仅根据查询框给出保守的结果作用域。"""
        if operation == "difference":
            return left
        if operation == "intersection":
            return left.intersection(right) or left
        return DbuBox(min(left.left, right.left), min(left.bottom, right.bottom),
                      max(left.right, right.right), max(left.top, right.top))
