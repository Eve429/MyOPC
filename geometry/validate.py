"""对局部轮廓批次执行不修改输入的结构校验。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layout import LayerSpec

from .contour import ContourBatch


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """一条顺序稳定、便于测试和报告的校验问题。"""

    code: str
    message: str
    layer: LayerSpec
    indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """不修改输入数据的几何校验汇总结果。"""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """未发现任何问题时返回 True。"""
        return not self.issues


def validate_contours(contours: ContourBatch, layer: LayerSpec) -> ValidationReport:
    """在昂贵的 OPC 迭代或原生修复前检测退化轮廓环。"""
    issues: list[ValidationIssue] = []
    for ring_id in range(contours.ring_count):
        start, end = contours.ring_offsets[ring_id : ring_id + 2]
        ring = contours.vertices[start:end]
        following = np.roll(ring, -1, axis=0)
        duplicate_edges = np.flatnonzero(np.all(ring == following, axis=1))
        if len(duplicate_edges):
            issues.append(
                ValidationIssue(
                    "zero_length_edge",
                    f"ring {ring_id} contains zero-length edges",
                    layer,
                    tuple(int(value) for value in duplicate_edges),
                )
            )
        x, y = ring[:, 0].astype(np.float64), ring[:, 1].astype(np.float64)
        signed_area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if signed_area2 == 0.0:
            issues.append(ValidationIssue("zero_area_ring", f"ring {ring_id} has zero signed area", layer, (ring_id,)))
    return ValidationReport(tuple(issues))
