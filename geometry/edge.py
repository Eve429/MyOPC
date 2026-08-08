"""从轮廓环向数学 Polygon 边的向量化转换。"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from layout.types import LayerSpec

from .types import ContourBatch, EdgeBatch


def extract_edges(contours: ContourBatch) -> EdgeBatch:
    """通过数组索引和环首尾闭合生成全部边，避免 Python 逐边循环。"""
    count = len(contours.vertices)
    if count == 0:
        empty_points = np.empty((0, 2), dtype=np.int64)
        empty_ids = np.empty(0, dtype=np.int64)
        return EdgeBatch(contours.layer, empty_points, empty_points.copy(),
                         empty_ids, empty_ids.copy(), np.empty(0, dtype=np.bool_))
    lengths = np.diff(contours.ring_offsets)
    next_indices = np.arange(count, dtype=np.int64) + 1
    # 关键性能步骤：ContourBatch 不重复保存每个环的首点。这里一次性构造 next_indices，
    # 再把每个环的末点指向该环首点，从而用 NumPy 批量生成闭合边，避免逐边 append。
    next_indices[contours.ring_offsets[1:] - 1] = contours.ring_offsets[:-1]
    ring_ids = np.repeat(np.arange(contours.ring_count, dtype=np.int64), lengths)
    return EdgeBatch(
        contours.layer,
        contours.vertices.copy(),
        contours.vertices[next_indices],
        ring_ids,
        contours.ring_polygon_ids[ring_ids],
        contours.ring_is_hole[ring_ids],
    )


def extract_edge_batches(contours: Mapping[LayerSpec, ContourBatch]) -> Mapping[LayerSpec, EdgeBatch]:
    """对 Layer 映射逐层执行一次向量化边提取。"""
    return MappingProxyType({layer: extract_edges(batch) for layer, batch in contours.items()})
