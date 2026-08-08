"""面向高频边邻域查询、可重复使用的 tile-local 均匀网格索引。"""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from layout.types import DbuBox

from .types import EdgeBatch


class UniformGridIndex:
    """一次构建边包围盒索引，并在多轮 OPC 迭代中重复使用。"""

    def __init__(self, edges: EdgeBatch, cell_size_dbu: int, max_cells_per_edge: int = 4096) -> None:
        """构建紧凑网格倒排表，同时单独处理跨越大量网格的超长边。"""
        if not isinstance(cell_size_dbu, Integral) or cell_size_dbu <= 0:
            raise ValueError("cell_size_dbu must be a positive integer")
        if not isinstance(max_cells_per_edge, Integral) or max_cells_per_edge <= 0:
            raise ValueError("max_cells_per_edge must be a positive integer")
        self.edges = edges
        self.cell_size_dbu = int(cell_size_dbu)
        self.max_cells_per_edge = int(max_cells_per_edge)
        self._bboxes = edges.bboxes
        postings: dict[tuple[int, int], list[int]] = defaultdict(list)
        oversized: list[int] = []
        for edge_id, (left, bottom, right, top) in enumerate(self._bboxes):
            x0, y0 = int(left) // self.cell_size_dbu, int(bottom) // self.cell_size_dbu
            x1, y1 = int(right) // self.cell_size_dbu, int(top) // self.cell_size_dbu
            cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
            # 超长边如果写入经过的每个网格，会使索引内存随边长失控；把它们放入
            # 独立候选集，查询时统一做精确 bbox 过滤，以小量计算换取内存上界。
            if cell_count > self.max_cells_per_edge:
                oversized.append(edge_id)
                continue
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    postings[(x, y)].append(edge_id)
        self._postings = {key: np.asarray(value, dtype=np.int64) for key, value in postings.items()}
        self._oversized = np.asarray(oversized, dtype=np.int64)

    @property
    def oversized_count(self) -> int:
        """返回为限制内存增长而未写入网格倒排表的边数量。"""
        return len(self._oversized)

    def query_box(self, box: DbuBox) -> NDArray[np.int64]:
        """返回包围盒与查询区域相交的已排序边 ID。"""
        x0, y0 = box.left // self.cell_size_dbu, box.bottom // self.cell_size_dbu
        x1, y1 = box.right // self.cell_size_dbu, box.top // self.cell_size_dbu
        postings = [self._postings[(x, y)] for x in range(x0, x1 + 1)
                    for y in range(y0, y1 + 1) if (x, y) in self._postings]
        if self.oversized_count:
            postings.append(self._oversized)
        if not postings:
            return np.empty(0, dtype=np.int64)
        candidates = np.unique(np.concatenate(postings))
        bboxes = self._bboxes[candidates]
        keep = ((bboxes[:, 0] <= box.right) & (bboxes[:, 2] >= box.left) &
                (bboxes[:, 1] <= box.top) & (bboxes[:, 3] >= box.bottom))
        return candidates[keep]

    def query_radius(self, x: int, y: int, radius_dbu: int) -> NDArray[np.int64]:
        """返回指定 DBU 点方形半径范围内的包围盒候选边。"""
        if not isinstance(radius_dbu, Integral) or radius_dbu < 0:
            raise ValueError("radius_dbu must be a non-negative integer")
        radius = int(radius_dbu)
        # 数据库单位矩形不允许零面积，因此半径为 0 时用一个 DBU 的最小查询框表达点查询。
        if radius == 0:
            return self.query_box(DbuBox(int(x), int(y), int(x) + 1, int(y) + 1))
        return self.query_box(DbuBox(int(x) - radius, int(y) - radius,
                                     int(x) + radius, int(y) + radius))
