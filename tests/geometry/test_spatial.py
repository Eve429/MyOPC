"""局部区域均匀网格边索引正确性测试。"""

import numpy as np
import pytest

from geometry import EdgeBatch, UniformGridIndex
from layout import DbuBox, LayerSpec


def _edges() -> EdgeBatch:
    """返回三条分离的边，其中包含一条超长候选边。"""
    starts = np.array([[0, 0], [100, 100], [-1000, 50]])
    ends = np.array([[20, 0], [120, 100], [1000, 50]])
    ids = np.arange(3)
    return EdgeBatch(LayerSpec(1, 0), starts, ends, ids, ids, np.zeros(3, dtype=bool))


def test_grid_query_filters_candidates_and_keeps_oversized_edges() -> None:
    """网格倒排表与超长边集合组合后应匹配精确 bbox 交集。"""
    index = UniformGridIndex(_edges(), cell_size_dbu=10, max_cells_per_edge=20)
    assert index.oversized_count == 1
    assert index.query_box(DbuBox(-5, -5, 25, 5)).tolist() == [0]
    assert index.query_box(DbuBox(95, 45, 105, 105)).tolist() == [1, 2]
    assert index.query_radius(110, 100, 5).tolist() == [1]


def test_grid_validation_and_zero_radius() -> None:
    """非法网格参数应尽早失败，零半径查询仍需可用。"""
    edges = _edges()
    with pytest.raises(ValueError):
        UniformGridIndex(edges, 0)
    index = UniformGridIndex(edges, 20)
    assert index.query_radius(0, 0, 0).tolist() == [0]
    with pytest.raises(ValueError):
        index.query_radius(0, 0, -1)
