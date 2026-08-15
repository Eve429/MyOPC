"""提供预检与生产切分共用的纯数组边段数量公式。"""

from __future__ import annotations

import numpy as np


def count_edge_fragments(lengths: np.ndarray, corner_dbu: float,
                         maximum_dbu: float) -> np.ndarray:
    """按角部短段和均衡中段策略计算每条数学边的切分数量。"""
    # 此函数只执行无额外副作用的 O(edge) 向量运算，既供物化前估算，也供真实
    # SegmentBatch 建造；两条路径共享同一公式，防止安全估算与实际分配逐渐漂移。
    counts = np.ceil(lengths / maximum_dbu).astype(np.int64)
    long_edges = lengths > 2.0 * maximum_dbu
    counts[long_edges] = 2 + np.ceil(
        (lengths[long_edges] - 2.0 * corner_dbu) / maximum_dbu).astype(np.int64)
    return counts
