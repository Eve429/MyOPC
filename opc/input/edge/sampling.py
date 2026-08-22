"""按边段中点批量生成与求解器一致的 inner/outer EPE 探针。"""

from __future__ import annotations

import numpy as np


def edge_probe_points(
    starts: object, ends: object, normals: object, distance_dbu: float
) -> tuple[np.ndarray, np.ndarray]:
    """返回每条边段中点沿负/正外法向偏移后的 inner 与 outer 坐标。"""
    starts_array = np.ascontiguousarray(starts, dtype=np.float64)
    ends_array = np.ascontiguousarray(ends, dtype=np.float64)
    normals_array = np.ascontiguousarray(normals, dtype=np.float64)
    if (
        starts_array.ndim != 2
        or starts_array.shape[1] != 2
        or ends_array.shape != starts_array.shape
        or normals_array.shape != starts_array.shape
    ):
        raise ValueError("starts, ends and normals must have equal shape (N, 2)")
    distance = float(distance_dbu)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_dbu must be finite and positive")
    # 目标边界在整个优化过程中固定，因此 inner/outer 都围绕参考边段中点定义。
    # 两个 N×2 返回数组正好对应评价器所需输入，不构造 line index、切向比例或
    # 模板对象；诊断预览调用同一函数，从根源上避免显示距离与求解距离分叉。
    midpoints = (starts_array + ends_array) * 0.5
    offsets = normals_array * distance
    return np.ascontiguousarray(midpoints - offsets), np.ascontiguousarray(midpoints + offsets)
