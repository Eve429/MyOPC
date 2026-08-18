"""NumPy 数组形状与内存布局的通用规范化。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_vector(value: object, dtype: np.dtype, name: str) -> NDArray:
    """把输入转换为指定类型的一维连续数组。"""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def as_matrix(value: object, dtype: np.dtype, columns: int, name: str) -> NDArray:
    """把输入转换为指定类型和列数的二维连续数组。"""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    return array


def as_points(value: object, name: str) -> NDArray[np.float64]:
    """把坐标输入规范化为有限的 N×2 连续浮点数组。"""
    array = as_matrix(value, np.dtype(np.float64), 2, name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array
