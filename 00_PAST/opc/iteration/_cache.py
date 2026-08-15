"""提供迭代方法共享的有界 NumPy tile 缓存。"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class ArrayTileCache:
    """按数组实际字节数维护最近使用的 tile，容量为零时关闭缓存。"""

    def __init__(self, max_bytes: int) -> None:
        """创建空缓存并拒绝负容量，防止配置错误变成无界常驻内存。"""
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("tile 缓存字节上限必须是非负整数")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.values: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, key: int) -> np.ndarray | None:
        """返回并提升命中项；未命中时返回 None。"""
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: int, value: np.ndarray) -> None:
        """保存数组，并从最旧项开始驱逐到不超过显式字节上限。"""
        if not isinstance(value, np.ndarray):
            raise TypeError("tile 缓存只接受 NumPy 数组")
        if self.max_bytes == 0 or value.nbytes > self.max_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.current_bytes -= old.nbytes
        self.values[key] = value
        self.current_bytes += value.nbytes
        while self.current_bytes > self.max_bytes:
            _, removed = self.values.popitem(last=False)
            self.current_bytes -= removed.nbytes
