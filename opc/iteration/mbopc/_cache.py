"""simple 与 gradient 共享的固定 target 画布 uint8 LRU 缓存。"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray


class TargetCanvasCache:
    """按显式字节上限保存跨状态复用的只读 uint8 target canvas。"""

    def __init__(self, max_bytes: int) -> None:
        """保存字节上限并建立空的 LRU 容器。"""
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes 必须是非负整数")
        self._max_bytes = max_bytes  # 0 表示完全禁用缓存
        # (macro_id, core_index) → uint8 canvas，最新在尾
        self._entries: OrderedDict[tuple[str, int], NDArray[np.uint8]] = OrderedDict()
        self._used_bytes = 0

    def get(self, macro_id: str, core_index: int) -> NDArray[np.uint8] | None:
        """命中时返回缓存 canvas 并把它标记为最新，未命中返回 None。"""
        key = (macro_id, core_index)  # key 必须包含 macro ID，防止跨 macro 误用
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)  # LRU：刚访问的移到最新端
        return entry  # 只读语义由调用方遵守

    def put(self, macro_id: str, core_index: int, value: NDArray[np.uint8]) -> None:
        """写入或替换一个 target canvas，超上限时从最旧端驱逐。"""
        if self._max_bytes == 0:  # 0 上限禁用缓存
            return
        nbytes = int(value.nbytes)
        if nbytes > self._max_bytes:  # 单项超上限不入缓存：宁可每次重算也不驱逐整缓存
            return
        key = (macro_id, core_index)
        old = self._entries.pop(key, None)  # 替换语义：先移除旧值
        if old is not None:
            self._used_bytes -= int(old.nbytes)
        while self._used_bytes + nbytes > self._max_bytes:
            _, evicted = self._entries.popitem(last=False)  # 驱逐最旧端
            self._used_bytes -= int(evicted.nbytes)
        self._entries[key] = value  # 存入最新端
        self._used_bytes += nbytes
