"""在版图输出前执行考虑 ownership 的 Patch 规范化。"""

from __future__ import annotations

from collections.abc import Iterator

import klayout.db as kdb

from layout.types import LayerSpec

from .errors import PatchConflictError
from .types import GeometryPatch


class PatchSet:
    """按确定顺序收集 core ownership 不重叠的 Patch。"""

    def __init__(self) -> None:
        """创建空 Patch 集合。"""
        self._patches: list[GeometryPatch] = []
        self._ids: set[str] = set()

    def add(self, patch: GeometryPatch) -> GeometryPatch:
        """按 ownership 裁剪，并拒绝重复 ID 或同层 core 重叠。"""
        if patch.patch_id in self._ids:
            raise PatchConflictError(f"duplicate patch_id: {patch.patch_id}")
        for existing in self._patches:
            if existing.layer == patch.layer and existing.ownership_box.overlaps(patch.ownership_box):
                raise PatchConflictError(
                    f"ownership overlap on {patch.layer.layer}/{patch.layer.datatype}: "
                    f"{existing.patch_id} vs {patch.patch_id}")
        # 关键正确性边界：输入 Region 可以包含跨越多个 core 的完整图形，但当前 Patch
        # 只能拥有 ownership_box 内的部分。此处先做精确相交，保证相邻 core 对同一图形
        # 的两部分既不丢失，也不会产生正面积重复；仅共享边界不构成 ownership 冲突。
        clipped = patch.region & kdb.Region(patch.ownership_box.to_native())
        normalized = GeometryPatch(patch.patch_id, patch.layer, clipped, patch.ownership_box)
        self._patches.append(normalized)
        self._ids.add(normalized.patch_id)
        self._patches.sort(key=lambda item: (item.layer, item.ownership_box, item.patch_id))
        return normalized

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        """返回至少含有一个 Patch 的所有 Layer。"""
        return tuple(sorted({patch.layer for patch in self._patches}))

    def region(self, layer: LayerSpec) -> kdb.Region:
        """拼接指定 Layer 中已经完成 ownership 裁剪的原始片段。"""
        result = kdb.Region()
        for patch in self._patches:
            if patch.layer == layer:
                result += patch.region
        return result

    def __len__(self) -> int:
        """返回 Patch 数量，其中包括显式记录的空 ownership 结果。"""
        return len(self._patches)

    def __iter__(self) -> Iterator[GeometryPatch]:
        """按照稳定的 Layer、矩形和 ID 顺序迭代。"""
        return iter(self._patches)
