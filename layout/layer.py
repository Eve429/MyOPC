"""图层规范化工具；该模块只处理元数据，不读取任何图形。"""

from __future__ import annotations

from collections.abc import Iterable

from .types import LayerSpec


def normalize_layers(layers: Iterable[LayerSpec | tuple[int, int]]) -> tuple[LayerSpec, ...]:
    """规范化、去重并排序外部传入的 Layer 描述。"""
    normalized = {item if isinstance(item, LayerSpec) else LayerSpec(*item) for item in layers}
    if not normalized:
        raise ValueError("at least one layer must be requested")
    return tuple(sorted(normalized))
