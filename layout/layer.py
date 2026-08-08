"""Layer resolution helpers kept separate from geometry extraction."""

from __future__ import annotations

from collections.abc import Iterable

from .types import LayerSpec


def normalize_layers(layers: Iterable[LayerSpec | tuple[int, int]]) -> tuple[LayerSpec, ...]:
    """Normalize, deduplicate, and sort external layer specifications."""
    normalized = {item if isinstance(item, LayerSpec) else LayerSpec(*item) for item in layers}
    if not normalized:
        raise ValueError("at least one layer must be requested")
    return tuple(sorted(normalized))
