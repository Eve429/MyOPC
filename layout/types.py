"""Small immutable contracts shared across layout queries and geometry batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import klayout.db as kdb


@dataclass(frozen=True, slots=True, order=True)
class DbuBox:
    """Axis-aligned area in integer database units (DBU)."""

    left: int
    bottom: int
    right: int
    top: int

    def __post_init__(self) -> None:
        """Normalize integral inputs and reject empty/inverted areas."""
        for name in ("left", "bottom", "right", "top"):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer DBU coordinate")
            object.__setattr__(self, name, int(value))
        if self.left >= self.right or self.bottom >= self.top:
            raise ValueError("DbuBox must have positive width and height")

    @property
    def width(self) -> int:
        """Return box width in DBU."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Return box height in DBU."""
        return self.top - self.bottom

    @property
    def area(self) -> int:
        """Return box area in squared DBU."""
        return self.width * self.height

    def expanded(self, margin: int) -> DbuBox:
        """Return a box expanded equally on all four sides."""
        if not isinstance(margin, Integral) or margin < 0:
            raise ValueError("margin must be a non-negative integer DBU value")
        margin = int(margin)
        return DbuBox(self.left - margin, self.bottom - margin,
                      self.right + margin, self.top + margin)

    def intersection(self, other: DbuBox) -> DbuBox | None:
        """Return the positive-area intersection, or None for disjoint/touching boxes."""
        left, bottom = max(self.left, other.left), max(self.bottom, other.bottom)
        right, top = min(self.right, other.right), min(self.top, other.top)
        return None if left >= right or bottom >= top else DbuBox(left, bottom, right, top)

    def overlaps(self, other: DbuBox) -> bool:
        """Return True only when the boxes overlap by positive area."""
        return self.intersection(other) is not None

    def to_native(self) -> kdb.Box:
        """Convert once at the KLayout batch boundary."""
        return kdb.Box(self.left, self.bottom, self.right, self.top)

    @classmethod
    def from_native(cls, box: kdb.Box) -> DbuBox:
        """Build a public DBU box from a KLayout box."""
        if box.empty():
            raise ValueError("an empty native box cannot become DbuBox")
        return cls(box.left, box.bottom, box.right, box.top)


@dataclass(frozen=True, slots=True, order=True)
class LayerSpec:
    """External GDS/OASIS layer and datatype identifier."""

    layer: int
    datatype: int = 0

    def __post_init__(self) -> None:
        """Reject invalid identifiers before they reach KLayout."""
        if not isinstance(self.layer, Integral) or not isinstance(self.datatype, Integral):
            raise TypeError("layer and datatype must be integers")
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "datatype", int(self.datatype))
        if self.layer < 0 or self.datatype < 0:
            raise ValueError("layer and datatype must be non-negative")


@dataclass(frozen=True, slots=True)
class CellRef:
    """Stable cell reference without exposing a mutable native Cell object."""

    name: str
    index: int


@dataclass(frozen=True, slots=True)
class LayerShapeStats:
    """Optional diagnostic counts for shapes returned by an ROI iterator."""

    polygon_like: int = 0
    text: int = 0
    edge: int = 0
    other: int = 0


@dataclass(frozen=True, slots=True)
class MaterializationStats:
    """Optional per-layer diagnostics plus native materialization duration."""

    elapsed_seconds: float
    shapes: Mapping[LayerSpec, LayerShapeStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the mapping so callers cannot desynchronize diagnostics."""
        object.__setattr__(self, "shapes", MappingProxyType(dict(self.shapes)))


@dataclass(frozen=True, slots=True)
class RegionBatch:
    """Tile-local, layer-keyed native regions in a common top-cell coordinate system."""

    regions: Mapping[LayerSpec, kdb.Region]
    query_box: DbuBox
    cell: CellRef
    stats: MaterializationStats | None = None
    backend: str = "klayout"

    def __post_init__(self) -> None:
        """Copy only the small mapping; native Region payloads stay in C++."""
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))
        if self.backend != "klayout":
            raise ValueError(f"unsupported region backend: {self.backend}")

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        """Return deterministic layer order."""
        return tuple(sorted(self.regions))

    def region(self, layer: LayerSpec) -> kdb.Region:
        """Return one native region without expanding its polygons in Python."""
        try:
            return self.regions[layer]
        except KeyError as exc:
            raise KeyError(f"batch does not contain layer {layer.layer}/{layer.datatype}") from exc

    def counts(self) -> Mapping[LayerSpec, int]:
        """Return flat polygon counts using native Region metadata."""
        return MappingProxyType({layer: region.count() for layer, region in self.regions.items()})
