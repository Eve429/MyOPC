"""Public high-performance hierarchical layout API."""

from .database import LayoutDB
from .errors import (
    AmbiguousTopCellError,
    CellNotFoundError,
    ClosedLayoutError,
    LayerNotFoundError,
    LayoutError,
    LayoutOpenError,
)
from .hierarchy import CellInfo, HierarchySummary
from .query import ShapeQuery
from .types import (
    CellRef,
    DbuBox,
    LayerShapeStats,
    LayerSpec,
    MaterializationStats,
    RegionBatch,
)

__all__ = [
    "AmbiguousTopCellError",
    "CellInfo",
    "CellNotFoundError",
    "CellRef",
    "ClosedLayoutError",
    "DbuBox",
    "HierarchySummary",
    "LayerNotFoundError",
    "LayerShapeStats",
    "LayerSpec",
    "LayoutDB",
    "LayoutError",
    "LayoutOpenError",
    "MaterializationStats",
    "RegionBatch",
    "ShapeQuery",
]
