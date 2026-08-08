"""面向 OPC 上层算法的高性能层级版图公共接口。"""

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
from .writer import PatchWriter

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
    "PatchWriter",
    "RegionBatch",
    "ShapeQuery",
]
