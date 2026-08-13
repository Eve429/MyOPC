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
    "CellNotFoundError",
    "CellRef",
    "ClosedLayoutError",
    "DbuBox",
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
