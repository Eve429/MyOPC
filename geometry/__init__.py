"""与具体 OPC 方法解耦的局部计算几何公共接口。"""

from .contour import contours_to_region, extract_contours
from .edge import extract_edge_batches, extract_edges
from .errors import (
    BackendMismatchError,
    CoordinateSystemError,
    GeometryError,
    PatchConflictError,
)
from .patch import PatchSet
from .region import GeometryEngine
from .spatial import UniformGridIndex
from .types import (
    ContourBatch,
    EdgeBatch,
    GeometryPatch,
    ValidationIssue,
    ValidationReport,
)
from .validate import validate_contours

__all__ = [
    "BackendMismatchError",
    "ContourBatch",
    "CoordinateSystemError",
    "EdgeBatch",
    "GeometryEngine",
    "GeometryError",
    "GeometryPatch",
    "PatchConflictError",
    "PatchSet",
    "UniformGridIndex",
    "ValidationIssue",
    "ValidationReport",
    "contours_to_region",
    "extract_contours",
    "extract_edge_batches",
    "extract_edges",
    "validate_contours",
]
