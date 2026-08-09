"""与具体 OPC 方法解耦的局部计算几何公共接口。"""

from .contour import contours_to_region, extract_contours
from .edge import extract_edge_batches, extract_edges
from .errors import (
    GeometryError,
    PatchConflictError,
    RasterizationError,
)
from .patch import PatchSet
from .raster import render_layout_region, render_region_batch
from .types import (
    ContourBatch,
    EdgeBatch,
    GeometryPatch,
    ValidationIssue,
    ValidationReport,
)
from .validate import validate_contours

__all__ = [
    "ContourBatch",
    "EdgeBatch",
    "GeometryError",
    "GeometryPatch",
    "PatchConflictError",
    "PatchSet",
    "RasterizationError",
    "ValidationIssue",
    "ValidationReport",
    "contours_to_region",
    "extract_contours",
    "extract_edge_batches",
    "extract_edges",
    "render_layout_region",
    "render_region_batch",
    "validate_contours",
]
