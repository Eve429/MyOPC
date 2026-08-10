"""与具体 OPC 方法解耦的局部计算几何公共接口。"""

from .contour import contours_to_region, extract_contour, extract_contours
from .errors import (
    GeometryError,
    PatchConflictError,
    RasterizationError,
)
from .patch import PatchSet
from .raster import render_layout_region, render_region_batch
from .types import (
    ContourBatch,
    GeometryPatch,
    ValidationIssue,
    ValidationReport,
)
from .validate import validate_contours

__all__ = [
    "ContourBatch",
    "GeometryError",
    "GeometryPatch",
    "PatchConflictError",
    "PatchSet",
    "RasterizationError",
    "ValidationIssue",
    "ValidationReport",
    "contours_to_region",
    "extract_contour",
    "extract_contours",
    "render_layout_region",
    "render_region_batch",
    "validate_contours",
]
