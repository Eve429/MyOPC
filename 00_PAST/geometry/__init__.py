"""与具体 OPC 方法解耦的局部计算几何公共接口。"""

from .contour import ContourBatch, contours_to_region, extract_contour, extract_contours
from .errors import (
    GeometryError,
    PatchConflictError,
    RasterizationError,
)
from .patch import GeometryPatch, PatchSet, PatchWriter
from .raster import (
    iter_region_coverage_tiles,
    render_layout_region,
    render_region_batch,
)
from .validate import ValidationIssue, ValidationReport, validate_contours

__all__ = [
    "ContourBatch",
    "GeometryError",
    "GeometryPatch",
    "PatchConflictError",
    "PatchSet",
    "PatchWriter",
    "RasterizationError",
    "ValidationIssue",
    "ValidationReport",
    "contours_to_region",
    "extract_contour",
    "extract_contours",
    "iter_region_coverage_tiles",
    "render_layout_region",
    "render_region_batch",
    "validate_contours",
]
