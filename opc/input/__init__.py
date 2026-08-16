"""MB-OPC、ILT 和后续方法可共同复用的 OPC 输入基础。"""

from .grid import CoreSpec, MacroSpec, plan_macros
from .mask import MaskPolarity, normalize_mask
from .raster import (
    ownership_canvas,
    points_to_canvas,
    rasterize_mask_canvas,
    rasterize_region_window,
)

__all__ = [
    "CoreSpec",
    "MacroSpec",
    "MaskPolarity",
    "normalize_mask",
    "ownership_canvas",
    "plan_macros",
    "points_to_canvas",
    "rasterize_mask_canvas",
    "rasterize_region_window",
]
