"""像素型 OPC 输入公共接口：macro 级 transmission 栅格问题与像素回写。"""

from .problem import (
    PixelMacroProblem,
    prepare_pixel_macro_problem,
    reconstruct_pixel_region,
)

__all__ = [
    "PixelMacroProblem",
    "prepare_pixel_macro_problem",
    "reconstruct_pixel_region",
]
