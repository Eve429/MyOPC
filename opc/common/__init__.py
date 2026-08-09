"""MB-OPC、ILT 和后续方法可共同复用的 OPC 前端能力。"""

from .mask import PhysicalMask, normalize_physical_mask
from .sampling import build_sample_template, sample_lines
from .types import (
    BoundarySampleBatch,
    BoundarySampleTemplate,
    CoreSpec,
    RectilinearCoreGrid,
)

__all__ = [
    "BoundarySampleBatch",
    "BoundarySampleTemplate",
    "CoreSpec",
    "PhysicalMask",
    "RectilinearCoreGrid",
    "build_sample_template",
    "normalize_physical_mask",
    "sample_lines",
]
