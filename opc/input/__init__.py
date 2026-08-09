"""MB-OPC、ILT 和后续方法可共同复用的 OPC 输入基础。"""

from .mask import PhysicalMask, normalize_physical_mask
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
    "normalize_physical_mask",
]
