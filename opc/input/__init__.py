"""MB-OPC、ILT 和后续方法可共同复用的 OPC 输入基础。"""

from .mask import PhysicalMask, normalize_physical_mask
from .types import CoreSpec, RectilinearCoreGrid

__all__ = [
    "CoreSpec",
    "PhysicalMask",
    "RectilinearCoreGrid",
    "normalize_physical_mask",
]
