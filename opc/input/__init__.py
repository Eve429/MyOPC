"""MB-OPC、ILT 和后续方法可共同复用的 OPC 输入基础。"""

from .grid import CoreSpec, RectilinearCoreGrid
from .mask import PhysicalMask, normalize_physical_mask
from .preflight import (
    default_memory_budget_bytes,
    preflight_layout,
    process_memory_snapshot,
    resolve_memory_budget_bytes,
)

__all__ = [
    "CoreSpec",
    "PhysicalMask",
    "RectilinearCoreGrid",
    "default_memory_budget_bytes",
    "normalize_physical_mask",
    "preflight_layout",
    "process_memory_snapshot",
    "resolve_memory_budget_bytes",
]
