"""最简 MB-OPC：固定步长、EPE 驱动的离散边段移动求解器。"""

from ._cache import TargetCanvasCache
from .simple import (
    IterationRecord,
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    SimpleMBOPCStep,
    evaluate_and_propose,
    optimize_macro,
)

__all__ = [
    "IterationRecord",
    "SimpleMBOPCConfig",
    "SimpleMBOPCResult",
    "SimpleMBOPCStep",
    "TargetCanvasCache",
    "evaluate_and_propose",
    "optimize_macro",
]
