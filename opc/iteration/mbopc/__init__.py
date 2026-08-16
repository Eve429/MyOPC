"""MB-OPC 求解器包：simple 离散方法、gradient 梯度方法与共享 target 缓存。"""

from ._cache import TargetCanvasCache
from .gradient import (
    GradientMBOPCConfig,
    GradientMBOPCIterationRecord,
    GradientMBOPCResult,
    optimize_gradient_macro,
)
from .simple import (
    IterationRecord,
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    SimpleMBOPCStep,
    evaluate_and_propose,
    optimize_macro,
)

__all__ = [
    "GradientMBOPCConfig",
    "GradientMBOPCIterationRecord",
    "GradientMBOPCResult",
    "IterationRecord",
    "SimpleMBOPCConfig",
    "SimpleMBOPCResult",
    "SimpleMBOPCStep",
    "TargetCanvasCache",
    "evaluate_and_propose",
    "optimize_gradient_macro",
    "optimize_macro",
]
