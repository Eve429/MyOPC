"""MB-OPC 求解器包：simple 离散方法、gradient 梯度方法与共享 target 缓存。"""

from ._cache import TargetCanvasCache
from .gradient import (
    GradientMBOPCConfig,
    GradientMBOPCIterationRecord,
    GradientMBOPCResult,
    optimize_gradient_macro,
)
from .simple import (
    SimpleMBOPCConfig,
    SimpleMBOPCIterationRecord,
    SimpleMBOPCResult,
    SimpleMBOPCStep,
    evaluate_state,
    optimize_simple_macro,
)

__all__ = [
    "GradientMBOPCConfig",
    "GradientMBOPCIterationRecord",
    "GradientMBOPCResult",
    "SimpleMBOPCConfig",
    "SimpleMBOPCIterationRecord",
    "SimpleMBOPCResult",
    "SimpleMBOPCStep",
    "TargetCanvasCache",
    "evaluate_state",
    "optimize_gradient_macro",
    "optimize_simple_macro",
]
