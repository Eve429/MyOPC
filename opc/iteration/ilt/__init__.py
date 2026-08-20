"""ILT 方法包：共享结果结构与 Simple/LevelSet 像素优化器，不建注册器。"""

from ._common import ILTMacroResult, ILTStateRecord
from .levelset import (
    LevelSetILTConfig,
    build_levelset_final_context_canvas,
    macro_gradient_magnitude,
    optimize_levelset_macro,
    signed_distance_initialization,
)
from .simple import (
    SimpleILTConfig,
    build_simple_final_context_canvas,
    optimize_simple_macro,
)

__all__ = [
    "ILTMacroResult",
    "ILTStateRecord",
    "LevelSetILTConfig",
    "SimpleILTConfig",
    "build_levelset_final_context_canvas",
    "build_simple_final_context_canvas",
    "macro_gradient_magnitude",
    "optimize_levelset_macro",
    "optimize_simple_macro",
    "signed_distance_initialization",
]
