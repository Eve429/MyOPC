"""ILT 方法包：共享结果结构与 Simple 像素优化器，不建注册器。"""

from ._common import ILTMacroResult, ILTStateRecord
from .simple import SimpleILTConfig, optimize_simple_macro

__all__ = [
    "ILTMacroResult",
    "ILTStateRecord",
    "SimpleILTConfig",
    "optimize_simple_macro",
]
