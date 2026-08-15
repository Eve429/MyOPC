"""提供基于边段 EPE 的简单同步 MB-OPC 迭代器。"""

from .contracts import IterationRecord, SimpleMBOPCConfig, SimpleMBOPCResult
from .solver import optimize

__all__ = ["IterationRecord", "SimpleMBOPCConfig", "SimpleMBOPCResult", "optimize"]
