"""提供独立的可微边段 OPC 迭代器。"""

from .contracts import DiffOPCConfig, DiffOPCIterationRecord, DiffOPCResult
from .solver import optimize

__all__ = ["DiffOPCConfig", "DiffOPCIterationRecord", "DiffOPCResult", "optimize"]
