"""面向多种 OPC 方法的公共前端与方法实现。"""

from .errors import OPCError, OwnershipError, PhysicalMaskError, ReconstructionError

__all__ = ["OPCError", "OwnershipError", "PhysicalMaskError", "ReconstructionError"]
