"""OPC 公共前端和具体方法共享的领域异常。"""


class OPCError(Exception):
    """所有可预期 OPC 前端错误的基类。"""


class PhysicalMaskError(OPCError):
    """物理 mask 无法规范化为合法边界时抛出。"""


class ReconstructionError(OPCError):
    """移动边段无法重建为合法闭合图形时抛出。"""
