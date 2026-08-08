"""版图模块异常定义，为调用方提供稳定且易识别的失败语义。"""


class LayoutError(RuntimeError):
    """所有版图数据库异常的基类。"""


class LayoutOpenError(LayoutError):
    """输入版图无法打开或解析时抛出。"""


class AmbiguousTopCellError(LayoutError):
    """版图存在多个顶层 Cell 且调用方未明确选择时抛出。"""


class CellNotFoundError(LayoutError):
    """请求的 Cell 不存在时抛出。"""


class LayerNotFoundError(LayoutError):
    """请求的 layer/datatype 组合不存在时抛出。"""


class ClosedLayoutError(LayoutError):
    """继续使用已经关闭的 LayoutDB 时抛出。"""
