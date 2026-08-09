"""几何与补丁领域异常定义。"""


class GeometryError(RuntimeError):
    """所有几何处理异常的基类。"""


class BackendMismatchError(GeometryError):
    """混用不兼容的原生后端批次时抛出。"""


class CoordinateSystemError(GeometryError):
    """混用不同 Cell 坐标系的几何数据时抛出。"""


class PatchConflictError(GeometryError):
    """同一 Layer 的两个 Patch ownership 区域重叠时抛出。"""


class RasterizationError(GeometryError):
    """版图区域无法按请求的物理像素规则栅格化时抛出。"""
