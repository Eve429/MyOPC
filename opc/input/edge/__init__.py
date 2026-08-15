"""供边段型 OPC 方法复用的高性能输入构造公共接口。"""

from .fragmentation import (
    FragmentationConfig,
    SegmentBatch,
    SegmentGeometry,
    fragment_edges,
)
from .problem import MacroProblem, prepare_macro_problem
from .reconstruction import reconstruct_contours, reconstruct_region
from .sampling import edge_probe_points

__all__ = [
    "FragmentationConfig",
    "MacroProblem",
    "SegmentBatch",
    "SegmentGeometry",
    "edge_probe_points",
    "fragment_edges",
    "prepare_macro_problem",
    "reconstruct_contours",
    "reconstruct_region",
]
