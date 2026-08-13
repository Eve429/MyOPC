"""供边段型 OPC 方法复用的高性能输入构造公共接口。"""

from .builder import MBOPCProblem, prepare_problem
from .fragmentation import (
    FragmentationConfig,
    SegmentBatch,
    SegmentGeometry,
    fragment_edges,
)
from .macro import MacroPreparation, prepare_macro
from .reconstruction import reconstruct_contours, reconstruct_region
from .sampling import edge_probe_points

__all__ = [
    "FragmentationConfig",
    "MacroPreparation",
    "MBOPCProblem",
    "SegmentBatch",
    "SegmentGeometry",
    "edge_probe_points",
    "fragment_edges",
    "prepare_macro",
    "prepare_problem",
    "reconstruct_contours",
    "reconstruct_region",
]
