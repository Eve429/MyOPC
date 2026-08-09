"""供边段型 OPC 方法复用的高性能输入构造公共接口。"""

from .builder import prepare_problem
from .fragmentation import fragment_edges
from .ownership import build_ownership
from .reconstruction import reconstruct_contours, reconstruct_region
from .sampling import edge_probe_points
from .types import (
    FragmentationConfig,
    MBOPCProblem,
    OwnershipBatch,
    SegmentBatch,
    SegmentGeometry,
)

__all__ = [
    "FragmentationConfig",
    "MBOPCProblem",
    "OwnershipBatch",
    "SegmentBatch",
    "SegmentGeometry",
    "build_ownership",
    "edge_probe_points",
    "fragment_edges",
    "prepare_problem",
    "reconstruct_contours",
    "reconstruct_region",
]
