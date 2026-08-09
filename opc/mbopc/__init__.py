"""高性能 MB-OPC 控制边段前端公共接口。"""

from .artifacts import save_problem_npz, write_debug_gds
from .fragment import fragment_edges
from .frontend import prepare_problem
from .ownership import MidpointOwnerPolicy, OwnershipPolicy
from .reconstruct import reconstruct_contours, reconstruct_region
from .types import (
    FragmentationConfig,
    MBOPCProblem,
    OwnershipBatch,
    SegmentBatch,
    SegmentGeometry,
    SegmentUpdateBatch,
    UpdateResult,
)
from .updates import merge_owner_updates
from .verification import build_geometry_cases, run_geometry_suite

__all__ = [
    "FragmentationConfig",
    "MBOPCProblem",
    "MidpointOwnerPolicy",
    "OwnershipBatch",
    "OwnershipPolicy",
    "SegmentBatch",
    "SegmentGeometry",
    "SegmentUpdateBatch",
    "UpdateResult",
    "build_geometry_cases",
    "fragment_edges",
    "merge_owner_updates",
    "prepare_problem",
    "reconstruct_contours",
    "reconstruct_region",
    "run_geometry_suite",
    "save_problem_npz",
    "write_debug_gds",
]
