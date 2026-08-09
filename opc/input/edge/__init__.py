"""供边段型 OPC 方法复用的高性能输入构造公共接口。"""

from .artifacts import save_problem_npz, write_debug_gds
from .builder import prepare_problem
from .fragmentation import fragment_edges
from .ownership import MidpointOwnerPolicy, OwnershipPolicy
from .reconstruction import reconstruct_contours, reconstruct_region
from .sampling import build_sample_template, sample_lines
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
from .visualize import render_boundary_overlay

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
    "build_sample_template",
    "fragment_edges",
    "merge_owner_updates",
    "prepare_problem",
    "reconstruct_contours",
    "reconstruct_region",
    "render_boundary_overlay",
    "run_geometry_suite",
    "sample_lines",
    "save_problem_npz",
    "write_debug_gds",
]
