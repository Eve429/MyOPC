"""组合公共物理 mask、MB-OPC 分段、归属和采样的准备入口。"""

from __future__ import annotations

from collections.abc import Sequence

from layout import LayerSpec, RegionBatch
from opc.common import (
    CoreSpec,
    RectilinearCoreGrid,
    build_sample_template,
    normalize_physical_mask,
)

from .fragment import fragment_edges
from .ownership import MidpointOwnerPolicy, OwnershipPolicy
from .types import FragmentationConfig, MBOPCProblem


def prepare_problem(batch: RegionBatch, layer: LayerSpec, config: FragmentationConfig,
                    cores: RectilinearCoreGrid | Sequence[CoreSpec] | None = None,
                    tangent_positions: Sequence[float] = (0.5,),
                    normal_offsets: Sequence[float] | None = None,
                    ownership_policy: OwnershipPolicy | None = None) -> MBOPCProblem:
    """一次性准备可供多轮 MB-OPC 复用的完整前端问题。"""
    physical = normalize_physical_mask(batch, layer)
    segments = fragment_edges(physical, config)
    if cores is None:
        core = CoreSpec("core0", batch.query_box, batch.query_box)
        cores = (core,)
    policy = ownership_policy or MidpointOwnerPolicy()
    ownership = policy.assign(segments, cores)
    offsets = (-config.corner_length_dbu, config.corner_length_dbu)
    template = build_sample_template(
        segments.segment_count, tangent_positions,
        offsets if normal_offsets is None else normal_offsets)
    return MBOPCProblem(physical, config, segments, ownership, template)
