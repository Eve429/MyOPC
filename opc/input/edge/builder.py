"""组合公共物理 mask、MB-OPC 分段、归属和采样的准备入口。"""

from __future__ import annotations

import numpy as np

from layout import LayerSpec, RegionBatch
from opc.input import RectilinearCoreGrid, normalize_physical_mask

from .fragmentation import fragment_edges
from .ownership import build_ownership
from .types import FragmentationConfig, MBOPCProblem


def prepare_problem(batch: RegionBatch, layer: LayerSpec, config: FragmentationConfig,
                    grid: RectilinearCoreGrid | None = None) -> MBOPCProblem:
    """一次性准备可供多轮 MB-OPC 复用的完整前端问题。"""
    physical = normalize_physical_mask(batch, layer)
    segments = fragment_edges(physical, config)
    if grid is None:
        box = batch.query_box
        # 单 core 仍走与整张 reticle 完全相同的规则网格代码，避免第二套显式 core
        # 校验和边界语义。两条切线只分配常数级数组，不影响真实多 core 路径。
        grid = RectilinearCoreGrid(
            np.array([box.left, box.right], dtype=np.int64),
            np.array([box.bottom, box.top], dtype=np.int64))
    return MBOPCProblem(physical, config, segments, build_ownership(segments, grid))
