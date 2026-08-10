"""组合公共物理 mask、MB-OPC 分段、归属和采样的准备入口。"""

from __future__ import annotations

import numpy as np

from geometry import extract_contour
from layout import LayerSpec, RegionBatch
from opc.input import RectilinearCoreGrid, normalize_physical_mask

from .fragmentation import fragment_edges
from .ownership import _build_ownership
from .types import FragmentationConfig, MBOPCProblem


def prepare_problem(batch: RegionBatch, layer: LayerSpec, config: FragmentationConfig,
                    grid: RectilinearCoreGrid | None = None) -> MBOPCProblem:
    """一次性准备可供多轮 MB-OPC 复用的完整前端问题。"""
    physical = normalize_physical_mask(batch, layer)
    # PhysicalMask 仅保留所有 OPC 方法共享的原生 Region。边段型输入在这里执行
    # 唯一一次数值轮廓提取，之后由 SegmentBatch 成为该拓扑的唯一持有者。
    segments = fragment_edges(extract_contour(physical.region), config)
    if grid is None:
        box = batch.query_box
        # 单 core 仍走与整张 reticle 完全相同的规则网格代码，避免第二套显式 core
        # 校验和边界语义。两条切线只分配常数级数组，不影响真实多 core 路径。
        grid = RectilinearCoreGrid(
            np.array([box.left, box.right], dtype=np.int64),
            np.array([box.bottom, box.top], dtype=np.int64))
    owners, offsets, members = _build_ownership(segments, grid)
    return MBOPCProblem(physical, config, segments, grid, owners, offsets, members)
