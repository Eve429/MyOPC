"""校验唯一 owner 提交，并把稀疏 segment 更新汇聚到全局位移向量。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from opc.errors import OwnershipError

from .types import MBOPCProblem, SegmentUpdateBatch, UpdateResult


def merge_owner_updates(problem: MBOPCProblem, updates: Sequence[SegmentUpdateBatch],
                        base_displacements: object | None = None) -> UpdateResult:
    """按稳定 key 汇聚绝对位移，并返回受影响 segment 与 polygon。"""
    segment_count = problem.segments.segment_count
    if base_displacements is None:
        result = np.zeros(segment_count, dtype=np.float64)
    else:
        result = np.ascontiguousarray(base_displacements, dtype=np.float64).copy()
        if result.ndim != 1 or len(result) != segment_count or not np.all(np.isfinite(result)):
            raise ValueError("base_displacements must be a finite segment-aligned vector")
    if not updates:
        empty = np.empty(0, dtype=np.int32)
        return UpdateResult(result, empty, empty.copy())
    keys = np.concatenate([batch.keys for batch in updates])
    sources = np.concatenate([batch.source_core_indices for batch in updates])
    values = np.concatenate([batch.normal_displacements for batch in updates])
    indices = problem.segments.lookup_keys(keys)
    if np.any(indices < 0):
        raise OwnershipError("segment update contains an unknown stable key")
    if np.any(sources < 0) or np.any(sources >= len(problem.ownership.cores)):
        raise OwnershipError("segment update source core is out of range")
    expected = problem.ownership.owner_indices[indices]
    if np.any(expected != sources):
        raise OwnershipError("only the unique owner may update a segment")
    order = np.argsort(indices, kind="stable")
    if len(order) > 1 and np.any(indices[order][1:] == indices[order][:-1]):
        raise OwnershipError("the same segment was updated more than once")
    if np.any(np.abs(values) > problem.config.max_displacement_dbu):
        raise OwnershipError("segment update exceeds maximum displacement")
    # 位移始终相对固定参考边界保存为绝对值，不做隐式增量累加；这样更新顺序不会
    # 改变结果，也能直接检查相对参考图形的最大漂移。所有 halo 视图读取同一向量。
    result[indices] = values
    changed = np.ascontiguousarray(indices, dtype=np.int32)
    dirty_edges = problem.segments.edge_ids[changed]
    dirty_polygons = np.unique(problem.segments.edges.polygon_ids[dirty_edges]).astype(np.int32)
    return UpdateResult(result, changed, dirty_polygons)
