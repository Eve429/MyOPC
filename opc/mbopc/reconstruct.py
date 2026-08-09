"""根据控制边段法向位移重建闭合 Polygon 环和物理 Region。"""

from __future__ import annotations

import klayout.db as kdb
import numpy as np

from geometry import ContourBatch, contours_to_region, validate_contours
from opc.errors import ReconstructionError

from .types import FragmentationConfig, SegmentBatch


def _validated_displacements(segments: SegmentBatch, displacements: object,
                             config: FragmentationConfig) -> np.ndarray:
    """规范化位移向量，并在几何计算前执行一次全局范围检查。"""
    values = np.ascontiguousarray(displacements, dtype=np.float64)
    if values.ndim != 1 or len(values) != segments.segment_count:
        raise ValueError("displacements must match segment count")
    if not np.all(np.isfinite(values)):
        raise ReconstructionError("displacements must be finite")
    if np.any(np.abs(values) > config.max_displacement_dbu):
        raise ReconstructionError("displacement exceeds configured maximum")
    return values


def reconstruct_contours(segments: SegmentBatch, displacements: object,
                         config: FragmentationConfig) -> ContourBatch:
    """批量计算 junction，并重建保留 polygon/hole 拓扑的整数轮廓。"""
    values = _validated_displacements(segments, displacements, config)
    geometry = segments.materialize(values)
    count = segments.segment_count
    if not count:
        return segments.contours
    previous = np.arange(count, dtype=np.int64) - 1
    previous[segments.ring_segment_offsets[:-1]] = segments.ring_segment_offsets[1:] - 1
    previous_ends = geometry.ends[previous]
    current_starts = geometry.starts
    previous_edges = segments.edge_ids[previous]
    current_edges = segments.edge_ids
    same_edge = previous_edges == current_edges
    same_position = same_edge & np.isclose(values[previous], values, atol=1e-12, rtol=0.0)
    junctions = (previous_ends + current_starts) * 0.5
    bevel = np.zeros(count, dtype=np.bool_)
    corners = ~same_edge
    if np.any(corners):
        first_vectors = (segments.edges.ends[previous_edges[corners]] -
                         segments.edges.starts[previous_edges[corners]]).astype(np.float64)
        second_vectors = (segments.edges.ends[current_edges[corners]] -
                          segments.edges.starts[current_edges[corners]]).astype(np.float64)
        delta = current_starts[corners] - previous_ends[corners]
        cross = first_vectors[:, 0] * second_vectors[:, 1] - first_vectors[:, 1] * second_vectors[:, 0]
        parallel = np.isclose(cross, 0.0, atol=1e-12, rtol=0.0)
        safe_cross = np.where(parallel, 1.0, cross)
        factor = (delta[:, 0] * second_vectors[:, 1] -
                  delta[:, 1] * second_vectors[:, 0]) / safe_cross
        intersections = previous_ends[corners] + factor[:, None] * first_vectors
        original = segments.edges.ends[previous_edges[corners]].astype(np.float64)
        distance = np.linalg.norm(intersections - original, axis=1)
        scale = np.maximum.reduce((np.abs(values[previous[corners]]),
                                   np.abs(values[corners]), np.ones(np.count_nonzero(corners))))
        corner_bevel = parallel | (distance > config.miter_limit * scale)
        junctions[corners] = intersections
        bevel[np.flatnonzero(corners)] = corner_bevel
    # 同一数学边且位移相同时不输出内部切分点；斜边的浮点参数点若先取整，可能偏离
    # 原直线并产生细小 XOR 毛刺。位移不同时输出两个端点形成 jog；原始拐角只有
    # miter 失控或平行时才使用两个 bevel 点，其余 junction 保存一个解析交点。
    two_points = (same_edge & ~same_position) | bevel
    output_counts = np.where(same_position, 0, np.where(two_points, 2, 1)).astype(np.int64)
    output_offsets = np.empty(count + 1, dtype=np.int64)
    output_offsets[0] = 0
    np.cumsum(output_counts, out=output_offsets[1:])
    points = np.empty((int(output_offsets[-1]), 2), dtype=np.float64)
    first_positions = output_offsets[:-1]
    single = np.flatnonzero(output_counts == 1)
    points[first_positions[single]] = junctions[single]
    doubled = np.flatnonzero(two_points)
    points[first_positions[doubled]] = previous_ends[doubled]
    points[first_positions[doubled] + 1] = current_starts[doubled]
    ring_counts = np.add.reduceat(output_counts, segments.ring_segment_offsets[:-1])
    raw_ring_offsets = np.empty(len(ring_counts) + 1, dtype=np.int64)
    raw_ring_offsets[0] = 0
    np.cumsum(ring_counts, out=raw_ring_offsets[1:])
    vertices = np.rint(points).astype(np.int64)
    keep = np.ones(len(vertices), dtype=np.bool_)
    if len(vertices) > 1:
        keep[1:] = np.any(vertices[1:] != vertices[:-1], axis=1)
    keep[raw_ring_offsets[:-1]] = True
    last_indices = raw_ring_offsets[1:] - 1
    duplicate_closure = np.all(vertices[last_indices] == vertices[raw_ring_offsets[:-1]], axis=1)
    keep[last_indices[duplicate_closure]] = False
    clean_counts = np.add.reduceat(keep.astype(np.int64), raw_ring_offsets[:-1])
    clean_offsets = np.empty(len(clean_counts) + 1, dtype=np.int64)
    clean_offsets[0] = 0
    np.cumsum(clean_counts, out=clean_offsets[1:])
    contours = ContourBatch(segments.contours.layer, vertices[keep], clean_offsets,
                            segments.contours.ring_polygon_ids,
                            segments.contours.ring_is_hole)
    report = validate_contours(contours)
    if not report.is_valid:
        codes = ", ".join(issue.code for issue in report.issues)
        raise ReconstructionError(f"reconstructed contours are invalid: {codes}")
    return contours


def reconstruct_region(segments: SegmentBatch, displacements: object,
                       config: FragmentationConfig) -> kdb.Region:
    """把重建轮廓转换为 KLayout Region，并拒绝原生无效 Polygon。"""
    region = contours_to_region(reconstruct_contours(segments, displacements, config))
    if not region.has_valid_polygons():
        raise ReconstructionError("reconstructed region contains invalid polygons")
    return region
