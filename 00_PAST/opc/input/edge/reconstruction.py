"""根据控制边段法向位移重建闭合 Polygon 环和物理 Region。"""

from __future__ import annotations

import klayout.db as kdb
import numpy as np

from geometry import ContourBatch, contours_to_region, validate_contours
from opc.errors import ReconstructionError

from .builder import MBOPCProblem
from .fragmentation import FragmentationConfig, SegmentBatch


def _ring_signed_areas2(contours: ContourBatch) -> np.ndarray:
    """向量化返回每个 ring 的两倍有向面积，供拓扑方向比较。"""
    if not len(contours.vertices):
        return np.empty(0, dtype=np.float64)
    current = np.arange(len(contours.vertices), dtype=np.int64)
    following = current + 1
    following[contours.ring_offsets[1:] - 1] = contours.ring_offsets[:-1]
    vertices = contours.vertices.astype(np.float64, copy=False)
    crosses = (vertices[:, 0] * vertices[following, 1] -
               vertices[:, 1] * vertices[following, 0])
    return np.add.reduceat(crosses, contours.ring_offsets[:-1])


def _validate_reference_topology(reference: ContourBatch,
                                 candidate: ContourBatch) -> None:
    """拒绝 ring 翻转及 hole 越出所属 hull 的候选轮廓。"""
    if (candidate.ring_count != reference.ring_count or
            not np.array_equal(candidate.polygon_ring_offsets,
                               reference.polygon_ring_offsets)):
        raise ReconstructionError("reconstructed contours changed ring topology")
    reference_areas = _ring_signed_areas2(reference)
    candidate_areas = _ring_signed_areas2(candidate)
    # 左边越过右边后仍可能形成 KLayout 认为“有效”的反向矩形；绕向比较专门
    # 捕获这类对边穿越。零面积候选会在后面的 validate_contours 中被拒绝。
    if np.any(np.signbit(reference_areas) != np.signbit(candidate_areas)):
        raise ReconstructionError("reconstructed ring reversed orientation")
    ring_is_hole = np.ones(candidate.ring_count, dtype=np.bool_)
    ring_is_hole[candidate.polygon_ring_offsets[:-1]] = False
    hole_ids = np.flatnonzero(ring_is_hole)
    if not len(hole_ids):
        return
    ring_polygon_ids = np.repeat(
        np.arange(candidate.polygon_count, dtype=np.int64),
        np.diff(candidate.polygon_ring_offsets))
    for hole_id in hole_ids:
        polygon_id = ring_polygon_ids[hole_id]
        hull_id = candidate.polygon_ring_offsets[polygon_id]
        hull_start, hull_end = candidate.ring_offsets[hull_id:hull_id + 2]
        hole_start, hole_end = candidate.ring_offsets[hole_id:hole_id + 2]
        hull = kdb.Region(kdb.Polygon([
            kdb.Point(int(x), int(y))
            for x, y in candidate.vertices[hull_start:hull_end]]))
        hole = kdb.Region(kdb.Polygon([
            kdb.Point(int(x), int(y))
            for x, y in candidate.vertices[hole_start:hole_end]]))
        # hole 与 hull 的关系不是单 ring 有效性可以表达的；任意 hole 面积落到
        # hull 外都意味着内外线交叉，必须整轮拒绝而不能让 Region 布尔运算修补。
        if not (hole - hull).is_empty():
            raise ReconstructionError("reconstructed hole escaped its hull")


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


def reconstruct_contours(problem: MBOPCProblem, displacements: object) -> ContourBatch:
    """批量计算 junction，并重建保留 polygon/hole 拓扑的整数轮廓。"""
    segments, config = problem.segments, problem.config
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
        vertices = segments.contours.vertices
        first_ids, second_ids = previous_edges[corners], current_edges[corners]
        first_vectors = (vertices[segments.edge_next_ids[first_ids]] -
                         vertices[first_ids]).astype(np.float64)
        second_vectors = (vertices[segments.edge_next_ids[second_ids]] -
                          vertices[second_ids]).astype(np.float64)
        delta = current_starts[corners] - previous_ends[corners]
        cross = first_vectors[:, 0] * second_vectors[:, 1] - first_vectors[:, 1] * second_vectors[:, 0]
        parallel = np.isclose(cross, 0.0, atol=1e-12, rtol=0.0)
        safe_cross = np.where(parallel, 1.0, cross)
        factor = (delta[:, 0] * second_vectors[:, 1] -
                  delta[:, 1] * second_vectors[:, 0]) / safe_cross
        intersections = previous_ends[corners] + factor[:, None] * first_vectors
        original = vertices[segments.edge_next_ids[first_ids]].astype(np.float64)
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
    contours = ContourBatch(vertices[keep], clean_offsets,
                            segments.contours.polygon_ring_offsets)
    _validate_reference_topology(segments.contours, contours)
    report = validate_contours(contours, problem.physical_mask.layer)
    if not report.is_valid:
        codes = ", ".join(issue.code for issue in report.issues)
        raise ReconstructionError(f"reconstructed contours are invalid: {codes}")
    return contours


def reconstruct_region(problem: MBOPCProblem, displacements: object) -> kdb.Region:
    """把重建轮廓转换为 KLayout Region，并拒绝原生无效 Polygon。"""
    region = contours_to_region(reconstruct_contours(problem, displacements))
    if not region.has_valid_polygons():
        raise ReconstructionError("reconstructed region contains invalid polygons")
    return region
