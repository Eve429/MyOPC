"""根据控制边段法向位移重建闭合 Polygon 环和物理 Region。"""

from __future__ import annotations

from dataclasses import dataclass

import klayout.db as kdb
import numpy as np

from geometry import ContourBatch, contours_to_region, validate_contours
from opc.errors import ReconstructionError

from .fragmentation import FragmentationConfig, SegmentBatch
from .problem import MacroProblem


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


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """保存一次重构的轻量产物：整数轮廓与可选的各段采样中点。"""

    contours: ContourBatch                # np.rint 取整与去重后的整数轮廓
    segment_midpoints: np.ndarray | None  # [S,2] float64；未请求时 None


def _reconstruct_geometry(problem: MacroProblem, displacements: object, *,
                          with_midpoints: bool = False) -> ReconstructionResult:
    """核心重构：junction/bevel 拼装整数轮廓；按需附带各段采样中点。

    中点需要 following 与三个 [S,2] 数组（约 56S 字节临时内存）——仅
    梯度路径请求；simple 等热路径调用方默认不承担该成本。
    """
    segments, config = problem.segments, problem.fragmentation
    # 检查 displacements 有效性
    values = _validated_displacements(segments, displacements, config)
    # 得到边段移动后的start、end
    geometry = segments.materialize(values)
    count = segments.segment_count
    if not count:
        return ReconstructionResult(
            segments.contours,
            np.empty((0, 2), dtype=np.float64) if with_midpoints else None)
    # 构造previous，让每条边段知道自己前一条是什么
    previous = np.arange(count, dtype=np.int64) - 1
    previous[segments.ring_segment_offsets[:-1]] = segments.ring_segment_offsets[1:] - 1
    # previous_edges、current_edges表示上一条边和当前边；相同就表示原本是一条边而不是corner
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
        # 只有存在真拐角（前一段与当前段属于不同数学边）才需要解析交点；
        # 同一条边的内部切分点由上方中点近似与 same_position 逻辑覆盖。
        # 角点方向一律取自原始（未位移）顶点表：位移只沿法向，不改变边方向。
        vertices = segments.contours.vertices
        # 压缩到拐角子集：前一/当前段的数学边起点顶点索引，后续全部向量化
        first_ids, second_ids = previous_edges[corners], current_edges[corners]
        # 前一条边的方向向量（终点顶点 − 起点顶点），定义拐角处第一条无限直线
        first_vectors = (vertices[segments.edge_next_ids[first_ids]] -
                         vertices[first_ids]).astype(np.float64)
        # 当前边的方向向量，定义第二条无限直线
        second_vectors = (vertices[segments.edge_next_ids[second_ids]] -
                          vertices[second_ids]).astype(np.float64)
        # 位移后两线不再共点：delta = 当前段起点 − 前一段终点，是两条直线的相对错位
        delta = current_starts[corners] - previous_ends[corners]
        # 二维叉积（z 分量）= |a||b|·sinθ：两条边方向的平行判定基础
        cross = first_vectors[:, 0] * second_vectors[:, 1] - first_vectors[:, 1] * second_vectors[:, 0]
        # 平行/共线时直线无交点，解析解不存在，必须走 bevel 退化路径
        parallel = np.isclose(cross, 0.0, atol=1e-12, rtol=0.0)
        # 防除零：平行行把分母换成 1.0，其 factor 是垃圾值，但 parallel 标志
        # 保证这些行最终走 bevel，垃圾结果不会被消费
        safe_cross = np.where(parallel, 1.0, cross)
        # 克莱姆法则解 previous_end + t·first = current_start + s·second 中的 t：
        # 分子是 delta 与 second 的叉积，分母是两方向的叉积
        factor = (delta[:, 0] * second_vectors[:, 1] -
                  delta[:, 1] * second_vectors[:, 0]) / safe_cross
        # 参数 t 代回第一条直线得到精确交点，即拐角的 miter（斜接）顶点
        intersections = previous_ends[corners] + factor[:, None] * first_vectors
        # 拐角的原始顶点（前一条边的终点顶点 = 位移前两人共享的角点）
        original = vertices[segments.edge_next_ids[first_ids]].astype(np.float64)
        # miter 尖点偏离原角点的距离：尖角在同等位移下会把交点推得很远（尖刺）
        distance = np.linalg.norm(intersections - original, axis=1)
        # 位移幅度标尺：取两侧位移绝对值与 1（DBU）的最大者；下界 1 保证零位移
        # 时阈值不退化为 0，miter_limit 以「位移的倍数」表达允许的尖刺长度
        scale = np.maximum.reduce((np.abs(values[previous[corners]]),
                                   np.abs(values[corners]), np.ones(np.count_nonzero(corners))))
        # 两种情况放弃 miter 改用 bevel（保留 previous_end/current_start 两个点平接）：
        # ① 平行无交点；② 交点偏移超过 miter_limit 倍位移（尖刺过长）
        corner_bevel = parallel | (distance > config.miter_limit * scale)
        # 非 bevel 拐角的 junction 从中点近似升级为解析交点（bevel 行写入值不被使用）
        junctions[corners] = intersections
        # 布尔掩码 → 压缩下标，把拐角子集的 bevel 标志写回全长度数组
        bevel[np.flatnonzero(corners)] = corner_bevel
    # 同一数学边且位移相同时不输出内部切分点；斜边的浮点参数点若先取整，可能偏离
    # 原直线并产生细小 XOR 毛刺。位移不同时输出两个端点形成 jog；原始拐角只有
    # miter 失控或平行时才使用两个 bevel 点，其余 junction 保存一个解析交点。
    two_points = (same_edge & ~same_position) | bevel
    segment_midpoints = None  # 默认不计算；仅 with_midpoints=True 的梯度路径填充
    if with_midpoints:
        # 各段实际采样中点：与轮廓拼接规则一一对应——two_points 边界
        # （jog/bevel）前段终于 previous_end、后段始于 current_start；
        # 普通边界共享一个 junction；same_position 无输出点，内部细分点
        # 取共线中点（落在合并直线上，各 fragment 采样等价）。保持
        # float64 连续域，不随下方 np.rint 整数化——梯度采样需要追踪
        # fragment 身份而非最终轮廓顶点。
        following = np.arange(count, dtype=np.int64) + 1  # 环内下一段
        following[segments.ring_segment_offsets[1:] - 1] = (
            segments.ring_segment_offsets[:-1])  # 各环尾段回卷到首段
        segment_starts = np.where(two_points[:, None], current_starts,
                                  junctions)
        segment_ends = np.where(two_points[following, None],
                                previous_ends[following],
                                junctions[following])
        segment_midpoints = (segment_starts + segment_ends) * 0.5
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
    report = validate_contours(contours, problem.layer)
    if not report.is_valid:
        codes = ", ".join(issue.code for issue in report.issues)
        raise ReconstructionError(f"reconstructed contours are invalid: {codes}")
    return ReconstructionResult(contours, segment_midpoints)


def reconstruct_contours(problem: MacroProblem, displacements: object) -> ContourBatch:
    """批量计算 junction，并重建保留 polygon/hole 拓扑的整数轮廓。"""
    return _reconstruct_geometry(problem, displacements).contours


def reconstruct_region(problem: MacroProblem, displacements: object) -> kdb.Region:
    """把重建轮廓转换为 KLayout Region，并拒绝原生无效 Polygon。"""
    region = contours_to_region(reconstruct_contours(problem, displacements))
    if not region.has_valid_polygons():
        raise ReconstructionError("reconstructed region contains invalid polygons")
    return region


def reconstruct_region_with_midpoints(
        problem: MacroProblem,
        displacements: object) -> tuple[kdb.Region, np.ndarray]:
    """一次重构同时返回 Region 与各段实际采样中点（梯度路径专用）。

    两产物来自同一次几何计算：Region 供栅格化与评价、中点供梯度采样，
    保证 forward 几何与 backward 采样点永不来自不同次重构。
    """
    result = _reconstruct_geometry(problem, displacements, with_midpoints=True)
    region = contours_to_region(result.contours)
    if not region.has_valid_polygons():
        raise ReconstructionError("reconstructed region contains invalid polygons")
    return region, result.segment_midpoints
