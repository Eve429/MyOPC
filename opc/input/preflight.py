"""在完整几何物化前估算 OPC 输入规模并采集进程内存状态。"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import klayout.db as kdb
import numpy as np
import psutil

from layout import DbuBox, LayerSpec

from .grid import RectilinearCoreGrid

_GIB = 1024 ** 3
_INT32_MAX = int(np.iinfo(np.int32).max)


def default_memory_budget_bytes() -> int:
    """返回启动时系统可用内存的七成，给操作系统和原生库保留余量。"""
    return max(1, int(psutil.virtual_memory().available * 0.70))


def resolve_memory_budget_bytes(value_gib: float | None) -> int:
    """把可选 GiB 参数转换为字节，省略时采用系统可用内存七成。"""
    if value_gib is None:
        return default_memory_budget_bytes()
    value = float(value_gib)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("memory-budget-gib 必须是有限正数")
    return int(value * _GIB)


def process_memory_snapshot() -> dict[str, int]:
    """读取 Python、NumPy 与 KLayout 均可见的进程级内存检查点。"""
    process = psutil.Process()
    basic = process.memory_info()
    try:
        full = process.memory_full_info()
    except (psutil.AccessDenied, AttributeError):
        full = basic
    # tracemalloc 看不到 KLayout 和 NumPy 的原生分配，因此以操作系统进程统计为准。
    # Windows 的 private/peak_wset 与其他平台字段不完全一致，缺失时使用可比较的
    # USS/RSS 退化值，摘要仍保留统一键，便于同一台机器前后对比。
    rss = int(basic.rss)
    uss = int(getattr(full, "uss", getattr(basic, "private", rss)))
    private = int(getattr(basic, "private", uss))
    peak = int(getattr(basic, "peak_wset", rss))
    return {
        "rss_bytes": rss, "uss_bytes": uss, "private_bytes": private,
        "peak_working_set_bytes": peak,
        "system_available_bytes": int(psutil.virtual_memory().available),
    }


def _shape_polygon(shape: kdb.Shape) -> kdb.Polygon:
    """把预检允许的 Box、Path、Polygon 统一转换为临时 Polygon。"""
    if shape.is_box():
        return kdb.Polygon(shape.box)
    if shape.is_path():
        return shape.path.polygon()
    if shape.is_polygon():
        return shape.polygon
    raise TypeError("预检迭代器返回了非 Polygon 类图形")


def _fragment_counts(lengths: np.ndarray, corner_dbu: float,
                     maximum_dbu: float) -> np.ndarray:
    """按生产切分公式估算每条数学边产生的 segment 数。"""
    counts = np.ceil(lengths / maximum_dbu).astype(np.int64)
    long_edges = lengths > 2.0 * maximum_dbu
    counts[long_edges] = 2 + np.ceil(
        (lengths[long_edges] - 2.0 * corner_dbu) / maximum_dbu).astype(np.int64)
    return counts


def _membership_upper_bound(starts: np.ndarray, ends: np.ndarray,
                            counts: np.ndarray, grid: RectilinearCoreGrid) -> int:
    """按数学边扩展 bbox 估算全部切分段触及的 context 数上界。"""
    halo = grid.halo_dbu
    left = np.minimum(starts[:, 0], ends[:, 0]) - halo
    right = np.maximum(starts[:, 0], ends[:, 0]) + halo
    bottom = np.minimum(starts[:, 1], ends[:, 1]) - halo
    top = np.maximum(starts[:, 1], ends[:, 1]) + halo
    ix0 = np.clip(np.searchsorted(grid.x_cuts[1:], left, side="left"),
                  0, grid.column_count - 1)
    ix1 = np.clip(np.searchsorted(grid.x_cuts[:-1], right, side="right") - 1,
                  0, grid.column_count - 1)
    iy0 = np.clip(np.searchsorted(grid.y_cuts[1:], bottom, side="left"),
                  0, grid.row_count - 1)
    iy1 = np.clip(np.searchsorted(grid.y_cuts[:-1], top, side="right") - 1,
                  0, grid.row_count - 1)
    spans = np.maximum(ix1 - ix0 + 1, 0) * np.maximum(iy1 - iy0 + 1, 0)
    value = int(np.sum(counts * spans, dtype=np.int64))
    if value < 0:
        raise OverflowError("membership 估算超过 int64 容量")
    return value


def estimate_prepare_peak_bytes(file_bytes: int, shapes: int, vertices: int,
                                segments: int, memberships: int) -> int:
    """用当前原生几何和 NumPy 热路径的安全系数估算准备阶段峰值。"""
    # 文件解析按 8 倍、原始对象按 256 B/shape、布尔几何按 256 B/vertex、
    # 切分临时数组按 128 B/segment、CSR 排序按 32 B/membership 计入。
    # 该值用于提前拒绝而非精确计费，宁可高估也不允许在预检后发生系统换页风暴。
    return (file_bytes * 8 + shapes * 256 + vertices * 256 +
            segments * 128 + memberships * 32)


def estimate_solver_peak_bytes(file_bytes: int, shapes: int, vertices: int,
                               segments: int, memberships: int) -> int:
    """估算当前全局 MB-OPC 求解路径的 CPU 峰值下限。"""
    # 求解阶段会同时存在 problem、current/next/best/written 和全局参考几何；
    # 128 B/segment 已覆盖这些对齐数组及重建临时量，membership 以 8 B 计入排序
    # 与持久索引重叠。KLayout Region 仍用 shape/vertex 系数保守保留。
    return (file_bytes * 2 + shapes * 128 + vertices * 128 +
            segments * 128 + memberships * 8)


def preflight_layout(
        layout_path: str | Path, *, top_cell: str, layer: LayerSpec, box: DbuBox,
        corner_dbu: float | None = None, maximum_segment_dbu: float | None = None,
        grid: RectilinearCoreGrid | None = None, memory_budget_bytes: int,
        max_file_bytes: int | None = None, max_shape_occurrences: int | None = None,
        max_source_vertices: int | None = None) -> dict[str, int | bool | str]:
    """扫描指定层级 ROI，并在任何完整 Region 或边段数组分配前返回容量判断。"""
    source = Path(layout_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"版图文件不存在：{source}")
    file_bytes = source.stat().st_size
    if max_file_bytes is not None and file_bytes > max_file_bytes:
        raise ValueError(f"版图文件 {file_bytes / _GIB:.3f} GiB 超过读取上限")
    if not isinstance(memory_budget_bytes, Integral) or memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes 必须是正整数")
    if (corner_dbu is None) != (maximum_segment_dbu is None):
        raise ValueError("边段预检必须同时提供 corner_dbu 和 maximum_segment_dbu")
    if corner_dbu is not None and (corner_dbu <= 0.0 or maximum_segment_dbu <= 0.0):
        raise ValueError("边段预检长度必须为正")
    layout = kdb.Layout()
    layout.read(str(source))
    top = layout.cell(top_cell)
    if top is None:
        raise ValueError(f"版图中不存在顶层 Cell：{top_cell}")
    layer_index = None
    for index in layout.layer_indexes():
        info = layout.get_info(index)
        if info.layer == layer.layer and info.datatype == layer.datatype:
            layer_index = index
            break
    if layer_index is None:
        raise ValueError(f"版图中不存在 Layer {layer.layer}/{layer.datatype}")
    iterator = kdb.RecursiveShapeIterator(layout, top, layer_index, box.to_native(), True)
    iterator.shape_flags = kdb.Shapes.SBoxes | kdb.Shapes.SPaths | kdb.Shapes.SPolygons
    shapes = vertices = segments = memberships = 0
    scan_complete = True
    for item in iterator:
        shapes += 1
        if max_shape_occurrences is not None and shapes > max_shape_occurrences:
            raise ValueError(f"ROI 层级展开图形数超过上限 {max_shape_occurrences:,}")
        polygon = _shape_polygon(item.shape()).transformed(item.trans())
        rings = [polygon.each_point_hull()]
        rings.extend(polygon.each_point_hole(index) for index in range(polygon.holes()))
        for ring_iterator in rings:
            points = np.asarray([(point.x, point.y) for point in ring_iterator],
                                dtype=np.float64)
            vertices += len(points)
            if max_source_vertices is not None and vertices > max_source_vertices:
                raise ValueError(f"ROI 原始顶点数超过上限 {max_source_vertices:,}")
            if corner_dbu is None or not len(points):
                continue
            ends = np.roll(points, -1, axis=0)
            lengths = np.hypot(*(ends - points).T)
            valid = lengths > 0.0
            counts = _fragment_counts(
                lengths[valid], float(corner_dbu), float(maximum_segment_dbu))
            segments += int(counts.sum(dtype=np.int64))
            if grid is not None and len(counts):
                memberships += _membership_upper_bound(
                    points[valid], ends[valid], counts, grid)
        prepare_bytes = estimate_prepare_peak_bytes(
            file_bytes, shapes, vertices, segments, memberships)
        solver_bytes = estimate_solver_peak_bytes(
            file_bytes, shapes, vertices, segments, memberships)
        # 已经超过预算或 int32 后，当前统计就是足以拒绝的严格下界。立即停止层级
        # 展开可避免仅为得到更大的数字而扫描数亿 occurrence；摘要显式标记未完成。
        if (max(prepare_bytes, solver_bytes) > memory_budget_bytes or
                segments > _INT32_MAX or memberships > _INT32_MAX):
            scan_complete = False
            break
    del layout
    prepare_bytes = estimate_prepare_peak_bytes(
        file_bytes, shapes, vertices, segments, memberships)
    solver_bytes = estimate_solver_peak_bytes(
        file_bytes, shapes, vertices, segments, memberships)
    int32_ok = segments <= _INT32_MAX and memberships <= _INT32_MAX
    memory_ok = max(prepare_bytes, solver_bytes) <= memory_budget_bytes
    accepted = scan_complete and int32_ok and memory_ok
    reason = "ok"
    if not int32_ok:
        reason = "segment 或 membership 超过当前全局 int32 容量"
    elif not memory_ok:
        reason = "当前全局准备/求解路径预计超过 CPU 内存预算"
    return {
        "source_file_bytes": file_bytes, "shape_occurrences": shapes,
        "source_vertices": vertices, "estimated_segments": segments,
        "estimated_memberships": memberships,
        "estimated_prepare_peak_bytes": prepare_bytes,
        "estimated_solver_peak_bytes": solver_bytes,
        "memory_budget_bytes": int(memory_budget_bytes),
        "int32_capacity_ok": int32_ok, "memory_budget_ok": memory_ok,
        "scan_complete": scan_complete, "counts_are_lower_bounds": not scan_complete,
        "accepted": accepted, "reason": reason,
        "recommended_mode": "in_memory" if accepted else "sharded_required",
    }
