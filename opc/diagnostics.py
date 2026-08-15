"""显式保存 OPC 数值快照、版图、边界标注图和确定性几何图集。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import klayout.db as kdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import render_region_batch
from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.input import CoreSpec, RectilinearCoreGrid
from opc.input.edge.builder import MBOPCProblem, prepare_problem
from opc.input.edge.fragmentation import FragmentationConfig
from opc.input.edge.reconstruction import reconstruct_region
from opc.input.edge.sampling import edge_probe_points

_OWNER_COLORS = (
    (255, 87, 87), (68, 204, 255), (255, 196, 64), (126, 226, 126),
    (207, 132, 255), (255, 142, 68), (86, 234, 205), (247, 114, 181),
)


def save_problem_npz(problem: MBOPCProblem, displacements: object,
                     output_path: str | Path) -> Path:
    """以 segment 全局索引为对齐契约原子保存前端诊断快照。"""
    values = np.ascontiguousarray(displacements, dtype=np.float64)
    if values.ndim != 1 or len(values) != problem.segments.segment_count:
        raise ValueError("displacements must match segment count")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    segments = problem.segments
    try:
        # NPZ 仅用于一次前端验证内部的数值检查，不承诺跨重新分段恢复。写入二进制
        # 流可阻止 NumPy 自动追加扩展名；同卷 os.replace 保证异常时不会留下半文件。
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, format_version=np.array([3], dtype=np.int32),
                contour_vertices=segments.contours.vertices,
                contour_ring_offsets=segments.contours.ring_offsets,
                contour_polygon_ring_offsets=segments.contours.polygon_ring_offsets,
                edge_next_ids=segments.edge_next_ids,
                edge_polygon_ids=segments.edge_polygon_ids,
                edge_normals=segments.edge_normals,
                segment_ring_offsets=segments.ring_segment_offsets,
                segment_edge_ids=segments.edge_ids, segment_t0=segments.t0,
                segment_t1=segments.t1, segment_displacements=values,
                owner_indices=problem.owner_indices, core_offsets=problem.core_offsets,
                member_segment_indices=problem.member_segment_indices,
                grid_x_cuts=problem.grid.x_cuts, grid_y_cuts=problem.grid.y_cuts,
                grid_halo_dbu=np.array(problem.grid.halo_dbu, dtype=np.int64))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_debug_gds(reference: kdb.Region, reconstructed: kdb.Region,
                    output_path: str | Path, dbu_um: float,
                    layer: int, datatype: int = 0) -> Path:
    """把参考与重建 mask 写入两个独立顶层 Cell，便于版图工具比较。"""
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".gds":
        raise ValueError("debug layout output must use .gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.gds")
    layout = kdb.Layout()
    layout.dbu = float(dbu_um)
    layer_index = layout.layer(int(layer), int(datatype))
    layout.create_cell("REFERENCE").shapes(layer_index).insert(reference)
    layout.create_cell("RECONSTRUCTED").shapes(layer_index).insert(reconstructed)
    try:
        layout.write(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _line_arrays(starts: object, ends: object,
                 normals: object) -> tuple[np.ndarray, ...]:
    """规范化绘图线段，并统一检查三个 N×2 数组。"""
    starts_array = np.ascontiguousarray(starts, dtype=np.float64)
    ends_array = np.ascontiguousarray(ends, dtype=np.float64)
    normals_array = np.ascontiguousarray(normals, dtype=np.float64)
    if (starts_array.ndim != 2 or starts_array.shape[1] != 2 or
            ends_array.shape != starts_array.shape or normals_array.shape != starts_array.shape):
        raise ValueError("starts, ends and normals must have equal shape (N, 2)")
    return starts_array, ends_array, normals_array


def _sample_array(samples: object | None, name: str) -> np.ndarray | None:
    """规范化可选诊断探针坐标并拒绝隐藏的一维广播。"""
    if samples is None:
        return None
    array = np.ascontiguousarray(samples, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (N, 2)")
    return array


def render_boundary_overlay(
        region: kdb.Region, layer: LayerSpec, box: DbuBox, dbu_um: float,
        starts: object, ends: object, normals: object, output_path: str | Path,
        owners: object | None = None, inner_samples: object | None = None,
        outer_samples: object | None = None, cores: tuple[CoreSpec, ...] = (),
        max_dimension: int = 1200, max_labels: int = 80,
        max_samples: int = 600) -> Path:
    """保存含 mask、segment、法向、EPE 探针和 core 归属的 PNG。"""
    starts_array, ends_array, normals_array = _line_arrays(starts, ends, normals)
    inner_array = _sample_array(inner_samples, "inner_samples")
    outer_array = _sample_array(outer_samples, "outer_samples")
    if ((inner_array is None) != (outer_array is None) or
            (inner_array is not None and len(inner_array) != len(outer_array))):
        raise ValueError("inner_samples and outer_samples must be paired")
    if max_dimension < 64 or max_labels < 0 or max_samples < 0:
        raise ValueError("visualization limits are invalid")
    pixel_dbu = max(1, int(np.ceil(max(box.width, box.height) / max_dimension)))
    pixel_nm = pixel_dbu * float(dbu_um) * 1000.0
    batch = RegionBatch({layer: region}, box, CellRef("OPC_VISUAL", 0))
    pixels = render_region_batch(batch, layer, dbu_um, pixel_size_nm=pixel_nm)
    # 公共 raster 数组统一以左下为原点；Pillow 图片第 0 行显示在顶部，所以标注
    # 边界只在构造显示底图时翻转一次，后续 DBU→图片坐标继续使用顶部向下公式。
    image = Image.fromarray(np.flipud(pixels), mode="L").convert("RGB")
    # 小版图只对诊断图做最多四倍最近邻放大；DBU 几何和底层栅格值不改变，
    # 放大仅为标签、法向箭头和两类探针留出可读空间。
    display_scale = min(4.0, max(1.0, max_dimension / max(image.width, image.height)))
    if display_scale > 1.0:
        size = (round(image.width * display_scale), round(image.height * display_scale))
        image = image.resize(size, resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf", max(10, round(10 * display_scale ** 0.5)))
    except OSError:
        font = ImageFont.load_default()
    height = image.height

    def point(value: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        """把全局 DBU 坐标转换为顶部向下的图片整数坐标。"""
        x = round((float(value[0]) - box.left) / pixel_dbu * display_scale)
        y = height - 1 - round((float(value[1]) - box.bottom) / pixel_dbu * display_scale)
        return x, y

    if owners is None:
        owner_array = np.full(len(starts_array), -1, dtype=np.int32)
    else:
        owner_array = np.ascontiguousarray(owners, dtype=np.int32)
        if owner_array.ndim != 1 or len(owner_array) != len(starts_array):
            raise ValueError("owners must match line count")
    # core 使用细框、segment 使用 owner 固定颜色；大量真实版图只抽样标签，
    # 但所有线段仍绘制，避免为了图像可读性隐藏跨 core 或斜边连接问题。
    for core_index, core in enumerate(cores):
        color = _OWNER_COLORS[core_index % len(_OWNER_COLORS)]
        draw.rectangle((*point((core.ownership_box.left, core.ownership_box.top)),
                        *point((core.ownership_box.right, core.ownership_box.bottom))),
                       outline=color, width=max(1, round(display_scale)))
    for index, (start, end) in enumerate(zip(starts_array, ends_array, strict=True)):
        owner = int(owner_array[index])
        color = (230, 230, 230) if owner < 0 else _OWNER_COLORS[owner % len(_OWNER_COLORS)]
        draw.line((point(start), point(end)), fill=color, width=max(2, round(display_scale)))
    label_step = max(1, int(np.ceil(len(starts_array) / max(max_labels, 1))))
    normal_length = max(6.0 * pixel_dbu, min(box.width, box.height) * 0.025)
    for index in range(0, len(starts_array), label_step):
        midpoint = (starts_array[index] + ends_array[index]) * 0.5
        arrow_end = midpoint + normals_array[index] * normal_length
        draw.line((point(midpoint), point(arrow_end)), fill=(255, 255, 0), width=1)
        draw.ellipse((*np.subtract(point(arrow_end), 2), *np.add(point(arrow_end), 2)),
                     fill=(255, 255, 0))
        draw.text(point(midpoint), f"S{index}/C{int(owner_array[index])}",
                  fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    if inner_array is not None and max_samples:
        # 探针数与 segment 数同阶，因此按统一步长抽样；同一边段的 cyan inner 和
        # red outer 始终成对保留，便于核对法向方向及真实 EPE 距离。
        sample_step = max(1, int(np.ceil(len(inner_array) / max_samples)))
        radius = max(2, round(display_scale))
        for inner, outer in zip(inner_array[::sample_step], outer_array[::sample_step], strict=True):
            for sample, color in ((inner, (64, 255, 255)), (outer, (255, 64, 64))):
                x, y = point(sample)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_geometry_cases() -> dict[str, kdb.Region]:
    """返回覆盖正交、孔洞、斜边、负坐标和接触语义的用例。"""
    orthogonal = kdb.Region()
    orthogonal.insert(kdb.Polygon([
        kdb.Point(-100, -70), kdb.Point(110, -70), kdb.Point(110, -25),
        kdb.Point(35, -25), kdb.Point(35, 30), kdb.Point(85, 30),
        kdb.Point(85, 90), kdb.Point(-100, 90), kdb.Point(-100, 45),
        kdb.Point(-25, 45), kdb.Point(-25, -15), kdb.Point(-100, -15),
    ]))
    hole_overlap = kdb.Region(kdb.Box(-110, -90, 80, 90))
    hole_overlap -= kdb.Region(kdb.Box(-55, -40, 25, 45))
    hole_overlap.insert(kdb.Box(45, -45, 135, 55))
    hole_overlap.insert(kdb.Box(70, 10, 155, 105))
    diagonal = kdb.Region()
    diagonal.insert(kdb.Polygon([
        kdb.Point(-110, -75), kdb.Point(5, -105), kdb.Point(95, -25),
        kdb.Point(120, 80), kdb.Point(15, 105), kdb.Point(-80, 35),
    ]))
    negative_long = kdb.Region()
    negative_long.insert(kdb.Box(-260, -25, 180, 25))
    negative_long.insert(kdb.Box(-35, -150, 35, 135))
    contact = kdb.Region()
    contact.insert(kdb.Box(-120, -75, 10, 35))
    contact.insert(kdb.Box(-40, -25, 85, 85))
    contact.insert(kdb.Box(85, 85, 135, 135))
    return {
        "orthogonal_concave": orthogonal, "hole_overlap": hole_overlap,
        "diagonal_angles": diagonal, "negative_cross_core": negative_long,
        "overlap_corner_touch": contact,
    }


def _case_batch(name: str, region: kdb.Region, layer: LayerSpec) -> RegionBatch:
    """为单个用例生成含固定留白的局部批次。"""
    box = region.bbox()
    query_box = DbuBox(box.left - 20, box.bottom - 20, box.right + 20, box.top + 20)
    return RegionBatch({layer: region}, query_box, CellRef(name.upper(), 0))


def _case_grid(box: DbuBox) -> RectilinearCoreGrid:
    """在用例中心建立 2×2 core，强制覆盖水平和垂直跨界。"""
    middle_x = box.left + box.width // 2
    middle_y = box.bottom + box.height // 2
    return RectilinearCoreGrid(np.array([box.left, middle_x, box.right]),
                               np.array([box.bottom, middle_y, box.top]), 24)


def run_geometry_suite(output_dir: str | Path, write_images: bool = True,
                       probe_distance_dbu: float = 8.0) -> dict[str, Any]:
    """运行零位移、分段上限、归属和显式探针图集验证。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    layer, dbu_um = LayerSpec(1, 0), 0.001
    config = FragmentationConfig(8.0, 25.0, 12.0)
    results: list[dict[str, Any]] = []
    for name, region in build_geometry_cases().items():
        batch = _case_batch(name, region, layer)
        problem = prepare_problem(batch, layer, config, _case_grid(batch.query_box))
        zero = np.zeros(problem.segments.segment_count, dtype=np.float64)
        reconstructed = reconstruct_region(problem, zero)
        xor_area = int((reconstructed ^ problem.physical_mask.region).area())
        geometry = problem.segments.materialize(zero)
        lengths = np.linalg.norm(geometry.ends - geometry.starts, axis=1)
        maximum_length = float(lengths.max(initial=0.0))
        if xor_area or maximum_length > config.max_segment_length_dbu + 1e-12:
            raise ValueError(f"几何用例 {name} 验证失败")
        if np.any(problem.owner_indices < 0):
            raise ValueError(f"几何用例 {name} 存在无 owner 边段")
        image_path = output / f"{name}.png"
        if write_images:
            inner, outer = edge_probe_points(
                geometry.starts, geometry.ends, geometry.normals, probe_distance_dbu)
            render_boundary_overlay(
                reconstructed, layer, batch.query_box, dbu_um, geometry.starts,
                geometry.ends, geometry.normals, image_path,
                problem.owner_indices, inner, outer, problem.grid.cores(),
                max_labels=48, max_samples=240)
        results.append({
            "name": name, "polygon_count": problem.segments.contours.polygon_count,
            "ring_count": problem.segments.contours.ring_count,
            "edge_count": len(problem.segments.contours.vertices),
            "segment_count": problem.segments.segment_count,
            "core_count": problem.core_count,
            "membership_count": len(problem.member_segment_indices),
            "maximum_segment_length_dbu": maximum_length,
            "probe_distance_dbu": float(probe_distance_dbu),
            "zero_displacement_xor_area": xor_area,
            # JSON 只保存相对文件名，便于 doc 图集随仓库移动；Python 调用方已经
            # 持有 output_dir，不把开发机绝对路径写入可跟踪报告。
            "image": image_path.name if write_images else None,
        })
    summary: dict[str, Any] = {
        "case_count": len(results),
        "all_zero_displacement_exact": all(
            not case["zero_displacement_xor_area"] for case in results),
        "cases": results,
    }
    summary_path = output / "geometry_suite.json"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, summary_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary
