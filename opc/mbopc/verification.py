"""构造确定性多图形用例，并输出 MB-OPC 边段标注图集。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import klayout.db as kdb
import numpy as np

from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.common import RectilinearCoreGrid, render_boundary_overlay, sample_lines

from .frontend import prepare_problem
from .reconstruct import reconstruct_region
from .types import FragmentationConfig


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
        "orthogonal_concave": orthogonal,
        "hole_overlap": hole_overlap,
        "diagonal_angles": diagonal,
        "negative_cross_core": negative_long,
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


def run_geometry_suite(output_dir: str | Path, write_images: bool = True) -> dict[str, Any]:
    """运行零位移不变性、分段上限、归属和图集输出验证。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    layer, dbu_um = LayerSpec(1, 0), 0.001
    config = FragmentationConfig(8.0, 25.0, 12.0)
    results: list[dict[str, Any]] = []
    for name, region in build_geometry_cases().items():
        batch = _case_batch(name, region, layer)
        problem = prepare_problem(batch, layer, config, _case_grid(batch.query_box))
        zero = np.zeros(problem.segments.segment_count, dtype=np.float64)
        reconstructed = reconstruct_region(problem.segments, zero, config)
        xor_area = int((reconstructed ^ problem.physical_mask.region).area())
        geometry = problem.segments.materialize(zero)
        maximum_length = float(geometry.lengths.max(initial=0.0))
        if xor_area or maximum_length > config.max_segment_length_dbu + 1e-12:
            raise ValueError(f"几何用例 {name} 验证失败")
        if np.any(problem.ownership.owner_indices < 0):
            raise ValueError(f"几何用例 {name} 存在无 owner 边段")
        image_path = output / f"{name}.png"
        if write_images:
            samples = sample_lines(geometry.starts, geometry.ends, geometry.normals,
                                   problem.sample_template)
            render_boundary_overlay(
                reconstructed, layer, batch.query_box, dbu_um, geometry.starts,
                geometry.ends, geometry.normals, image_path,
                problem.ownership.owner_indices, samples, problem.ownership.cores,
                max_labels=48, max_samples=240)
        results.append({
            "name": name,
            "polygon_count": problem.physical_mask.contours.polygon_count,
            "ring_count": problem.physical_mask.contours.ring_count,
            "edge_count": problem.segments.edges.edge_count,
            "segment_count": problem.segments.segment_count,
            "core_count": len(problem.ownership.cores),
            "membership_count": len(problem.ownership.member_segment_indices),
            "maximum_segment_length_dbu": maximum_length,
            "zero_displacement_xor_area": xor_area,
            "image": str(image_path) if write_images else None,
        })
    summary: dict[str, Any] = {
        "case_count": len(results),
        "all_zero_displacement_exact": all(not case["zero_displacement_xor_area"]
                                            for case in results),
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
