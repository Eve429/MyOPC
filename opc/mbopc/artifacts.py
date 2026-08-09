"""保存 MB-OPC 调试数组和参考/重建 GDS 的原子化产物。"""

from __future__ import annotations

import os
from pathlib import Path

import klayout.db as kdb
import numpy as np

from .types import MBOPCProblem


def save_problem_npz(problem: MBOPCProblem, displacements: object,
                     output_path: str | Path) -> Path:
    """以纯数值 NPZ 保存公共边界、紧凑 segment、归属和当前位移。"""
    values = np.ascontiguousarray(displacements, dtype=np.float64)
    if values.ndim != 1 or len(values) != problem.segments.segment_count:
        raise ValueError("displacements must match segment count")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    segments, ownership = problem.segments, problem.ownership
    try:
        # 写入打开的二进制流可避免 NumPy 根据临时文件名自动追加 `.npz`，从而让
        # 同目录 os.replace 保持原子性。所有字段均为数值或 Unicode，不启用 pickle。
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, format_version=np.array([1], dtype=np.int32),
                contour_vertices=segments.contours.vertices,
                contour_ring_offsets=segments.contours.ring_offsets,
                contour_polygon_ids=segments.contours.ring_polygon_ids,
                contour_is_hole=segments.contours.ring_is_hole,
                edge_starts=segments.edges.starts, edge_ends=segments.edges.ends,
                edge_polygon_ids=segments.edges.polygon_ids,
                edge_ring_ids=segments.edges.ring_ids, edge_lengths=segments.edge_lengths,
                edge_normals=segments.edge_normals, edge_keys=segments.edge_keys,
                segment_edge_offsets=segments.edge_segment_offsets,
                segment_ring_offsets=segments.ring_segment_offsets,
                segment_edge_ids=segments.edge_ids, segment_t0=segments.t0, segment_t1=segments.t1,
                segment_keys=segments.keys,
                segment_displacements=values, owner_indices=ownership.owner_indices,
                core_offsets=ownership.core_offsets,
                member_segment_indices=ownership.member_segment_indices,
                core_ids=np.asarray([core.core_id for core in ownership.cores]),
                core_boxes=np.asarray([[core.ownership_box.left, core.ownership_box.bottom,
                                        core.ownership_box.right, core.ownership_box.top]
                                       for core in ownership.cores], dtype=np.int64),
                context_boxes=np.asarray([[core.context_box.left, core.context_box.bottom,
                                           core.context_box.right, core.context_box.top]
                                          for core in ownership.cores], dtype=np.int64))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_debug_gds(reference: kdb.Region, reconstructed: kdb.Region, output_path: str | Path,
                    dbu_um: float, layer: int, datatype: int = 0) -> Path:
    """把参考与重建 mask 写入两个独立顶层 Cell，便于版图工具切换比较。"""
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
