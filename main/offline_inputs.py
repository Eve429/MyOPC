"""准备并恢复可脱离原始版图运行的光刻与 MB-OPC 测试输入。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# 这些脚本明确要求能从任意工作目录直接执行。Python 直接运行深层脚本时只把
# 当前文件目录加入 sys.path，因此按文件位置加入仓库根；这不是安装包，也不会
# 修改用户环境。后续第一方导入仍走项目正常公共接口。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry import (
    ContourBatch,
    contours_to_region,
)
from layout import DbuBox, LayerSpec, LayoutDB
from opc.input import PhysicalMask, RectilinearCoreGrid, preflight_layout
from opc.input.edge import (
    FragmentationConfig,
    MBOPCProblem,
    SegmentBatch,
    prepare_problem,
)
from opc.input.grid import axis_cuts_by_size
from opc.input.preflight import estimate_prepare_peak_bytes
from opc.input.raster import rasterize_region_canvas

_GIB = 1024 ** 3
_RASTER_FORMAT = "myopc.raster-input"
_SEGMENT_FORMAT = "myopc.mbopc-input"
_RASTER_VERSION = 1
_SEGMENT_VERSION = 2


def _output_npz(path: str | Path) -> Path:
    """规范化 NPZ 输出路径，并拒绝容易误判格式的其他扩展名。"""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("离线输入输出必须使用 .npz 扩展名")
    return output


def _positive_limit(value: float, name: str) -> float:
    """把安全上限规范化为有限正数，避免零值意外关闭保护。"""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _exact_dbu(value_nm: float, dbu_nm: float, name: str,
               allow_zero: bool = False) -> int:
    """把必须落在版图格点上的纳米配置严格换算为整数 DBU。"""
    value = float(value_nm)
    if (not np.isfinite(value) or value < 0.0 or
            (value == 0.0 and not allow_zero)):
        requirement = "有限非负数" if allow_zero else "有限正数"
        raise ValueError(f"{name} 必须是{requirement}")
    raw = value / dbu_nm
    rounded = round(raw)
    # tile、halo、像素和边段配置共同决定离线文件的坐标契约。静默取整会让
    # 保存时的边界与后续迭代配置不一致，因此只接受当前 DBU 精确可表达的值。
    if (rounded < 0 or (rounded == 0 and not allow_zero) or
            not np.isclose(raw, rounded, atol=1e-9, rtol=0.0)):
        raise ValueError(f"{name}={value_nm} nm 不能由当前 {dbu_nm} nm/DBU 精确表达")
    return int(rounded)


def _normalize_box(box: DbuBox | tuple[int, int, int, int] | None,
                   fallback: DbuBox | None) -> DbuBox:
    """选择显式 ROI 或顶层 bbox，并统一为公共 DBU 框。"""
    if box is None:
        if fallback is None:
            raise ValueError("输入顶层 Cell 为空")
        return fallback
    if isinstance(box, DbuBox):
        return box
    if len(box) != 4:
        raise ValueError("box 必须包含 LEFT BOTTOM RIGHT TOP 四个整数")
    return DbuBox(*box)


def _select_database_input(
        source: Path, top_cell: str | None, layer: LayerSpec | None,
        box: DbuBox | tuple[int, int, int, int] | None,
        max_file_gib: float) -> tuple[LayoutDB, LayerSpec, DbuBox]:
    """打开公共 LayoutDB，并在不物化图形时解析离线输入范围。"""
    if not source.is_file():
        raise FileNotFoundError(f"版图文件不存在：{source}")
    file_limit = _positive_limit(max_file_gib, "max_file_gib") * _GIB
    if source.stat().st_size > file_limit:
        raise ValueError(
            f"版图文件 {source.stat().st_size / _GIB:.3f} GiB 超过 {max_file_gib} GiB 上限")
    database = LayoutDB.open(source, top_cell=top_cell)
    layers = database.layers()
    if layer is None:
        if len(layers) != 1:
            names = ", ".join(f"{item.layer}/{item.datatype}" for item in layers)
            database.close()
            raise ValueError(f"版图包含多个 Layer，请显式指定 layer：{names}")
        selected_layer = layers[0]
    elif layer not in layers:
        database.close()
        raise ValueError(f"版图中不存在 Layer {layer.layer}/{layer.datatype}")
    else:
        selected_layer = layer
    selected_box = _normalize_box(box, database.bbox())
    return database, selected_layer, selected_box


def _atomic_npz(path: str | Path, arrays: dict[str, object],
                compressed: bool = False) -> Path:
    """把数组归档原子写入目标文件，异常时删除同目录临时文件。"""
    output = _output_npz(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer = np.savez_compressed if compressed else np.savez
            writer(stream, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _atomic_json(path: str | Path, value: dict[str, Any]) -> Path:
    """以 UTF-8 中文和稳定缩进原子保存测试汇总 JSON。"""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _atomic_png(path: str | Path, values: np.ndarray) -> Path:
    """原子保存左下原点的零到一数组为顶部原点八位灰度 PNG。"""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    # PNG 仅是显式请求的诊断产物；在写盘边界才执行裁剪、量化和纵轴翻转，
    # 不让图片坐标或 uint8 表示回流到模型与优化器的数值路径。
    image = np.flipud(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))
    try:
        Image.fromarray(np.ascontiguousarray(image), mode="L").save(
            temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _archive_members(path: str | Path, max_archive_gib: float) -> tuple[Path, set[str]]:
    """在 NumPy 分配数组前校验归档文件和各成员声明的总解压尺寸。"""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"离线输入不存在：{source}")
    byte_limit = int(_positive_limit(max_archive_gib, "max_archive_gib") * _GIB)
    if source.stat().st_size > byte_limit:
        raise ValueError("离线输入文件本身超过读取上限")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if sum(item.file_size for item in members) > byte_limit:
                raise ValueError("离线输入解压后数组总量超过读取上限")
            if any(item.flag_bits & 0x1 for item in members):
                raise ValueError("不支持加密 NPZ 输入")
            return source, {Path(item.filename).stem for item in members}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"离线输入不是有效 NPZ：{source}") from exc


def _metadata(data: np.lib.npyio.NpzFile, expected_format: str,
              expected_version: int) -> dict[str, Any]:
    """读取并验证无 pickle 的格式标识、版本和 JSON 元数据。"""
    try:
        format_name = str(data["format_name"].item())
        version = int(data["format_version"].item())
        value = json.loads(str(data["metadata_json"].item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("离线输入缺少有效格式标识或元数据") from exc
    if format_name != expected_format or version != expected_version:
        suffix = "；请重新生成离线边段输入" if expected_format == _SEGMENT_FORMAT else ""
        raise ValueError(
            f"不支持的离线输入格式：{format_name} version {version}{suffix}")
    if not isinstance(value, dict):
        raise TypeError("离线输入 metadata_json 必须是对象")
    return value


def _post_prepare_bytes(problem: MBOPCProblem, source_bytes: int) -> int:
    """按实际构造出的数组数量复核准备结果的保守峰值估计。"""
    contours = problem.segments.contours
    return estimate_prepare_peak_bytes(
        source_bytes, problem.physical_mask.region.count(), len(contours.vertices),
        problem.segments.segment_count, len(problem.member_segment_indices))


def materialize_raster_input(
        layout_path: str | Path, *,
        layer: LayerSpec | None = None, top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256, max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0
        ) -> tuple[np.ndarray, dict[str, object]]:
    """预检并把版图 ROI 栅格化为模型方向的内存画布。"""
    source = Path(layout_path).expanduser().resolve()
    database, selected_layer, selected_box = _select_database_input(
        source, top_cell, layer, box, max_file_gib)
    with database:
        dbu_um = database.dbu_um
        dbu_nm = dbu_um * 1000.0
        pixel_dbu = _exact_dbu(pixel_nm, dbu_nm, "pixel_nm")
        if not isinstance(canvas, int) or canvas <= 0:
            raise ValueError("canvas 必须是正整数")
        width = (selected_box.width + pixel_dbu - 1) // pixel_dbu
        height = (selected_box.height + pixel_dbu - 1) // pixel_dbu
        # 像素规模在任何 Region 物化前检查。一个归档严格对应一个可直接送入模型的
        # canvas，不隐式切 tile，避免光刻专项测试混入跨 tile 拼接语义。
        if width > canvas or height > canvas:
            raise ValueError(
                f"ROI 需要 {width}x{height} 像素，超过 {canvas}x{canvas} canvas；"
                "请缩小 box 或增大 pixel_nm")
        selected_top = database.top_cell.name
        preflight = preflight_layout(
            source, top_cell=selected_top, layer=selected_layer, box=selected_box,
            memory_budget_bytes=int(_positive_limit(
                max_estimated_gib, "max_estimated_gib") * _GIB),
            max_file_bytes=int(_positive_limit(max_file_gib, "max_file_gib") * _GIB),
            max_shape_occurrences=int(_positive_limit(
                max_shape_occurrences, "max_shape_occurrences")),
            max_source_vertices=int(_positive_limit(
                max_source_vertices, "max_source_vertices")))
        if not preflight["accepted"]:
            raise ValueError(f"ROI 物化前预检拒绝：{preflight['reason']}")
        batch = database.query([selected_layer], selected_box).materialize()
        mask = rasterize_region_canvas(
            batch.region(selected_layer), selected_box, pixel_dbu, canvas)
        polygon_count = batch.region(selected_layer).count()
    metadata = {
        "source": str(source), "top_cell": selected_top,
        "layer": [selected_layer.layer, selected_layer.datatype],
        "box_dbu": [selected_box.left, selected_box.bottom,
                    selected_box.right, selected_box.top],
        "dbu_um": dbu_um, "pixel_nm": float(pixel_nm), "pixel_dbu": pixel_dbu,
        "canvas": canvas, "active_width": width, "active_height": height,
        "orientation": "bottom_left", "polygon_count": polygon_count,
        "preflight": preflight,
    }
    return np.ascontiguousarray(mask, dtype=np.float32), metadata


def prepare_raster_input(
        layout_path: str | Path, output_path: str | Path, *,
        layer: LayerSpec | None = None, top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256, max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0) -> Path:
    """预检并栅格化一个版图 ROI，然后保存可重复使用的像素输入。"""
    # 归档路径与直接运行路径共享完全相同的版图读取和栅格化实现；此处只增加
    # 原子写盘，使专项调试可以缓存输入，而正常 GDS 入口不会生成隐式中间文件。
    mask, metadata = materialize_raster_input(
        layout_path, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib)
    return _atomic_npz(output_path, {
        "format_name": np.array(_RASTER_FORMAT),
        "format_version": np.array(_RASTER_VERSION, dtype=np.int32),
        "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        "mask": mask,
    }, compressed=True)


def load_raster_input(
        input_path: str | Path, *,
        max_archive_gib: float = 8.0) -> tuple[np.ndarray, dict[str, object]]:
    """读取并完整校验模型方向的离线像素输入。"""
    source, members = _archive_members(input_path, max_archive_gib)
    required = {"format_name", "format_version", "metadata_json", "mask"}
    if not required <= members:
        raise ValueError(f"像素输入缺少字段：{', '.join(sorted(required - members))}")
    with np.load(source, allow_pickle=False) as data:
        metadata = _metadata(data, _RASTER_FORMAT, _RASTER_VERSION)
        mask = np.ascontiguousarray(data["mask"], dtype=np.float32)
    try:
        canvas = int(metadata["canvas"])
        active_width, active_height = int(metadata["active_width"]), int(metadata["active_height"])
        orientation = str(metadata["orientation"])
        layer_values, box_values = metadata["layer"], metadata["box_dbu"]
        LayerSpec(int(layer_values[0]), int(layer_values[1]))
        DbuBox(*(int(value) for value in box_values))
        dbu_um, pixel_dbu = float(metadata["dbu_um"]), int(metadata["pixel_dbu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("像素输入元数据缺少有效 Layer、ROI、DBU、canvas 或方向") from exc
    if (mask.ndim != 2 or mask.shape != (canvas, canvas) or
            active_width <= 0 or active_height <= 0 or
            active_width > canvas or active_height > canvas):
        raise ValueError("像素输入 mask 尺寸或有效范围无效")
    if (orientation != "bottom_left" or not np.isfinite(dbu_um) or dbu_um <= 0.0 or
            pixel_dbu <= 0 or not np.all(np.isfinite(mask))):
        raise ValueError("像素输入方向或数值无效")
    if np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("像素输入覆盖率必须位于 [0, 1]")
    return mask, metadata


def resolve_raster_input(
        input_path: str | Path, *, layer: LayerSpec | None = None,
        top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0
        ) -> tuple[np.ndarray, dict[str, object]]:
    """自动读取 raster NPZ，或直接把 GDS/OASIS ROI 物化为内存 mask。"""
    source = Path(input_path).expanduser().resolve()
    # NPZ 是明确的离线输入契约，其 metadata 已经固定 Layer、ROI、像素和方向；
    # 其他扩展名统一交给 LayoutDB，从而继续支持 KLayout 可读取的版图格式。
    if source.suffix.lower() == ".npz":
        return load_raster_input(source, max_archive_gib=max_estimated_gib)
    return materialize_raster_input(
        source, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib)


def prepare_segment_input(
        layout_path: str | Path, output_path: str | Path, *,
        layer: LayerSpec | None = None, top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        tile_size_nm: float = 1024.0, halo_nm: float = 512.0,
        corner_nm: float = 16.0, segment_nm: float = 32.0,
        max_displacement_nm: float = 24.0, max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0) -> Path:
    """预检、物化并保存可直接恢复为 MBOPCProblem 的完整边段输入。"""
    source = Path(layout_path).expanduser().resolve()
    database, selected_layer, selected_box = _select_database_input(
        source, top_cell, layer, box, max_file_gib)
    with database:
        dbu_um, source_bytes = database.dbu_um, source.stat().st_size
        dbu_nm = dbu_um * 1000.0
        tile_dbu = _exact_dbu(tile_size_nm, dbu_nm, "tile_size_nm")
        halo_dbu = _exact_dbu(halo_nm, dbu_nm, "halo_nm", allow_zero=True)
        config = FragmentationConfig(
            float(_exact_dbu(corner_nm, dbu_nm, "corner_nm")),
            float(_exact_dbu(segment_nm, dbu_nm, "segment_nm")),
            float(_exact_dbu(
                max_displacement_nm, dbu_nm, "max_displacement_nm", allow_zero=True)))
        grid = RectilinearCoreGrid(
            axis_cuts_by_size(selected_box.left, selected_box.right, tile_dbu),
            axis_cuts_by_size(selected_box.bottom, selected_box.top, tile_dbu), halo_dbu)
        selected_top = database.top_cell.name
        preflight = preflight_layout(
            source, top_cell=selected_top, layer=selected_layer, box=selected_box,
            corner_dbu=config.corner_length_dbu,
            maximum_segment_dbu=config.max_segment_length_dbu, grid=grid,
            memory_budget_bytes=int(_positive_limit(
                max_estimated_gib, "max_estimated_gib") * _GIB),
            max_file_bytes=int(_positive_limit(max_file_gib, "max_file_gib") * _GIB),
            max_shape_occurrences=int(_positive_limit(
                max_shape_occurrences, "max_shape_occurrences")),
            max_source_vertices=int(_positive_limit(
                max_source_vertices, "max_source_vertices")))
        if not preflight["accepted"]:
            raise ValueError(f"ROI 物化前预检拒绝：{preflight['reason']}")
        batch = database.query([selected_layer], selected_box).materialize()
        if batch.region(selected_layer).is_empty():
            raise ValueError("选定 ROI 和 Layer 中没有可提取的物理图形")
        problem = prepare_problem(batch, selected_layer, config, grid)
    actual_peak = _post_prepare_bytes(problem, source_bytes)
    if actual_peak > max_estimated_gib * _GIB:
        raise ValueError(
            f"物化后实际规模估计 {actual_peak / _GIB:.3f} GiB 超过 "
            f"{max_estimated_gib} GiB 上限")
    contours, segments = problem.segments.contours, problem.segments
    metadata = {
        "source": str(source), "top_cell": selected_top,
        "layer": [selected_layer.layer, selected_layer.datatype],
        "box_dbu": [selected_box.left, selected_box.bottom,
                    selected_box.right, selected_box.top],
        "dbu_um": dbu_um,
        "fragmentation": {
            "corner_length_dbu": config.corner_length_dbu,
            "max_segment_length_dbu": config.max_segment_length_dbu,
            "max_displacement_dbu": config.max_displacement_dbu,
            "miter_limit": config.miter_limit,
        },
        "tiling": {"tile_size_nm": float(tile_size_nm), "tile_dbu": tile_dbu,
                   "halo_nm": float(halo_nm), "halo_dbu": halo_dbu,
                   "columns": grid.column_count, "rows": grid.row_count},
        "counts": {"polygons": contours.polygon_count, "rings": contours.ring_count,
                   "edges": len(contours.vertices), "segments": segments.segment_count,
                   "cores": problem.core_count,
                   "memberships": len(problem.member_segment_indices)},
        "preflight": preflight, "post_prepare_estimated_peak_bytes": actual_peak,
    }
    arrays: dict[str, object] = {
        "format_name": np.array(_SEGMENT_FORMAT),
        "format_version": np.array(_SEGMENT_VERSION, dtype=np.int32),
        "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        "contour_vertices": contours.vertices,
        "contour_ring_offsets": contours.ring_offsets,
        "contour_polygon_ring_offsets": contours.polygon_ring_offsets,
        "edge_next_ids": segments.edge_next_ids,
        "edge_polygon_ids": segments.edge_polygon_ids,
        "edge_normals": segments.edge_normals,
        "segment_ring_offsets": segments.ring_segment_offsets,
        "segment_edge_ids": segments.edge_ids,
        "segment_t0": segments.t0, "segment_t1": segments.t1,
        "owner_indices": problem.owner_indices,
        "core_offsets": problem.core_offsets,
        "member_segment_indices": problem.member_segment_indices,
        "grid_x_cuts": problem.grid.x_cuts, "grid_y_cuts": problem.grid.y_cuts,
        "grid_halo_dbu": np.array(problem.grid.halo_dbu, dtype=np.int64),
    }
    # 大边段归档不压缩，避免一次性压缩整张 reticle 时增加 CPU 时间和临时内存；
    # 输入只准备一次，后续模型/迭代专项测试可以直接顺序读取连续数组。
    return _atomic_npz(output_path, arrays, compressed=False)


def _validate_loaded_problem(problem: MBOPCProblem) -> None:
    """校验各数据类未覆盖的跨数组拓扑和 owner/membership 不变量。"""
    contours = problem.segments.contours
    normals = problem.segments.edge_normals
    if len(normals) and not np.allclose(
            np.hypot(normals[:, 0], normals[:, 1]), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("边段输入法向不是单位向量")
    edge_ids = problem.segments.edge_ids
    if len(edge_ids) and np.any(np.diff(edge_ids) < 0):
        raise ValueError("segment_edge_ids 必须保持全局数学边顺序")
    if len(edge_ids):
        boundaries = np.flatnonzero(np.r_[True, np.diff(edge_ids) != 0])
        ends = np.r_[boundaries[1:] - 1, len(edge_ids) - 1]
        if (not np.allclose(problem.segments.t0[boundaries], 0.0, atol=1e-12, rtol=0.0) or
                not np.allclose(problem.segments.t1[ends], 1.0, atol=1e-12, rtol=0.0)):
            raise ValueError("每条数学边的 segment 参数区间必须覆盖 [0, 1]")
        same_edge = edge_ids[1:] == edge_ids[:-1]
        if not np.allclose(problem.segments.t1[:-1][same_edge],
                           problem.segments.t0[1:][same_edge], atol=1e-12, rtol=0.0):
            raise ValueError("同一数学边的相邻 segment 参数区间存在空隙或重叠")
    offsets = problem.segments.ring_segment_offsets
    edge_ring_ids = np.repeat(
        np.arange(contours.ring_count, dtype=np.int64), np.diff(contours.ring_offsets))
    for ring_id, (start, end) in enumerate(pairwise(offsets)):
        selected = edge_ids[start:end]
        if (not len(selected) or edge_ring_ids[selected[0]] != ring_id or
                edge_ring_ids[selected[-1]] != ring_id):
            raise ValueError("segment ring offsets 与 edge ring 不一致")
    owners, members = problem.owner_indices, problem.member_segment_indices
    segment_count = problem.segments.segment_count
    core_lengths = np.diff(problem.core_offsets)
    member_cores = np.repeat(np.arange(len(core_lengths), dtype=np.int32), core_lengths)
    same_core = member_cores[1:] == member_cores[:-1]
    if len(members) > 1 and np.any((members[1:] <= members[:-1]) & same_core):
        raise ValueError("每个 core 的 membership 必须按 segment 严格递增且不重复")
    owner_hits = member_cores == owners[members]
    counts = np.bincount(members[owner_hits], minlength=segment_count)
    if len(counts) != segment_count or np.any(counts != 1):
        raise ValueError("每个 segment 必须在自身 owner 的 context 中恰好出现一次")
    if not problem.physical_mask.region.has_valid_polygons():
        raise ValueError("边段输入重建出的物理 Region 无效")


def load_segment_input(
        input_path: str | Path, *,
        max_archive_gib: float = 8.0) -> tuple[MBOPCProblem, dict[str, object]]:
    """读取离线边段归档并恢复现有 MBOPCProblem 公共数据结构。"""
    source, members = _archive_members(input_path, max_archive_gib)
    header = {"format_name", "format_version", "metadata_json"}
    if not header <= members:
        raise ValueError(f"边段输入缺少字段：{', '.join(sorted(header - members))}")
    required = {
        "format_name", "format_version", "metadata_json", "contour_vertices",
        "contour_ring_offsets", "contour_polygon_ring_offsets",
        "edge_next_ids", "edge_polygon_ids", "edge_normals", "segment_ring_offsets",
        "segment_edge_ids", "segment_t0", "segment_t1", "owner_indices",
        "core_offsets", "member_segment_indices", "grid_x_cuts", "grid_y_cuts",
        "grid_halo_dbu",
    }
    with np.load(source, allow_pickle=False) as data:
        metadata = _metadata(data, _SEGMENT_FORMAT, _SEGMENT_VERSION)
        # 先判格式版本，再检查 v2 数组字段。真实 v1 本来就没有新缓存和 grid 字段，
        # 若反过来检查只能得到误导性的“缺字段”，用户无法知道应重新生成输入。
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"边段输入缺少字段：{', '.join(sorted(missing))}")
        arrays = {name: np.ascontiguousarray(data[name]) for name in required
                  if name not in {"format_name", "format_version", "metadata_json"}}
    try:
        layer_values = metadata["layer"]
        box_values = metadata["box_dbu"]
        config_values = metadata["fragmentation"]
        layer = LayerSpec(int(layer_values[0]), int(layer_values[1]))
        query_box = DbuBox(*(int(value) for value in box_values))
        config = FragmentationConfig(
            float(config_values["corner_length_dbu"]),
            float(config_values["max_segment_length_dbu"]),
            float(config_values["max_displacement_dbu"]),
            float(config_values["miter_limit"]))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("边段输入的 Layer、ROI 或分段配置无效") from exc
    contours = ContourBatch(
        arrays["contour_vertices"], arrays["contour_ring_offsets"],
        arrays["contour_polygon_ring_offsets"])
    segments = SegmentBatch(
        contours, arrays["edge_next_ids"], arrays["edge_polygon_ids"],
        arrays["edge_normals"], arrays["segment_ring_offsets"],
        arrays["segment_edge_ids"], arrays["segment_t0"], arrays["segment_t1"])
    grid = RectilinearCoreGrid(
        arrays["grid_x_cuts"], arrays["grid_y_cuts"],
        int(arrays["grid_halo_dbu"].item()))
    if grid.bounds != query_box:
        raise ValueError("core grid 范围与离线 ROI 不一致")
    region = contours_to_region(contours)
    physical = PhysicalMask(layer, region, query_box)
    problem = MBOPCProblem(
        physical, config, segments, grid, arrays["owner_indices"],
        arrays["core_offsets"], arrays["member_segment_indices"])
    _validate_loaded_problem(problem)
    try:
        expected_counts = metadata["counts"]
        if not isinstance(expected_counts, dict):
            raise TypeError("counts must be an object")
        actual_counts = {
            "polygons": contours.polygon_count, "rings": contours.ring_count,
            "edges": len(contours.vertices), "segments": segments.segment_count,
            "cores": problem.core_count, "memberships": len(problem.member_segment_indices),
        }
        archived_counts = {name: int(expected_counts[name]) for name in actual_counts}
        dbu_um = float(metadata["dbu_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("边段输入缺少有效计数或 dbu_um 元数据") from exc
    if (not np.isfinite(dbu_um) or dbu_um <= 0.0 or
            archived_counts != actual_counts):
        raise ValueError("边段输入元数据计数或 dbu_um 与数组不一致")
    return problem, metadata


def parse_layer(value: str) -> LayerSpec:
    """解析命令行中的 `layer` 或 `layer/datatype`。"""
    parts = value.replace(":", "/").split("/")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("Layer 格式应为 layer 或 layer/datatype")
    try:
        return LayerSpec(int(parts[0]), int(parts[1]) if len(parts) == 2 else 0)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"非法 Layer：{value}") from exc


def add_layout_source_arguments(parser: argparse.ArgumentParser) -> None:
    """加入直接版图输入共用的范围选择和物化安全参数。"""
    parser.add_argument("--top-cell", help="多顶层版图必须指定")
    parser.add_argument("--layer", type=parse_layer, help="目标 layer/datatype")
    parser.add_argument("--box", nargs=4, type=int,
                        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"), help="可选 DBU ROI")
    parser.add_argument("--max-file-gib", type=float, default=4.0)
    parser.add_argument("--max-shapes", type=int, default=5_000_000)
    parser.add_argument("--max-vertices", type=int, default=20_000_000)
    parser.add_argument("--max-estimated-gib", type=float, default=8.0)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """向两个准备子命令加入输入输出位置参数和公共版图参数。"""
    parser.add_argument("layout", type=Path, help="输入 GDS/OASIS")
    parser.add_argument("output", type=Path, help="输出 NPZ")
    add_layout_source_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    """构造像素和边段离线输入准备命令行。"""
    parser = argparse.ArgumentParser(description="准备可重复使用的光刻/MB-OPC 离线输入。")
    commands = parser.add_subparsers(dest="command", required=True)
    raster = commands.add_parser("raster", help="保存模型方向固定画布")
    _add_common_arguments(raster)
    raster.add_argument("--pixel-nm", type=float, default=8.0)
    raster.add_argument("--canvas", type=int, default=256)
    segments = commands.add_parser("segments", help="保存完整可恢复 MBOPCProblem")
    _add_common_arguments(segments)
    segments.add_argument("--tile-size-nm", type=float, default=1024.0)
    segments.add_argument("--halo-nm", type=float, default=512.0)
    segments.add_argument("--corner-nm", type=float, default=16.0)
    segments.add_argument("--segment-nm", type=float, default=32.0)
    segments.add_argument("--max-displacement-nm", type=float, default=24.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行一种离线输入准备流程。"""
    args = build_parser().parse_args(argv)
    common = {
        "layer": args.layer, "top_cell": args.top_cell,
        "box": None if args.box is None else tuple(args.box),
        "max_file_gib": args.max_file_gib,
        "max_shape_occurrences": args.max_shapes,
        "max_source_vertices": args.max_vertices,
        "max_estimated_gib": args.max_estimated_gib,
    }
    try:
        if args.command == "raster":
            output = prepare_raster_input(
                args.layout, args.output, pixel_nm=args.pixel_nm,
                canvas=args.canvas, **common)
        else:
            output = prepare_segment_input(
                args.layout, args.output, tile_size_nm=args.tile_size_nm,
                halo_nm=args.halo_nm, corner_nm=args.corner_nm,
                segment_nm=args.segment_nm,
                max_displacement_nm=args.max_displacement_nm, **common)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
