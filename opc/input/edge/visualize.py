"""把物理 mask、边界法向、采样点和 core 归属绘制为标注 PNG。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import render_region_batch
from layout import CellRef, DbuBox, LayerSpec, RegionBatch

from ..types import BoundarySampleBatch, CoreSpec

_OWNER_COLORS = (
    (255, 87, 87), (68, 204, 255), (255, 196, 64), (126, 226, 126),
    (207, 132, 255), (255, 142, 68), (86, 234, 205), (247, 114, 181),
)


def _line_arrays(starts: object, ends: object, normals: object) -> tuple[np.ndarray, ...]:
    """规范化绘图线段，并统一检查三个 N×2 数组。"""
    starts_array = np.ascontiguousarray(starts, dtype=np.float64)
    ends_array = np.ascontiguousarray(ends, dtype=np.float64)
    normals_array = np.ascontiguousarray(normals, dtype=np.float64)
    if (starts_array.ndim != 2 or starts_array.shape[1] != 2 or
            ends_array.shape != starts_array.shape or normals_array.shape != starts_array.shape):
        raise ValueError("starts, ends and normals must have equal shape (N, 2)")
    return starts_array, ends_array, normals_array


def render_boundary_overlay(region, layer: LayerSpec, box: DbuBox, dbu_um: float,
                            starts: object, ends: object, normals: object, output_path: str | Path,
                            owners: object | None = None, samples: BoundarySampleBatch | None = None,
                            cores: tuple[CoreSpec, ...] = (), max_dimension: int = 1200,
                            max_labels: int = 80, max_samples: int = 600) -> Path:
    """保存含 mask、segment、法向、采样点、core 和稳定短标签的 PNG。"""
    starts_array, ends_array, normals_array = _line_arrays(starts, ends, normals)
    if max_dimension < 64 or max_labels < 0 or max_samples < 0:
        raise ValueError("visualization limits are invalid")
    pixel_dbu = max(1, int(np.ceil(max(box.width, box.height) / max_dimension)))
    pixel_nm = pixel_dbu * float(dbu_um) * 1000.0
    batch = RegionBatch({layer: region}, box, CellRef("OPC_VISUAL", 0))
    pixels = render_region_batch(batch, layer, dbu_um, pixel_size_nm=pixel_nm)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    # 当版图小于最大显示尺寸时，仅对诊断图做最多 4 倍的最近邻放大。
    # 这一步不改动 DBU 几何和栅格化结果，只为给标签、法向箭头留出可读空间。
    display_scale = min(4.0, max(1.0, max_dimension / max(image.width, image.height)))
    if display_scale > 1.0:
        display_size = (round(image.width * display_scale), round(image.height * display_scale))
        image = image.resize(display_size, resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(10, round(10 * display_scale ** 0.5)))
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
    # core 使用细框显示，segment 使用 owner 固定颜色。大量真实版图只抽样标签，
    # 但所有线段仍会绘制，保证图片既能看整体归属又不会被文字完全遮挡。
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
    if samples is not None and max_samples:
        # 采样点可达到 segment 数的整数倍，因此和文字分开抽样，避免真实大版图诊断图被点云完全覆盖。
        sample_step = max(1, int(np.ceil(len(samples.points) / max_samples)))
        for sample_point, offset in zip(samples.points[::sample_step],
                                        samples.normal_offsets[::sample_step], strict=True):
            x, y = point(sample_point)
            color = (255, 64, 64) if offset > 0 else (64, 255, 255)
            radius = max(2, round(display_scale))
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
