"""对任意直线批次生成可缓存模板和向量化边界采样坐标。"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .types import BoundarySampleBatch, BoundarySampleTemplate


def build_sample_template(line_count: int, tangent_positions: Sequence[float] = (0.5,),
                          normal_offsets: Sequence[float] = (0.0,)) -> BoundarySampleTemplate:
    """为每条线构造切向位置与法向偏移的笛卡尔积模板。"""
    if not isinstance(line_count, Integral) or line_count < 0:
        raise ValueError("line_count must be a non-negative integer")
    tangents = np.ascontiguousarray(tangent_positions, dtype=np.float64)
    offsets = np.ascontiguousarray(normal_offsets, dtype=np.float64)
    if tangents.ndim != 1 or offsets.ndim != 1 or not len(tangents) or not len(offsets):
        raise ValueError("sample positions and offsets must be non-empty vectors")
    per_line = len(tangents) * len(offsets)
    # 模板只在问题准备阶段构建一次。这里用 repeat/tile 直接生成结构化顺序，后续
    # 每轮只做数组索引和广播，不再嵌套遍历 line、切向点和法向偏移三个维度。
    indices = np.repeat(np.arange(int(line_count), dtype=np.int32), per_line)
    tangent_grid = np.tile(np.repeat(tangents, len(offsets)), int(line_count))
    offset_grid = np.tile(offsets, int(line_count) * len(tangents))
    return BoundarySampleTemplate(indices, tangent_grid, offset_grid)


def sample_lines(starts: object, ends: object, normals: object,
                 template: BoundarySampleTemplate, out: object | None = None) -> BoundarySampleBatch:
    """按模板批量物化线段采样点，可复用调用方提供的 N×2 缓冲区。"""
    starts_array = np.ascontiguousarray(starts, dtype=np.float64)
    ends_array = np.ascontiguousarray(ends, dtype=np.float64)
    normals_array = np.ascontiguousarray(normals, dtype=np.float64)
    if (starts_array.ndim != 2 or starts_array.shape[1] != 2 or
            ends_array.shape != starts_array.shape or normals_array.shape != starts_array.shape):
        raise ValueError("starts, ends and normals must have equal shape (N, 2)")
    if len(template.line_indices) and int(template.line_indices.max()) >= len(starts_array):
        raise ValueError("sample template references an unknown line")
    if out is None:
        points = np.empty((len(template.line_indices), 2), dtype=np.float64)
    else:
        points = np.asarray(out)
        if points.dtype != np.float64 or points.shape != (len(template.line_indices), 2):
            raise ValueError("out must be a float64 array with shape (sample_count, 2)")
    indices = template.line_indices
    # 分步写入同一缓冲区，避免同时保留 midpoint、切向增量和法向增量三个 N×2
    # 临时数组。ILT 或 MB-OPC 多轮调用时可以重复传入同一 out，进一步稳定峰值内存。
    np.take(starts_array, indices, axis=0, out=points)
    points += (ends_array[indices] - starts_array[indices]) * template.tangent_positions[:, None]
    points += normals_array[indices] * template.normal_offsets[:, None]
    return BoundarySampleBatch(points, indices, template.tangent_positions, template.normal_offsets)
