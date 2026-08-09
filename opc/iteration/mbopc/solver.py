"""流式执行同步 owner-only 的简单 EPE 驱动 MB-OPC。"""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import klayout.db as kdb
import numpy as np
import torch

from evaluation import evaluate_edge_probes, evaluate_process_window
from geometry import ContourBatch, contours_to_region
from lithography import ICCAD13Lithography
from opc.errors import ReconstructionError
from opc.input.edge import MBOPCProblem, reconstruct_contours
from opc.input.raster import ownership_canvas, rasterize_region_canvas

from .types import IterationRecord, SimpleMBOPCConfig, SimpleMBOPCResult


def _subset_contours(contours: ContourBatch,
                     polygon_ids: np.ndarray) -> ContourBatch:
    """批量提取完整 Polygon 的 hull/hole rings，并保留原 polygon ID。"""
    ids = np.ascontiguousarray(polygon_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("polygon_ids 必须是一维数组")
    ids = np.unique(ids)
    ring_polygon_ids = contours.ring_polygon_ids
    if len(ring_polygon_ids) > 1 and np.any(np.diff(ring_polygon_ids) < 0):
        raise ValueError("ContourBatch 的 polygon/ring 顺序必须单调")
    left = np.searchsorted(ring_polygon_ids, ids, side="left")
    right = np.searchsorted(ring_polygon_ids, ids, side="right")
    exists = (left < len(ring_polygon_ids)) & (right > left)
    left, right = left[exists], right[exists]
    ring_group_counts = right - left
    ring_prefix = np.empty(len(ring_group_counts), dtype=np.int64)
    if len(ring_prefix):
        ring_prefix[0] = 0
        np.cumsum(ring_group_counts[:-1], out=ring_prefix[1:])
    # Polygon ID 对应的 hull/hole rings 在规范化批次中连续排列。以下区间展开只
    # 分配“本 tile 选中的 ring/vertex”大小，不再对整张 reticle 执行 np.isin 和
    # repeat；tile 越多时，避免 O(tile×全局顶点数) 成为 CPU 主耗时。
    ring_indices = (np.repeat(left - ring_prefix, ring_group_counts) +
                    np.arange(int(np.sum(ring_group_counts)), dtype=np.int64))
    all_ring_counts = np.diff(contours.ring_offsets)
    selected_counts = all_ring_counts[ring_indices]
    vertex_prefix = np.empty(len(selected_counts), dtype=np.int64)
    if len(vertex_prefix):
        vertex_prefix[0] = 0
        np.cumsum(selected_counts[:-1], out=vertex_prefix[1:])
    vertex_starts = contours.ring_offsets[ring_indices]
    vertex_indices = (np.repeat(vertex_starts - vertex_prefix, selected_counts) +
                      np.arange(int(np.sum(selected_counts)), dtype=np.int64))
    offsets = np.empty(len(selected_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(selected_counts, out=offsets[1:])
    return ContourBatch(
        contours.layer, contours.vertices[vertex_indices], offsets,
        ring_polygon_ids[ring_indices], contours.ring_is_hole[ring_indices])


def _polygon_ids_for_core(problem: MBOPCProblem, core_index: int) -> np.ndarray:
    """返回一个 core context 内可能受位移影响的稳定 Polygon ID。"""
    members = problem.ownership.segments_for_core(core_index)
    if not len(members):
        return np.empty(0, dtype=np.int64)
    edge_ids = problem.segments.edge_ids[members]
    return np.unique(problem.segments.edges.polygon_ids[edge_ids]).astype(np.int64)


class _TargetCache:
    """按字节上限缓存固定 target tile，避免每轮重复调用 KLayout 栅格器。"""

    def __init__(self, max_bytes: int) -> None:
        """创建空 LRU，并允许零上限显式关闭缓存。"""
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.values: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, key: int) -> np.ndarray | None:
        """返回并提升命中项；未命中返回 None。"""
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: int, value: np.ndarray) -> None:
        """保存 uint8 tile，并从最旧项开始驱逐直到满足字节上限。"""
        if self.max_bytes <= 0 or value.nbytes > self.max_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.current_bytes -= old.nbytes
        self.values[key] = value
        self.current_bytes += value.nbytes
        while self.current_bytes > self.max_bytes:
            _, removed = self.values.popitem(last=False)
            self.current_bytes -= removed.nbytes


def _target_tile(problem: MBOPCProblem, core_index: int,
                 config: SimpleMBOPCConfig, cache: _TargetCache) -> np.ndarray:
    """读取或生成固定参考 mask 的 context 画布。"""
    cached = cache.get(core_index)
    if cached is not None:
        # 缓存用 uint8 把每 tile 常驻内存降为 float32 的四分之一；命中时必须恢复
        # 到模型约定的 [0,1]，否则第二轮会把 255 当作 mask 强度并破坏光刻结果。
        return cached.astype(np.float32) / 255.0
    context = problem.ownership.cores[core_index].context_box
    raster = rasterize_region_canvas(
        problem.physical_mask.region, context, config.pixel_dbu, config.canvas)
    compact = np.rint(raster * 255.0).astype(np.uint8)
    cache.put(core_index, compact)
    return compact.astype(np.float32) / 255.0


def _current_tile(problem: MBOPCProblem, contours: ContourBatch, core_index: int,
                  polygon_ids: np.ndarray, target: np.ndarray,
                  config: SimpleMBOPCConfig) -> np.ndarray:
    """用参考 tile 加邻近 Polygon 差分生成当前 mask，不创建完整 reticle Region。"""
    if not len(polygon_ids):
        return target.copy()
    context = problem.ownership.cores[core_index].context_box
    reference = contours_to_region(_subset_contours(problem.segments.contours, polygon_ids))
    current = contours_to_region(_subset_contours(contours, polygon_ids))
    local_target = problem.physical_mask.region & kdb.Region(context.to_native())
    local_current = (local_target - reference) + current
    return rasterize_region_canvas(local_current, context, config.pixel_dbu, config.canvas)


def _owner_indices(problem: MBOPCProblem) -> tuple[np.ndarray, ...]:
    """一次性建立每个 core 的 owner segment 索引，供所有轮次复用。"""
    return tuple(np.flatnonzero(problem.ownership.owner_indices == core_index).astype(np.int32)
                 for core_index in range(len(problem.ownership.cores)))


def optimize(problem: MBOPCProblem, model: ICCAD13Lithography,
             config: SimpleMBOPCConfig) -> SimpleMBOPCResult:
    """以 tile batch 评价当前状态，并在轮次屏障后统一发布 owner 更新。"""
    if config.max_displacement_dbu > problem.config.max_displacement_dbu + 1e-12:
        raise ValueError("求解器最大位移不能超过前端重建配置")
    cores = problem.ownership.cores
    for core in cores:
        required_width = (core.context_box.width + config.pixel_dbu - 1) // config.pixel_dbu
        required_height = (core.context_box.height + config.pixel_dbu - 1) // config.pixel_dbu
        if required_width > config.canvas or required_height > config.canvas:
            raise ValueError(f"core {core.core_id} context 超过固定光刻画布")
    owners = _owner_indices(problem)
    polygon_ids = tuple(_polygon_ids_for_core(problem, index) for index in range(len(cores)))
    reference_geometry = problem.segments.materialize()
    cache = _TargetCache(config.target_cache_bytes)
    current = np.zeros(problem.segments.segment_count, dtype=np.float64)
    current_contours = reconstruct_contours(problem.segments, current, problem.config)
    best_displacements = current.copy()
    best_contours = current_contours
    best_score: tuple[int, float, float] | None = None
    best_iteration = 0
    records: list[IterationRecord] = []
    stop_reason = "iteration_limit"
    for iteration in range(config.iterations):
        started = perf_counter()
        step = config.initial_step_dbu * (0.5 ** (iteration // config.decay_every))
        next_values = current.copy()
        written = np.zeros(problem.segments.segment_count, dtype=np.bool_)
        total_l2 = total_pvb = 0.0
        total_epe = total_valid = total_ambiguous = 0
        for batch_start in range(0, len(cores), config.batch_size):
            batch_indices = list(range(batch_start, min(len(cores), batch_start + config.batch_size)))
            targets: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            ownership_masks: list[np.ndarray] = []
            for core_index in batch_indices:
                target = _target_tile(problem, core_index, config, cache)
                targets.append(target)
                masks.append(_current_tile(
                    problem, current_contours, core_index, polygon_ids[core_index],
                    target, config))
                core = cores[core_index]
                ownership_masks.append(ownership_canvas(
                    core.ownership_box, core.context_box, config.pixel_dbu, config.canvas))
            target_tensor = torch.as_tensor(np.stack(targets), device=model.device)
            mask_tensor = torch.as_tensor(np.stack(masks), device=model.device)
            owned_tensor = torch.as_tensor(np.stack(ownership_masks), device=model.device)
            with torch.no_grad():
                printed = model(mask_tensor)
                quality = evaluate_process_window(
                    target_tensor, printed.nominal, printed.maximum, printed.minimum,
                    owned_tensor)
            total_l2 += quality.l2
            total_pvb += quality.pvband
            probe_batches: list[np.ndarray] = []
            probe_inner: list[np.ndarray] = []
            probe_outer: list[np.ndarray] = []
            probe_segments: list[np.ndarray] = []
            for local_index, core_index in enumerate(batch_indices):
                indices = owners[core_index]
                if not len(indices):
                    continue
                geometry_indices = reference_geometry.segment_indices[indices]
                if not np.array_equal(geometry_indices, indices):
                    raise RuntimeError("参考几何索引与 segment 全局顺序不一致")
                midpoints = (reference_geometry.starts[indices] +
                             reference_geometry.ends[indices]) * 0.5
                normals = reference_geometry.normals[indices]
                context = cores[core_index].context_box
                origin = np.array([context.left, context.bottom], dtype=np.float64)
                probe_batches.append(np.full(len(indices), local_index, dtype=np.int64))
                probe_inner.append((midpoints - normals * config.epe_distance_dbu - origin) /
                                   config.pixel_dbu)
                probe_outer.append((midpoints + normals * config.epe_distance_dbu - origin) /
                                   config.pixel_dbu)
                probe_segments.append(indices)
            if probe_segments:
                with torch.no_grad():
                    epe = evaluate_edge_probes(
                        target_tensor, printed.nominal,
                        torch.as_tensor(np.concatenate(probe_batches), device=model.device),
                        torch.as_tensor(np.concatenate(probe_inner), device=model.device),
                        torch.as_tensor(np.concatenate(probe_outer), device=model.device),
                        config.print_threshold)
                segments = np.concatenate(probe_segments)
                if np.any(written[segments]):
                    raise RuntimeError("一个 segment 在同一轮被多个 core 重复写入")
                written[segments] = True
                directions = epe.directions.detach().cpu().numpy().astype(np.float64)
                next_values[segments] = np.clip(
                    current[segments] + directions * step,
                    -config.max_displacement_dbu, config.max_displacement_dbu)
                total_epe += epe.violation_count
                total_valid += int(torch.count_nonzero(epe.valid).item())
                total_ambiguous += int(torch.count_nonzero(epe.ambiguous).item())
            # 只有标量和 owner 方向离开本 batch；四张 B×H×W GPU tensor 在循环
            # 末尾失去引用。`current` 仍未变化，后续 batch 不可能看到已计算的 next。
            del target_tensor, mask_tensor, owned_tensor, printed
        score = (total_epe, total_l2, total_pvb)
        if best_score is None or score < best_score:
            best_score = score
            best_displacements = current.copy()
            best_contours = current_contours
            best_iteration = iteration
        moved = int(np.count_nonzero(~np.isclose(next_values, current, atol=1e-12, rtol=0.0)))
        rejected = 0
        if iteration < config.iterations - 1 and moved:
            try:
                candidate_contours = reconstruct_contours(
                    problem.segments, next_values, problem.config)
            except (ValueError, ReconstructionError):
                # 第一版以全轮回滚保证拓扑安全；不会保留半个 Polygon 的更新，也不会
                # 为特定错误引入补偿点。报告 rejected 数量，后续可在有真实需求时细化。
                rejected = moved
                next_values = current.copy()
                candidate_contours = current_contours
        else:
            candidate_contours = current_contours
        records.append(IterationRecord(
            iteration, step, total_epe, total_l2, total_pvb, total_valid,
            total_ambiguous, moved - rejected, rejected, perf_counter() - started))
        if total_epe == 0:
            stop_reason = "zero_epe"
            break
        if iteration == config.iterations - 1:
            break
        if moved == 0 or rejected == moved:
            stop_reason = "no_legal_update"
            break
        # 屏障位于此处：只有全部 core/batch 完成且候选轮廓合法后，下一轮才同时
        # 看到新的全局绝对位移。不存在边计算边覆盖 `current` 的顺序依赖。
        current = next_values
        current_contours = candidate_contours
    return SimpleMBOPCResult(
        best_displacements, best_contours, tuple(records), best_iteration, stop_reason)
