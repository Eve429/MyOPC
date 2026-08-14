"""流式执行同步 owner-only 的简单 EPE 驱动 MB-OPC。"""

from __future__ import annotations

from time import perf_counter

import klayout.db as kdb
import numpy as np
import torch

from evaluation import evaluate_binary_l2, evaluate_edge_probes, evaluate_pvband
from geometry import ContourBatch, contours_to_region
from lithography import LithographyModel
from opc.errors import ReconstructionError
from opc.input import CoreSpec
from opc.input.edge import MBOPCProblem, edge_probe_points, reconstruct_contours
from opc.input.raster import ownership_canvas, rasterize_mask_canvas
from opc.iteration._cache import ArrayTileCache

from .contracts import IterationRecord, SimpleMBOPCConfig, SimpleMBOPCResult


def _subset_contours(contours: ContourBatch,
                     polygon_ids: np.ndarray) -> ContourBatch:
    """批量提取完整 Polygon 的 hull/hole rings，并局部重建两级 CSR。"""
    ids = np.ascontiguousarray(polygon_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("polygon_ids 必须是一维数组")
    ids = np.unique(ids)
    ids = ids[(ids >= 0) & (ids < contours.polygon_count)]
    left = contours.polygon_ring_offsets[ids]
    right = contours.polygon_ring_offsets[ids + 1]
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
    polygon_offsets = np.empty(len(ring_group_counts) + 1, dtype=np.int64)
    polygon_offsets[0] = 0
    np.cumsum(ring_group_counts, out=polygon_offsets[1:])
    return ContourBatch(contours.vertices[vertex_indices], offsets, polygon_offsets)


def _polygon_ids_for_core(problem: MBOPCProblem, core_index: int) -> np.ndarray:
    """返回一个 core context 内可能受位移影响的稳定 Polygon ID。"""
    members = problem.segments_for_core(core_index)
    if not len(members):
        return np.empty(0, dtype=np.int64)
    edge_ids = problem.segments.edge_ids[members]
    return np.unique(problem.segments.edge_polygon_ids[edge_ids]).astype(np.int64)


def _target_tile(problem: MBOPCProblem, core_index: int, core: CoreSpec,
                 config: SimpleMBOPCConfig, cache: ArrayTileCache) -> np.ndarray:
    """读取或生成固定参考 mask 的 uint8 context 画布。"""
    cached = cache.get(core_index)
    if cached is not None:
        return cached
    context = core.context_box
    raster = rasterize_mask_canvas(
        problem.physical_mask.region, context, config.pixel_dbu, config.canvas,
        polarity=problem.physical_mask.polarity, field_box=problem.physical_mask.query_box)
    compact = np.rint(raster * 255.0).astype(np.uint8)
    cache.put(core_index, compact)
    return compact


def _current_tile(problem: MBOPCProblem, contours: ContourBatch,
                  displacements: np.ndarray, core_index: int,
                  core: CoreSpec, polygon_ids: np.ndarray, target: np.ndarray,
                  config: SimpleMBOPCConfig) -> np.ndarray:
    """用参考 tile 加邻近 Polygon 差分生成当前 mask，不创建完整 reticle Region。"""
    if not len(polygon_ids):
        return target.astype(np.float32) / 255.0
    context = core.context_box
    members = problem.segments_for_core(core_index)
    if not np.count_nonzero(displacements[members]):
        # 当前 core 能看到的 ownership/halo segment 全为精确零位移时，局部图形在数学
        # 上就是参考 Region。直接栅格化参考 Region 可跳过两次轮廓子集物化和三次
        # KLayout Region 布尔运算。这里不能直接返回 target：target 为节省缓存内存已
        # 量化到 uint8，原路径的 current mask 则保持浮点覆盖率，混用会改变评价结果。
        return rasterize_mask_canvas(
            problem.physical_mask.region, context, config.pixel_dbu, config.canvas,
            polarity=problem.physical_mask.polarity,
            field_box=problem.physical_mask.query_box)
    reference = contours_to_region(_subset_contours(problem.segments.contours, polygon_ids))
    current = contours_to_region(_subset_contours(contours, polygon_ids))
    local_target = problem.physical_mask.region & kdb.Region(context.to_native())
    local_current = (local_target - reference) + current
    return rasterize_mask_canvas(
        local_current, context, config.pixel_dbu, config.canvas,
        polarity=problem.physical_mask.polarity, field_box=problem.physical_mask.query_box)


def optimize(problem: MBOPCProblem, model: LithographyModel,
             config: SimpleMBOPCConfig) -> SimpleMBOPCResult:
    """以 tile batch 评价当前状态，并在轮次屏障后统一发布 owner 更新。"""
    if config.canvas > model.config.canvas:
        raise ValueError("求解器 tile canvas 不能超过光刻模型 canvas")
    # 网格常驻只保存两轴切线；优化入口一次性展开 CoreSpec 并在所有轮次复用，
    # 避免问题对象长期持有数百个可由切线和 halo 直接推导的 Python 小对象。
    cores = problem.grid.cores()
    for core in cores:
        required_width = (core.context_box.width + config.pixel_dbu - 1) // config.pixel_dbu
        required_height = (core.context_box.height + config.pixel_dbu - 1) // config.pixel_dbu
        if required_width > config.canvas or required_height > config.canvas:
            raise ValueError(f"core {core.core_id} context 超过固定光刻画布")
    owners = tuple(problem.owner_segments_for_core(index)
                   for index in range(problem.core_count))
    polygon_ids = tuple(_polygon_ids_for_core(problem, index) for index in range(len(cores)))
    reference_geometry = problem.segments.materialize()
    cache = ArrayTileCache(config.target_cache_bytes)
    current = np.zeros(problem.segments.segment_count, dtype=np.float64)
    # 零位移初态与 prepare_problem 保存的参考轮廓完全相同，直接共享只读 ContourBatch
    # 即可，避免求解开始前对整张 reticle 再做一次 O(vertex+segment) 全局重建。
    current_contours = problem.segments.contours
    nominal_condition = model.condition("nominal")
    maximum_condition = model.condition("dose_max")
    minimum_condition = model.condition("defocus_min")
    best_displacements = current.copy()
    best_epe: int | None = None
    best_iteration = 0
    records: list[IterationRecord] = []
    stop_reason = "iteration_limit"
    # `iterations` 表示最多提交多少次全局同步更新。初态先评价一次，此后每个合法
    # 更新发布后再评价新状态，因此完整执行 N 次更新会产生 N+1 条状态记录；最后
    # 一条只评价最终状态而不再生成候选，保证最佳位移一定对应真实光刻/EPE 结果。
    for iteration in range(config.iterations + 1):
        started = perf_counter()
        can_update = iteration < config.iterations
        step = (config.initial_step_dbu * (0.5 ** (iteration // config.decay_every))
                if can_update else 0.0)
        next_values = current.copy() if can_update else current
        written = (np.zeros(problem.segments.segment_count, dtype=np.bool_)
                   if can_update else None)
        total_l2 = total_pvb = 0
        total_epe = total_valid = total_ambiguous = 0
        for batch_start in range(0, len(cores), config.batch_size):
            batch_indices = list(range(batch_start, min(len(cores), batch_start + config.batch_size)))
            batch_count = len(batch_indices)
            # 三个 CPU batch 一次预分配并原位填充，避免逐 tile 小数组列表再 stack。
            # target 在 CPU 保持 uint8，传到设备时才一次性转 float32，批内峰值为旧
            # 路径的八分之一；mask 需要保留未量化覆盖率，仍使用 float32。
            targets = np.empty((batch_count, config.canvas, config.canvas), dtype=np.uint8)
            masks = np.empty((batch_count, config.canvas, config.canvas), dtype=np.float32)
            ownership_masks = np.empty(
                (batch_count, config.canvas, config.canvas), dtype=np.bool_)
            for local_index, core_index in enumerate(batch_indices):
                core = cores[core_index]
                target = _target_tile(problem, core_index, core, config, cache)
                targets[local_index] = target
                masks[local_index] = _current_tile(
                    problem, current_contours, current, core_index, core,
                    polygon_ids[core_index],
                    target, config)
                ownership_masks[local_index] = ownership_canvas(
                    core.ownership_box, core.context_box, config.pixel_dbu, config.canvas)
            target_tensor = torch.as_tensor(
                targets, dtype=torch.float32, device=model.device).div_(255.0)
            mask_tensor = torch.as_tensor(masks, device=model.device)
            owned_tensor = torch.as_tensor(ownership_masks, device=model.device)
            # CUDA tensor 已拥有设备内数据；CPU 模式 tensor 会持有底层 NumPy 引用。
            # 删除局部名称即可让两种设备都在 batch 末尾按真实依赖及时释放。
            del targets, masks, ownership_masks
            with torch.no_grad():
                printed = model.forward_many(
                    mask_tensor,
                    (nominal_condition, maximum_condition, minimum_condition))
                total_l2 += evaluate_binary_l2(
                    target_tensor, printed["nominal"],
                    model.config.print_threshold, owned_tensor)
                total_pvb += evaluate_pvband(
                    printed["dose_max"], printed["defocus_min"],
                    model.config.print_threshold, owned_tensor)
            probe_batches: list[np.ndarray] = []
            probe_inner: list[np.ndarray] = []
            probe_outer: list[np.ndarray] = []
            probe_segments: list[np.ndarray] = []
            for local_index, core_index in enumerate(batch_indices):
                indices = owners[core_index]
                if not len(indices):
                    continue
                inner, outer = edge_probe_points(
                    reference_geometry.starts[indices], reference_geometry.ends[indices],
                    reference_geometry.normals[indices], config.epe_distance_dbu)
                context = cores[core_index].context_box
                origin = np.array([context.left, context.bottom], dtype=np.float64)
                probe_batches.append(np.full(len(indices), local_index, dtype=np.int64))
                probe_inner.append((inner - origin) / config.pixel_dbu - 0.5)
                probe_outer.append((outer - origin) / config.pixel_dbu - 0.5)
                probe_segments.append(indices)
            if probe_segments:
                with torch.no_grad():
                    epe = evaluate_edge_probes(
                        target_tensor, printed["nominal"],
                        torch.as_tensor(np.concatenate(probe_batches), device=model.device),
                        torch.as_tensor(np.concatenate(probe_inner), device=model.device),
                        torch.as_tensor(np.concatenate(probe_outer), device=model.device),
                        model.config.print_threshold)
                segments = np.concatenate(probe_segments)
                if can_update:
                    # 最终评价态没有下一次更新，不分配/写入候选数组；其余状态仍用
                    # 本轮 bool 表检测唯一 owner 契约，防止错误 membership 重复写边。
                    if np.any(written[segments]):
                        raise RuntimeError("一个 segment 在同一轮被多个 core 重复写入")
                    written[segments] = True
                    directions = epe.directions.detach().cpu().numpy()
                    next_values[segments] = np.clip(
                        current[segments] + directions * step,
                        -problem.config.max_displacement_dbu,
                        problem.config.max_displacement_dbu)
                total_epe += epe.violation_count
                total_valid += int(torch.count_nonzero(epe.valid).item())
                total_ambiguous += int(torch.count_nonzero(epe.ambiguous).item())
                del epe
            # 只有标量和 owner 方向离开本 batch；四张 B×H×W GPU tensor 在循环
            # 末尾失去引用。`current` 仍未变化，后续 batch 不可能看到已计算的 next。
            del target_tensor, mask_tensor, owned_tensor, printed
        # L2/PVBand 只作为工艺诊断输出，不能改变 EPE 驱动方法的最佳轮次选择；
        # EPE 相同保留更早状态，避免非驱动指标暗中改变最终几何。
        if best_epe is None or total_epe < best_epe:
            best_epe = total_epe
            best_displacements = current.copy()
            best_iteration = iteration
        moved = (int(np.count_nonzero(~np.isclose(
            next_values, current, atol=1e-12, rtol=0.0))) if can_update else 0)
        rejected = 0
        if moved:
            try:
                candidate_contours = reconstruct_contours(problem, next_values)
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
        if not can_update:
            break
        if moved == 0 or rejected == moved:
            stop_reason = "no_legal_update"
            break
        # 屏障位于此处：只有全部 core/batch 完成且候选轮廓合法后，下一状态才同时
        # 看到新的全局绝对位移。末次允许更新也必须经过该重建守卫，随后额外评价
        # 一次发布结果；不存在未经评价候选冒充最佳状态或边算边覆盖 current。
        current = next_values
        current_contours = candidate_contours
    return SimpleMBOPCResult(
        best_displacements, tuple(records), best_iteration, stop_reason)
