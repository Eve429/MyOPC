"""按 owner 唯一计分和逐 batch 反传执行可微边段 OPC。"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from torch.nn import functional

from evaluation import evaluate_binary_l2, evaluate_edge_probes, evaluate_pvband
from lithography import ICCAD13Lithography
from opc.errors import ReconstructionError
from opc.input.edge import MBOPCProblem, edge_probe_points, reconstruct_region
from opc.input.raster import ownership_canvas, rasterize_region_canvas
from opc.iteration._cache import ArrayTileCache

from .contracts import DiffOPCConfig, DiffOPCIterationRecord, DiffOPCResult
from .rasterizer import rasterize_soft_edges


def _sample_probe(image: torch.Tensor, points: np.ndarray, origin: torch.Tensor,
                  pixel_dbu: int, *, mode: str = "bilinear") -> torch.Tensor:
    """按左下原点像素中心坐标在单张 tile 上采样 DBU 探针。"""
    if image.ndim != 2 or mode not in ("bilinear", "nearest"):
        raise ValueError("probe image 必须二维且采样模式必须为 bilinear 或 nearest")
    # raster 第 0 列中心位于 origin+0.5*pixel，因此先减 0.5 才是 grid_sample
    # 的像素索引。旧实现遗漏该项，会把每个探针系统性向右上偏移半个像素。
    xy = ((torch.as_tensor(points, device=image.device, dtype=torch.float32) - origin) /
          float(pixel_dbu) - 0.5)
    height, width = image.shape
    grid = torch.stack((xy[:, 0] / max(width - 1, 1) * 2.0 - 1.0,
                        xy[:, 1] / max(height - 1, 1) * 2.0 - 1.0), dim=-1)
    return functional.grid_sample(
        image[None, None], grid.reshape(1, -1, 1, 2), mode=mode,
        padding_mode="zeros", align_corners=True)[0, 0, :, 0]


def _owner_segments(problem: MBOPCProblem) -> tuple[np.ndarray, ...]:
    """从既有 membership CSR 一次建立每个 core 唯一拥有的 segment 视图。"""
    result: list[np.ndarray] = []
    for core_index in range(problem.core_count):
        members = problem.segments_for_core(core_index)
        result.append(members[problem.owner_indices[members] == core_index])
    return tuple(result)


def _target_tile(problem: MBOPCProblem, core_index: int, core: object,
                 pixel_dbu: int, canvas: int,
                 cache: ArrayTileCache) -> np.ndarray:
    """读取或生成固定参考 context mask，缓存受显式字节上限约束。"""
    cached = cache.get(core_index)
    if cached is not None:
        return cached
    target = rasterize_region_canvas(
        problem.physical_mask.region, core.context_box, pixel_dbu, canvas)
    # 固定 target 缓存使用 uint8，把用户设置的字节上限真正用于像素数量而不是
    # float32 临时精度；进入连续损失前统一除以 255，量化误差小于 1/255。
    compact = np.rint(target * 255.0).astype(np.uint8)
    cache.put(core_index, compact)
    return compact


def _validate_problem(problem: MBOPCProblem, model: ICCAD13Lithography,
                      config: DiffOPCConfig) -> tuple[object, ...]:
    """在分配优化器和 GPU 张量前校验画布、位移及非空问题。"""
    if config.canvas > model.config.canvas:
        raise ValueError("DiffOPC canvas 不能超过光刻模型 canvas")
    if problem.segments.segment_count <= 0:
        raise ValueError("DiffOPC 问题不包含可优化边段")
    if config.max_displacement_dbu > problem.config.max_displacement_dbu:
        raise ValueError("DiffOPC 位移上限不能超过前端重建上限")
    cores = problem.grid.cores()
    for core in cores:
        width = (core.context_box.width + config.pixel_dbu - 1) // config.pixel_dbu
        height = (core.context_box.height + config.pixel_dbu - 1) // config.pixel_dbu
        if width > config.canvas or height > config.canvas:
            raise ValueError(f"core {core.core_id} context 超过 DiffOPC 固定画布")
    return cores


def optimize(problem: MBOPCProblem, model: ICCAD13Lithography,
             config: DiffOPCConfig) -> DiffOPCResult:
    """流式累计全局位移梯度，并只在轮次屏障后发布合法候选状态。"""
    cores = _validate_problem(problem, model, config)
    geometry = problem.segments.materialize()
    owners = _owner_segments(problem)
    cache = ArrayTileCache(config.target_cache_bytes)
    # ownership 像素和 owner segment 在一次已准备问题内固定。全局分母使损失及
    # 梯度不随 tile 数、batch 大小或 halo membership 重复次数变化；只逐 core
    # 生成一个临时 bool 画布，不把所有 ownership mask 常驻内存。
    owned_pixel_count = sum(int(np.count_nonzero(ownership_canvas(
        core.ownership_box, core.context_box, config.pixel_dbu, config.canvas)))
        for core in cores)
    if owned_pixel_count <= 0:
        raise ValueError("DiffOPC 网格在当前像素尺寸下没有 ownership 像素")
    probe_denominator = float(max(problem.segments.segment_count * 2, 1))
    current = torch.zeros(
        problem.segments.segment_count, device=model.device,
        dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam((current,), lr=config.learning_rate)
    nominal = model.condition("nominal")
    maximum = model.condition("dose_max")
    minimum = model.condition("defocus_min")
    conditions = (nominal, maximum, minimum)
    threshold = model.config.print_threshold
    relative_centers = torch.stack(torch.meshgrid(
        (torch.arange(config.canvas, device=model.device, dtype=torch.float32) + 0.5) * config.pixel_dbu,
        (torch.arange(config.canvas, device=model.device, dtype=torch.float32) + 0.5) * config.pixel_dbu,
        indexing="ij"), dim=-1)[..., [1, 0]]
    best_loss = float("inf")
    best_iteration = 0
    best = np.zeros(problem.segments.segment_count, dtype=np.float64)
    records: list[DiffOPCIterationRecord] = []
    stop_reason = "iteration_limit"
    for iteration in range(config.iterations):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        l2_numerator = pvb_numerator = epe_numerator = 0.0
        binary_l2 = binary_pvb = binary_epe = valid_probes = ambiguous_probes = 0
        for batch_start in range(0, len(cores), config.batch_size):
            batch_indices = range(batch_start, min(len(cores), batch_start + config.batch_size))
            local_info: list[tuple[int, object, np.ndarray, np.ndarray]] = []
            targets: list[np.ndarray] = []
            masks: list[torch.Tensor] = []
            ownership_masks: list[np.ndarray] = []
            for core_index in batch_indices:
                core = cores[core_index]
                members = problem.segments_for_core(core_index)
                target = _target_tile(
                    problem, core_index, core,
                    config.pixel_dbu, config.canvas, cache)
                targets.append(target)
                ownership_masks.append(ownership_canvas(
                    core.ownership_box, core.context_box,
                    config.pixel_dbu, config.canvas))
                masks.append(rasterize_soft_edges(
                    target.astype(np.float32) / 255.0,
                    geometry.starts[members], geometry.ends[members],
                    geometry.normals[members], current[members],
                    pixel_dbu=config.pixel_dbu,
                    temperature=config.soft_temperature,
                    origin_dbu=(core.context_box.left, core.context_box.bottom),
                    chunk_size=config.raster_chunk_size,
                    relative_pixel_centers=relative_centers))
                local_info.append((core_index, core, members, owners[core_index]))
            mask_tensor = torch.stack(masks)
            target_tensor = torch.as_tensor(
                np.stack(targets), device=model.device,
                dtype=torch.float32).div_(255.0)
            owned_tensor = torch.as_tensor(
                np.stack(ownership_masks), device=model.device, dtype=torch.bool)
            printed = model.forward_many(mask_tensor, conditions)
            owned_values = owned_tensor.to(dtype=torch.float32)
            batch_l2 = torch.sum(
                (printed[nominal.name] - target_tensor).square() * owned_values)
            batch_pvb = torch.sum(
                (printed[maximum.name] - printed[minimum.name]).square() * owned_values)
            batch_epe = current.sum() * 0.0
            for local, (core_index, core, _members, owner_ids) in enumerate(local_info):
                if not len(owner_ids):
                    continue
                # 所有 probe 基于固定参考 segment；owner 只出现一次，halo segment
                # 仍参与 mask 光学上下文，却不会第二次产生自身 EPE 梯度或诊断计数。
                inner, outer = edge_probe_points(
                    geometry.starts[owner_ids], geometry.ends[owner_ids],
                    geometry.normals[owner_ids], config.epe_distance_dbu)
                origin = torch.tensor(
                    [core.context_box.left, core.context_box.bottom],
                    device=model.device, dtype=torch.float32)
                image = printed[nominal.name][local]
                inner_values = _sample_probe(
                    image, inner, origin, config.pixel_dbu)
                outer_values = _sample_probe(
                    image, outer, origin, config.pixel_dbu)
                target_inner = _sample_probe(
                    target_tensor[local], inner, origin,
                    config.pixel_dbu, mode="nearest")
                target_outer = _sample_probe(
                    target_tensor[local], outer, origin,
                    config.pixel_dbu, mode="nearest")
                local_inner = (inner - np.array(
                    [core.context_box.left, core.context_box.bottom])) / config.pixel_dbu - 0.5
                local_outer = (outer - np.array(
                    [core.context_box.left, core.context_box.bottom])) / config.pixel_dbu - 0.5
                in_bounds = ((local_inner[:, 0] >= 0.0) &
                             (local_inner[:, 0] <= config.canvas - 1) &
                             (local_inner[:, 1] >= 0.0) &
                             (local_inner[:, 1] <= config.canvas - 1) &
                             (local_outer[:, 0] >= 0.0) &
                             (local_outer[:, 0] <= config.canvas - 1) &
                             (local_outer[:, 1] >= 0.0) &
                             (local_outer[:, 1] <= config.canvas - 1))
                valid = (torch.as_tensor(in_bounds, device=model.device) &
                         (target_inner >= threshold) & (target_outer < threshold))
                batch_epe = batch_epe + torch.sum(
                    torch.where(valid, functional.relu(
                        threshold - inner_values).square(), 0.0))
                batch_epe = batch_epe + torch.sum(
                    torch.where(valid, functional.relu(
                        outer_values - threshold).square(), 0.0))
                with torch.no_grad():
                    diagnostic = evaluate_edge_probes(
                        target_tensor[local], image,
                        torch.zeros(len(owner_ids), device=model.device),
                        torch.as_tensor(local_inner, device=model.device),
                        torch.as_tensor(local_outer, device=model.device), threshold)
                    binary_epe += diagnostic.violation_count
                    valid_probes += int(torch.count_nonzero(diagnostic.valid).item())
                    ambiguous_probes += int(torch.count_nonzero(
                        diagnostic.ambiguous).item())
            # 每个 batch 基于同一只读 current 反传并立即释放光刻图；optimizer.step
            # 仍位于全部 batch 之后，因此这里只有梯度累加，没有边段提前更新。
            batch_loss = (config.weight_l2 * batch_l2 / owned_pixel_count +
                          config.weight_pvband * batch_pvb / owned_pixel_count +
                          config.weight_epe * batch_epe / probe_denominator)
            batch_loss.backward()
            l2_numerator += float(batch_l2.detach())
            pvb_numerator += float(batch_pvb.detach())
            epe_numerator += float(batch_epe.detach())
            with torch.no_grad():
                binary_l2 += evaluate_binary_l2(
                    target_tensor, printed[nominal.name], threshold, owned_tensor)
                binary_pvb += evaluate_pvband(
                    printed[maximum.name], printed[minimum.name], threshold, owned_tensor)
            del (batch_loss, batch_l2, batch_pvb, batch_epe, printed, mask_tensor,
                 target_tensor, owned_tensor, owned_values, masks, targets,
                 ownership_masks, local_info)
        l2_loss = l2_numerator / owned_pixel_count
        pvb_loss = pvb_numerator / owned_pixel_count
        epe_loss = epe_numerator / probe_denominator
        total_loss = (config.weight_l2 * l2_loss +
                      config.weight_pvband * pvb_loss +
                      config.weight_epe * epe_loss)
        # 记录、best 和位移都来自同一个已完整评价快照；不会再用 step 前损失
        # 选择 step 后参数。相同损失保留更早状态，保证确定性。
        snapshot = current.detach().cpu().numpy().astype(np.float64, copy=True)
        if total_loss < best_loss:
            best_loss = total_loss
            best_iteration = iteration
            best = snapshot
        records.append(DiffOPCIterationRecord(
            iteration, total_loss, l2_loss, pvb_loss, epe_loss,
            binary_l2, binary_pvb, binary_epe, valid_probes, ambiguous_probes,
            int(np.count_nonzero(np.abs(snapshot) > 1e-12)),
            perf_counter() - started))
        if total_loss == 0.0:
            stop_reason = "zero_loss"
            break
        # 最后一个配置轮次已完成评价；再执行一个无人评价的 step 既浪费时间，
        # 又可能让调用方误以为该候选属于 best，因此在这里结束。
        if iteration == config.iterations - 1:
            break
        if current.grad is None or not bool(torch.all(torch.isfinite(current.grad)).item()):
            stop_reason = "invalid_gradient"
            break
        if config.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_((current,), config.gradient_clip)
        before_step = current.detach().clone()
        optimizer.step()
        with torch.no_grad():
            current.clamp_(-config.max_displacement_dbu,
                           config.max_displacement_dbu)
        candidate = current.detach().cpu().numpy().astype(np.float64, copy=False)
        if np.allclose(candidate, snapshot, atol=1e-12, rtol=0.0):
            stop_reason = "no_update"
            break
        try:
            # 位移范围只是局部 MRC；全局重建继续检查环翻转、自交、孔洞越界和
            # 对边穿越。失败时整轮候选回滚并停止，不发布半个 Polygon 的状态。
            reconstruct_region(problem, candidate)
        except (ValueError, ReconstructionError):
            with torch.no_grad():
                current.copy_(before_step)
            stop_reason = "invalid_geometry"
            break
    return DiffOPCResult(best, best_iteration, tuple(records), stop_reason)
