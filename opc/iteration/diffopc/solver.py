"""流式执行独立的可微边段 OPC。"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from evaluation import evaluate_pvband
from lithography import ICCAD13Lithography
from opc.input.edge import MBOPCProblem, edge_probe_points
from opc.input.raster import rasterize_region_canvas

from .contracts import DiffOPCConfig, DiffOPCIterationRecord, DiffOPCResult
from .rasterizer import rasterize_soft_edges


def _sample_probe(image: torch.Tensor, points: np.ndarray, origin: torch.Tensor,
                  pixel_dbu: int) -> torch.Tensor:
    """在一个 batch tile 上对 DBU probe 做可微双线性采样。"""
    xy = (torch.as_tensor(points, device=image.device, dtype=torch.float32) - origin)
    xy = xy / max(float(pixel_dbu), 1.0)
    height, width = image.shape[-2:]
    grid = torch.stack((xy[:, 0] / max(width - 1, 1) * 2.0 - 1.0,
                        xy[:, 1] / max(height - 1, 1) * 2.0 - 1.0), dim=-1)
    return torch.nn.functional.grid_sample(
        image[None, None], grid.reshape(1, -1, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True)[0, 0, :, 0]


def optimize(problem: MBOPCProblem, model: ICCAD13Lithography,
             config: DiffOPCConfig) -> DiffOPCResult:
    """按 owner 唯一写入原则反传边段位移并返回最佳状态。"""
    if config.canvas > model.config.canvas:
        raise ValueError("DiffOPC canvas 不能超过光刻模型 canvas")
    cores = problem.grid.cores()
    geometry = problem.segments.materialize()
    current = torch.zeros(problem.segments.segment_count, device=model.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam((current,), lr=config.learning_rate)
    best_loss, best_iteration = float("inf"), 0
    best = current.detach().cpu().numpy().astype(np.float64, copy=True)
    records: list[DiffOPCIterationRecord] = []
    nominal = model.condition("nominal")
    maximum = model.condition("dose_max")
    minimum = model.condition("defocus_min")
    for iteration in range(config.iterations):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=model.device)
        l2_value = pvband_value = epe_value = 0.0
        for batch_start in range(0, len(cores), config.batch_size):
            batch = cores[batch_start:batch_start + config.batch_size]
            masks, local_info = [], []
            for core_index, core in enumerate(batch, batch_start):
                members = problem.segments_for_core(core_index)
                masks.append(rasterize_region_canvas(problem.physical_mask.region, core.context_box, config.pixel_dbu, config.canvas))
                local_info.append((core_index, core, members))
            mask_tensor = []
            for core_index, core, members in local_info:
                mask_tensor.append(rasterize_soft_edges(
                    masks[core_index - batch_start], geometry.starts[members], geometry.ends[members],
                    geometry.normals[members], current[members], pixel_dbu=config.pixel_dbu,
                    temperature=config.soft_temperature, origin_dbu=(core.context_box.left, core.context_box.bottom)))
            soft_mask = torch.stack(mask_tensor)
            printed = model.forward_many(soft_mask, (nominal, maximum, minimum))
            target = torch.as_tensor(np.stack(masks), device=model.device, dtype=torch.float32)
            batch_l2 = torch.sum((printed[nominal.name] - target).square())
            batch_pvb = evaluate_pvband(printed[maximum.name], printed[minimum.name], model.config.print_threshold)
            epe_loss = torch.zeros((), device=model.device)
            for local, (core_index, core, members) in enumerate(local_info):
                if not len(members):
                    continue
                inner, outer = edge_probe_points(
                    geometry.starts[members], geometry.ends[members], geometry.normals[members],
                    max(float(config.pixel_dbu), 1.0))
                origin = torch.tensor(
                    [core.context_box.left, core.context_box.bottom],
                    device=model.device, dtype=torch.float32)
                image = printed[nominal.name][local]
                inner_value = _sample_probe(image, inner, origin, config.pixel_dbu)
                outer_value = _sample_probe(image, outer, origin, config.pixel_dbu)
                epe_loss = epe_loss + torch.relu(0.5 - inner_value).square().mean()
                epe_loss = epe_loss + torch.relu(outer_value - 0.5).square().mean()
                epe_value += float((torch.relu(0.5 - inner_value).sum() +
                                    torch.relu(outer_value - 0.5).sum()).detach())
            total = total + config.weight_l2 * batch_l2 + config.weight_pvband * batch_pvb + config.weight_epe * epe_loss
            l2_value += float(batch_l2.detach()); pvband_value += float(batch_pvb)
        total.backward()
        if config.gradient_clip:
            torch.nn.utils.clip_grad_norm_((current,), config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            current.clamp_(-config.max_displacement_dbu, config.max_displacement_dbu)
        value = float(total.detach())
        if value < best_loss:
            best_loss, best_iteration = value, iteration
            best = current.detach().cpu().numpy().astype(np.float64, copy=True)
        moved = int(np.count_nonzero(np.abs(current.detach().cpu().numpy()) > 1e-12))
        records.append(DiffOPCIterationRecord(iteration, value, l2_value,
                                              pvband_value, epe_value, moved,
                                              perf_counter() - started))
    return DiffOPCResult(best, best_iteration, tuple(records))
