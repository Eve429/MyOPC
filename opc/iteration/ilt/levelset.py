"""实现基于水平集参数化的可微 ILT。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.nn import functional

from lithography import ICCAD13Lithography, ProcessCondition

from .simple import ILTIterationRecord, SimpleILTResult, _image_batch


class _LevelSetBinarize(torch.autograd.Function):
    """用水平集梯度幅值近似硬二值边界的代理反向。"""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                levelset: torch.Tensor) -> torch.Tensor:
        """前向输出 phi 小于零的开窗掩膜。"""
        ctx.save_for_backward(levelset)
        return (levelset < 0.0).to(levelset.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """把梯度集中到 phi 的离散零等值线附近。"""
        (levelset,) = ctx.saved_tensors
        padded = functional.pad(levelset[:, None], (1, 1, 1, 1), mode="replicate")
        dx = (padded[:, 0, 1:-1, 2:] - padded[:, 0, 1:-1, :-2]) * 0.5
        dy = (padded[:, 0, 2:, 1:-1] - padded[:, 0, :-2, 1:-1]) * 0.5
        return (-torch.sqrt(dx.square() + dy.square() + 1e-12) * grad_output,)


@dataclass(frozen=True, slots=True)
class LevelSetConfig:
    """保存水平集 ILT 的迭代、损失和二值阈值配置。"""

    iterations: int = 20
    step_size: float = 0.2
    weight_process_l2: float = 1.0
    weight_pvband: float = 0.0
    curvature_weight: float = 0.0
    mask_threshold: float = 0.5


def optimize_levelset(target: torch.Tensor, model: ICCAD13Lithography,
                      config: LevelSetConfig,
                      initial_levelset: torch.Tensor | None = None,
                      optimization_mask: torch.Tensor | None = None,
                      process_conditions: tuple[ProcessCondition, ...] | None = None
                      ) -> SimpleILTResult:
    """优化水平集 phi，并返回与 SimpleILT 相同的结果契约。"""
    target_batch, squeeze = _image_batch(target, "target", model.device)
    target_batch = target_batch.detach()
    if initial_levelset is None:
        parameters = target_batch.mul(-2.0).add(1.0)
    else:
        parameters, initial_squeeze = _image_batch(initial_levelset, "initial_levelset", model.device)
        if initial_squeeze != squeeze or parameters.shape != target_batch.shape:
            raise ValueError("initial_levelset 必须与 target 形状一致")
    if optimization_mask is None:
        movable = torch.ones_like(target_batch)
    else:
        movable, mask_squeeze = _image_batch(optimization_mask, "optimization_mask", model.device)
        if mask_squeeze != squeeze or movable.shape != target_batch.shape:
            raise ValueError("optimization_mask 必须与 target 形状一致")
        movable = movable.detach().clamp(0.0, 1.0)
    conditions = process_conditions or (model.condition("dose_max"), model.condition("defocus_min"))
    nominal = model.condition("nominal")
    fixed = parameters.detach().clone()
    parameters = parameters.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((parameters,), lr=config.step_size)
    best_loss, best_iteration = float("inf"), 0
    best_parameters, best_mask = parameters.detach().clone(), torch.zeros_like(parameters)
    records: list[ILTIterationRecord] = []
    for iteration in range(config.iterations):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        phi = parameters * movable + fixed * (1.0 - movable)
        mask = _LevelSetBinarize.apply(phi)
        printed = model.forward_many(mask, (nominal, *conditions))
        nominal_l2 = torch.sum((printed[nominal.name] - target_batch).square())
        process_stack = torch.stack(tuple(printed[item.name] for item in conditions))
        process_l2 = torch.sum((process_stack - target_batch.unsqueeze(0)).square())
        pvband = torch.sum((torch.amax(process_stack, 0) - torch.amin(process_stack, 0)).square())
        curvature = torch.zeros((), device=model.device)
        if config.curvature_weight:
            curvature = torch.sum(functional.avg_pool2d(mask[:, None], 3, 1, 1)[:, 0].square())
        loss = nominal_l2 + config.weight_process_l2 * process_l2 + config.weight_pvband * pvband + config.curvature_weight * curvature
        values = (float(loss.detach()), float(nominal_l2.detach()), float(process_l2.detach()),
                  float(pvband.detach()), float(curvature.detach()))
        if values[0] < best_loss:
            best_loss, best_iteration = values[0], iteration
            best_parameters, best_mask = phi.detach().clone(), mask.detach().clone()
        loss.backward(); optimizer.step()
        records.append(ILTIterationRecord(iteration, *values, perf_counter() - started))
    binary = best_mask >= config.mask_threshold
    if squeeze:
        best_parameters, best_mask, binary = best_parameters[0], best_mask[0], binary[0]
    return SimpleILTResult(best_parameters, best_mask, binary, best_iteration, tuple(records))
