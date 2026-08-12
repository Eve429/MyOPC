"""实现可替换工艺条件、可限制优化区域的基础梯度 ILT。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch

from lithography import LithographyModel, ProcessCondition

from ._common import curvature_loss, image_batch


@dataclass(frozen=True, slots=True)
class SimpleILTConfig:
    """保存 SimpleILT 当前实际使用的优化参数。"""

    iterations: int
    step_size: float
    sigmoid_steepness: float = 4.0
    weight_pvband: float = 0.0
    weight_process_l2: float = 1.0
    curvature_weight: float = 0.0
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        """在分配优化张量前拒绝空迭代、非有限权重和无效阈值。"""
        values = (
            self.step_size, self.sigmoid_steepness, self.weight_pvband,
            self.weight_process_l2, self.curvature_weight, self.mask_threshold,
        )
        if self.iterations <= 0 or not all(isfinite(value) for value in values):
            raise ValueError("ILT 迭代次数和所有浮点配置必须有效且有限")
        if (self.step_size <= 0.0 or self.sigmoid_steepness <= 0.0 or
                self.weight_pvband < 0.0 or self.weight_process_l2 < 0.0 or
                self.curvature_weight < 0.0 or not 0.0 < self.mask_threshold < 1.0):
            raise ValueError("ILT 步长、陡度、权重或阈值超出有效范围")


@dataclass(frozen=True, slots=True)
class ILTIterationRecord:
    """保存一轮连续损失和耗时，便于判断优化是否收敛。"""

    iteration: int
    total_loss: float
    nominal_l2: float
    process_l2: float
    pvband_loss: float
    curvature_loss: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SimpleILTResult:
    """保存总损失最优轮的参数、软掩膜、二值掩膜和记录。"""

    best_parameters: torch.Tensor
    soft_mask: torch.Tensor
    binary_mask: torch.Tensor
    best_iteration: int
    records: tuple[ILTIterationRecord, ...]


def optimize(target: torch.Tensor, model: LithographyModel,
             config: SimpleILTConfig, initial_parameters: torch.Tensor | None = None,
             optimization_mask: torch.Tensor | None = None,
             nominal_condition: ProcessCondition | None = None,
             process_conditions: Sequence[ProcessCondition] | None = None) -> SimpleILTResult:
    """优化连续像素参数，使独立工艺条件下的 wafer 接近目标图。"""
    target_batch, squeeze = image_batch(target, "target", model.device)
    # target 是固定监督，不应把调用方可能携带的计算图带进每轮 backward；提前
    # detach 同时避免无意义的 target.grad 累积和跨轮图引用。
    target_batch = target_batch.detach()
    height, width = target_batch.shape[-2:]
    if height > model.config.canvas or width > model.config.canvas:
        raise ValueError("ILT target 超过光刻模型 canvas")
    if initial_parameters is None:
        # OpenILT 的像素初始化采用目标 0/1 到参数 -1/+1 的直接映射；保留该基线
        # 行为，使 sigmoid 后初始 mask 已接近目标，同时不把初始化器另拆成无必要文件。
        initial = target_batch.mul(2.0).sub(1.0)
    else:
        initial, initial_squeeze = image_batch(
            initial_parameters, "initial_parameters", model.device)
        if initial.shape != target_batch.shape or initial_squeeze != squeeze:
            raise ValueError("initial_parameters 必须与 target 形状一致")
    if optimization_mask is None:
        movable = torch.ones_like(target_batch)
    else:
        movable, movable_squeeze = image_batch(
            optimization_mask, "optimization_mask", model.device)
        if movable.shape != target_batch.shape or movable_squeeze != squeeze:
            raise ValueError("optimization_mask 必须与 target 形状一致")
        if torch.any((movable < 0.0) | (movable > 1.0)):
            raise ValueError("optimization_mask 必须位于 [0,1]")
        movable = movable.detach()
    if config.curvature_weight > 0.0 and (height < 3 or width < 3):
        raise ValueError("启用曲率正则时 ILT 图像边长不能小于 3")
    nominal = model.condition("nominal") if nominal_condition is None else nominal_condition
    process = tuple(process_conditions) if process_conditions is not None else (
        model.condition("dose_max"), model.condition("defocus_min"))
    if not isinstance(nominal, ProcessCondition) or any(
            not isinstance(condition, ProcessCondition) for condition in process):
        raise TypeError("ILT 工艺条件必须是 ProcessCondition")
    all_conditions = (nominal, *process)
    if len({condition.name for condition in all_conditions}) != len(all_conditions):
        raise ValueError("ILT 工艺条件名称不能重复")

    # 固定区使用初始参数对应的软 mask；优化变量仅在 movable 区域参与输出。这里不对
    # 参数逐轮 clamp，避免截断梯度；sigmoid 自然把送入光刻模型的 mask 限制到 [0,1]。
    fixed_soft = torch.sigmoid(config.sigmoid_steepness * initial).detach()
    parameters = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD((parameters,), lr=config.step_size)
    best_loss = float("inf")
    best_iteration = 0
    best_parameters = parameters.detach().clone()
    best_mask = fixed_soft.clone()
    records: list[ILTIterationRecord] = []
    for iteration in range(config.iterations):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimized_soft = torch.sigmoid(config.sigmoid_steepness * parameters)
        mask = optimized_soft * movable + fixed_soft * (1.0 - movable)
        printed = model.forward_many(mask, all_conditions)
        nominal_l2 = torch.sum((printed[nominal.name] - target_batch).square())
        if process:
            process_stack = torch.stack(tuple(printed[item.name] for item in process))
            process_l2 = torch.sum((process_stack - target_batch.unsqueeze(0)).square())
            # 任意数量独立条件用逐像素范围表达连续工艺窗，不再把固定 max/min 三元组
            # 写入接口；只有调用者传入的条件参与该项，可用于 ILT 或梯度 OPC 实验。
            pvband_loss = torch.sum(
                (torch.amax(process_stack, dim=0) -
                 torch.amin(process_stack, dim=0)).square())
        else:
            process_l2 = nominal_l2.new_zeros(())
            pvband_loss = nominal_l2.new_zeros(())
        curvature_value = (curvature_loss(mask) if config.curvature_weight > 0.0 else
                          nominal_l2.new_zeros(()))
        loss = (nominal_l2 + config.weight_process_l2 * process_l2 +
                config.weight_pvband * pvband_loss +
                config.curvature_weight * curvature_value)
        values = (
            float(loss.detach().item()), float(nominal_l2.detach().item()),
            float(process_l2.detach().item()), float(pvband_loss.detach().item()),
            float(curvature_value.detach().item()),
        )
        if values[0] < best_loss:
            best_loss = values[0]
            best_iteration = iteration
            best_parameters = parameters.detach().clone()
            best_mask = mask.detach().clone()
        loss.backward()
        optimizer.step()
        records.append(ILTIterationRecord(
            iteration, *values, perf_counter() - started))
    binary = best_mask >= config.mask_threshold
    if squeeze:
        best_parameters, best_mask, binary = (
            best_parameters[0], best_mask[0], binary[0])
    return SimpleILTResult(
        best_parameters, best_mask, binary, best_iteration, tuple(records))
