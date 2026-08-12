"""实现连续平滑参数化和粗到细 warm-start 的 CurvMultiILT。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch

from lithography import LithographyModel, ProcessCondition

from .simple import (
    ILTIterationRecord,
    SimpleILTResult,
    _curvature_loss,
    _image_batch,
    _resize_image,
    _smooth_sigmoid_mask,
)


@dataclass(frozen=True, slots=True)
class CurvMultiConfig:
    """保存 CurvMultiILT 的尺度、连续参数化、损失和优化配置。"""

    scales: tuple[int, ...] = (4, 2, 1)
    iterations_per_stage: int = 10
    step_size: float = 0.5
    smoothing_kernel: int = 7
    sigmoid_steepness: float = 4.0
    sigmoid_offset: float = 0.5
    weight_process_l2: float = 0.0
    weight_pvband: float = 1.0
    curvature_weight: float = 200.0
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        """在分配多尺度张量前拒绝无效尺度、平滑核和浮点配置。"""
        values = (self.step_size, self.sigmoid_steepness, self.sigmoid_offset,
                  self.weight_process_l2, self.weight_pvband,
                  self.curvature_weight, self.mask_threshold)
        if self.iterations_per_stage <= 0 or not all(isfinite(value) for value in values):
            raise ValueError("CurvMultiILT 迭代次数和浮点配置必须有效且有限")
        if (not self.scales or any(not isinstance(scale, int) or scale <= 0
                                   for scale in self.scales)):
            raise ValueError("CurvMultiILT scales 必须包含正整数")
        if self.scales[-1] != 1 or any(
                left <= right for left, right in zip(self.scales, self.scales[1:])):
            raise ValueError("CurvMultiILT scales 必须严格递减并以 1 结束")
        if (not isinstance(self.smoothing_kernel, int) or self.smoothing_kernel <= 0 or
                self.smoothing_kernel % 2 == 0):
            raise ValueError("CurvMultiILT smoothing_kernel 必须是正奇数")
        if (self.step_size <= 0.0 or self.sigmoid_steepness <= 0.0 or
                not 0.0 <= self.sigmoid_offset <= 1.0 or
                self.weight_process_l2 < 0.0 or self.weight_pvband < 0.0 or
                self.curvature_weight < 0.0 or not 0.0 < self.mask_threshold < 1.0):
            raise ValueError("CurvMultiILT 步长、sigmoid 参数、权重或阈值超出范围")


def optimize_curvmulti(
        target: torch.Tensor, model: LithographyModel,
        config: CurvMultiConfig,
        initial_parameters: torch.Tensor | None = None,
        optimization_mask: torch.Tensor | None = None,
        nominal_condition: ProcessCondition | None = None,
        process_conditions: Sequence[ProcessCondition] | None = None
        ) -> SimpleILTResult:
    """按粗到细尺度优化连续平滑参数，并返回最终尺度的统一 ILT 结果。"""
    target_batch, squeeze = _image_batch(target, "target", model.device)
    target_batch = target_batch.detach()
    height, width = target_batch.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("CurvMultiILT target 的高度和宽度必须为正")
    if height > model.config.canvas or width > model.config.canvas:
        raise ValueError("CurvMultiILT target 超过光刻模型 canvas")
    if not torch.all(torch.isfinite(target_batch)) or torch.any(
            (target_batch < 0.0) | (target_batch > 1.0)):
        raise ValueError("CurvMultiILT target 必须为 [0,1] 内的有限数")
    if any(height % scale or width % scale for scale in config.scales):
        raise ValueError("CurvMultiILT target 高宽必须能被全部 scale 整除")
    if any(min(height // scale, width // scale) < config.smoothing_kernel
           for scale in config.scales):
        raise ValueError("CurvMultiILT 最粗尺度不能小于 smoothing_kernel")
    if config.curvature_weight > 0.0 and min(height, width) < 3:
        raise ValueError("启用曲率正则时 CurvMultiILT 图像边长不能小于 3")

    if initial_parameters is None:
        # OpenILT CurvMulti 的 sigmoid offset 为 0.5，直接使用 [0,1] target 作为
        # 参数初值可在平滑后形成对称软边；无需复制 LevelSet 的 SDF 表示。
        initial = target_batch
    else:
        initial, initial_squeeze = _image_batch(
            initial_parameters, "initial_parameters", model.device)
        if initial_squeeze != squeeze or initial.shape != target_batch.shape:
            raise ValueError("initial_parameters 必须与 target 形状一致")
        if not torch.all(torch.isfinite(initial)):
            raise ValueError("initial_parameters 不能包含 NaN 或 Inf")
        initial = initial.detach()
    if optimization_mask is None:
        full_movable = torch.ones_like(target_batch)
    else:
        full_movable, movable_squeeze = _image_batch(
            optimization_mask, "optimization_mask", model.device)
        if movable_squeeze != squeeze or full_movable.shape != target_batch.shape:
            raise ValueError("optimization_mask 必须与 target 形状一致")
        if not torch.all(torch.isfinite(full_movable)) or torch.any(
                (full_movable < 0.0) | (full_movable > 1.0)):
            raise ValueError("optimization_mask 必须为 [0,1] 内的有限数")
        full_movable = full_movable.detach()

    nominal = model.condition("nominal") if nominal_condition is None else nominal_condition
    process = (tuple(process_conditions) if process_conditions is not None else
               (model.condition("dose_max"), model.condition("defocus_min")))
    if not isinstance(nominal, ProcessCondition) or any(
            not isinstance(condition, ProcessCondition) for condition in process):
        raise TypeError("CurvMultiILT 工艺条件必须是 ProcessCondition")
    all_conditions = (nominal, *process)
    if len({condition.name for condition in all_conditions}) != len(all_conditions):
        raise ValueError("CurvMultiILT 工艺条件名称不能重复")

    previous_parameters: torch.Tensor | None = None
    fixed_full_mask = _smooth_sigmoid_mask(
        initial, config.smoothing_kernel,
        config.sigmoid_steepness, config.sigmoid_offset).detach()
    final_parameters = initial.detach().clone()
    final_mask = fixed_full_mask.clone()
    final_iteration = 0
    records: list[ILTIterationRecord] = []
    global_iteration = 0
    for scale in config.scales:
        shape = (height // scale, width // scale)
        stage_reference = _resize_image(initial, shape, "area")
        # 当前尺度参考用 area 保持覆盖率；跨阶段参数按 OpenILT 使用 nearest
        # warm-start，不在阶段间凭空引入新灰度。窗口也用 nearest 保持边界明确。
        stage_initial = (stage_reference if previous_parameters is None else
                         _resize_image(previous_parameters, shape, "nearest"))
        movable = _resize_image(full_movable, shape, "nearest")
        parameters = stage_initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD((parameters,), lr=config.step_size)
        stage_best_loss = float("inf")
        stage_best_parameters = stage_initial.detach().clone()
        stage_best_mask = fixed_full_mask.clone()
        stage_best_iteration = global_iteration
        for _ in range(config.iterations_per_stage):
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # 在平滑前钉住固定区，防止不可动参数通过 7×7 邻域影响可动边界；平滑后
            # 再混合一次 fixed_mask，保证窗口外输出逐像素保持本尺度参考值。
            effective = parameters * movable + stage_reference * (1.0 - movable)
            stage_mask = _smooth_sigmoid_mask(
                effective, config.smoothing_kernel,
                config.sigmoid_steepness, config.sigmoid_offset)
            # Hopkins 核只在模型固定像素网格上具有既定物理含义；粗尺度只减少
            # 参数自由度，不能把小图直接补零送入模型，否则图形物理尺寸会缩小。
            # 因此每轮把控制网格的 mask 近邻恢复到完整网格后再仿真和计算损失。
            optimized_mask = _resize_image(stage_mask, (height, width), "nearest")
            mask = optimized_mask * full_movable + fixed_full_mask * (1.0 - full_movable)
            printed = model.forward_many(mask, all_conditions)
            # OpenILT CurvMulti 源码把 printedMax 误作 nominal；这里按具名条件计算，
            # 避免 nominal L2 与第一个 process 条件重复计权。
            nominal_l2 = torch.sum((printed[nominal.name] - target_batch).square())
            if process:
                process_stack = torch.stack(tuple(printed[item.name] for item in process))
                process_l2 = torch.sum(
                    (process_stack - target_batch.unsqueeze(0)).square())
                pvband = torch.sum((torch.amax(process_stack, dim=0) -
                                    torch.amin(process_stack, dim=0)).square())
            else:
                process_l2 = nominal_l2.new_zeros(())
                pvband = nominal_l2.new_zeros(())
            # CurvMulti 的曲率作用于 nominal wafer，而不是输入 mask；这正是它与
            # SimpleILT/LevelSetILT 的主要算法差异，并复用同一零和离散曲率核。
            curvature = (_curvature_loss(printed[nominal.name])
                         if config.curvature_weight > 0.0 else nominal_l2.new_zeros(()))
            loss = (nominal_l2 + config.weight_process_l2 * process_l2 +
                    config.weight_pvband * pvband +
                    config.curvature_weight * curvature)
            values = (float(loss.detach()), float(nominal_l2.detach()),
                      float(process_l2.detach()), float(pvband.detach()),
                      float(curvature.detach()))
            if values[0] < stage_best_loss:
                stage_best_loss = values[0]
                stage_best_iteration = global_iteration
                stage_best_parameters = effective.detach().clone()
                stage_best_mask = mask.detach().clone()
            loss.backward()
            optimizer.step()
            records.append(ILTIterationRecord(
                global_iteration, *values, perf_counter() - started))
            global_iteration += 1
        # 只把本阶段最优有效参数带入下一尺度，丢弃 optimizer 和 autograd 图；因此
        # 常驻设备内存由当前阶段 O(BHW) 决定，不累计所有尺度的图像或优化器状态。
        previous_parameters = stage_best_parameters
        final_parameters = stage_best_parameters
        final_mask = stage_best_mask
        final_iteration = stage_best_iteration
    binary = final_mask >= config.mask_threshold
    if squeeze:
        final_parameters, final_mask, binary = (
            final_parameters[0], final_mask[0], binary[0])
    return SimpleILTResult(
        final_parameters, final_mask, binary, final_iteration, tuple(records))
