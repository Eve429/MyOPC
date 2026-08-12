"""实现分级监督、完整物理光刻仿真的 MultilevelILT。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch

from lithography import ICCAD13Lithography, ProcessCondition

from .simple import (
    ILTIterationRecord,
    SimpleILTResult,
    _curvature_loss,
    _image_batch,
    _resize_image,
    _smooth_sigmoid_mask,
)


@dataclass(frozen=True, slots=True)
class MultilevelConfig:
    """保存 MultilevelILT 每级尺度、迭代、Adam 步长和损失配置。"""

    scales: tuple[int, ...] = (2, 1)
    stage_iterations: tuple[int, ...] = (20, 100)
    stage_step_sizes: tuple[float, ...] = (0.2, 0.2)
    smoothing_kernel: int = 7
    sigmoid_steepness: float = 4.0
    sigmoid_offset: float = 0.5
    weight_process_l2: float = 0.0
    weight_pvband: float = 1.0
    curvature_weight: float = 0.0
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        """在分配层级张量前校验每级配置和全部有限浮点参数。"""
        values = (*self.stage_step_sizes, self.sigmoid_steepness,
                  self.sigmoid_offset, self.weight_process_l2,
                  self.weight_pvband, self.curvature_weight,
                  self.mask_threshold)
        if (not self.scales or len(self.stage_iterations) != len(self.scales) or
                len(self.stage_step_sizes) != len(self.scales)):
            raise ValueError("MultilevelILT 尺度、迭代数和步长数量必须相同且非空")
        if (any(not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0
                for scale in self.scales) or self.scales[-1] != 1 or
                any(left <= right for left, right in zip(self.scales, self.scales[1:]))):
            raise ValueError("MultilevelILT scales 必须是严格递减并以 1 结束的正整数")
        if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0
               for count in self.stage_iterations):
            raise ValueError("MultilevelILT 每级迭代次数必须是正整数")
        if not all(isfinite(value) for value in values):
            raise ValueError("MultilevelILT 浮点配置必须有限")
        if (not isinstance(self.smoothing_kernel, int) or
                isinstance(self.smoothing_kernel, bool) or
                self.smoothing_kernel <= 0 or self.smoothing_kernel % 2 == 0):
            raise ValueError("MultilevelILT smoothing_kernel 必须是正奇数")
        if (any(step <= 0.0 for step in self.stage_step_sizes) or
                self.sigmoid_steepness <= 0.0 or
                not 0.0 <= self.sigmoid_offset <= 1.0 or
                self.weight_process_l2 < 0.0 or self.weight_pvband < 0.0 or
                self.curvature_weight < 0.0 or not 0.0 < self.mask_threshold < 1.0):
            raise ValueError("MultilevelILT 步长、sigmoid 参数、权重或阈值超出范围")


def optimize_multilevel(
        target: torch.Tensor, model: ICCAD13Lithography,
        config: MultilevelConfig,
        initial_parameters: torch.Tensor | None = None,
        optimization_mask: torch.Tensor | None = None,
        nominal_condition: ProcessCondition | None = None,
        process_conditions: Sequence[ProcessCondition] | None = None
        ) -> SimpleILTResult:
    """逐级运行独立 Adam，并以完整物理网格光刻结果监督当前级别。"""
    target_batch, squeeze = _image_batch(target, "target", model.device)
    target_batch = target_batch.detach()
    height, width = target_batch.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("MultilevelILT target 的高度和宽度必须为正")
    if height > model.config.canvas or width > model.config.canvas:
        raise ValueError("MultilevelILT target 超过光刻模型 canvas")
    if not torch.all(torch.isfinite(target_batch)) or torch.any(
            (target_batch < 0.0) | (target_batch > 1.0)):
        raise ValueError("MultilevelILT target 必须为 [0,1] 内的有限数")
    if any(height % scale or width % scale for scale in config.scales):
        raise ValueError("MultilevelILT target 高宽必须能被全部 scale 整除")
    if any(min(height // scale, width // scale) < config.smoothing_kernel
           for scale in config.scales):
        raise ValueError("MultilevelILT 最粗级别不能小于 smoothing_kernel")
    if config.curvature_weight > 0.0 and any(
            min(height // scale, width // scale) < 3 for scale in config.scales):
        raise ValueError("启用曲率正则时每个 MultilevelILT 级别边长不能小于 3")

    if initial_parameters is None:
        # OpenILT Low 级实际以 target 而不是 PixelInit 返回的 [-1,1] 参数启动；这里
        # 保留该可复现实义，同时让调用方可以显式传入别的连续初值。
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
        raise TypeError("MultilevelILT 工艺条件必须是 ProcessCondition")
    all_conditions = (nominal, *process)
    if len({condition.name for condition in all_conditions}) != len(all_conditions):
        raise ValueError("MultilevelILT 工艺条件名称不能重复")

    fixed_full_mask = _smooth_sigmoid_mask(
        initial, config.smoothing_kernel,
        config.sigmoid_steepness, config.sigmoid_offset).detach()
    previous_parameters: torch.Tensor | None = None
    final_parameters = initial.detach().clone()
    final_mask = fixed_full_mask.clone()
    final_iteration = 0
    records: list[ILTIterationRecord] = []
    global_iteration = 0
    for scale, iteration_count, step_size in zip(
            config.scales, config.stage_iterations, config.stage_step_sizes):
        shape = (height // scale, width // scale)
        stage_target = _resize_image(target_batch, shape, "area")
        stage_reference = _resize_image(initial, shape, "area")
        stage_initial = (stage_reference if previous_parameters is None else
                         _resize_image(previous_parameters, shape, "nearest"))
        movable = _resize_image(full_movable, shape, "nearest")
        parameters = stage_initial.detach().clone().requires_grad_(True)
        # OpenILT 在每级重建 Adam，因此低级动量不会污染细级参数；step_size 已保存
        # 实际 Adam 学习率，不再隐藏乘 0.2 的二次换算。
        optimizer = torch.optim.Adam((parameters,), lr=step_size)
        stage_best_loss = float("inf")
        stage_best_parameters = stage_initial.detach().clone()
        stage_best_mask = fixed_full_mask.clone()
        stage_best_iteration = global_iteration
        for _ in range(iteration_count):
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # 参数平滑前固定不可动区，避免邻域卷积从窗口外引入可训练自由度；恢复到
            # 完整网格后再次混合，确保最终输出的固定像素逐点保持初始软 mask。
            effective = parameters * movable + stage_reference * (1.0 - movable)
            stage_mask = _smooth_sigmoid_mask(
                effective, config.smoothing_kernel,
                config.sigmoid_steepness, config.sigmoid_offset)
            optimized_full = _resize_image(stage_mask, (height, width), "nearest")
            full_mask = (optimized_full * full_movable +
                         fixed_full_mask * (1.0 - full_movable))
            # 所有级别都在已标定的完整物理像素网格调用 Hopkins；仿真结果再用 area
            # 汇聚到本级监督网格。这样粗级确实减少损失/参数自由度，却不改变 PSF 尺度。
            printed_full = model.forward_many(full_mask, all_conditions)
            printed = {name: _resize_image(image, shape, "area")
                       for name, image in printed_full.items()}
            # 级别损失只再引用降采样结果；立即释放完整 wafer 字典，避免本轮 backward
            # 之外还由 Python 局部变量延长一套完整图引用的生命周期。
            del printed_full
            nominal_l2 = torch.sum((printed[nominal.name] - stage_target).square())
            if process:
                process_stack = torch.stack(tuple(printed[item.name] for item in process))
                process_l2 = torch.sum(
                    (process_stack - stage_target.unsqueeze(0)).square())
                pvband = torch.sum((torch.amax(process_stack, dim=0) -
                                    torch.amin(process_stack, dim=0)).square())
            else:
                process_l2 = nominal_l2.new_zeros(())
                pvband = nominal_l2.new_zeros(())
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
                stage_best_mask = full_mask.detach().clone()
            loss.backward()
            optimizer.step()
            records.append(ILTIterationRecord(
                global_iteration, *values, perf_counter() - started))
            global_iteration += 1
        # 每级只向后传递历史最优有效参数，不保留旧 Adam 状态、wafer 或 autograd 图；
        # 最终 scale=1，故统一结果仍与完整 target 同形。
        previous_parameters = stage_best_parameters
        final_parameters = stage_best_parameters
        final_mask = stage_best_mask
        final_iteration = stage_best_iteration
        del optimizer, parameters
    binary = final_mask >= config.mask_threshold
    if squeeze:
        final_parameters, final_mask, binary = (
            final_parameters[0], final_mask[0], binary[0])
    return SimpleILTResult(
        final_parameters, final_mask, binary, final_iteration, tuple(records))
