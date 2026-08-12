"""实现基于水平集参数化的可微 ILT。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional

from lithography import ICCAD13Lithography, ProcessCondition

from .simple import ILTIterationRecord, SimpleILTResult, _curvature_loss, _image_batch


class _LevelSetBinarize(torch.autograd.Function):
    """为硬二值前向提供受水平集空间梯度调制的代理反向。"""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                levelset: torch.Tensor) -> torch.Tensor:
        """前向输出 phi 小于零的开窗掩膜。"""
        ctx.save_for_backward(levelset)
        return (levelset < 0.0).to(levelset.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """按负空间梯度幅值缩放光刻损失对硬 mask 的上游梯度。"""
        (levelset,) = ctx.saved_tensors
        padded = functional.pad(levelset[:, None], (1, 1, 1, 1), mode="replicate")
        dx = (padded[:, 0, 1:-1, 2:] - padded[:, 0, 1:-1, :-2]) * 0.5
        dy = (padded[:, 0, 2:, 1:-1] - padded[:, 0, :-2, 1:-1]) * 0.5
        return (-torch.sqrt(dx.square() + dy.square()) * grad_output,)


@dataclass(frozen=True, slots=True)
class LevelSetConfig:
    """保存水平集 ILT 的迭代步长和损失权重。"""

    iterations: int = 20
    step_size: float = 0.2
    weight_process_l2: float = 1.0
    weight_pvband: float = 0.0
    curvature_weight: float = 0.0

    def __post_init__(self) -> None:
        """在分配水平集和 Adam 状态前拒绝无效迭代与损失权重。"""
        values = (self.step_size, self.weight_process_l2, self.weight_pvband,
                  self.curvature_weight)
        if self.iterations <= 0 or not all(isfinite(value) for value in values):
            raise ValueError("LevelSetILT 迭代次数和浮点配置必须有效且有限")
        if (self.step_size <= 0.0 or self.weight_process_l2 < 0.0 or
                self.weight_pvband < 0.0 or self.curvature_weight < 0.0):
            raise ValueError("LevelSetILT 步长或权重超出有效范围")


def _distance_transform_1d(values: np.ndarray, result: np.ndarray,
                           sites: np.ndarray, boundaries: np.ndarray) -> None:
    """用复用工作区的线性下包络算法写入一维精确平方距离。"""
    length = len(values)
    count = 0
    sites[0], boundaries[0], boundaries[1] = 0, -np.inf, np.inf
    for index in range(1, length):
        intersection = ((values[index] + index * index) -
                        (values[sites[count]] + sites[count] * sites[count]))
        intersection /= 2.0 * (index - sites[count])
        while intersection <= boundaries[count]:
            count -= 1
            intersection = ((values[index] + index * index) -
                            (values[sites[count]] + sites[count] * sites[count]))
            intersection /= 2.0 * (index - sites[count])
        count += 1
        sites[count], boundaries[count], boundaries[count + 1] = index, intersection, np.inf
    count = 0
    for index in range(length):
        while boundaries[count + 1] < index:
            count += 1
        delta = index - sites[count]
        result[index] = delta * delta + values[sites[count]]


def _euclidean_distance(binary_features: np.ndarray) -> np.ndarray:
    """返回每个像素到最近 True 像素中心的精确欧氏距离。"""
    height, width = binary_features.shape
    # 有限上界必须大于画布内任意平方距离，避免 inf-inf 进入下包络交点。
    # 两次一维变换的临时内存均为 O(HW)，初始化只执行一次，不进入迭代热路径。
    infinity = float(height * height + width * width + 1)
    work = np.where(binary_features, 0.0, infinity)
    horizontal = np.empty_like(work)
    sites = np.empty(width, dtype=np.int64)
    boundaries = np.empty(width + 1, dtype=np.float64)
    for row in range(height):
        _distance_transform_1d(work[row], horizontal[row], sites, boundaries)
    # 第一遍输入和横向工作区不会同时参与第二遍；主动释放并复用一维 scratch，
    # 使初始化峰值接近两个 float64 画布，而不是保留全部中间数组到函数返回。
    del work, sites, boundaries
    squared = np.empty_like(horizontal)
    sites = np.empty(height, dtype=np.int64)
    boundaries = np.empty(height + 1, dtype=np.float64)
    for column in range(width):
        _distance_transform_1d(
            horizontal[:, column], squared[:, column], sites, boundaries)
    np.sqrt(squared, out=squared)
    return squared


def signed_distance_initialization(target: torch.Tensor,
                                   threshold: float = 0.5) -> torch.Tensor:
    """从二值目标构造前景为负、背景为正的精确像素中心 SDF。"""
    if target.ndim not in (2, 3):
        raise ValueError("LevelSet target 必须具有 [H,W] 或 [B,H,W] 形状")
    if target.shape[-2] <= 0 or target.shape[-1] <= 0:
        raise ValueError("LevelSet target 的高度和宽度必须为正")
    if not isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("LevelSet SDF 阈值必须是 (0,1) 内的有限数")
    if not torch.all(torch.isfinite(target)):
        raise ValueError("LevelSet target 不能包含 NaN 或 Inf")
    batch = target.unsqueeze(0) if target.ndim == 2 else target
    binary = batch.detach().cpu().numpy() >= threshold
    fields = np.empty(binary.shape, dtype=np.float32)
    for index, image in enumerate(binary):
        if np.all(image):
            field = -np.full(image.shape, max(image.shape), dtype=np.float64)
        elif not np.any(image):
            field = np.full(image.shape, max(image.shape), dtype=np.float64)
        else:
            field = _euclidean_distance(image)
            inside = _euclidean_distance(~image)
            field[image] = -inside[image]
        # 预分配 batch 输出，避免 list 中常驻每张结果后再由 np.stack 整体复制；
        # float64 仅服务距离计算，跨 CPU/GPU 边界的权威初值统一为 float32。
        fields[index] = field
    result = torch.as_tensor(fields, device=target.device)
    return result[0] if target.ndim == 2 else result


def optimize_levelset(target: torch.Tensor, model: ICCAD13Lithography,
                      config: LevelSetConfig,
                      initial_levelset: torch.Tensor | None = None,
                      optimization_mask: torch.Tensor | None = None,
                      nominal_condition: ProcessCondition | None = None,
                      process_conditions: Sequence[ProcessCondition] | None = None
                      ) -> SimpleILTResult:
    """优化水平集 phi，并返回与 SimpleILT 相同的结果契约。"""
    target_batch, squeeze = _image_batch(target, "target", model.device)
    target_batch = target_batch.detach()
    if not torch.all(torch.isfinite(target_batch)) or torch.any(
            (target_batch < 0.0) | (target_batch > 1.0)):
        raise ValueError("LevelSetILT target 必须为 [0,1] 内的有限数")
    height, width = target_batch.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("LevelSetILT target 的高度和宽度必须为正")
    if height > model.config.canvas or width > model.config.canvas:
        raise ValueError("LevelSetILT target 超过光刻模型 canvas")
    if initial_levelset is None:
        # OpenILT 的代理梯度依赖 |grad(phi)|；±1 二值初值只在一层像素上有梯度，
        # 因此在 CPU 一次性生成精确 SDF，再传回模型设备，迭代中不重复计算。
        parameters, _ = _image_batch(
            signed_distance_initialization(target_batch), "initial_levelset", model.device)
    else:
        parameters, initial_squeeze = _image_batch(initial_levelset, "initial_levelset", model.device)
        if initial_squeeze != squeeze or parameters.shape != target_batch.shape:
            raise ValueError("initial_levelset 必须与 target 形状一致")
        if not torch.all(torch.isfinite(parameters)):
            raise ValueError("initial_levelset 不能包含 NaN 或 Inf")
    if optimization_mask is None:
        movable = torch.ones_like(target_batch)
    else:
        movable, mask_squeeze = _image_batch(optimization_mask, "optimization_mask", model.device)
        if mask_squeeze != squeeze or movable.shape != target_batch.shape:
            raise ValueError("optimization_mask 必须与 target 形状一致")
        if not torch.all(torch.isfinite(movable)) or torch.any(
                (movable < 0.0) | (movable > 1.0)):
            raise ValueError("optimization_mask 必须为 [0,1] 内的有限数")
        movable = movable.detach()
    if config.curvature_weight > 0.0 and (height < 3 or width < 3):
        raise ValueError("启用曲率正则时 LevelSetILT 图像边长不能小于 3")
    conditions = (tuple(process_conditions) if process_conditions is not None else
                  (model.condition("dose_max"), model.condition("defocus_min")))
    nominal = model.condition("nominal") if nominal_condition is None else nominal_condition
    if not isinstance(nominal, ProcessCondition) or any(
            not isinstance(condition, ProcessCondition) for condition in conditions):
        raise TypeError("LevelSetILT 工艺条件必须是 ProcessCondition")
    all_conditions = (nominal, *conditions)
    if len({condition.name for condition in all_conditions}) != len(all_conditions):
        raise ValueError("LevelSetILT 工艺条件名称不能重复")
    fixed = parameters.detach().clone()
    parameters = parameters.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((parameters,), lr=config.step_size)
    best_loss, best_iteration = float("inf"), 0
    best_parameters = parameters.detach().clone()
    records: list[ILTIterationRecord] = []
    for iteration in range(config.iterations):
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        phi = parameters * movable + fixed * (1.0 - movable)
        mask = _LevelSetBinarize.apply(phi)
        printed = model.forward_many(mask, all_conditions)
        nominal_l2 = torch.sum((printed[nominal.name] - target_batch).square())
        if conditions:
            process_stack = torch.stack(tuple(printed[item.name] for item in conditions))
            process_l2 = torch.sum((process_stack - target_batch.unsqueeze(0)).square())
            pvband = torch.sum((torch.amax(process_stack, 0) -
                               torch.amin(process_stack, 0)).square())
        else:
            process_l2 = nominal_l2.new_zeros(())
            pvband = nominal_l2.new_zeros(())
        curvature = (_curvature_loss(mask) if config.curvature_weight > 0.0 else
                     nominal_l2.new_zeros(()))
        loss = nominal_l2 + config.weight_process_l2 * process_l2 + config.weight_pvband * pvband + config.curvature_weight * curvature
        values = (float(loss.detach()), float(nominal_l2.detach()), float(process_l2.detach()),
                  float(pvband.detach()), float(curvature.detach()))
        if values[0] < best_loss:
            best_loss, best_iteration = values[0], iteration
            best_parameters = phi.detach().clone()
        # 固定区通过 phi 混合从计算图上切断参数影响；一次 backward 后 Adam 只更新
        # movable 对应梯度。参数和两份 Adam 状态是本算法主要的 O(BHW) 常驻显存。
        loss.backward()
        optimizer.step()
        records.append(ILTIterationRecord(iteration, *values, perf_counter() - started))
    # 求解前向必须保持 OpenILT 的 phi<0 硬边界；对外 soft_mask 仅由最优 phi
    # 生成连续诊断表示。binary 直接使用同一个零等值线，避免 phi==0 时结果反转。
    soft_mask = torch.sigmoid(-best_parameters)
    binary = best_parameters < 0.0
    if squeeze:
        best_parameters, soft_mask, binary = best_parameters[0], soft_mask[0], binary[0]
    return SimpleILTResult(best_parameters, soft_mask, binary, best_iteration, tuple(records))
