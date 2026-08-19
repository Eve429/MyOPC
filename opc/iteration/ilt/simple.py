"""Simple ILT：macro 像素参数 sigmoid 化、core 批梯度累加与同步 SGD。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from lithography import LithographyModel
from opc.input.pixel import PixelMacroProblem

from ._common import (
    ILTMacroResult,
    ILTStateRecord,
    curvature_loss,
    owned_continuous_losses,
    weighted_macro_loss,
)

# 进度回调类型：参数是本批真正完成评价、backward 与释放的 tile 数。
OnTilesCompleted = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class SimpleILTConfig:
    """Simple ILT 的优化参数（[simple_ilt] 段直接注册，无派生换算）。"""

    iterations: int          # macro 级同步 SGD 更新次数（评价 N+1 个状态）
    step_size: float         # SGD 步长
    sigmoid_steepness: float # 参数化 sigmoid 陡度 β
    weight_process_l2: float # process L2 权重
    weight_pvband: float     # 连续 PV 权重
    curvature_weight: float  # 掩膜曲率权重（0 = 不构建卷积）
    mask_threshold: float    # 最终二值化阈值
    batch_size: int          # 一次 forward 的 core 数

    def __post_init__(self) -> None:
        """在分配优化张量前拒绝空迭代、非法步长与越界阈值。"""
        for name in ("iterations", "batch_size"):
            # 布尔是 int 子类，必须显式排除（TOML 的 true 不许当 1 用）。
            # 配置层全部错误统一 ValueError（与 simple/gradient 版一致）。
            if (not isinstance(getattr(self, name), int)
                    or isinstance(getattr(self, name), bool)):
                raise ValueError(f"{name} 必须是整数")  # noqa: TRY004
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations/batch_size 必须为正")
        values = (self.step_size, self.sigmoid_steepness,
                  self.weight_process_l2, self.weight_pvband,
                  self.curvature_weight, self.mask_threshold)
        if not all(isfinite(value) for value in values):
            raise ValueError("step/权重/阈值必须是有限数")
        if (self.step_size <= 0.0 or self.sigmoid_steepness <= 0.0
                or self.weight_process_l2 < 0.0 or self.weight_pvband < 0.0
                or self.curvature_weight < 0.0
                or not 0.0 < self.mask_threshold < 1.0):
            raise ValueError("步长、陡度、权重或阈值超出有效范围")


def optimize_simple_macro(
        problem: PixelMacroProblem,
        model: LithographyModel,
        config: SimpleILTConfig,
        *,
        on_tiles_completed: OnTilesCompleted | None = None,
) -> ILTMacroResult:
    """优化一个 macro 的 ownership 像素参数并返回最佳已评价状态。

    macro 参数常驻 CPU（[Hm,Wm] float32）；每个 core batch 把快照参数
    按 trainable 索引取到 GPU 画布、一次 forward 出三工艺角、loss 只在
    各自 ownership 求和，backward 后把 local 梯度 scatter-add 回 CPU 宏
    梯度（求和不平均）；全部 core 完成后才执行一次同步 SGD step。
    """
    # 入口契约：进入批量前向前挡住画布不一致。
    canvas_pixels = int(problem.macro.canvas_pixels)
    if int(model.config.canvas) != canvas_pixels:
        raise ValueError("模型画布与 problem 画布不一致")
    beta = float(config.sigmoid_steepness)
    device = model.device
    pixel_dbu = int(problem.macro.pixel_dbu)
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    core_count = problem.macro.core_count
    # macro ownership 在 query 栅格中的位置：初始参数与监督都取自这一块。
    mrow0 = (box.bottom - query.bottom) // pixel_dbu
    mcol0 = (box.left - query.left) // pixel_dbu
    target_ownership_u8 = problem.target_u8[mrow0:mrow0 + hm, mcol0:mcol0 + wm]
    # OpenILT 初始化：params = 2·T − 1（P1-1 修复）。端点参数 ±1 远离
    # 机器精度区，0/1 像素 sigmoid 斜率 ≈ β·σ(β)σ(−β)（β=4 时 0.0707），
    # 内部像素保持可优化——拓扑变化/开孔/SRAF 的前提。代价：state0
    # soft = σ(β(2T−1)) 不再精确等于 T，但其二值化（threshold 0.5）仍与
    # T ≥ 0.5 逐格一致（σ(β(2T−1)) ≥ 0.5 ⟺ T ≥ 0.5）。
    target_float = target_ownership_u8.astype(np.float32) / 255.0
    flat_parameters = (2.0 * target_float - 1.0).reshape(-1)
    # 三工艺角一次前向（同一 state 全部 batch 共享同一 FFT 约定）
    conditions = (model.condition("nominal"), model.condition("dose_max"),
                  model.condition("defocus_min"))
    use_curvature = config.curvature_weight > 0.0
    best_loss = float("inf")  # 严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_flat = flat_parameters.copy()
    records: list[ILTStateRecord] = []
    for state_index in range(config.iterations + 1):
        build_gradient = state_index < config.iterations  # 末状态纯评价
        macro_gradient = (np.zeros_like(flat_parameters)
                          if build_gradient else None)
        sums = {"nominal": 0.0, "process": 0.0, "pvband": 0.0,
                "curvature": 0.0}
        started = time.perf_counter()  # 本状态全部 core 评价计时
        for batch_start in range(0, core_count, config.batch_size):
            core_indices = list(range(
                batch_start,
                min(batch_start + config.batch_size, core_count)))
            batch_count = len(core_indices)
            # CPU 组批：target/计分/trainable 索引三种画布一次取出
            targets = np.stack(
                [problem.target_canvas(c) for c in core_indices])
            ownerships = np.stack(
                [problem.ownership_canvas(c) for c in core_indices])
            trainables = np.stack(
                [problem.trainable_index_canvas(c) for c in core_indices])
            trainable_flat = trainables.reshape(batch_count, -1)  # [B,P]
            valid = trainable_flat >= 0
            safe = np.where(valid, trainable_flat, 0)
            # 同一 state 全部 batch 读同一宏参数快照：numpy 取值即快照，
            # 无 autograd 直通；梯度经 local 叶子张量回散（见下）。
            local_values = flat_parameters[safe]
            target_tensor = torch.from_numpy(targets).to(
                device=device, dtype=torch.float32).div_(255.0)
            ownership_tensor = torch.from_numpy(ownerships).to(device=device)
            index_tensor = torch.from_numpy(
                trainable_flat.astype(np.int64)).to(device=device)
            index3 = index_tensor.view(batch_count, canvas_pixels, canvas_pixels)
            # local 是叶子张量：forward 数值来自快照，backward 梯度经
            # scatter-add 汇入宏梯度——评价与更新职责分离的最小机制。
            local = torch.from_numpy(local_values).to(
                device=device, dtype=torch.float32)
            if build_gradient:
                local.requires_grad_(True)
            soft = torch.sigmoid(beta * local.view(
                batch_count, canvas_pixels, canvas_pixels))
            # 固定 context 统一为初始版图的 state0 transmission σ(β(2T−1))：
            # 由常量 target 推导、无梯度边。与 trainable 像素同一套定义，
            # 消除 macro seam 上人为的 ~1.8% transmission 跳变，并保证邻宏
            # 在本宏画布中的初始值恰为其自身 state0（监督目标仍是 raw T）。
            context_soft = torch.sigmoid(
                beta * (target_tensor * 2.0 - 1.0)).detach()
            mask = torch.where(index3 >= 0, soft, context_soft)
            printed = model.forward_many(mask, conditions)  # 一次共享 FFT
            nominal_l2, process_l2, pvband_loss = owned_continuous_losses(
                printed["nominal"], printed["dose_max"],
                printed["defocus_min"], target_tensor, ownership_tensor)
            curvature_value = (curvature_loss(mask, ownership_tensor)
                               if use_curvature else 0.0)
            sums["nominal"] += float(nominal_l2.detach())
            sums["process"] += float(process_l2.detach())
            sums["pvband"] += float(pvband_loss.detach())
            if use_curvature:
                sums["curvature"] += float(curvature_value.detach())
            if build_gradient:
                batch_total = weighted_macro_loss(
                    nominal_l2, process_l2, pvband_loss, curvature_value,
                    weight_process_l2=config.weight_process_l2,
                    weight_pvband=config.weight_pvband,
                    curvature_weight=config.curvature_weight)
                batch_total.backward()  # 批间梯度直接累加（同一参数快照）
                grad_np = local.grad.detach().cpu().numpy()
                # scatter-add 求和：同一像素出现在多个 core context 时
                # 梯度相加，绝不按出现次数平均。
                np.add.at(macro_gradient, trainable_flat[valid],
                          grad_np[valid])
            # 释放：批结束只保留标量与宏梯度，画布与 autograd 图失去引用
            del printed, mask, soft, local, target_tensor, ownership_tensor
            del index_tensor, index3, nominal_l2, process_l2, pvband_loss
            if use_curvature:
                del curvature_value
            if on_tiles_completed is not None:  # backward 且释放后才报进度
                on_tiles_completed(batch_count)
        total_loss = float(weighted_macro_loss(
            sums["nominal"], sums["process"], sums["pvband"],
            sums["curvature"], weight_process_l2=config.weight_process_l2,
            weight_pvband=config.weight_pvband,
            curvature_weight=config.curvature_weight))
        if not (isfinite(total_loss) and isfinite(sums["nominal"])
                and isfinite(sums["process"]) and isfinite(sums["pvband"])
                and isfinite(sums["curvature"])):
            raise FloatingPointError(
                f"{problem.macro.macro_id} state {state_index} 连续 loss 非有限")
        records.append(ILTStateRecord(
            state_index=state_index, stage_index=0,
            stage_state_index=state_index, scale=1,
            total_loss=total_loss, nominal_l2=sums["nominal"],
            process_l2=sums["process"], pvband_loss=sums["pvband"],
            curvature_loss=sums["curvature"],
            elapsed_seconds=time.perf_counter() - started))
        # best 只能来自完整已评价的宏状态；严格更小才替换
        if total_loss < best_loss:
            best_loss = total_loss
            best_state_index = state_index
            best_flat = flat_parameters.copy()
        if state_index == config.iterations:  # 轮次自然用尽
            break
        # 屏障：全部 core 梯度完成后的唯一检查与唯一 step
        if not np.isfinite(macro_gradient).all():
            raise FloatingPointError(
                f"{problem.macro.macro_id} state {state_index} 宏梯度非有限")
        flat_parameters = flat_parameters - config.step_size * macro_gradient
        if not np.isfinite(flat_parameters).all():
            raise FloatingPointError(
                f"{problem.macro.macro_id} state {state_index} 更新后参数非有限")
    # best 物化：软掩膜/二值都在 CPU float32 域完成，不携带 autograd
    best_parameters = best_flat.reshape(hm, wm)
    soft_mask = torch.sigmoid(
        beta * torch.from_numpy(best_parameters)).numpy()
    binary_mask = soft_mask >= config.mask_threshold
    return ILTMacroResult(
        best_parameters=best_parameters, soft_mask=soft_mask,
        binary_mask=binary_mask, best_state_index=best_state_index,
        records=tuple(records))
