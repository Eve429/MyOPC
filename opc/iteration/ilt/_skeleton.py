"""像素 ILT 求解器公共骨架：批画布打包、三值 context 与 state 批循环。

三个现行方法（Simple/LevelSet/CurvMulti）共享同一 state×batch 循环体，
差异只在参数化（槽位值来源）与更新器；骨架固定循环、组装与聚合次序，
方法以钩子注入差异点。BatchPack 每 macro 打包一次（画布是静态数据），
CPU 常驻、每 state 每批转移后即释放——GPU 每 batch 只保留当前张量的
内存纪律不变。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from lithography import LithographyModel, ProcessCondition
from opc.input.pixel import PixelMacroProblem

from ._common import curvature_loss, owned_continuous_losses, weighted_macro_loss

# 进度回调类型：参数是本批真正完成评价、backward 与释放的 tile 数。
OnTilesCompleted = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class BatchPack:
    """一个 core 批的静态画布组与派生索引（CPU 常驻，state 维度复用）。"""

    count: int  # 批内 core 数
    targets_u8: np.ndarray  # [B,C,C] uint8 监督目标
    ownerships: np.ndarray  # [B,C,C] bool 计分像素
    trainables: np.ndarray  # [B,C,C] int64 宏参数索引（macro 外 -1）
    valids: np.ndarray  # [B,C,C] bool 物理 context 窗口
    trainable_flat: np.ndarray  # [B,P] 摊平索引
    owned: np.ndarray  # [B,P] trainable 槽位
    safe: np.ndarray  # [B,P] 越界槽位填 0 的合法索引


@dataclass(frozen=True, slots=True)
class SlotForward:
    """方法钩子产物：可微槽位值与 backward 后的梯度收集器。

    collect_gradient 在批 backward 之后由骨架调用：Simple/LevelSet 在
    闭包里把 local.grad 散布回宏梯度（np.add.at raw sum）；CurvMulti 为
    None（梯度经 autograd 链路直接累加进控制张量 .grad）。
    """

    values: torch.Tensor
    collect_gradient: Callable[[BatchPack], None] | None


def pack_batches(problem: PixelMacroProblem, batch_size: int) -> list[BatchPack]:
    """按 batch_size 把全部 core 打包为静态画布组（每 macro 恰一次）。"""
    core_count = problem.macro.core_count
    packs: list[BatchPack] = []
    for batch_start in range(0, core_count, batch_size):
        core_indices = list(range(batch_start, min(batch_start + batch_size, core_count)))
        # CPU 组批：target/计分/trainable/valid 四种画布一次取出；画布是
        # problem 静态数据的纯函数，state 循环不再重复构造。
        targets = np.stack([problem.target_canvas(c) for c in core_indices])
        ownerships = np.stack([problem.ownership_canvas(c) for c in core_indices])
        trainables = np.stack([problem.trainable_index_canvas(c) for c in core_indices])
        valids = np.stack([problem.context_valid_canvas(c) for c in core_indices])
        trainable_flat = trainables.reshape(len(core_indices), -1)  # [B,P]
        owned = trainable_flat >= 0  # trainable 槽位（区别于窗口掩码）
        safe = np.where(owned, trainable_flat, 0)  # 越界填 0，垃圾值被 where 覆盖
        packs.append(
            BatchPack(
                count=len(core_indices),
                targets_u8=targets,
                ownerships=ownerships,
                trainables=trainables,
                valids=valids,
                trainable_flat=trainable_flat,
                owned=owned,
                safe=safe,
            )
        )
    return packs


def check_common_entry(
    problem: PixelMacroProblem,
    model: LithographyModel,
    *,
    curvature_weight: float = 0.0,
    require_context_pixel: bool = False,
    reason: str = "",
) -> None:
    """三方法共用的入口契约：画布一致 + context ≥ 1 像素联合约束。

    LevelSet 无条件要求 context ≥ 1 像素（中心差分需真实物理邻域）；
    Simple/CurvMulti 仅在启用曲率时要求（3×3 valid 卷积的 ownership 边缘
    一圈不计曲率，context < 1 像素时曲率随 core 切分方式变化——网格切分
    不应改变损失语义）。reason 前缀保留方法语义。
    """
    canvas_pixels = int(problem.macro.canvas_pixels)
    if int(model.config.canvas) != canvas_pixels:
        raise ValueError("模型画布与 problem 画布不一致")
    if (require_context_pixel or curvature_weight > 0.0) and int(problem.macro.context_dbu) < int(
        problem.macro.pixel_dbu
    ):
        raise ValueError(f"{reason}要求 context 不小于 1 像素（context_dbu >= pixel_dbu），当前 context=0")


def run_state_batches(
    model: LithographyModel,
    packs: list[BatchPack],
    conditions: tuple[ProcessCondition, ...],
    *,
    slot_values: Callable[[BatchPack], SlotForward],
    context_mode: str,
    curvature_source: str,
    context_beta: float | None = None,
    weight_process_l2: float,
    weight_pvband: float,
    curvature_weight: float,
    build_gradient: bool,
    on_tiles_completed: OnTilesCompleted | None = None,
) -> dict[str, float]:
    """执行一个 state 的全部 core 批：组装画布、forward、损失与梯度。

    context_mode：soft（σ(β(2T−1))，Simple/CurvMulti）或 hard
    （target ≥ 0.5，LevelSet）；curvature_source：mask（Simple/LevelSet）
    或 nominal_wafer（CurvMulti，曲率作用于 printed nominal）。返回四项
    损失的跨批浮点和，records/best/更新留给方法层。
    """
    device = model.device
    canvas_pixels = packs[0].targets_u8.shape[-1]
    use_curvature = curvature_weight > 0.0
    sums = {"nominal": 0.0, "process": 0.0, "pvband": 0.0, "curvature": 0.0}
    for pack in packs:
        target_tensor = torch.from_numpy(pack.targets_u8).to(device=device, dtype=torch.float32).div_(255.0)
        ownership_tensor = torch.from_numpy(pack.ownerships).to(device=device)
        index3 = torch.from_numpy(pack.trainables).to(device=device).view(pack.count, canvas_pixels, canvas_pixels)
        valid_tensor = torch.from_numpy(pack.valids).to(device=device)
        # 方法差异点：槽位值（可微）与梯度收集器由钩子给出
        forward = slot_values(pack)
        # 三值语义的固定 context：soft 模式与 trainable 初始值同一公式
        # （由常量 target 推导、无梯度边）；hard 模式取二值 target。
        # window 外的数值 padding 不是物理 T=0 像素，必须恒 0。
        if context_mode == "soft":
            context_source = torch.sigmoid(context_beta * (target_tensor * 2.0 - 1.0)).detach()
        else:
            context_source = (target_tensor >= 0.5).to(torch.float32)
        context = torch.where(valid_tensor, context_source, torch.zeros_like(context_source))
        mask = torch.where(index3 >= 0, forward.values, context)
        printed = model.forward_many(mask, conditions)  # 一次共享 FFT
        nominal_l2, process_l2, pvband_loss = owned_continuous_losses(
            printed["nominal"], printed["dose_max"], printed["defocus_min"], target_tensor, ownership_tensor
        )
        curvature_value = (
            curvature_loss(mask if curvature_source == "mask" else printed["nominal"], ownership_tensor)
            if use_curvature
            else 0.0
        )
        sums["nominal"] += float(nominal_l2.detach())
        sums["process"] += float(process_l2.detach())
        sums["pvband"] += float(pvband_loss.detach())
        if use_curvature:
            sums["curvature"] += float(curvature_value.detach())
        if build_gradient:
            batch_total = weighted_macro_loss(
                nominal_l2,
                process_l2,
                pvband_loss,
                curvature_value,
                weight_process_l2=weight_process_l2,
                weight_pvband=weight_pvband,
                curvature_weight=curvature_weight,
            )
            batch_total.backward()  # 批间梯度直接累加（同一参数快照）
            if forward.collect_gradient is not None:
                forward.collect_gradient(pack)
        # 释放：批结束只保留标量与宏梯度，画布与 autograd 图失去引用
        del printed, mask, forward, context, context_source, valid_tensor
        del target_tensor, ownership_tensor, index3
        del nominal_l2, process_l2, pvband_loss
        if use_curvature:
            del curvature_value
        if on_tiles_completed is not None:  # backward 且释放后才报进度
            on_tiles_completed(pack.count)
    return sums


def total_loss_of(
    sums: dict[str, float], *, weight_process_l2: float, weight_pvband: float, curvature_weight: float
) -> float:
    """把 run_state_batches 的四项和聚合为加权总损失（张量同式）。"""
    return float(
        weighted_macro_loss(
            sums["nominal"],
            sums["process"],
            sums["pvband"],
            sums["curvature"],
            weight_process_l2=weight_process_l2,
            weight_pvband=weight_pvband,
            curvature_weight=curvature_weight,
        )
    )
