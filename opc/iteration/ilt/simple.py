"""Simple ILT：macro 像素参数 sigmoid 化、core 批梯度累加与同步 SGD。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from lithography import LithographyModel
from opc.input.pixel import PixelMacroProblem

from ._common import ILTMacroResult, ILTStateRecord
from ._skeleton import (
    BatchPack,
    OnTilesCompleted,
    SlotForward,
    check_common_entry,
    pack_batches,
    run_state_batches,
    total_loss_of,
)


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

    macro 参数常驻 CPU（[Hm,Wm] float32）；公共骨架固定画布组装、
    forward 与聚合次序，本方法只注入槽位前向（快照 gather + sigmoid）
    与梯度收集（local 叶子 scatter-add 回 CPU 宏梯度，求和不平均）；
    全部 core 完成后才执行一次同步 SGD step。
    """
    check_common_entry(problem, model,
                       curvature_weight=config.curvature_weight,
                       reason="curvature_weight > 0 ")
    beta = float(config.sigmoid_steepness)
    device = model.device
    pixel_dbu = int(problem.macro.pixel_dbu)
    canvas_pixels = int(problem.macro.canvas_pixels)
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
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
    # 静态画布每 macro 打包一次；state 循环只做槽位 gather 与更新
    packs = pack_batches(problem, config.batch_size)
    best_loss = float("inf")  # 严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_flat = flat_parameters.copy()
    records: list[ILTStateRecord] = []
    for state_index in range(config.iterations + 1):
        build_gradient = state_index < config.iterations  # 末状态纯评价
        macro_gradient = (np.zeros_like(flat_parameters)
                          if build_gradient else None)
        started = time.perf_counter()  # 本状态全部 core 评价计时

        def slot_values(pack: BatchPack,
                        _build: bool = build_gradient,
                        _gradient=macro_gradient,
                        _parameters=flat_parameters) -> SlotForward:
            """同一 state 全部 batch 读同一宏参数快照：numpy 取值即快照，
            无 autograd 直通；backward 后 local 叶子梯度回散宏梯度。"""
            local_values = _parameters[pack.safe]
            local = torch.from_numpy(local_values).to(
                device=device, dtype=torch.float32)
            if _build:
                local.requires_grad_(True)
            values = torch.sigmoid(beta * local.view(
                pack.count, canvas_pixels, canvas_pixels))

            def collect(finished: BatchPack, _local=local,
                        _gradient=_gradient) -> None:
                """scatter-add 求和：同一像素出现在多个 core context 时
                梯度相加，绝不按出现次数平均。"""
                grad_np = _local.grad.detach().cpu().numpy()
                np.add.at(_gradient, finished.trainable_flat[finished.owned],
                          grad_np[finished.owned])

            return SlotForward(
                values=values,
                collect_gradient=collect if _build else None)

        sums = run_state_batches(
            model, packs, conditions, slot_values=slot_values,
            context_mode="soft", context_beta=beta, curvature_source="mask",
            weight_process_l2=config.weight_process_l2,
            weight_pvband=config.weight_pvband,
            curvature_weight=config.curvature_weight,
            build_gradient=build_gradient,
            on_tiles_completed=on_tiles_completed)
        total_loss = total_loss_of(
            sums, weight_process_l2=config.weight_process_l2,
            weight_pvband=config.weight_pvband,
            curvature_weight=config.curvature_weight)
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


def build_simple_final_context_canvas(
        problem: PixelMacroProblem, core_index: int,
        config: SimpleILTConfig) -> np.ndarray:
    """组装 Simple 终评的固定 context 画布：真实 context 取初始 soft，padding 恒 0。

    与训练热路径同一套 transmission 定义 σ(β(2T−1))（trainable 槽位上的
    值由公共 `_binary_canvas` 的 where 覆盖，不进入终评结果）。公式与
    solver 内联版刻意保持代码重复：让训练留在 GPU torch、终评留在 CPU
    numpy，避免为消除几行重复引入 CPU/GPU round-trip（REQ-012）。
    """
    beta = float(config.sigmoid_steepness)
    target = problem.target_canvas(core_index).astype(np.float32) / 255.0
    context_soft = 1.0 / (1.0 + np.exp(-beta * (2.0 * target - 1.0)))
    # 三值语义：window 外的数值 padding 不是物理 T=0 像素，恒 0
    return np.where(problem.context_valid_canvas(core_index),
                    context_soft, 0.0).astype(np.float32)
