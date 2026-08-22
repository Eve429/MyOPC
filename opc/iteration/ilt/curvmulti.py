"""CurvMulti ILT：粗到细控制网格、平滑 sigmoid 掩膜与 nominal wafer 曲率。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from lithography import LithographyModel
from opc.input.pixel import PixelMacroProblem

from ._common import (
    ILTMacroResult,
    ILTStateRecord,
    resize_image,
    smooth_sigmoid_mask,
)
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
class CurvMultiConfig:
    """CurvMulti ILT 的多尺度优化参数（[curvmulti_ilt] 段直接注册）。"""

    scales: tuple[int, ...]  # 控制网格缩放比，严格递减且以 1 结尾
    iterations_per_stage: int  # 每 stage 宏级同步 SGD 更新次数
    step_size: float  # SGD 步长
    smoothing_kernel: int  # sigmoid 前均值平滑窗口（正奇数）
    sigmoid_steepness: float  # sigmoid 陡度 β
    sigmoid_offset: float  # sigmoid 阈值偏移
    weight_process_l2: float  # process L2 权重
    weight_pvband: float  # 连续 PV 权重
    curvature_weight: float  # nominal wafer 曲率权重（0 = 不构建卷积）
    mask_threshold: float  # 最终二值化阈值
    batch_size: int  # 一次 forward 的 core 数

    def __post_init__(self) -> None:
        """在分配多尺度张量前拒绝无效尺度、平滑核与浮点配置。"""
        for name in ("iterations_per_stage", "batch_size", "smoothing_kernel"):
            # 布尔是 int 子类，必须显式排除（TOML 的 true 不许当 1 用）。
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须是整数")  # noqa: TRY004
        if self.iterations_per_stage < 1 or self.batch_size < 1:
            raise ValueError("iterations_per_stage/batch_size 必须为正")
        if (
            not isinstance(self.scales, tuple)
            or not self.scales
            or any(not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0 for scale in self.scales)
        ):
            raise ValueError("scales 必须是非空正整数元组")
        if self.scales[-1] != 1 or any(left <= right for left, right in zip(self.scales, self.scales[1:])):
            raise ValueError("scales 必须严格递减并以 1 结尾")
        if self.smoothing_kernel <= 0 or self.smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel 必须是正奇数")
        values = (
            self.step_size,
            self.sigmoid_steepness,
            self.sigmoid_offset,
            self.weight_process_l2,
            self.weight_pvband,
            self.curvature_weight,
            self.mask_threshold,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("step/权重/阈值必须是有限数")
        if (
            self.step_size <= 0.0
            or self.sigmoid_steepness <= 0.0
            or not 0.0 <= self.sigmoid_offset <= 1.0
            or self.weight_process_l2 < 0.0
            or self.weight_pvband < 0.0
            or self.curvature_weight < 0.0
            or not 0.0 < self.mask_threshold < 1.0
        ):
            raise ValueError("步长、sigmoid 参数、权重或阈值超出有效范围")


def optimize_curvmulti_macro(
    problem: PixelMacroProblem,
    model: LithographyModel,
    config: CurvMultiConfig,
    *,
    on_tiles_completed: OnTilesCompleted | None = None,
) -> ILTMacroResult:
    """按粗到细控制网格优化一个 macro，返回全局最佳已评价状态。

    每个 stage 的控制网格 [Hm/s,Wm/s] 是 SGD 参数；画布取值链为
    控制网格 -> 平滑 sigmoid -> nearest 上采样到宏全分辨率 -> 经
    trainable 索引进各 core 画布（光刻恒在完整物理网格执行）。同一
    state 全部 core/batch 经同一控制张量前向、梯度跨批累加，屏障后
    恰一次 step；stage 切换只以 nearest 带走本 stage best 参数。
    """
    # 入口契约：画布一致 + 曲率启用的 context≥1px 联合约束统一走骨架
    # （3×3 valid 卷积的 ownership 边缘一圈不计曲率，context < 1 像素时
    # 曲率随 core 切分方式变化——网格切分不应改变损失语义）。
    check_common_entry(problem, model, curvature_weight=config.curvature_weight, reason="curvature_weight > 0 ")
    beta = float(config.sigmoid_steepness)
    offset = float(config.sigmoid_offset)
    kernel = int(config.smoothing_kernel)
    device = model.device
    pixel_dbu = int(problem.macro.pixel_dbu)
    canvas_pixels = int(problem.macro.canvas_pixels)
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    # 多尺度契约：宏参数域必须能整除全部 scale，且最粗控制网格放得下平滑核。
    if any(hm % scale or wm % scale for scale in config.scales):
        raise ValueError(f"宏 ownership {wm}x{hm} 必须整除全部 scale {config.scales}")
    if min(hm // config.scales[0], wm // config.scales[0]) < kernel:
        raise ValueError(
            f"最粗控制网格不能小于 smoothing_kernel（{min(hm // config.scales[0], wm // config.scales[0])} < {kernel}）"
        )
    if config.curvature_weight > 0.0 and min(hm, wm) < 3:
        raise ValueError("启用 wafer 曲率时宏 ownership 边长不能小于 3")
    # macro ownership 在 query 栅格中的位置：初始参考与监督都取自这一块。
    mrow0 = (box.bottom - query.bottom) // pixel_dbu
    mcol0 = (box.left - query.left) // pixel_dbu
    target_ownership_u8 = problem.target_u8[mrow0 : mrow0 + hm, mcol0 : mcol0 + wm]
    # OpenILT CurvMulti 初值：直接用 [0,1] target 作参数（offset=0.5 时平滑后
    # 形成对称软边）；无需 Simple 的 2T−1 logit 或 LevelSet 的 SDF 表示。
    target_float = target_ownership_u8.astype(np.float32) / 255.0
    target_grid = torch.from_numpy(target_float).unsqueeze(0)  # [1,Hm,Wm]
    # 三工艺角一次前向（同一 state 全部 batch 共享同一 FFT 约定）
    conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
    # 静态画布每 macro 打包一次；stage/state 循环只做控制链 gather 与更新
    packs = pack_batches(problem, config.batch_size)
    best_loss = float("inf")  # 全局严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_control: torch.Tensor | None = None  # 全局 best 的控制网格（CPU）
    records: list[ILTStateRecord] = []
    state_index = 0  # 跨 stage 全宏单调连续
    previous_stage_best: torch.Tensor | None = None  # 上一 stage best（CPU）
    for stage_index, scale in enumerate(config.scales):
        control_shape = (hm // scale, wm // scale)
        # stage 参考 area 保覆盖率（仅首 stage 使用）；跨 stage 参数 nearest
        # 不引入新灰度——两值不可混用，也不为后续 stage 计算废弃参考。
        stage_initial = (
            resize_image(target_grid, control_shape, "area")
            if previous_stage_best is None
            else resize_image(previous_stage_best, control_shape, "nearest")
        )
        control = stage_initial.detach().clone().to(device=device)
        # 每 stage 独立 SGD：跨 stage 禁止继承 optimizer state（REQ-003）。
        optimizer = torch.optim.SGD([control], lr=config.step_size)
        stage_best_loss = float("inf")
        stage_best_control = control.detach().to("cpu").clone()
        for stage_state in range(config.iterations_per_stage + 1):
            build_gradient = stage_state < config.iterations_per_stage
            control.requires_grad_(build_gradient)
            if build_gradient:
                optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()

            def slot_values(pack: BatchPack, _control=control) -> SlotForward:
                """可微链：控制网格 -> 平滑 sigmoid -> nearest 上采样 -> 槽位
                gather。控制张量在同一 state 内不被修改，全部 batch 因此
                读同一快照；backward 梯度经链路自然累加进 control.grad
                （无需叶子收集器）。"""
                mask_control = smooth_sigmoid_mask(_control, kernel, beta, offset)  # [1,hc,wc]
                full_mask = resize_image(mask_control, (hm, wm), "nearest").view(-1)
                gathered = full_mask[torch.from_numpy(pack.safe).to(device=device)]  # [B,P] 可微 gather
                values = gathered.view(pack.count, canvas_pixels, canvas_pixels)
                return SlotForward(values=values, collect_gradient=None)

            # CurvMulti 曲率作用于 nominal wafer（printed）而非输入 mask
            # ——与 Simple/LevelSet 的 mask 曲率是本方法的算法差异所在。
            sums = run_state_batches(
                model,
                packs,
                conditions,
                slot_values=slot_values,
                context_mode="soft",
                context_beta=beta,
                curvature_source="nominal_wafer",
                weight_process_l2=config.weight_process_l2,
                weight_pvband=config.weight_pvband,
                curvature_weight=config.curvature_weight,
                build_gradient=build_gradient,
                on_tiles_completed=on_tiles_completed,
            )
            total_loss = total_loss_of(
                sums,
                weight_process_l2=config.weight_process_l2,
                weight_pvband=config.weight_pvband,
                curvature_weight=config.curvature_weight,
            )
            if not (
                isfinite(total_loss)
                and isfinite(sums["nominal"])
                and isfinite(sums["process"])
                and isfinite(sums["pvband"])
                and isfinite(sums["curvature"])
            ):
                raise FloatingPointError(f"{problem.macro.macro_id} state {state_index} 连续 loss 非有限")
            records.append(
                ILTStateRecord(
                    state_index=state_index,
                    stage_index=stage_index,
                    stage_state_index=stage_state,
                    scale=scale,
                    total_loss=total_loss,
                    nominal_l2=sums["nominal"],
                    process_l2=sums["process"],
                    pvband_loss=sums["pvband"],
                    curvature_loss=sums["curvature"],
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            # 全局 best 只能来自完整已评价的宏状态；严格更小才替换
            if total_loss < best_loss:
                best_loss = total_loss
                best_state_index = state_index
                best_control = control.detach().to("cpu").clone()
            # stage best 只服务下一 stage 的 nearest warm-start（REQ-003）
            if total_loss < stage_best_loss:
                stage_best_loss = total_loss
                stage_best_control = control.detach().to("cpu").clone()
            # 记录计数器在 best 判定后递增：与是否发生 step 无关，保证跨 stage
            # 单调连续（每 stage 的末状态同样占用一个编号）。
            state_index += 1
            if stage_state == config.iterations_per_stage:  # 本 stage 状态用尽
                break
            # 屏障：全部 core 梯度完成后的唯一 finite 检查与唯一 step
            gradient = control.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"{problem.macro.macro_id} state {state_index - 1} 控制网格梯度非有限")
            optimizer.step()
            if not torch.isfinite(control).all():
                raise FloatingPointError(f"{problem.macro.macro_id} state {state_index - 1} 更新后控制网格非有限")
        # stage 结束丢弃 optimizer 与图，只带走本 stage best 控制网格
        previous_stage_best = stage_best_control
    # best 物化：全局 best 控制网格在 CPU float32 域恢复全分辨率，不携带
    # autograd。参数与软掩膜都按 nearest 上采样——正是画布槽位取值的定义。
    assert best_control is not None  # 至少评价过 state 0
    best_parameters = resize_image(best_control, (hm, wm), "nearest")[0].numpy()
    soft_mask = resize_image(smooth_sigmoid_mask(best_control, kernel, beta, offset), (hm, wm), "nearest")[0].numpy()
    binary_mask = soft_mask >= config.mask_threshold
    return ILTMacroResult(
        best_parameters=best_parameters,
        soft_mask=soft_mask,
        binary_mask=binary_mask,
        best_state_index=best_state_index,
        records=tuple(records),
    )


def build_curvmulti_final_context_canvas(
    problem: PixelMacroProblem, core_index: int, config: CurvMultiConfig
) -> np.ndarray:
    """组装 CurvMulti 终评的固定 context 画布：真实 context 取初始 soft，padding 恒 0。

    与训练热路径同一套 transmission 定义 σ(β(2T−1))，并与 Simple 的
    build_simple_final_context_canvas 逐值一致（两者同为 sigmoid 参数化、
    context 不含控制网格信息；trainable 槽位上的值由公共 `_binary_canvas`
    的 where 覆盖）。公式与 solver 内联版刻意保持代码重复：训练留在 GPU
    torch、终评留在 CPU numpy，避免为消除几行重复引入 CPU/GPU round-trip。
    """
    beta = float(config.sigmoid_steepness)
    target = problem.target_canvas(core_index).astype(np.float32) / 255.0
    context_soft = 1.0 / (1.0 + np.exp(-beta * (2.0 * target - 1.0)))
    # 三值语义：window 外的数值 padding 不是物理 T=0 像素，恒 0
    return np.where(problem.context_valid_canvas(core_index), context_soft, 0.0).astype(np.float32)
