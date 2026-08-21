"""LevelSet ILT：SDF 参数化、外部梯度 STE 与宏级 Adam 的像素优化。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

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
class LevelSetILTConfig:
    """LevelSet ILT 的优化参数（[levelset_ilt] 段直接注册，无派生换算）。"""

    iterations: int          # macro 级 Adam 更新次数（评价 N+1 个状态）
    step_size: float         # Adam 学习率，物理单位 nm
    weight_process_l2: float # process L2 权重
    weight_pvband: float     # 连续 PV 权重
    curvature_weight: float  # hard mask 曲率权重（0 = 不构建卷积）
    batch_size: int          # 一次 forward 的 core 数

    def __post_init__(self) -> None:
        """在分配优化张量前拒绝空迭代、非法步长与负权重（bool 拒当 int）。"""
        for name in ("iterations", "batch_size"):
            # 布尔是 int 子类，必须显式排除（TOML 的 true 不许当 1 用）。
            if (not isinstance(getattr(self, name), int)
                    or isinstance(getattr(self, name), bool)):
                raise ValueError(f"{name} 必须是整数")  # noqa: TRY004
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations/batch_size 必须为正")
        values = (self.step_size, self.weight_process_l2,
                  self.weight_pvband, self.curvature_weight)
        if not all(isfinite(value) for value in values):
            raise ValueError("步长/权重必须是有限数")
        if (self.step_size <= 0.0 or self.weight_process_l2 < 0.0
                or self.weight_pvband < 0.0 or self.curvature_weight < 0.0):
            raise ValueError("步长或权重超出有效范围")


def signed_distance_initialization(
        target_u8: np.ndarray, pixel_nm: float = 1.0) -> np.ndarray:
    """从 query transmission 栅格生成前景为负、单位为 nm 的像素中心精确 SDF。

    阈值事实源唯一：target_u8/255 >= 0.5（127/128 分界），任何调用位置
    不得换用不等价的整数比较。距离定义在像素中心之间（raster 定义，
    刻意不与 OpenILT polygon-edge 初值对齐，规格 §2.2 已冻结）；生产
    路径必须走 SciPy compiled EDT，旧纯 Python EDT 只作测试 oracle。
    """
    if not isfinite(pixel_nm) or pixel_nm <= 0.0:
        raise ValueError("pixel_nm 必须是正有限数")
    spacing = float(pixel_nm)
    target = np.asarray(target_u8)
    if target.ndim != 2:
        raise ValueError("target_u8 必须是二维栅格")
    binary = target.astype(np.float32) / 255.0 >= 0.5
    height, width = binary.shape
    # 单类退化显式给有限常量场：无对面类时 EDT 无参照（REQ-003）
    if bool(binary.all()):
        return np.full(
            (height, width), -float(max(height, width)) * spacing, np.float32)
    if not bool(binary.any()):
        return np.full(
            (height, width), float(max(height, width)) * spacing, np.float32)
    # 顺序执行 outside/inside EDT 并即时释放 float64 中间量：初始化峰值
    # 不超过一份 float64 距离场 + 一份 float32 结果（REQ-003 SHOULD）。
    phi = distance_transform_edt(
        ~binary, sampling=(spacing, spacing)).astype(np.float32)
    inside = distance_transform_edt(binary, sampling=(spacing, spacing))  # float64 临时，用后即弃
    phi[binary] = -inside[binary]
    del inside
    return phi


def macro_gradient_magnitude(problem: PixelMacroProblem,
                             initial_query_phi: np.ndarray,
                             macro_phi: np.ndarray,
                             pixel_nm: float = 1.0) -> np.ndarray:
    """计算 macro 参数域唯一的 |grad(phi)| 代理梯度系数 [Hm,Wm]。

    中心差分的 1px 邻域取自 initial_query_phi 的固定物理 context；
    phi 以 nm 表示、差分除以物理像素间距，因此 |grad(phi)| 为无量纲——
    macro 外像素永不训练，其 phi 恒为初值；外围不用 replicate padding：
    core 切分方式不得改变同一参数的系数（DEC-003）。只在需要 backward
    的 state 调用，末纯评价状态不调用（REQ-004）。
    """
    if not isfinite(pixel_nm) or pixel_nm <= 0.0:
        raise ValueError("pixel_nm 必须是正有限数")
    spacing = float(pixel_nm)
    hm, wm = problem.ownership_shape
    if (np.asarray(initial_query_phi).shape != problem.query_shape
            or np.asarray(macro_phi).shape != (hm, wm)):
        raise ValueError("initial_query_phi/macro_phi 形状与 problem 不一致")
    pixel_dbu = int(problem.macro.pixel_dbu)
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    # macro ownership 在 query 栅格中的位置（context>=1 像素时四周各余一圈）
    mrow0 = (box.bottom - query.bottom) // pixel_dbu
    mcol0 = (box.left - query.left) // pixel_dbu
    halo = np.array(initial_query_phi[
        mrow0 - 1:mrow0 + hm + 1, mcol0 - 1:mcol0 + wm + 1],
        dtype=np.float32, copy=True)
    halo[1:-1, 1:-1] = macro_phi  # 中心覆盖为当前快照，外围保留初始 context
    dx = (halo[1:-1, 2:] - halo[1:-1, :-2]) / (2.0 * spacing)
    dy = (halo[2:, 1:-1] - halo[:-2, 1:-1]) / (2.0 * spacing)
    return np.sqrt(dx * dx + dy * dy)


class _LevelSetBinarize(torch.autograd.Function):
    """hard 前向 + 外部梯度调制反向的 LevelSet STE（REQ-002）。"""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                levelset: torch.Tensor,
                grad_magnitude: torch.Tensor | None) -> torch.Tensor:
        """前向输出 phi<0 的透光掩膜（phi==0 不透光）。"""
        if grad_magnitude is not None:  # 纯评价路径不保存（不会 backward）
            ctx.save_for_backward(grad_magnitude)
        return (levelset < 0.0).to(levelset.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor) -> tuple[torch.Tensor | None,
                                                     None]:
        """代理梯度 -|grad(phi)|·grad_output；系数只读，返回 None。

        本函数不做任何空间差分——|grad(phi)| 已在 macro 参数域统一算好
        （DEC-003），这里只做逐元素缩放，因此对任意形状张量成立。
        """
        (coefficient,) = ctx.saved_tensors
        return -coefficient * grad_output, None


def build_levelset_final_context_canvas(
        problem: PixelMacroProblem, core_index: int,
        config: LevelSetILTConfig) -> np.ndarray:
    """组装 LevelSet 终评固定 context 画布：真实 context 取 hard target，padding 恒 0。

    与训练同一套三值语义；不运行 SDF（终评只消费 best 二值结果）。
    config 形参与公共策略签名对齐，本方法不读任何算法字段。
    """
    target = problem.target_canvas(core_index).astype(np.float32) / 255.0
    hard = (target >= 0.5).astype(np.float32)
    return np.where(problem.context_valid_canvas(core_index),
                    hard, 0.0).astype(np.float32)


def optimize_levelset_macro(
        problem: PixelMacroProblem,
        model: LithographyModel,
        config: LevelSetILTConfig,
        *,
        pixel_nm: float = 1.0,
        on_tiles_completed: OnTilesCompleted | None = None,
) -> ILTMacroResult:
    """优化一个 macro 的 ownership 像素 phi 并返回最佳已评价状态。

    phi 以 nm 为单位常驻 CPU（[Hm,Wm] float32，Adam 状态同域）；每 backward state
    在 macro 域算一次唯一 |grad(phi)|，各 core 批把快照 phi/系数按
    trainable 索引 gather 到 GPU、STE hard 前向，loss 只在各 core 自己
    的 ownership 求和，backward 后 local 梯度 scatter-add 回 CPU 宏梯度
    （raw sum 不平均）；全部 core 完成后执行恰一次 Adam step。
    """
    if not isfinite(pixel_nm) or pixel_nm <= 0.0:
        raise ValueError("pixel_nm 必须是正有限数")
    pixel_nm = float(pixel_nm)
    # 入口契约：中心差分至少需要一圈真实物理 context（replicate padding 会让
    # 边缘参数系数依赖 core 切分，网格切分不应改变算法语义），故无条件要求。
    check_common_entry(problem, model, require_context_pixel=True,
                       reason="LevelSet ")
    pixel_dbu = int(problem.macro.pixel_dbu)
    canvas_pixels = int(problem.macro.canvas_pixels)
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    # macro ownership 在 query 栅格中的位置（SDF crop 与 halo 差分共用）
    mrow0 = (box.bottom - query.bottom) // pixel_dbu
    mcol0 = (box.left - query.left) // pixel_dbu
    # SDF once/macro：初始化在 state 循环外，不随 core/batch 重复（REQ-003）
    initial_query_phi = signed_distance_initialization(
        problem.target_u8, pixel_nm=pixel_nm)
    if not np.isfinite(initial_query_phi).all():
        raise FloatingPointError(
            f"{problem.macro.macro_id} 初始 SDF 非有限")
    macro_phi = torch.from_numpy(
        initial_query_phi[mrow0:mrow0 + hm, mcol0:mcol0 + wm].copy())
    optimizer = torch.optim.Adam(
        [macro_phi], lr=config.step_size, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False)  # 超参与 REQ-008 契约锁定
    conditions = (model.condition("nominal"), model.condition("dose_max"),
                  model.condition("defocus_min"))
    device = model.device
    # 静态画布每 macro 打包一次；state 循环只做 phi 快照 gather 与更新
    packs = pack_batches(problem, config.batch_size)
    best_loss = float("inf")  # 严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_phi = macro_phi.detach().clone()
    records: list[ILTStateRecord] = []
    for state_index in range(config.iterations + 1):
        build_gradient = state_index < config.iterations  # 末状态纯评价
        phi_flat = macro_phi.numpy().reshape(-1)  # 当前快照（numpy 视图）
        if build_gradient:
            grad_magnitude = macro_gradient_magnitude(
                problem, initial_query_phi,
                macro_phi.numpy(), pixel_nm=pixel_nm).reshape(-1)
            if not np.isfinite(grad_magnitude).all():
                raise FloatingPointError(
                    f"{problem.macro.macro_id} state {state_index} "
                    "grad magnitude 非有限")
            macro_gradient = np.zeros_like(phi_flat)
        started = time.perf_counter()  # 本状态全部 core 评价计时

        def slot_values(pack: BatchPack,
                        _build: bool = build_gradient,
                        _phi=phi_flat,
                        _coefficient=grad_magnitude if build_gradient else None,
                        _macro_gradient=macro_gradient
                        ) -> SlotForward:
            """同一 state 全部批读同一宏参数快照：numpy 取值即快照，无
            autograd 直通；STE 前向 hard、backward −系数×上游，梯度经
            local 叶子回散宏梯度（与 Simple 同机制）。"""
            local = torch.from_numpy(_phi[pack.safe]).to(
                device=device, dtype=torch.float32)
            if _build:
                local.requires_grad_(True)
            local_grad = (torch.from_numpy(_coefficient[pack.safe]).to(
                device=device, dtype=torch.float32)
                if _build else None)
            values = _LevelSetBinarize.apply(
                local.view(pack.count, canvas_pixels, canvas_pixels),
                None if local_grad is None else local_grad.view(
                    pack.count, canvas_pixels, canvas_pixels))

            def collect(finished: BatchPack, _local=local,
                        _macro_gradient=_macro_gradient) -> None:
                """scatter-add raw sum：同一像素出现在多个 core context 时
                梯度相加，绝不按出现次数平均（REQ-007）。"""
                grad_np = _local.grad.detach().cpu().numpy()
                np.add.at(_macro_gradient,
                          finished.trainable_flat[finished.owned],
                          grad_np[finished.owned])

            return SlotForward(
                values=values,
                collect_gradient=collect if _build else None)

        sums = run_state_batches(
            model, packs, conditions, slot_values=slot_values,
            context_mode="hard", curvature_source="mask",
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
            best_phi = macro_phi.detach().clone()
        if state_index == config.iterations:  # 轮次自然用尽
            break
        # 屏障：全部 core 梯度完成后的唯一检查、唯一赋值与唯一 step
        if not np.isfinite(macro_gradient).all():
            raise FloatingPointError(
                f"{problem.macro.macro_id} state {state_index} 宏梯度非有限")
        macro_phi.grad = torch.from_numpy(
            macro_gradient.reshape(hm, wm))  # 唯一宏梯度（raw sum 结果）
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(macro_phi).all()):
            raise FloatingPointError(
                f"{problem.macro.macro_id} state {state_index} 更新后 phi 非有限")
        for adam_state in optimizer.state.values():  # m/v 常驻 CPU
            for value in adam_state.values():
                if (isinstance(value, torch.Tensor)
                        and not bool(torch.isfinite(value).all())):
                    raise FloatingPointError(
                        f"{problem.macro.macro_id} state {state_index} "
                        "Adam 状态非有限")
    # best 物化：soft 仅诊断（sigmoid(-phi)），二值走 phi<0 硬边界
    best_parameters = best_phi.numpy()
    soft_mask = torch.sigmoid(-best_phi).numpy()
    binary_mask = best_parameters < 0.0
    return ILTMacroResult(
        best_parameters=best_parameters, soft_mask=soft_mask,
        binary_mask=binary_mask, best_state_index=best_state_index,
        records=tuple(records))
