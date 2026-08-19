"""基于梯度的 MB-OPC：midpoint 边梯度代理、连续 loss 与同步 Adam 法向位移。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from numpy.typing import NDArray

from evaluation import (
    evaluate_binary_l2,
    evaluate_edge_probes,
    evaluate_pvband,
)
from lithography import LithographyModel
from opc.errors import ReconstructionError
from opc.input import ownership_canvas, points_to_canvas, rasterize_mask_canvas
from opc.input.edge import MacroProblem, reconstruct_region_with_midpoints
from opc.input.edge.sampling import edge_probe_points

from ._cache import TargetCanvasCache

# 进度回调类型：参数是本批真正完成评价、backward 与释放的 tile 数。
OnTilesCompleted = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class GradientMBOPCConfig:
    """保存已经转换到 DBU 的梯度迭代参数。"""

    iterations: int              # 最多发布更新次数
    learning_rate_dbu: float     # Adam 学习率（连续 DBU，允许非整数）
    weight_nominal_l2: float     # nominal 连续 loss 权重
    weight_process_l2: float     # dose_max/defocus_min 对 target 连续 loss 权重
    weight_pvband: float         # dose_max-defocus_min 连续差 loss 权重
    epe_distance_dbu: float      # 仅离散诊断的探针距离
    batch_size: int              # 一次 forward 的 core 数
    target_cache_bytes: int      # CPU uint8 target LRU 上限

    def __post_init__(self) -> None:
        """校验梯度参数自身的数值契约，跨层参数（上下文）由求解器入口复验。"""
        for name in ("iterations", "batch_size", "target_cache_bytes"):
            value = getattr(self, name)
            # 布尔是 int 子类，必须显式排除（TOML 的 true 不许当 1 用）。
            # 配置层全部错误统一 ValueError（与 simple 版校验一致）。
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} 必须是整数，不接受布尔")  # noqa: TRY004
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations/batch_size 必须为正")
        if self.target_cache_bytes < 0:
            raise ValueError("target_cache_bytes 必须为非负")
        if not isfinite(self.learning_rate_dbu) or self.learning_rate_dbu <= 0.0:
            raise ValueError("learning_rate_dbu 必须是有限正数")
        weights = (self.weight_nominal_l2, self.weight_process_l2,
                   self.weight_pvband)
        if any(not isfinite(w) or w < 0.0 for w in weights):
            raise ValueError("loss 权重必须是有限非负数")
        if not any(w > 0.0 for w in weights):
            raise ValueError("三个 loss 权重至少一个为正")
        if not isfinite(self.epe_distance_dbu) or self.epe_distance_dbu <= 0.0:
            raise ValueError("epe_distance_dbu 必须是有限正数")


@dataclass(frozen=True, slots=True)
class GradientMBOPCIterationRecord:
    """保存一个 macro 已评价状态的连续 loss、离散诊断与耗时。"""

    state_index: int             # 0=baseline；N=第 N 次更新后状态
    total_loss: float            # 本状态加权连续 loss（所有权像素归一）
    nominal_l2_loss: float       # L_nom
    process_l2_loss: float       # L_process
    pvband_loss: float           # L_pv
    l2: int                      # 本状态离散二值 L2（诊断）
    pvband: int                  # 本状态离散 PVBand（诊断）
    epe: int                     # 本状态离散 EPE（诊断）
    valid_probes: int            # 本状态有效探针
    ambiguous_probes: int        # 本状态歧义探针
    displaced_segments: int      # 本状态非零 owner 位移段数
    elapsed_seconds: float       # 本状态全部 tile 评价耗时


@dataclass(frozen=True, slots=True)
class GradientMBOPCResult:
    """保存单 macro 的最佳已评价位移、全部状态记录和停止原因。"""

    best_displacements: NDArray[np.float64]  # 按全局段序；context 恒 0
    records: tuple[GradientMBOPCIterationRecord, ...]  # records[0]=baseline
    best_state_index: int                    # 指向 records 中 best
    stop_reason: str          # zero_loss/no_update/invalid_geometry/no_owned_segments/iteration_limit
    stop_detail: str | None   # 非法候选的明确原因；正常停止为 None


class _EdgeGradientMask(torch.autograd.Function):
    """hard 面积覆盖率前向与 DiffOPC Algorithm 4 的 midpoint STE 反向。"""

    @staticmethod
    def forward(ctx, hard_masks, local_displacements, batch_indices,
                midpoints_xy):
        """前向不改 mask 数值，只保存反向采样需要的批号与中点坐标。

        local_displacements 只用于建立 autograd 边（STE 硬几何直通），
        不参与 forward 计算。
        """
        ctx.save_for_backward(batch_indices, midpoints_xy)
        return hard_masks

    @staticmethod
    def backward(ctx, grad_output):
        """按当前已发布重构几何的段中点双线性采样 dL/dMask，梯度为 2·g_mid。

        2 倍来源：论文对两个 endpoint 各采样一次 g_mid；本项目标量位移
        同时驱动两端点，中点随位移移动同一单位法向，两端链式求和恰为
        2·g_mid。
        """
        batch_indices, midpoints_xy = ctx.saved_tensors
        # 布局契约：图像 [B,H,W]、像素索引 [y,x]、坐标 (x,y) 连续；行 0 是
        # 最低 Y，与 points_to_canvas 输出一致，直接按连续坐标插值。
        size = grad_output.shape[-1]  # 正方形 canvas 边长
        x = midpoints_xy[:, 0]  # [M]
        y = midpoints_xy[:, 1]
        # 越界中点整体置零：先夹回边界取值，再按 inside 清零。
        inside = (x >= 0.0) & (x <= size - 1.0) & (y >= 0.0) & (y <= size - 1.0)
        xc = x.clamp(0.0, float(size - 1))
        yc = y.clamp(0.0, float(size - 1))
        x0 = torch.floor(xc)  # 左列整数格点
        y0 = torch.floor(yc)  # 下行整数格点
        x1 = (x0 + 1.0).clamp(max=float(size - 1))  # 右列（边界折回，权重为 0）
        y1 = (y0 + 1.0).clamp(max=float(size - 1))  # 上行（边界折回，权重为 0）
        wx = xc - x0  # x 方向权重
        wy = yc - y0  # y 方向权重
        # 扁平索引四角一次取值：每条 membership 只读 4 个像素，避免整图 gather。
        plane = size * size  # 单张图像素数
        base = batch_indices * plane  # [M] 扁平基址
        flat = grad_output.reshape(-1)  # [B*H*W]
        x0i = x0.long()
        y0i = y0.long()
        x1i = x1.long()
        y1i = y1.long()
        v00 = flat[base + y0i * size + x0i]
        v01 = flat[base + y0i * size + x1i]
        v10 = flat[base + y1i * size + x0i]
        v11 = flat[base + y1i * size + x1i]
        g_mid = (v00 * (1.0 - wx) * (1.0 - wy) + v01 * wx * (1.0 - wy)
                 + v10 * (1.0 - wx) * wy + v11 * wx * wy)
        g_mid = torch.where(inside, g_mid, torch.zeros_like(g_mid))
        return None, 2.0 * g_mid, None, None


def optimize_gradient_macro(
        problem: MacroProblem,
        model: LithographyModel,
        config: GradientMBOPCConfig,
        target_cache: TargetCanvasCache,
        *,
        on_tiles_completed: OnTilesCompleted | None = None,
) -> GradientMBOPCResult:
    """优化一个 macro 的 owner 边段法向位移并返回最佳已评价合法状态。"""
    # 入口契约：进入 GPU 大分配前挡住不兼容。
    segment_count = problem.segments.segment_count  # 段数 S
    canvas_pixels = int(problem.macro.canvas_pixels)
    if int(model.config.canvas) != canvas_pixels:
        raise ValueError("模型画布与 problem 画布不一致")
    if config.epe_distance_dbu > float(problem.macro.context_dbu):
        raise ValueError("epe_distance_dbu 超过 problem 的 context 宽度")
    owner_ids = np.flatnonzero(problem.owner_indices >= 0)  # owner 段全局号
    if len(owner_ids) == 0:
        # 空或纯 context macro：没有可训练参数，O=0 必然没有计分像素，
        # 任何评价都只会得到 0/0；直接以全零 baseline 停止，不建 optimizer。
        empty = GradientMBOPCIterationRecord(
            state_index=0, total_loss=0.0, nominal_l2_loss=0.0,
            process_l2_loss=0.0, pvband_loss=0.0, l2=0, pvband=0, epe=0,
            valid_probes=0, ambiguous_probes=0, displaced_segments=0,
            elapsed_seconds=0.0)
        return GradientMBOPCResult(
            np.zeros(segment_count, dtype=np.float64), (empty,), 0,
            "no_owned_segments", None)
    macro_id = problem.macro.macro_id  # cache 键与错误消息的 macro 部分
    pixel_dbu = int(problem.macro.pixel_dbu)
    core_count = problem.macro.core_count
    max_displacement = float(problem.fragmentation.max_displacement_dbu)
    # 初始化：owner 映射、参考中点/法向与探针坐标只建一次，全部状态迭代
    # 复用（同轮内不得重建 mapping）。
    # segment_to_parameter 把 owner 段全局号压缩成 Adam 参数下标 [0, O)：
    # 非 owner 段恒 -1，owner 段按 owner_ids 顺序编号——parameters 的每个
    # 元素经它反向定位到唯一段（如 5 段中第 1、2 段是 owner，则
    # [-1,0,1,-1,-1]，其中 0、1 即两个可训练参数的下标）。
    segment_to_parameter = np.full(segment_count, -1, dtype=np.int32)
    segment_to_parameter[owner_ids] = np.arange(len(owner_ids), dtype=np.int32)
    reference = problem.segments.materialize()  # 参考几何唯一物化（探针用）
    # 零位移参考几何与段采样中点：target/EPE 基准与 state0 采样共用一次
    # 重构；中点由重构几何提供（含 corner miter 切向调整），非刚体推算。
    reference_region, current_segment_midpoints = (
        reconstruct_region_with_midpoints(
            problem, np.zeros(segment_count, dtype=np.float64)))
    core_owner_members = []  # 每 core 的 owner 段号（EPE 探针专用，语义不变）
    core_sampling_members = []  # 每 core 全部可见段中的 owner 段（梯度采样）
    probe_inner_xy = []  # 每 core 的参考探针 canvas 坐标（None=无 owner 段）
    probe_outer_xy = []
    total_pixels = 0  # loss 归一分母 P：全部 core ownership 像素数
    for core_index in range(core_count):
        spec = problem.macro.core(core_index)  # 即时构造 CoreSpec，不常驻
        total_pixels += int(ownership_canvas(
            spec.ownership_box, spec.context_box, pixel_dbu,
            canvas_pixels).sum())
        # 梯度采样按 membership：该 core 可见的所有段中，凡 owner
        # 段都在本 core 的 canvas 采样一次并累加到同一参数——跨 core 边界段
        # 的邻 tile 贡献不丢弃；采样与 owner（发布归属）职责分离。
        members = np.asarray(problem.segments_for_core(core_index))
        core_sampling_members.append(
            members[segment_to_parameter[members] >= 0])
        owner_members = problem.owner_segments_for_core(core_index)
        core_owner_members.append(owner_members)
        if len(owner_members):  # 探针围绕参考边定义，坐标与状态无关
            inner_dbu, outer_dbu = edge_probe_points(
                reference.starts[owner_members], reference.ends[owner_members],
                reference.normals[owner_members], config.epe_distance_dbu)
            probe_inner_xy.append(points_to_canvas(
                inner_dbu, spec.context_box, pixel_dbu, canvas_pixels))
            probe_outer_xy.append(points_to_canvas(
                outer_dbu, spec.context_box, pixel_dbu, canvas_pixels))
        else:
            probe_inner_xy.append(None)
            probe_outer_xy.append(None)
    del reference  # 探针已提取，释放全量段几何数组
    if total_pixels == 0:
        # 有 owner 段却算不出任何计分像素，属于数据损坏，不能静默除零。
        raise ValueError("存在 owner 段但 ownership 计分像素为 0（数据不一致）")
    device = model.device  # 参数与批张量的目标设备
    threshold = float(model.config.print_threshold)  # 离散诊断二值阈值
    # 三工艺角一次前向
    conditions = (model.condition("nominal"), model.condition("dose_max"),
                  model.condition("defocus_min"))
    # 唯一可训练参数：owner 法向位移 [O]
    parameters = torch.zeros(
        len(owner_ids), dtype=torch.float32, device=device, requires_grad=True)
    # 固定超参（规格钉死，不新增配置面）
    optimizer = torch.optim.Adam(
        [parameters], lr=config.learning_rate_dbu, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0, amsgrad=False)
    current_region = reference_region  # 当前已发布合法几何
    records = []  # 已评价状态记录（records[0] 恒为 baseline）
    best_loss = float("inf")  # 严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_owner = np.zeros(len(owner_ids), dtype=np.float64)
    stop_reason = None
    stop_detail = None
    candidate_full = np.zeros(segment_count, dtype=np.float64)  # 展开缓冲
    for state_index in range(config.iterations + 1):
        can_update = state_index < config.iterations  # 末状态纯评价
        if can_update:
            optimizer.zero_grad(set_to_none=True)  # 梯度按状态清零后累积
        started = time.perf_counter()  # 本状态评价计时
        current_owner = parameters.detach().cpu().numpy().astype(np.float64)
        sums = {"nominal": 0.0, "process": 0.0, "pvband": 0.0}  # 连续分量累计
        diag = {"l2": 0, "pvband": 0, "epe": 0, "valid": 0, "ambiguous": 0}
        for batch_start in range(0, core_count, config.batch_size):
            # 本批 core（行优先稳定序）
            core_indices = list(range(
                batch_start, min(batch_start + config.batch_size, core_count)))
            batch_count = len(core_indices)
            # target 批（uint8 缓存格式）
            targets = np.empty((batch_count, canvas_pixels, canvas_pixels),
                               dtype=np.uint8)
            # 当前 mask 批
            masks = np.empty((batch_count, canvas_pixels, canvas_pixels),
                             dtype=np.float32)
            # 计分像素批
            ownership = np.empty((batch_count, canvas_pixels, canvas_pixels),
                                 dtype=np.bool_)
            member_slots = []  # 梯度采样条目的 batch 槽位（int64）
            member_params = []  # 梯度采样条目指向的参数索引（int64）
            member_mids = []  # 梯度采样条目的当前中点 canvas 坐标
            probe_slots = []  # EPE 探针条目的 batch 槽位（与梯度条目独立）
            for slot, core_index in enumerate(core_indices):  # 逐 core 组批
                spec = problem.macro.core(core_index)
                cached = target_cache.get(macro_id, core_index)
                if cached is None:  # 未命中：参考几何栅格化并回填缓存
                    cached = np.rint(rasterize_mask_canvas(
                        reference_region, spec.context_box, pixel_dbu,
                        canvas_pixels, polarity=problem.polarity)
                        * 255.0).astype(np.uint8)
                    target_cache.put(macro_id, core_index, cached)
                targets[slot] = cached
                # 当前候选直接栅格
                masks[slot] = rasterize_mask_canvas(
                    current_region, spec.context_box, pixel_dbu,
                    canvas_pixels, polarity=problem.polarity)
                # 唯一计分像素
                ownership[slot] = ownership_canvas(
                    spec.ownership_box, spec.context_box, pixel_dbu,
                    canvas_pixels)
                sampling_members = core_sampling_members[core_index]
                if len(sampling_members):  # 梯度采样：全部 membership 中 owner 段
                    # 采样中点由当前已发布的重构几何提供（corner miter 后含
                    # 切向调整），与栅格化用 Region 恒来自同一次合法重构。
                    midpoints_dbu = current_segment_midpoints[sampling_members]
                    # DBU→canvas 唯一换算
                    member_mids.append(points_to_canvas(
                        midpoints_dbu, spec.context_box, pixel_dbu,
                        canvas_pixels))
                    member_slots.append(np.full(
                        len(sampling_members), slot, dtype=np.int64))
                    member_params.append(segment_to_parameter[
                        sampling_members].astype(np.int64))
                owner_members = core_owner_members[core_index]  # 探针语义
                if len(owner_members):  # 无 owner 段的 core 仍计完成 tile
                    # 探针槽位独立于梯度条目
                    probe_slots.append(np.full(
                        len(owner_members), slot, dtype=np.int64))
            trainable = bool(member_params)  # 本批是否有可训练 membership
            build_graph = trainable and can_update  # 末状态纯评价不建图
            # uint8→float32/255
            target_tensor = torch.from_numpy(targets).to(
                device=device, dtype=torch.float32).div_(255.0)
            ownership_tensor = torch.from_numpy(ownership).to(device=device)
            hard = torch.from_numpy(masks).to(device=device)
            if build_graph:  # STE：forward 数值不变，autograd 边接到位移
                # 批号与参数索引一次上设备
                slots = torch.from_numpy(
                    np.concatenate(member_slots)).to(device)
                owned = torch.from_numpy(
                    np.concatenate(member_params)).to(device)
                # 中点坐标转 float32 上设备
                mids = torch.from_numpy(
                    np.concatenate(member_mids)).to(device=device,
                                                     dtype=torch.float32)
                local = parameters[owned]  # gather 出 [M]，autograd 边
                mask_tensor = _EdgeGradientMask.apply(hard, local, slots, mids)
            else:
                mask_tensor = hard  # 无梯度路径直通
            printed = model.forward_many(mask_tensor, conditions)  # 一次 FFT
            nominal = printed["nominal"]
            dose_max = printed["dose_max"]
            defocus_min = printed["defocus_min"]
            if build_graph:  # 建图版连续 loss：backward 累积到 parameters.grad
                l_nom = ((nominal - target_tensor) ** 2
                         * ownership_tensor).sum()
                l_proc = (((dose_max - target_tensor) ** 2
                           + (defocus_min - target_tensor) ** 2)
                          * ownership_tensor).sum()
                l_pv = ((dose_max - defocus_min) ** 2
                        * ownership_tensor).sum()
                batch_loss = (config.weight_nominal_l2 * l_nom
                              + config.weight_process_l2 * l_proc
                              + config.weight_pvband * l_pv) / total_pixels
                batch_loss.backward()  # 批间梯度直接累加（同一参数快照）
                triple = (l_nom, l_proc, l_pv)
            else:
                with torch.no_grad():  # 纯评价路径不建图
                    triple = (((nominal - target_tensor) ** 2
                               * ownership_tensor).sum(),
                              (((dose_max - target_tensor) ** 2
                                + (defocus_min - target_tensor) ** 2)
                               * ownership_tensor).sum(),
                              ((dose_max - defocus_min) ** 2
                               * ownership_tensor).sum())
            with torch.no_grad():  # 离散诊断只读数值，不进入训练
                sums["nominal"] += float(triple[0]) / total_pixels
                sums["process"] += float(triple[1]) / total_pixels
                sums["pvband"] += float(triple[2]) / total_pixels
                diag["l2"] += evaluate_binary_l2(
                    target_tensor, nominal, threshold=threshold,
                    ownership_mask=ownership_tensor)
                diag["pvband"] += evaluate_pvband(
                    dose_max, defocus_min, threshold=threshold,
                    ownership_mask=ownership_tensor)
                if probe_slots:  # 本批 owner 探针一次批量评价（owner-core 语义）
                    batch_index_tensor = torch.from_numpy(
                        np.concatenate(probe_slots))
                    inner_xy = torch.from_numpy(np.concatenate(
                        [probe_inner_xy[c] for c in core_indices
                         if len(core_owner_members[c])]))
                    outer_xy = torch.from_numpy(np.concatenate(
                        [probe_outer_xy[c] for c in core_indices
                         if len(core_owner_members[c])]))
                    # 阈值跟随模型 PrintThresh
                    epe_result = evaluate_edge_probes(
                        target_tensor, nominal, batch_index_tensor,
                        inner_xy, outer_xy, threshold=threshold)
                    diag["epe"] += epe_result.violation_count
                    diag["valid"] += int(epe_result.valid.cpu().numpy().sum())
                    diag["ambiguous"] += int(
                        epe_result.ambiguous.cpu().numpy().sum())
            # 释放：批结束只保留标量与梯度，光刻图和批张量立即失去引用。
            del printed, nominal, dose_max, defocus_min, mask_tensor, hard
            del target_tensor, ownership_tensor
            if on_tiles_completed is not None:  # backward 且释放后才报进度
                on_tiles_completed(batch_count)
        nominal_loss = sums["nominal"]
        process_loss = sums["process"]
        pvband_loss = sums["pvband"]
        total_loss = (config.weight_nominal_l2 * nominal_loss
                      + config.weight_process_l2 * process_loss
                      + config.weight_pvband * pvband_loss)
        if not (isfinite(total_loss) and isfinite(nominal_loss)
                and isfinite(process_loss) and isfinite(pvband_loss)):
            raise FloatingPointError(
                f"{macro_id} state {state_index} 连续 loss 非有限")
        records.append(GradientMBOPCIterationRecord(
            state_index=state_index, total_loss=total_loss,
            nominal_l2_loss=nominal_loss, process_l2_loss=process_loss,
            pvband_loss=pvband_loss, l2=diag["l2"], pvband=diag["pvband"],
            epe=diag["epe"], valid_probes=diag["valid"],
            ambiguous_probes=diag["ambiguous"],
            displaced_segments=int(np.count_nonzero(current_owner)),
            elapsed_seconds=time.perf_counter() - started))
        if total_loss < best_loss:  # 严格更小才更新；相同保留较早状态
            best_loss = total_loss
            best_state_index = state_index
            best_owner = current_owner.copy()
        if total_loss == 0.0:  # 连续 loss 恰为零即达目的
            stop_reason = "zero_loss"
            break
        if state_index == config.iterations:  # 轮次自然用尽
            stop_reason = "iteration_limit"
            break
        grad = parameters.grad  # 全部 batch 完成后的唯一屏障内检查
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise FloatingPointError(
                f"{macro_id} state {state_index} 梯度缺失或非有限")
        before = parameters.detach().clone()  # 更新前快照（no_update 判据）
        optimizer.step()  # 每 state 至多一次
        with torch.no_grad():
            parameters.clamp_(-max_displacement, max_displacement)  # 先裁上限
        if not bool(torch.isfinite(parameters).all()):
            raise FloatingPointError(
                f"{macro_id} state {state_index} 候选参数非有限")
        if torch.equal(parameters.detach(), before):  # 梯度全零时步长为零
            stop_reason = "no_update"
            break
        candidate_full[owner_ids] = parameters.detach().cpu().numpy().astype(
            np.float64)
        try:  # 候选必须先通过方向/hole/有效性守卫才可发布
            candidate_region, candidate_midpoints = (
                reconstruct_region_with_midpoints(problem, candidate_full))
        except (ValueError, ReconstructionError) as exc:
            # 宽捕获有实测依据：几何退化（如位移共线使 ring 顶点不足）会以
            # ValueError 从 KLayout 冒出而非 ReconstructionError（simple.py
            # 同款证据）；收窄需改 reconstruction.py 包装。
            stop_reason = "invalid_geometry"
            stop_detail = f"state {state_index + 1} 候选重建失败：{exc}"
            break
        # Region 与采样中点绑定发布：下一状态的栅格化与梯度采样恒来自
        # 同一次合法候选重构（失败时两者都不更新）。
        current_region = candidate_region
        current_segment_midpoints = candidate_midpoints
    if stop_reason is None:  # 防御兜底（iterations>=1 时循环内必设）
        stop_reason = "iteration_limit"
    best_full = np.zeros(segment_count, dtype=np.float64)
    best_full[owner_ids] = best_owner  # context 段恒 0
    return GradientMBOPCResult(
        best_displacements=best_full, records=tuple(records),
        best_state_index=best_state_index, stop_reason=stop_reason,
        stop_detail=stop_detail)
