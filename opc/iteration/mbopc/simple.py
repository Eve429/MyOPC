"""固定步长、EPE 驱动的最简 MB-OPC：状态评价、提案与单 macro 完整迭代。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import klayout.db as kdb
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
from opc.input import points_to_canvas, rasterize_mask_canvas
from opc.input.edge import MacroProblem, reconstruct_region
from opc.input.edge.fragmentation import SegmentGeometry

from ._batching import (
    MacroStaticPack,
    assemble_probe_batch,
    discrete_batch_diagnostics,
    iter_core_batches,
    pack_macro_statics,
    upload_eval_batch,
)
from ._cache import TargetCanvasCache

# 进度回调类型：参数是本批真正完成评价与释放的 tile 数。
OnTilesCompleted = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class SimpleMBOPCConfig:
    """保存已经转换到 DBU 的离散 EPE 迭代参数。"""

    iterations: int  # 最多发布更新次数
    initial_step_dbu: float  # 初始绝对法向步长
    decay_every: int  # 步长减半周期（每过这么多状态步长减半）
    epe_distance_dbu: float  # inner/outer 探针距离
    batch_size: int  # 一次 forward 的 core 数
    target_cache_bytes: int  # CPU target uint8 LRU 上限

    def __post_init__(self) -> None:
        """校验迭代参数自身的数值契约，跨层参数（步长/上下文）由求解器入口复验。"""
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations 必须是至少 1 的整数")
        if not np.isfinite(self.initial_step_dbu) or self.initial_step_dbu <= 0.0:
            raise ValueError("initial_step_dbu 必须是有限正数")
        if not isinstance(self.decay_every, int) or self.decay_every < 1:
            raise ValueError("decay_every 必须是至少 1 的整数")
        if not np.isfinite(self.epe_distance_dbu) or self.epe_distance_dbu <= 0.0:
            raise ValueError("epe_distance_dbu 必须是有限正数")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size 必须是至少 1 的整数")
        if not isinstance(self.target_cache_bytes, int) or self.target_cache_bytes < 0:
            raise ValueError("target_cache_bytes 必须是非负整数")


@dataclass(frozen=True, slots=True)
class SimpleMBOPCStep:
    """保存一个 macro 已评价状态的指标及下一状态提案。"""

    next_displacements: NDArray[np.float64]
    epe: int
    l2: int
    pvband: int
    valid_probes: int
    ambiguous_probes: int
    moved_segments: int


@dataclass(frozen=True, slots=True)
class SimpleMBOPCIterationRecord:
    """保存 baseline 或一次移动后状态的实际评价结果。"""

    state_index: int  # 0=baseline；1..N=对应位移完成后的状态
    step_dbu: float  # 产生本状态时使用的步长；baseline 为 0
    epe: int  # 本状态实际 EPE
    l2: int  # 本状态实际二值 L2
    pvband: int  # 本状态实际 PVBand
    valid_probes: int  # 本状态有效探针
    ambiguous_probes: int  # 本状态歧义探针
    moved_segments: int  # 从上一状态移动到本状态的段数
    elapsed_seconds: float  # 重建并评价本状态的耗时


@dataclass(frozen=True, slots=True)
class SimpleMBOPCResult:
    """保存单 macro 的最佳已评价位移、全部状态记录和停止原因。"""

    best_displacements: NDArray[np.float64]
    records: tuple[SimpleMBOPCIterationRecord, ...]  # records[0] 固定为 baseline
    best_state_index: int  # 0 表示零位移 baseline 最优
    stop_reason: str  # zero_epe/no_update/invalid_geometry/iteration_limit
    stop_detail: str | None  # 非法候选的明确原因；正常停止为 None


def evaluate_state(
    problem: MacroProblem,
    current_region: kdb.Region,
    current_displacements: NDArray[np.float64],
    model: LithographyModel,
    config: SimpleMBOPCConfig,
    step_dbu: float,
    target_cache: TargetCanvasCache,
    *,
    can_update: bool,
    reference: SegmentGeometry | None = None,
    pack: MacroStaticPack | None = None,
    on_tiles_completed: OnTilesCompleted | None = None,
) -> SimpleMBOPCStep:
    """评价一个 macro 当前状态，并产生同步 owner 位移提案。

    pack（计分画布/参考探针/零位移参考候选）由 optimize_simple_macro
    预打包逐状态复用，缺省时现算，reference 同理（两条 None 路径等价）；
    pack 优先时 reference 不参与本调用。
    """
    # 入口契约：位移形状/有限性/context 归零与 canvas 一致性。MacroProblem 构造
    # 已保证 owner/CSR 不变量，这里不重复校验。
    segment_count = problem.segments.segment_count  # 段数 S
    current = np.ascontiguousarray(current_displacements, dtype=np.float64)
    if current.shape != (segment_count,) or not np.all(np.isfinite(current)):
        raise ValueError("current_displacements 必须是长度等于段数的有限向量")
    context_mask = problem.owner_indices < 0  # 只读 context 段
    if len(current) and np.any(current[context_mask] != 0.0):
        raise ValueError("context 段（owner=-1）位移必须恒为 0")
    canvas_pixels = int(problem.macro.canvas_pixels)
    if int(model.config.canvas) != canvas_pixels:
        raise ValueError("模型画布与 problem 画布不一致")
    if not np.isfinite(step_dbu) or step_dbu < 0.0:
        raise ValueError("step_dbu 必须是非负有限数")
    # 固定几何：参考（零位移）端点/法向与探针坐标、计分画布均每 macro 一次。
    if pack is None:
        if reference is None:
            reference = problem.segments.materialize()
        pack = pack_macro_statics(
            problem,
            epe_distance_dbu=config.epe_distance_dbu,
            reference_geometry=reference,
            reference_region=reconstruct_region(problem, np.zeros(segment_count, dtype=np.float64)),
            to_canvas=points_to_canvas,
        )
    max_displacement = float(problem.fragmentation.max_displacement_dbu)
    next_values = current.copy()  # 提案缓冲（can_update=False 时不写方向）
    written = np.zeros(segment_count, dtype=np.bool_)  # 方向唯一写标记
    # 批间标量累计
    totals = {"epe": 0, "l2": 0, "pvband": 0, "valid": 0, "ambiguous": 0}
    threshold = float(model.config.print_threshold)  # 像素指标二值阈值
    device = model.device
    # 三工艺角条件：标称 / 大剂量 / 离焦小剂量（每 macro 一次，与
    # gradient 的 ctx.conditions 同语义）
    conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
    # 公共组批（target 缓存/当前候选栅格/静态计分画布）单源共用；
    # rasterize 钩子传本模块全局，monkeypatch 锚点保持在求解器模块。
    for core_indices, targets, masks, ownership in iter_core_batches(
        problem, pack, current_region, target_cache, batch_size=config.batch_size, rasterize=rasterize_mask_canvas
    ):
        # 本批 owner 探针（静态坐标）槽位与 canvas 坐标一次拼接
        probe_slots, inner_xy, outer_xy = assemble_probe_batch(pack, core_indices)
        # 光刻：一次 forward_many 出三工艺角。
        with torch.no_grad():  # 离散方法不需要梯度图
            target_tensor, mask_tensor, ownership_tensor = upload_eval_batch(targets, masks, ownership, device)
            printed = model.forward_many(mask_tensor, conditions)  # 共享一次 FFT
            # 像素指标与 EPE：公共离散诊断（evaluate_* 补丁锚在本模块）
            l2, pvband, epe_result = discrete_batch_diagnostics(
                target_tensor,
                printed,
                ownership_tensor,
                threshold,
                probe_slots,
                inner_xy,
                outer_xy,
                binary_l2=evaluate_binary_l2,
                pvband=evaluate_pvband,
                edge_probes=evaluate_edge_probes,
            )
            totals["l2"] += l2
            totals["pvband"] += pvband
            if epe_result is not None:
                totals["epe"] += epe_result.violation_count  # 违规段数
                # 回切整 batch 化：每张小张量只做一次设备→主机搬运，随后全部
                # 统计与写回在 numpy 侧切片完成，避免逐 core 的 GPU 同步。
                valid_all = epe_result.valid.cpu().numpy()
                ambiguous_all = epe_result.ambiguous.cpu().numpy()
                totals["valid"] += int(valid_all.sum())
                totals["ambiguous"] += int(ambiguous_all.sum())
                if can_update:  # 方向只写提案缓冲，current 全程只读
                    # -1/0/+1 方向 × 当前提案步长（一次取回）
                    moves = epe_result.directions.cpu().numpy().astype(np.float64) * step_dbu
                    cursor = 0  # 探针游标（按批内 core 顺序回切）
                    for core_index in core_indices:
                        idx = pack.owner_members[core_index]
                        if not len(idx):  # 空 owner core 无探针
                            continue
                        piece = slice(cursor, cursor + len(idx))
                        cursor += len(idx)
                        next_values[idx] += moves[piece]
                        written[idx] = True
            # 释放：批结束只保留标量与方向，GPU 张量立即失去引用。
            del printed, mask_tensor, target_tensor, ownership_tensor
        if on_tiles_completed is not None:  # 释放后才报告进度
            on_tiles_completed(len(core_indices))
    # 出口：核对方向写集与 context 归零，提案裁到位移上限。
    if can_update:  # 评价专用调用不产生提案，无需核对写集
        if not np.array_equal(written, problem.owner_indices >= 0):
            raise RuntimeError("owner 段未全部产生方向或出现重复方向")
        np.clip(next_values, -max_displacement, max_displacement, out=next_values)
        if np.any(next_values[context_mask] != 0.0):  # clip 不会触碰 context，防御
            raise RuntimeError("context 段位移被意外修改")
    moved = int(np.count_nonzero(next_values != current))  # 提案改变段数
    # 指标属于刚评价的输入状态，next 只是提案
    return SimpleMBOPCStep(
        next_displacements=next_values,
        epe=totals["epe"],
        l2=totals["l2"],
        pvband=totals["pvband"],
        valid_probes=totals["valid"],
        ambiguous_probes=totals["ambiguous"],
        moved_segments=moved,
    )


def optimize_simple_macro(
    problem: MacroProblem,
    model: LithographyModel,
    config: SimpleMBOPCConfig,
    target_cache: TargetCanvasCache,
    *,
    on_tiles_completed: OnTilesCompleted | None = None,
) -> SimpleMBOPCResult:
    """让单个 macro 独立完成 baseline 和全部离散 EPE 迭代。"""
    # 跨层参数复验（工作流配置层已查过，纯函数入口再挡一次直接调用方）。
    max_displacement = float(problem.fragmentation.max_displacement_dbu)
    if config.initial_step_dbu > max_displacement:
        raise ValueError("initial_step_dbu 超过 problem 的位移上限")
    if config.epe_distance_dbu > float(problem.macro.context_dbu):
        raise ValueError("epe_distance_dbu 超过 problem 的 context 宽度")

    def step_for(target_state: int) -> float:
        """返回产生第 target_state 次位移的步长（每 decay_every 状态减半）。"""
        return config.initial_step_dbu * 0.5 ** ((target_state - 1) // config.decay_every)

    segment_count = problem.segments.segment_count  # 段数 S
    owner_count = int(np.count_nonzero(problem.owner_indices >= 0))
    zeros = np.zeros(segment_count, dtype=np.float64)  # 零位移状态
    # 参考几何整个迭代只物化一次：baseline 与每个移动后状态的评价复用同一
    # 份端点/法向（探针始终围绕参考边定义，与位移状态无关）。
    reference = problem.segments.materialize()
    # 静态打包每 macro 一次：计分画布/参考探针坐标/零位移参考候选；target
    # 缓存 miss 源与 baseline mask 共用同一次零位移重构（纯函数，确定性）。
    started = time.perf_counter()
    baseline_region = reconstruct_region(problem, zeros)
    pack = pack_macro_statics(
        problem,
        epe_distance_dbu=config.epe_distance_dbu,
        reference_geometry=reference,
        reference_region=baseline_region,
        to_canvas=points_to_canvas,
    )
    pending_step = step_for(1)  # baseline 提案使用的步长
    # 评价 + State 1 提案
    proposal = evaluate_state(
        problem,
        baseline_region,
        zeros,
        model,
        config,
        pending_step,
        target_cache,
        can_update=True,
        reference=reference,
        pack=pack,
        on_tiles_completed=on_tiles_completed,
    )
    # records[0] 固定是 baseline
    records = [
        SimpleMBOPCIterationRecord(
            state_index=0,
            step_dbu=0.0,
            epe=proposal.epe,
            l2=proposal.l2,
            pvband=proposal.pvband,
            valid_probes=proposal.valid_probes,
            ambiguous_probes=proposal.ambiguous_probes,
            moved_segments=0,
            elapsed_seconds=time.perf_counter() - started,
        )
    ]
    best_epe = proposal.epe  # 最佳状态 EPE（EPE 相同保留较早状态，由严格小于实现）
    best_state_index = 0  # baseline 先当最佳
    best_displacements = zeros.copy()
    stop_reason: str | None = None
    stop_detail: str | None = None
    if owner_count and proposal.valid_probes == 0:  # 有段却无有效探针
        # 「无法评价」不是「零违规」：探针越过窄特征落入异侧（如 2nm 壁 +
        # 8nm 探针距离）时全部探针被判无效，epe 恒为 0；此时以零位移为 best
        # 终止并显式记录原因，不冒充收敛。
        stop_reason = "insufficient_probes"
        stop_detail = f"有效 EPE 探针 0 个 / owner 段 {owner_count} 个，无法评价（探针距离可能大于最窄特征）"
    elif proposal.epe == 0:  # baseline 已无违规
        stop_reason = "zero_epe"  # 直接以零位移为最佳
    else:  # 常规路径：逐状态移动并评价
        for state_index in range(1, config.iterations + 1):  # 移动后状态序
            candidate = proposal.next_displacements  # 上一评价的提案
            candidate_moved = proposal.moved_segments  # 该提案改变的段数
            if not candidate_moved:  # 提案与当前完全相同
                # 同一状态再评一次不产生任何新信息（指标、几何全部不变），
                # 直接停止，省去一次完整重建与光刻前向。
                stop_reason = "no_update"
                break
            started = time.perf_counter()
            try:  # 候选必须先通过方向/hole/有效性守卫
                candidate_region = reconstruct_region(problem, candidate)
            except (ValueError, ReconstructionError) as exc:  # 非法几何终止
                # 宽捕获有实测依据：几何退化（如共线 ring 少于三顶点）会以
                # ValueError 从 KLayout 冒出而非 ReconstructionError，收窄需
                # 改 reconstruction.py 包装；位移 shape/有限性已在
                # evaluate_state 入口拦截，此处 ValueError 几乎只可能是几何退化。
                stop_reason = "invalid_geometry"  # 保留最后合法 best
                # 错误原因不得吞掉
                stop_detail = f"state {state_index} 候选重建失败：{exc}"
                break
            can_propose = state_index < config.iterations  # 末状态不再生成
            pending_next = step_for(state_index + 1)  # 被丢弃提案的步长（末状态）
            # 移动后状态评价（末状态纯评价）
            proposal = evaluate_state(
                problem,
                candidate_region,
                candidate,
                model,
                config,
                pending_next,
                target_cache,
                can_update=can_propose,
                reference=reference,
                pack=pack,
                on_tiles_completed=on_tiles_completed,
            )
            # State N 指标属第 N 次位移后状态；moved 为产生本状态时移动的段数
            records.append(
                SimpleMBOPCIterationRecord(
                    state_index=state_index,
                    step_dbu=pending_step,
                    epe=proposal.epe,
                    l2=proposal.l2,
                    pvband=proposal.pvband,
                    valid_probes=proposal.valid_probes,
                    ambiguous_probes=proposal.ambiguous_probes,
                    moved_segments=candidate_moved,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            pending_step = pending_next  # 下一状态记录使用的步长
            if owner_count and proposal.valid_probes == 0:  # 移动后无法评价
                # 必须先于 best 比较终止：valid_probes==0 时 epe 恒 0，若放行
                # 会被 epe<best 误当成改善状态。
                stop_reason = "insufficient_probes"
                stop_detail = f"state {state_index} 有效 EPE 探针 0 个 / owner 段 {owner_count} 个，无法评价"
                break
            if proposal.epe < best_epe:  # 严格更小才更新；相同保留较早状态
                best_epe = proposal.epe
                best_state_index = state_index
                best_displacements = candidate.copy()
            if proposal.epe == 0:  # 无违规即达目的
                stop_reason = "zero_epe"
                break
            if can_propose and proposal.moved_segments == 0:  # 提案不再移动
                # 末状态 can_update=False 时 moved 恒 0，不构成 no_update 证据。
                stop_reason = "no_update"
                break
        if stop_reason is None:  # 状态数自然用尽
            stop_reason = "iteration_limit"
    return SimpleMBOPCResult(
        best_displacements=best_displacements,
        records=tuple(records),
        best_state_index=best_state_index,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
    )
