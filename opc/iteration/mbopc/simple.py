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
from opc.input import ownership_canvas, points_to_canvas, rasterize_mask_canvas
from opc.input.edge import MacroProblem, reconstruct_region
from opc.input.edge.fragmentation import SegmentGeometry
from opc.input.edge.sampling import edge_probe_points

from ._cache import TargetCanvasCache

# 进度回调类型：参数是本批真正完成评价与释放的 tile 数。
OnTilesCompleted = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class SimpleMBOPCConfig:
    """保存已经转换到 DBU 的离散 EPE 迭代参数。"""

    iterations: int              # 最多发布更新次数
    initial_step_dbu: float      # 初始绝对法向步长
    decay_every: int             # 步长减半周期（每过这么多轮步长减半）
    epe_distance_dbu: float      # inner/outer 探针距离
    batch_size: int              # 一次 forward 的 core 数
    target_cache_bytes: int      # CPU target uint8 LRU 上限

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
        if (not isinstance(self.target_cache_bytes, int)
                or self.target_cache_bytes < 0):
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
class IterationRecord:
    """保存 baseline 或一次移动后状态的实际评价结果。"""

    round_index: int             # 0=baseline；1..N=对应位移完成后的状态
    step_dbu: float              # 产生本状态时使用的步长；baseline 为 0
    epe: int                     # 本状态实际 EPE
    l2: int                      # 本状态实际二值 L2
    pvband: int                  # 本状态实际 PVBand
    valid_probes: int            # 本状态有效探针
    ambiguous_probes: int        # 本状态歧义探针
    moved_segments: int          # 从上一状态移动到本状态的段数
    elapsed_seconds: float       # 重建并评价本状态的耗时


@dataclass(frozen=True, slots=True)
class SimpleMBOPCResult:
    """保存单 macro 的最佳已评价位移、全部状态记录和停止原因。"""

    best_displacements: NDArray[np.float64]
    records: tuple[IterationRecord, ...]  # records[0] 固定为 baseline
    best_round: int                       # 0 表示零位移 baseline 最优
    stop_reason: str                      # zero_epe/no_update/invalid_geometry/iteration_limit
    stop_detail: str | None               # 非法候选的明确原因；正常停止为 None


def evaluate_and_propose(
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
        on_tiles_completed: OnTilesCompleted | None = None,
) -> SimpleMBOPCStep:
    """评价一个 macro 当前状态，并产生同步 owner 位移提案。"""
    # 入口契约：位移形状/有限性/context 归零与 canvas 一致性。MacroProblem 构造
    # 已保证 owner/CSR 不变量，这里不重复校验。
    segment_count = problem.segments.segment_count  # 段数 S
    current = np.ascontiguousarray(current_displacements, dtype=np.float64)
    if current.shape != (segment_count,) or not np.all(np.isfinite(current)):
        raise ValueError("current_displacements 必须是长度等于段数的有限向量")
    context_mask = problem.owner_indices < 0  # 只读 context 段
    if len(current) and np.any(current[context_mask] != 0.0):
        raise ValueError("context 段（owner=-1）位移必须恒为 0")
    canvas_pixels = int(problem.macro.canvas_pixels)  # 问题侧画布
    if int(model.config.canvas) != canvas_pixels:
        raise ValueError("模型画布与 problem 画布不一致")
    if not np.isfinite(step_dbu) or step_dbu < 0.0:
        raise ValueError("step_dbu 必须是非负有限数")
    # 固定几何：参考（零位移）端点与法向只物化一次，探针始终围绕参考边定义；
    # 多轮迭代经 reference 参数复用同一物化结果（默认 None 时本调用自算）。
    if reference is None:  # 独立调用方未提供
        reference = problem.segments.materialize()  # 现算参考几何
    macro_id = problem.macro.macro_id  # cache 键的 macro 部分
    pixel_dbu = int(problem.macro.pixel_dbu)  # 栅格像素
    max_displacement = float(problem.fragmentation.max_displacement_dbu)  # 位移上限
    next_values = current.copy()  # 提案缓冲（can_update=False 时不写方向）
    written = np.zeros(segment_count, dtype=np.bool_)  # 方向唯一写标记
    # 批间标量累计
    totals = {"epe": 0, "l2": 0, "pvband": 0,
              "valid": 0, "ambiguous": 0}
    reference_region: kdb.Region | None = None  # cache miss 时才重建参考 Region
    core_count = problem.macro.core_count  # tile 总数
    threshold = float(model.config.print_threshold)  # 像素指标二值阈值
    device = model.device  # 目标设备
    for batch_start in range(0, core_count, config.batch_size):  # 分批评价
        # 本批 core（行优先）
        core_indices = list(range(
            batch_start, min(batch_start + config.batch_size, core_count)))
        batch_count = len(core_indices)  # 本批 tile 数
        # target 批（uint8 缓存格式）
        targets = np.empty((batch_count, canvas_pixels, canvas_pixels),
                           dtype=np.uint8)
        # 当前 mask 批
        masks = np.empty((batch_count, canvas_pixels, canvas_pixels),
                         dtype=np.float32)
        # 计分像素批
        ownership = np.empty((batch_count, canvas_pixels, canvas_pixels),
                             dtype=np.bool_)
        probes: list[tuple[int, np.ndarray]] = []  # (batch 槽位, owner 全局索引)
        inner_parts: list[np.ndarray] = []  # 各 core inner canvas 坐标
        outer_parts: list[np.ndarray] = []  # 各 core outer canvas 坐标
        for slot, core_index in enumerate(core_indices):  # 逐 core 组批
            spec = problem.macro.core(core_index)  # 即时构造 CoreSpec，不常驻
            cached = target_cache.get(macro_id, core_index)  # LRU 查询
            if cached is None:  # 未命中：首次用零位移参考几何栅格化
                if reference_region is None:  # 参考候选只重建一次
                    reference_region = reconstruct_region(
                        problem, np.zeros(segment_count, dtype=np.float64))
                # 参考透光率 → uint8
                cached = np.rint(rasterize_mask_canvas(
                    reference_region, spec.context_box, pixel_dbu,
                    canvas_pixels, polarity=problem.polarity) * 255.0
                ).astype(np.uint8)
                target_cache.put(macro_id, core_index, cached)  # 回填缓存
            targets[slot] = cached  # 批内拷贝
            # 当前候选直接栅格
            masks[slot] = rasterize_mask_canvas(
                current_region, spec.context_box, pixel_dbu,
                canvas_pixels, polarity=problem.polarity)
            # 唯一计分像素
            ownership[slot] = ownership_canvas(
                spec.ownership_box, spec.context_box, pixel_dbu, canvas_pixels)
            owner_indices = problem.owner_segments_for_core(core_index)  # 唯一可写段
            if len(owner_indices):  # 空 owner core 无探针，但仍计入完成 tile
                # 参考边中点 ± 法向
                inner_dbu, outer_dbu = edge_probe_points(
                    reference.starts[owner_indices], reference.ends[owner_indices],
                    reference.normals[owner_indices], config.epe_distance_dbu)
                # DBU → 居中 canvas 连续坐标
                inner_parts.append(points_to_canvas(
                    inner_dbu, spec.context_box, pixel_dbu, canvas_pixels))
                outer_parts.append(points_to_canvas(
                    outer_dbu, spec.context_box, pixel_dbu, canvas_pixels))
                probes.append((slot, owner_indices))  # 记录探针归属
        # 光刻：target 送设备转 float32/255，一次 forward_many 出三工艺角。
        with torch.no_grad():  # 离散方法不需要梯度图
            # uint8 → float32/255
            target_tensor = torch.from_numpy(targets).to(
                device=device, dtype=torch.float32).div_(255.0)
            mask_tensor = torch.from_numpy(masks).to(device=device)
            ownership_tensor = torch.from_numpy(ownership).to(device=device)
            # 三工艺角条件：标称 / 大剂量 / 离焦小剂量
            conditions = (model.condition("nominal"),
                          model.condition("dose_max"),
                          model.condition("defocus_min"))
            printed = model.forward_many(mask_tensor, conditions)  # 共享一次 FFT
            nominal = printed["nominal"]  # 标称胶图
            # 像素指标：L2/PVBand 只在 ownership 像素累计，context/padding 不重复计分。
            totals["l2"] += evaluate_binary_l2(
                target_tensor, nominal, threshold=threshold,
                ownership_mask=ownership_tensor)
            totals["pvband"] += evaluate_pvband(
                printed["dose_max"], printed["defocus_min"],
                threshold=threshold, ownership_mask=ownership_tensor)
            # EPE：本批全部 owner 探针一次批量评价（batch 索引指向各自 core 图）。
            if probes:
                # 每个探针的 batch 槽位
                batch_index_tensor = torch.cat([
                    torch.full((len(idx),), slot, dtype=torch.long)
                    for slot, idx in probes])
                inner_xy = torch.from_numpy(np.concatenate(inner_parts))
                outer_xy = torch.from_numpy(np.concatenate(outer_parts))
                # 阈值跟随模型 PrintThresh
                epe_result = evaluate_edge_probes(
                    target_tensor, nominal, batch_index_tensor, inner_xy,
                    outer_xy, threshold=threshold)
                totals["epe"] += epe_result.violation_count  # 违规段数
                # 回切整 batch 化：每张小张量只做一次设备→主机搬运，随后全部
                # 统计与写回在 numpy 侧切片完成，避免逐 core 的 GPU 同步。
                valid_all = epe_result.valid.cpu().numpy()  # 一次取回
                ambiguous_all = epe_result.ambiguous.cpu().numpy()  # 一次取回
                totals["valid"] += int(valid_all.sum())  # 整批求和
                totals["ambiguous"] += int(ambiguous_all.sum())  # 整批求和
                if can_update:  # 方向只写提案缓冲，current 全程只读
                    # -1/0/+1 方向 × 当前提案步长（一次取回）
                    moves = (
                        epe_result.directions.cpu().numpy()
                        .astype(np.float64) * step_dbu)
                    cursor = 0  # 探针游标（按 core 顺序回切）
                    for _, idx in probes:
                        piece = slice(cursor, cursor + len(idx))  # 该 core 段
                        cursor += len(idx)
                        next_values[idx] += moves[piece]  # 在 current 基础上移动
                        written[idx] = True  # 唯一写标记
            # 释放：批结束只保留标量与方向，GPU 张量立即失去引用。
            del printed, nominal, mask_tensor, target_tensor, ownership_tensor
        if on_tiles_completed is not None:  # 释放后才报告进度
            on_tiles_completed(batch_count)
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
        epe=totals["epe"], l2=totals["l2"], pvband=totals["pvband"],
        valid_probes=totals["valid"], ambiguous_probes=totals["ambiguous"],
        moved_segments=moved)


def optimize_macro(
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

    def step_for(target_round: int) -> float:
        """返回产生第 target_round 轮位移的步长（每 decay_every 轮减半）。"""
        return config.initial_step_dbu * 0.5 ** (
            (target_round - 1) // config.decay_every)

    segment_count = problem.segments.segment_count  # 段数 S
    owner_count = int(np.count_nonzero(problem.owner_indices >= 0))  # owner 段数
    zeros = np.zeros(segment_count, dtype=np.float64)  # 零位移状态
    # 参考几何整个迭代只物化一次：baseline 与每个移动后状态的评价复用同一
    # 份端点/法向（探针始终围绕参考边定义，与位移状态无关）。
    reference = problem.segments.materialize()  # 唯一物化
    # baseline：零位移重建并评价；它同时产生 Round 1 提案。
    started = time.perf_counter()  # baseline 计时
    baseline_region = reconstruct_region(problem, zeros)  # 零位移候选
    pending_step = step_for(1)  # baseline 提案使用的步长
    # 评价 + Round 1 提案
    proposal = evaluate_and_propose(
        problem, baseline_region, zeros, model, config, pending_step,
        target_cache, can_update=True, reference=reference,
        on_tiles_completed=on_tiles_completed)
    # records[0] 固定是 baseline
    records = [IterationRecord(
        round_index=0, step_dbu=0.0, epe=proposal.epe, l2=proposal.l2,
        pvband=proposal.pvband, valid_probes=proposal.valid_probes,
        ambiguous_probes=proposal.ambiguous_probes, moved_segments=0,
        elapsed_seconds=time.perf_counter() - started)]
    best_epe = proposal.epe  # 最佳状态 EPE（EPE 相同保留较早轮，由严格小于实现）
    best_round = 0  # baseline 先当最佳
    best_displacements = zeros.copy()  # 零位移副本
    stop_reason: str | None = None  # 停止原因
    stop_detail: str | None = None  # 非法候选原因
    if owner_count and proposal.valid_probes == 0:  # 有段却无有效探针
        # 「无法评价」不是「零违规」：探针越过窄特征落入异侧（如 2nm 壁 +
        # 8nm 探针距离）时全部探针被判无效，epe 恒为 0；此时以零位移为 best
        # 终止并显式记录原因，不冒充收敛。
        stop_reason = "insufficient_probes"
        stop_detail = (f"有效 EPE 探针 0 个 / owner 段 {owner_count} 个，"
                       "无法评价（探针距离可能大于最窄特征）")
    elif proposal.epe == 0:  # baseline 已无违规
        stop_reason = "zero_epe"  # 直接以零位移为最佳
    else:  # 常规路径：逐轮移动并评价
        for round_index in range(1, config.iterations + 1):  # 移动后状态轮次
            candidate = proposal.next_displacements  # 上一评价的提案
            candidate_moved = proposal.moved_segments  # 该提案改变的段数
            if not candidate_moved:  # 提案与当前完全相同
                # 同一状态再评一次不产生任何新信息（指标、几何全部不变），
                # 直接停止，省去一整轮重建与光刻前向。
                stop_reason = "no_update"
                break
            started = time.perf_counter()  # 本轮计时（重建 + 评价）
            try:  # 候选必须先通过方向/hole/有效性守卫
                candidate_region = reconstruct_region(problem, candidate)
            except (ValueError, ReconstructionError) as exc:  # 非法几何终止
                # 捕获 ValueError 有实测依据：几何退化（如共线 ring 少于三
                # 顶点）会以 ValueError 从 KLayout 数组校验冒出，并非只有
                # ReconstructionError；把它包装进 ReconstructionError 需要
                # 改 reconstruction.py，故维持宽捕获。
                # 位移 shape/有限性由 evaluate_and_propose 入口契约先行拦截，
                # 此处的 ValueError 几乎只可能是几何退化。
                stop_reason = "invalid_geometry"  # 保留最后合法 best
                # 错误原因不得吞掉
                stop_detail = (
                    f"round {round_index} 候选重建失败：{exc}")
                break
            can_propose = round_index < config.iterations  # 末轮不再生成
            pending_next = step_for(round_index + 1)  # 被丢弃提案的步长（末轮）
            # 移动后状态评价（末轮纯评价）
            proposal = evaluate_and_propose(
                problem, candidate_region, candidate, model, config,
                pending_next, target_cache, can_update=can_propose,
                reference=reference,
                on_tiles_completed=on_tiles_completed)
            # Round N 指标属第 N 次位移后状态；moved 为产生本状态时移动的段数
            records.append(IterationRecord(
                round_index=round_index, step_dbu=pending_step, epe=proposal.epe,
                l2=proposal.l2, pvband=proposal.pvband,
                valid_probes=proposal.valid_probes,
                ambiguous_probes=proposal.ambiguous_probes,
                moved_segments=candidate_moved,
                elapsed_seconds=time.perf_counter() - started))
            pending_step = pending_next  # 下轮记录使用的步长
            if owner_count and proposal.valid_probes == 0:  # 移动后无法评价
                # 必须先于 best 比较终止：valid_probes==0 时 epe 恒 0，若放行
                # 会被 epe<best 误当成改善状态。
                stop_reason = "insufficient_probes"
                stop_detail = (f"round {round_index} 有效 EPE 探针 0 个 / "
                               f"owner 段 {owner_count} 个，无法评价")
                break
            if proposal.epe < best_epe:  # 严格更小才更新；相同保留较早轮
                best_epe = proposal.epe
                best_round = round_index
                best_displacements = candidate.copy()
            if proposal.epe == 0:  # 无违规即达目的
                stop_reason = "zero_epe"
                break
            if can_propose and proposal.moved_segments == 0:  # 提案不再移动
                # 末轮 can_update=False 时 moved 恒 0，不构成 no_update 证据。
                stop_reason = "no_update"
                break
        if stop_reason is None:  # 轮次自然用尽
            stop_reason = "iteration_limit"
    return SimpleMBOPCResult(
        best_displacements=best_displacements,
        records=tuple(records),
        best_round=best_round,
        stop_reason=stop_reason,
        stop_detail=stop_detail)
