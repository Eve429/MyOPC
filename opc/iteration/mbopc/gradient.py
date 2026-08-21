"""基于梯度的 MB-OPC：midpoint 边梯度代理、连续 loss 与同步 Adam 法向位移。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import klayout.db as kdb
import numpy as np
import torch
from numpy.typing import NDArray

from evaluation import (
    evaluate_binary_l2,
    evaluate_edge_probes,
    evaluate_pvband,
)
from lithography import LithographyModel, ProcessCondition
from opc.errors import ReconstructionError
from opc.input import points_to_canvas, rasterize_mask_canvas
from opc.input.edge import MacroProblem, reconstruct_region_with_midpoints

from ._batching import (
    MacroStaticPack,
    assemble_probe_batch,
    cached_target_canvas,
    discrete_batch_diagnostics,
    pack_macro_statics,
)
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
    epe_distance_dbu: float      # 离散诊断与 EPE profile 共用的法向距离
    batch_size: int              # 一次 forward 的 core 数
    target_cache_bytes: int      # CPU uint8 target LRU 上限
    weight_epe: float = 0.0      # 可微 EPE loss 权重（0 = 完全关闭）
    epe_steepness: float = 4.0   # EPE penalty sigmoid 陡度 gamma

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
                   self.weight_pvband, self.weight_epe)
        if any(not isfinite(w) or w < 0.0 for w in weights):
            raise ValueError("loss 权重必须是有限非负数")
        if not any(w > 0.0 for w in weights):
            raise ValueError("四个 loss 权重至少一个为正")
        if not isfinite(self.epe_steepness) or self.epe_steepness <= 0.0:
            raise ValueError("epe_steepness 必须是有限正数")
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
    epe_loss: float = 0.0        # 加权前 L_epe（关闭路径恒 0.0）


@dataclass(frozen=True, slots=True)
class GradientMBOPCResult:
    """保存单 macro 的最佳已评价位移、全部状态记录和停止原因。"""

    best_displacements: NDArray[np.float64]  # 按全局段序；context 恒 0
    records: tuple[GradientMBOPCIterationRecord, ...]  # records[0]=baseline
    best_state_index: int                    # 指向 records 中 best
    stop_reason: str          # zero_loss/no_update/invalid_geometry/no_owned_segments/iteration_limit
    stop_detail: str | None   # 非法候选的明确原因；正常停止为 None


@dataclass(frozen=True, slots=True)
class _GradientMacroContext:
    """保存单个 macro 优化期间完全不变的静态输入与预计算映射。

    只收集"全部状态迭代复用"的数据；parameters/optimizer/current 几何与
    best 跟踪属于迭代状态，刻意不入本类。列表内容按只读约定消费。
    """

    macro_id: str                    # cache 键与错误消息的 macro 部分
    segment_count: int               # 段数 S
    canvas_pixels: int               # 问题侧画布
    pixel_dbu: int                   # 栅格像素
    core_count: int                  # tile 总数
    max_displacement: float          # 位移上限（step 后裁剪用）
    owner_ids: NDArray[np.int64]     # owner 段全局号升序
    segment_to_parameter: NDArray[np.int32]  # 段号→参数下标；非 owner 恒 -1
    reference_region: kdb.Region     # 零位移参考几何（target 缓存未命中源）
    reference_segment_midpoints: NDArray[np.float64]  # 零位移各段采样中点 [S,2]
    core_sampling_members: list[NDArray[np.int32]]  # 每 core 梯度采样段（owner∩membership）
    core_owner_members: list[NDArray[np.int32]]     # 每 core EPE 探针段（owner 语义）
    probe_inner_xy: list[NDArray[np.float64] | None]  # 每 core 探针 canvas 坐标
    probe_outer_xy: list[NDArray[np.float64] | None]
    total_pixels: int                # loss 归一分母 P：全部 core ownership 像素数
    epe_profiles: list               # 每 core 的 owner profile 画布坐标 [E_c,Q,2]（关闭为 None 表元）
    epe_lengths: list                # 每 core 的 owner 段参考长度 [E_c]（DBU；关闭为 None 表元）
    epe_length_sum: float            # L_epe 分母 L_sum = Σ len_s（macro 内常量）
    device: torch.device            # 参数与批张量的目标设备
    threshold: float                # 离散诊断二值阈值
    conditions: tuple[ProcessCondition, ...]  # 三工艺角一次前向
    pack: MacroStaticPack           # 共享静态打包（计分画布/探针坐标/target 源）


@dataclass(frozen=True, slots=True)
class _GradientStateEvaluation:
    """保存一次已评价状态的连续 loss、离散诊断与纯评价耗时。"""

    total_loss: float       # 加权连续 loss（所有权像素归一）
    nominal_loss: float     # L_nom
    process_loss: float     # L_process
    pvband_loss: float      # L_pv
    l2: int                 # 离散二值 L2（诊断）
    pvband: int             # 离散 PVBand（诊断）
    epe: int                # 离散 EPE（诊断）
    valid_probes: int       # 有效探针
    ambiguous_probes: int   # 歧义探针
    epe_loss: float         # 加权前 L_epe（关闭路径恒 0.0）
    elapsed_seconds: float  # 全部 tile 评价耗时（不含参数快照与记录组装）


class _EdgeGradientMask(torch.autograd.Function):
    """hard 面积覆盖率前向与 DiffOPC Algorithm 4 的 midpoint STE 反向。"""

    @staticmethod
    def forward(ctx, hard_masks, local_displacements, batch_indices,
                midpoints_xy, pixel_dbu):
        """前向不改 mask 数值，只保存反向采样需要的批号、中点与像素。

        local_displacements 只用于建立 autograd 边（STE 硬几何直通），
        不参与 forward 计算；pixel_dbu 为 DBU→pixel 的换算尺度（非张量，
        存普通 ctx 属性），backward 用它把采样梯度换算回 DBU 参数。
        """
        ctx.pixel_dbu = pixel_dbu
        ctx.save_for_backward(batch_indices, midpoints_xy)
        return hard_masks

    @staticmethod
    def backward(ctx, grad_output):
        """按段中点双线性采样，返回给 DBU 位移参数的梯度为 2·g_mid/pixel_dbu。

        单位契约：g_mid 是 canvas/pixel 坐标下的 STE 边梯度，
        local_displacements 的单位是 DBU——x_canvas ≈ x_dbu/pixel_dbu，
        故 dx_canvas/dd_dbu = 1/pixel_dbu，链式换算后除以 pixel_dbu。
        2 倍来源与单位换算相互独立：论文对两个 endpoint 各采样一次
        g_mid，本项目标量位移同时驱动两端点，两端链式求和恰为 2·g_mid。
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
        # 末位 None 对应 forward 的 pixel_dbu（无梯度）；除以 pixel_dbu
        # 把 pixel 域采样梯度换算回 DBU 位移参数。
        return None, 2.0 * g_mid / ctx.pixel_dbu, None, None, None


def _profile_d_s(error: torch.Tensor, slots: torch.Tensor,
                  xy: torch.Tensor) -> torch.Tensor:
    """在 nominal 平方误差图上按 profile 双线性采样并求和，返回每段 d_s。

    布局契约与 _EdgeGradientMask.backward 同源：图像 [B,H,W]、像素索引
    [y,x]、坐标 (x,y) 连续、行 0 为最低 Y。采样保持 error 的 autograd 边
    （EPE 梯度经此回到 mask）；坐标已在构造期验证落在闭区间
    [0, canvas-1]，最外侧整数像素中心的邻居按契约退化为同一边界像素。
    """
    count, q_slots, _ = xy.shape  # [N,Q,2]
    points = xy.reshape(count * q_slots, 2)
    height, width = error.shape[-2:]
    x, y = points[:, 0], points[:, 1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = (x0 + 1.0).clamp(max=float(width - 1))   # 边界折回（权重为 0）
    y1 = (y0 + 1.0).clamp(max=float(height - 1))
    wx = x - x0
    wy = y - y0
    x0i, y0i = x0.long(), y0.long()
    x1i, y1i = x1.long(), y1.long()
    plane = height * width
    base = slots.repeat_interleave(q_slots) * plane
    flat = error.reshape(-1)
    v00 = flat[base + y0i * width + x0i]
    v01 = flat[base + y0i * width + x1i]
    v10 = flat[base + y1i * width + x0i]
    v11 = flat[base + y1i * width + x1i]
    sample = (v00 * (1.0 - wx) * (1.0 - wy) + v01 * wx * (1.0 - wy)
              + v10 * (1.0 - wx) * wy + v11 * wx * wy)
    # profile 内 sum 聚合（DEC-002）：d_s 近似"边缘偏移的 pixel 数"，
    # 与 epe_distance（Q）解耦；mean 会随 Q 线性稀释。
    return sample.view(count, q_slots).sum(dim=1)


def _prepare_macro_context(
        problem: MacroProblem, model: LithographyModel,
        config: GradientMBOPCConfig) -> _GradientMacroContext:
    """一次性构造整个优化期间不变的 owner 映射、参考几何与探针静态输入。"""
    macro_id = problem.macro.macro_id  # cache 键与错误消息的 macro 部分
    pixel_dbu = int(problem.macro.pixel_dbu)  # 栅格像素
    core_count = problem.macro.core_count  # tile 总数
    canvas_pixels = int(problem.macro.canvas_pixels)  # 问题侧画布
    max_displacement = float(problem.fragmentation.max_displacement_dbu)
    owner_ids = np.flatnonzero(problem.owner_indices >= 0)  # owner 段全局号
    segment_count = problem.segments.segment_count  # 段数 S
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
    reference_region, reference_segment_midpoints = (
        reconstruct_region_with_midpoints(
            problem, np.zeros(segment_count, dtype=np.float64)))
    # 共享静态打包（A1）：计分画布/参考探针坐标每 macro 一次，全状态复用
    pack = pack_macro_statics(
        problem, epe_distance_dbu=config.epe_distance_dbu,
        reference_geometry=reference, reference_region=reference_region,
        to_canvas=points_to_canvas)
    core_sampling_members = []  # 每 core 全部可见段中的 owner 段（梯度采样）
    # EPE profile 静态预计算（weight_epe>0 时）：固定在参考段中点/法向，
    # 与位移状态无关；Q = 2R 个对称半像素中心槽位。
    # R 的整数性已由 optimize 入口在空 owner 快速返回之前校验（单点）。
    epe_enabled = config.weight_epe > 0.0
    epe_radius = (round(config.epe_distance_dbu / pixel_dbu)
                  if epe_enabled else 0)
    # q = (−R+0.5, …, −0.5, 0.5, …, R−0.5)·p：避开几何边界本身、对称覆盖
    epe_offsets = ((np.arange(2 * epe_radius, dtype=np.float64)
                    - epe_radius + 0.5) * pixel_dbu if epe_enabled
                   else np.empty(0))
    epe_profiles: list = []  # 每 core [E_c,Q,2] canvas 坐标（无 owner 段为 None）
    epe_lengths: list = []  # 每 core [E_c] 参考段长（DBU；无 owner 段为 None）
    epe_length_sum = 0.0  # L_epe 分母 L_sum = Σ len_s（macro 常量）
    for core_index in range(core_count):
        spec = problem.macro.core(core_index)  # 即时构造 CoreSpec，不常驻
        # 梯度采样按 membership：该 core 可见的所有段中，凡 owner
        # 段都在本 core 的 canvas 采样一次并累加到同一参数——跨 core 边界段
        # 的邻 tile 贡献不丢弃；采样与 owner（发布归属）职责分离。
        members = np.asarray(problem.segments_for_core(core_index))
        core_sampling_members.append(
            members[segment_to_parameter[members] >= 0])
        owner_members = pack.owner_members[core_index]
        if epe_enabled and len(owner_members):
            # EPE profile 固定在参考段（不随 current mask 移动）：中点沿
            # 单位法向的 2R 个半像素中心；每段恰在本 owner core 建一条。
            midpoints = (reference.starts[owner_members]
                         + reference.ends[owner_members]) * 0.5
            normals = reference.normals[owner_members]
            lengths = np.linalg.norm(
                reference.ends[owner_members] - reference.starts[owner_members],
                axis=1)
            profile_dbu = (midpoints[:, None, :]
                           + epe_offsets[None, :, None] * normals[:, None, :])
            profile_xy = points_to_canvas(
                profile_dbu.reshape(-1, 2), spec.context_box, pixel_dbu,
                canvas_pixels).reshape(len(owner_members), -1, 2)
            # 越界守卫：坐标必须落在闭区间 [0, canvas-1]（epe≤context 在
            # 数学上保证；违反即输入/网格契约错误，不裁剪不跳过）。
            if (float(profile_xy.min()) < 0.0
                    or float(profile_xy.max()) > float(canvas_pixels - 1)):
                raise ValueError(
                    f"{macro_id} core {core_index} EPE profile 越出画布闭区间")
            epe_profiles.append(profile_xy)
            epe_lengths.append(lengths)
            epe_length_sum += float(lengths.sum())
        else:
            epe_profiles.append(None)
            epe_lengths.append(None)
    del reference  # 探针已提取，释放全量段几何数组
    if pack.total_pixels == 0:
        # 有 owner 段却算不出任何计分像素，属于数据损坏，不能静默除零。
        raise ValueError("存在 owner 段但 ownership 计分像素为 0（数据不一致）")
    device = model.device  # 参数与批张量的目标设备
    threshold = float(model.config.print_threshold)  # 离散诊断二值阈值
    # 三工艺角一次前向
    conditions = (model.condition("nominal"), model.condition("dose_max"),
                  model.condition("defocus_min"))
    return _GradientMacroContext(
        macro_id=macro_id, segment_count=segment_count,
        canvas_pixels=canvas_pixels, pixel_dbu=pixel_dbu,
        core_count=core_count, max_displacement=max_displacement,
        owner_ids=owner_ids, segment_to_parameter=segment_to_parameter,
        reference_region=reference_region,
        reference_segment_midpoints=reference_segment_midpoints,
        core_sampling_members=core_sampling_members,
        core_owner_members=pack.owner_members,
        probe_inner_xy=pack.probe_inner_xy,
        probe_outer_xy=pack.probe_outer_xy,
        total_pixels=pack.total_pixels,
        epe_profiles=epe_profiles, epe_lengths=epe_lengths,
        epe_length_sum=epe_length_sum,
        device=device, threshold=threshold,
        conditions=conditions,
        pack=pack)


def _evaluate_state(
        ctx: _GradientMacroContext, model: LithographyModel,
        problem: MacroProblem, config: GradientMBOPCConfig,
        target_cache: TargetCanvasCache, parameters: torch.Tensor,
        current_region: kdb.Region,
        current_segment_midpoints: NDArray[np.float64], *,
        build_gradient: bool,
        on_tiles_completed: OnTilesCompleted | None = None,
) -> _GradientStateEvaluation:
    """评价当前已发布几何的全部 core，按批累积连续 loss 梯度与离散诊断。

    只允许 backward：zero_grad 与 optimizer.step 属于调用方的更新职责，
    本函数绝不触碰参数值——同轮多批梯度因此天然累积在同一参数快照上。
    """
    sums = {"nominal": 0.0, "process": 0.0, "pvband": 0.0,
            "epe": 0.0}  # 连续分量累计（epe 为加权前 L_epe）
    diag = {"l2": 0, "pvband": 0, "epe": 0, "valid": 0, "ambiguous": 0}
    started = time.perf_counter()  # 本状态评价计时
    for batch_start in range(0, ctx.core_count, config.batch_size):
        # 本批 core（行优先稳定序）
        core_indices = list(range(
            batch_start, min(batch_start + config.batch_size, ctx.core_count)))
        batch_count = len(core_indices)
        # target 批（uint8 缓存格式）
        targets = np.empty((batch_count, ctx.canvas_pixels, ctx.canvas_pixels),
                           dtype=np.uint8)
        # 当前 mask 批
        masks = np.empty((batch_count, ctx.canvas_pixels, ctx.canvas_pixels),
                         dtype=np.float32)
        # 计分像素批
        ownership = np.empty((batch_count, ctx.canvas_pixels, ctx.canvas_pixels),
                             dtype=np.bool_)
        member_slots = []  # 梯度采样条目的 batch 槽位（int64）
        member_params = []  # 梯度采样条目指向的参数索引（int64）
        member_mids = []  # 梯度采样条目的当前中点 canvas 坐标
        epe_slots = []  # EPE profile 条目的 batch 槽位（仅 owner core 语义）
        epe_xy_parts = []  # EPE profile 画布坐标 [E_c,Q,2]
        epe_len_parts = []  # EPE 段参考长度 [E_c]
        for slot, core_index in enumerate(core_indices):  # 逐 core 组批
            spec = problem.macro.core(core_index)
            # target：缓存命中直接用，miss 栅格化参考几何并回填
            targets[slot] = cached_target_canvas(
                problem, ctx.pack, target_cache, core_index,
                rasterize=rasterize_mask_canvas)
            # 当前候选直接栅格
            masks[slot] = rasterize_mask_canvas(
                current_region, spec.context_box, ctx.pixel_dbu,
                ctx.canvas_pixels, polarity=problem.polarity,
                dark_box=problem.dark_box)
            # 唯一计分像素（静态打包，逐态不重算）
            ownership[slot] = ctx.pack.ownership[core_index]
            sampling_members = ctx.core_sampling_members[core_index]
            if len(sampling_members):  # 梯度采样：全部 membership 中 owner 段
                # 采样中点由当前已发布的重构几何提供（corner miter 后含
                # 切向调整），与栅格化用 Region 恒来自同一次合法重构。
                midpoints_dbu = current_segment_midpoints[sampling_members]
                # DBU→canvas 唯一换算
                member_mids.append(points_to_canvas(
                    midpoints_dbu, spec.context_box, ctx.pixel_dbu,
                    ctx.canvas_pixels))
                member_slots.append(np.full(
                    len(sampling_members), slot, dtype=np.int64))
                member_params.append(ctx.segment_to_parameter[
                    sampling_members].astype(np.int64))
            owner_members = ctx.core_owner_members[core_index]  # 探针语义
            if len(owner_members) and ctx.epe_length_sum > 0.0:
                # EPE 只由 owner core 计一次（探针坐标经静态打包批量拼接）
                epe_slots.append(np.full(
                    len(owner_members), slot, dtype=np.int64))
                epe_xy_parts.append(ctx.epe_profiles[core_index])
                epe_len_parts.append(ctx.epe_lengths[core_index])
        trainable = bool(member_params)  # 本批是否有可训练 membership
        build_graph = trainable and build_gradient  # 末状态纯评价不建图
        # uint8→float32/255
        target_tensor = torch.from_numpy(targets).to(
            device=ctx.device, dtype=torch.float32).div_(255.0)
        ownership_tensor = torch.from_numpy(ownership).to(device=ctx.device)
        hard = torch.from_numpy(masks).to(device=ctx.device)
        if build_graph:  # STE：forward 数值不变，autograd 边接到位移
            # 批号与参数索引一次上设备
            slots = torch.from_numpy(
                np.concatenate(member_slots)).to(ctx.device)
            owned = torch.from_numpy(
                np.concatenate(member_params)).to(ctx.device)
            # 中点坐标转 float32 上设备
            mids = torch.from_numpy(
                np.concatenate(member_mids)).to(device=ctx.device,
                                                 dtype=torch.float32)
            local = parameters[owned]  # gather 出 [M]，autograd 边
            # pixel_dbu 随批传入：backward 把采样梯度换算回 DBU 单位。
            mask_tensor = _EdgeGradientMask.apply(hard, local, slots, mids,
                                                  ctx.pixel_dbu)
        else:
            mask_tensor = hard  # 无梯度路径直通
        printed = model.forward_many(mask_tensor, ctx.conditions)  # 一次 FFT
        nominal = printed["nominal"]
        dose_max = printed["dose_max"]
        defocus_min = printed["defocus_min"]
        # EPE profile 批张量：坐标/长度常驻 CPU，仅当前批上设备（O(E·Q)）
        epe_slot_tensor = epe_xy_tensor = epe_len_tensor = None
        if epe_slots:
            epe_slot_tensor = torch.from_numpy(
                np.concatenate(epe_slots)).to(device=ctx.device)
            epe_xy_tensor = torch.from_numpy(
                np.concatenate(epe_xy_parts)).to(
                    device=ctx.device, dtype=torch.float32)
            epe_len_tensor = torch.from_numpy(
                np.concatenate(epe_len_parts)).to(
                    device=ctx.device, dtype=torch.float32)
        epe_value = 0.0  # 本批 L_epe 分量（Σ len·pen / L_sum；无条目为 0）
        if build_graph:  # 建图版连续 loss：backward 累积到 parameters.grad
            # nominal 平方误差张量供 L2 与 EPE 共用（PERF-002 复用）
            nominal_error = (nominal - target_tensor) ** 2
            l_nom = (nominal_error * ownership_tensor).sum()
            l_proc = (((dose_max - target_tensor) ** 2
                       + (defocus_min - target_tensor) ** 2)
                      * ownership_tensor).sum()
            l_pv = ((dose_max - defocus_min) ** 2
                    * ownership_tensor).sum()
            batch_loss = (config.weight_nominal_l2 * l_nom
                          + config.weight_process_l2 * l_proc
                          + config.weight_pvband * l_pv) / ctx.total_pixels
            if epe_slot_tensor is not None:
                # 第四项 loss 与三项共用同一次 backward；penalty 为
                # zero-based sigmoid，len 归一保证切段不变性（DEC-007）。
                d_s = _profile_d_s(nominal_error, epe_slot_tensor,
                                   epe_xy_tensor)
                penalty = 2.0 * (torch.sigmoid(
                    config.epe_steepness * d_s) - 0.5)
                batch_epe = ((epe_len_tensor * penalty).sum()
                             / ctx.epe_length_sum)
                batch_loss = batch_loss + config.weight_epe * batch_epe
                epe_value = float(batch_epe.detach())
            batch_loss.backward()  # 批间梯度直接累加（同一参数快照）
            triple = (l_nom, l_proc, l_pv)
        else:
            with torch.no_grad():  # 纯评价路径不建图
                nominal_error = (nominal - target_tensor) ** 2
                triple = ((nominal_error * ownership_tensor).sum(),
                          (((dose_max - target_tensor) ** 2
                            + (defocus_min - target_tensor) ** 2)
                           * ownership_tensor).sum(),
                          ((dose_max - defocus_min) ** 2
                           * ownership_tensor).sum())
                if epe_slot_tensor is not None:
                    d_s = _profile_d_s(nominal_error, epe_slot_tensor,
                                       epe_xy_tensor)
                    penalty = 2.0 * (torch.sigmoid(
                        config.epe_steepness * d_s) - 0.5)
                    batch_epe = ((epe_len_tensor * penalty).sum()
                                 / ctx.epe_length_sum)
                    epe_value = float(batch_epe)
        with torch.no_grad():  # 离散诊断只读数值，不进入训练
            sums["nominal"] += float(triple[0]) / ctx.total_pixels
            sums["process"] += float(triple[1]) / ctx.total_pixels
            sums["pvband"] += float(triple[2]) / ctx.total_pixels
            sums["epe"] += epe_value
            # 本批 owner 探针（静态坐标）与 L2/PVBand/EPE 公共离散诊断
            # （evaluate_* 补丁锚在本模块）
            probe_slots_np, inner_np, outer_np = assemble_probe_batch(
                ctx.pack, core_indices)
            l2, pvband, epe_result = discrete_batch_diagnostics(
                target_tensor, printed, ownership_tensor, ctx.threshold,
                probe_slots_np, inner_np, outer_np,
                binary_l2=evaluate_binary_l2, pvband=evaluate_pvband,
                edge_probes=evaluate_edge_probes)
            diag["l2"] += l2
            diag["pvband"] += pvband
            if epe_result is not None:
                diag["epe"] += epe_result.violation_count
                diag["valid"] += int(epe_result.valid.cpu().numpy().sum())
                diag["ambiguous"] += int(
                    epe_result.ambiguous.cpu().numpy().sum())
        # 释放：批结束只保留标量与梯度，光刻图和批张量立即失去引用。
        del printed, nominal, dose_max, defocus_min, mask_tensor, hard
        del target_tensor, ownership_tensor, nominal_error
        del epe_slot_tensor, epe_xy_tensor, epe_len_tensor
        if on_tiles_completed is not None:  # backward 且释放后才报进度
            on_tiles_completed(batch_count)
    nominal_loss = sums["nominal"]
    process_loss = sums["process"]
    pvband_loss = sums["pvband"]
    epe_loss = sums["epe"]  # 加权前 L_epe（关闭路径恒 0.0）
    total_loss = (config.weight_nominal_l2 * nominal_loss
                  + config.weight_process_l2 * process_loss
                  + config.weight_pvband * pvband_loss)
    if ctx.epe_length_sum > 0.0:  # 启用时 total 与 best 含加权 EPE
        total_loss += config.weight_epe * epe_loss
    return _GradientStateEvaluation(
        total_loss=total_loss, nominal_loss=nominal_loss,
        process_loss=process_loss, pvband_loss=pvband_loss, l2=diag["l2"],
        pvband=diag["pvband"], epe=diag["epe"], valid_probes=diag["valid"],
        ambiguous_probes=diag["ambiguous"], epe_loss=epe_loss,
        elapsed_seconds=time.perf_counter() - started)


def _take_optimizer_step(
        problem: MacroProblem, parameters: torch.Tensor,
        optimizer: torch.optim.Optimizer, owner_ids: NDArray[np.int64],
        candidate_full: NDArray[np.float64], max_displacement: float, *,
        macro_id: str, state_index: int,
) -> tuple[kdb.Region, NDArray[np.float64]] | None:
    """执行一次 Adam 更新，并把新参数重构为可发布的合法候选几何。

    返回 None 表示参数无变化（no_update 判据）；重构失败以 ValueError/
    ReconstructionError 原样上抛，停止决策留给调用方。macro_id 与
    state_index 仅供异常消息定位，不参与计算。
    """
    before = parameters.detach().clone()  # 更新前快照（no_update 判据）
    optimizer.step()  # 每 state 至多一次
    with torch.no_grad():
        parameters.clamp_(-max_displacement, max_displacement)  # 先裁上限
    if not bool(torch.isfinite(parameters).all()):
        raise FloatingPointError(
            f"{macro_id} state {state_index} 候选参数非有限")
    if torch.equal(parameters.detach(), before):  # 梯度全零时步长为零
        return None
    candidate_full[owner_ids] = parameters.detach().cpu().numpy().astype(
        np.float64)
    # 候选必须先通过方向/hole/有效性守卫才可发布；返回的 Region 与采样
    # 中点来自同一次重构，调用方必须成对发布（失败时两者都不更新）。
    return reconstruct_region_with_midpoints(problem, candidate_full)


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
    if config.weight_epe > 0.0:  # EPE 启用时 R 必须为正整数（GPU 分配前）
        entry_pixel = int(problem.macro.pixel_dbu)
        radius = config.epe_distance_dbu / entry_pixel
        if radius < 1.0 or radius != int(radius):
            raise ValueError(
                "weight_epe>0 要求 epe_distance_dbu 是 pixel_dbu 的正整数倍"
                f"（R≥1），实际 R={radius}")
    owner_ids = np.flatnonzero(problem.owner_indices >= 0)  # owner 段全局号
    if len(owner_ids) == 0:
        # 空或纯 context macro：没有可训练参数，O=0 必然没有计分像素，
        # 任何评价都只会得到 0/0；直接以全零 baseline 停止，不建 optimizer。
        empty = GradientMBOPCIterationRecord(
            state_index=0, total_loss=0.0, nominal_l2_loss=0.0,
            process_l2_loss=0.0, pvband_loss=0.0, l2=0, pvband=0, epe=0,
            valid_probes=0, ambiguous_probes=0, displaced_segments=0,
            elapsed_seconds=0.0, epe_loss=0.0)
        return GradientMBOPCResult(
            np.zeros(segment_count, dtype=np.float64), (empty,), 0,
            "no_owned_segments", None)
    # 静态输入只准备一次，全部状态迭代复用。
    ctx = _prepare_macro_context(problem, model, config)
    # 唯一可训练参数：owner 法向位移 [O]
    parameters = torch.zeros(
        len(owner_ids), dtype=torch.float32, device=ctx.device,
        requires_grad=True)
    # 固定超参（规格钉死，不新增配置面）
    optimizer = torch.optim.Adam(
        [parameters], lr=config.learning_rate_dbu, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0, amsgrad=False)
    current_region = ctx.reference_region  # 当前已发布合法几何
    current_segment_midpoints = ctx.reference_segment_midpoints
    records = []  # 已评价状态记录（records[0] 恒为 baseline）
    best_loss = float("inf")  # 严格更小才更新（平局保留较早状态）
    best_state_index = 0
    best_owner = np.zeros(len(owner_ids), dtype=np.float64)
    stop_reason: str | None = None
    stop_detail: str | None = None
    candidate_full = np.zeros(segment_count, dtype=np.float64)  # 展开缓冲
    for state_index in range(config.iterations + 1):
        can_update = state_index < config.iterations  # 末状态纯评价
        if can_update:
            optimizer.zero_grad(set_to_none=True)  # 梯度按状态清零后累积
        current_owner = parameters.detach().cpu().numpy().astype(np.float64)
        evaluation = _evaluate_state(
            ctx, model, problem, config, target_cache, parameters,
            current_region, current_segment_midpoints,
            build_gradient=can_update, on_tiles_completed=on_tiles_completed)
        if not (isfinite(evaluation.total_loss)
                and isfinite(evaluation.nominal_loss)
                and isfinite(evaluation.process_loss)
                and isfinite(evaluation.pvband_loss)
                and isfinite(evaluation.epe_loss)):
            raise FloatingPointError(
                f"{ctx.macro_id} state {state_index} 连续 loss 非有限")
        records.append(GradientMBOPCIterationRecord(
            state_index=state_index, total_loss=evaluation.total_loss,
            nominal_l2_loss=evaluation.nominal_loss,
            process_l2_loss=evaluation.process_loss,
            pvband_loss=evaluation.pvband_loss, l2=evaluation.l2,
            pvband=evaluation.pvband, epe=evaluation.epe,
            valid_probes=evaluation.valid_probes,
            ambiguous_probes=evaluation.ambiguous_probes,
            displaced_segments=int(np.count_nonzero(current_owner)),
            elapsed_seconds=evaluation.elapsed_seconds,
            epe_loss=evaluation.epe_loss))
        if evaluation.total_loss < best_loss:  # 严格更小才更新；相同保留较早状态
            best_loss = evaluation.total_loss
            best_state_index = state_index
            best_owner = current_owner.copy()
        if evaluation.total_loss == 0.0:  # 连续 loss 恰为零即达目的
            stop_reason = "zero_loss"
            break
        if state_index == config.iterations:  # 轮次自然用尽
            stop_reason = "iteration_limit"
            break
        grad = parameters.grad  # 全部 batch 完成后的唯一屏障内检查
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise FloatingPointError(
                f"{ctx.macro_id} state {state_index} 梯度缺失或非有限")
        try:  # 候选必须先通过方向/hole/有效性守卫才可发布
            candidate = _take_optimizer_step(
                problem, parameters, optimizer, owner_ids, candidate_full,
                ctx.max_displacement, macro_id=ctx.macro_id,
                state_index=state_index)
        except (ValueError, ReconstructionError) as exc:
            # 宽捕获有实测依据：几何退化（如位移共线使 ring 顶点不足）会以
            # ValueError 从 KLayout 冒出而非 ReconstructionError（simple.py
            # 同款证据）；收窄需改 reconstruction.py 包装。
            stop_reason = "invalid_geometry"
            stop_detail = f"state {state_index + 1} 候选重建失败：{exc}"
            break
        if candidate is None:  # 参数无变化
            stop_reason = "no_update"
            break
        # Region 与采样中点绑定发布：下一状态的栅格化与梯度采样恒来自
        # 同一次合法候选重构（失败时两者都不更新）。
        current_region, current_segment_midpoints = candidate
    if stop_reason is None:  # 防御兜底（iterations>=1 时循环内必设）
        stop_reason = "iteration_limit"
    best_full = np.zeros(segment_count, dtype=np.float64)
    best_full[owner_ids] = best_owner  # context 段恒 0
    return GradientMBOPCResult(
        best_displacements=best_full, records=tuple(records),
        best_state_index=best_state_index, stop_reason=stop_reason,
        stop_detail=stop_detail)
