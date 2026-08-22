"""simple 与 gradient 共享的静态评价打包与批量离散诊断（A1 骨架）。

只承载与"更新策略"无关的公共计算（防两求解器漂移）：每 macro 打包一次
的静态画布与参考探针坐标、target 缓存 miss 回填、公共组批与批后离散
诊断。凡被测试按求解器模块名 monkeypatch 的函数（rasterize_mask_canvas /
points_to_canvas / evaluate_*）由调用方以自身模块全局显式传入——补丁
锚点不随代码搬迁失效。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import klayout.db as kdb
import numpy as np
import torch
from numpy.typing import NDArray

from evaluation import EPEEvaluation
from opc.input import ownership_canvas
from opc.input.edge import MacroProblem
from opc.input.edge.fragmentation import SegmentGeometry
from opc.input.edge.sampling import edge_probe_points


@dataclass(frozen=True, slots=True)
class MacroStaticPack:
    """一个 macro 全部状态迭代期间不变的评价静态输入（CPU 常驻）。"""

    macro_id: str
    pixel_dbu: int
    canvas_pixels: int
    core_count: int
    ownership: list[NDArray[np.bool_]]  # 每 core 唯一计分画布（静态）
    total_pixels: int  # loss 归一分母：全部 core 计分像素数
    owner_members: list[NDArray[np.int32]]  # 每 core owner 段号（探针/写回）
    probe_inner_xy: list[NDArray[np.float64] | None]  # 参考探针 canvas 坐标
    probe_outer_xy: list[NDArray[np.float64] | None]
    reference_region: kdb.Region  # target 缓存 miss 的零位移候选源


def pack_macro_statics(
    problem: MacroProblem,
    *,
    epe_distance_dbu: float,
    reference_geometry: SegmentGeometry,
    reference_region: kdb.Region,
    to_canvas,
) -> MacroStaticPack:
    """一次构造全部状态迭代复用的计分画布与参考探针坐标。"""
    pixel_dbu = int(problem.macro.pixel_dbu)
    canvas_pixels = int(problem.macro.canvas_pixels)
    ownership: list[NDArray[np.bool_]] = []
    owner_members: list[NDArray[np.int32]] = []
    inner_list: list = []
    outer_list: list = []
    total_pixels = 0
    for core_index in range(problem.macro.core_count):
        spec = problem.macro.core(core_index)  # 即时构造 CoreSpec，不常驻
        canvas = ownership_canvas(spec.ownership_box, spec.context_box, pixel_dbu, canvas_pixels)
        ownership.append(canvas)
        total_pixels += int(canvas.sum())
        members = problem.owner_segments_for_core(core_index)
        owner_members.append(members)
        if len(members):  # 探针围绕参考边中点 ± 法向（与位移状态无关）
            inner_dbu, outer_dbu = edge_probe_points(
                reference_geometry.starts[members],
                reference_geometry.ends[members],
                reference_geometry.normals[members],
                epe_distance_dbu,
            )
            inner_list.append(to_canvas(inner_dbu, spec.context_box, pixel_dbu, canvas_pixels))
            outer_list.append(to_canvas(outer_dbu, spec.context_box, pixel_dbu, canvas_pixels))
        else:
            inner_list.append(None)
            outer_list.append(None)
    return MacroStaticPack(
        macro_id=problem.macro.macro_id,
        pixel_dbu=pixel_dbu,
        canvas_pixels=canvas_pixels,
        core_count=problem.macro.core_count,
        ownership=ownership,
        total_pixels=total_pixels,
        owner_members=owner_members,
        probe_inner_xy=inner_list,
        probe_outer_xy=outer_list,
        reference_region=reference_region,
    )


def cached_target_canvas(
    problem: MacroProblem,
    pack: MacroStaticPack,
    target_cache,
    core_index: int,
    *,
    rasterize,
) -> NDArray[np.uint8]:
    """返回该 core 的 target uint8 画布：命中直接用，miss 栅格化并回填。

    rasterize 注入调用方模块的 rasterize_mask_canvas（补丁锚保持在
    求解器模块）；miss 源恒为 pack 的零位移参考几何。
    """
    cached = target_cache.get(pack.macro_id, core_index)
    if cached is None:
        spec = problem.macro.core(core_index)
        cached = np.rint(
            rasterize(
                pack.reference_region, spec.context_box, pack.pixel_dbu, pack.canvas_pixels, polarity=problem.polarity
            )
            * 255.0
        ).astype(np.uint8)
        target_cache.put(pack.macro_id, core_index, cached)
    return cached


def iter_core_batches(
    problem: MacroProblem,
    pack: MacroStaticPack,
    current_region: kdb.Region,
    target_cache,
    *,
    batch_size: int,
    rasterize,
) -> Iterator[tuple[list[int], NDArray[np.uint8], NDArray[np.float32], NDArray[np.bool_]]]:
    """按批产出 (core_indices, targets, masks, ownership) 的 numpy 组批结果。

    与更新策略无关的公共组批段（simple/gradient 评价函数共用，防漂移）：
    target 走缓存 miss 回填、当前候选直接栅格、计分画布取静态打包。
    rasterize 注入调用方模块的 rasterize_mask_canvas（补丁锚保持在
    求解器模块）。
    """
    for batch_start in range(0, pack.core_count, batch_size):
        core_indices = list(range(batch_start, min(batch_start + batch_size, pack.core_count)))
        batch_count = len(core_indices)
        targets = np.empty((batch_count, pack.canvas_pixels, pack.canvas_pixels), dtype=np.uint8)
        masks = np.empty((batch_count, pack.canvas_pixels, pack.canvas_pixels), dtype=np.float32)
        ownership = np.empty((batch_count, pack.canvas_pixels, pack.canvas_pixels), dtype=np.bool_)
        for slot, core_index in enumerate(core_indices):
            spec = problem.macro.core(core_index)  # 即时构造 CoreSpec，不常驻
            targets[slot] = cached_target_canvas(problem, pack, target_cache, core_index, rasterize=rasterize)
            masks[slot] = rasterize(
                current_region, spec.context_box, pack.pixel_dbu, pack.canvas_pixels, polarity=problem.polarity
            )
            ownership[slot] = pack.ownership[core_index]
        yield core_indices, targets, masks, ownership


def upload_eval_batch(
    targets: NDArray[np.uint8],
    masks: NDArray[np.float32],
    ownership: NDArray[np.bool_],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批 numpy 上设备：target 转 float32/255，mask/ownership 直传。"""
    target_tensor = torch.from_numpy(targets).to(device=device, dtype=torch.float32).div_(255.0)
    mask_tensor = torch.from_numpy(masks).to(device=device)
    ownership_tensor = torch.from_numpy(ownership).to(device=device)
    return target_tensor, mask_tensor, ownership_tensor


def assemble_probe_batch(
    pack: MacroStaticPack,
    core_indices: list[int],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """按批内 slot 顺序拼接有 owner 段 core 的探针槽位与坐标。"""
    slots: list[NDArray[np.int64]] = []
    inner_parts: list[NDArray[np.float64]] = []
    outer_parts: list[NDArray[np.float64]] = []
    for slot, core_index in enumerate(core_indices):
        members = pack.owner_members[core_index]
        if len(members):
            slots.append(np.full(len(members), slot, dtype=np.int64))
            inner_parts.append(pack.probe_inner_xy[core_index])
            outer_parts.append(pack.probe_outer_xy[core_index])
    if not slots:  # 整批无 owner 段：返回空数组，诊断层按无探针跳过
        return (np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64))
    return (np.concatenate(slots), np.concatenate(inner_parts), np.concatenate(outer_parts))


def discrete_batch_diagnostics(
    target_tensor,
    printed,
    ownership_tensor,
    threshold: float,
    probe_slots: NDArray[np.int64],
    inner_xy: NDArray[np.float64],
    outer_xy: NDArray[np.float64],
    *,
    binary_l2,
    pvband,
    edge_probes,
) -> tuple[int, int, EPEEvaluation | None]:
    """批后离散诊断：ownership 像素 L2/PVBand +（有探针时）EPE 评价。

    调用顺序固定 L2 → PVBand → EPE（与两求解器原实现一致）；三个
    evaluate_* 由调用方注入自身模块全局，保持补丁锚点。无探针批次
    返回 epe_result=None，由调用方跳过探针统计与方向消费。
    """
    nominal = printed["nominal"]
    l2 = binary_l2(target_tensor, nominal, threshold=threshold, ownership_mask=ownership_tensor)
    pv = pvband(printed["dose_max"], printed["defocus_min"], threshold=threshold, ownership_mask=ownership_tensor)
    if len(probe_slots):
        epe_result = edge_probes(
            target_tensor,
            nominal,
            torch.from_numpy(probe_slots),
            torch.from_numpy(inner_xy),
            torch.from_numpy(outer_xy),
            threshold=threshold,
        )
    else:
        epe_result = None
    return int(l2), int(pv), epe_result
