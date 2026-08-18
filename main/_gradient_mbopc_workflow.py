"""梯度 MB-OPC 工作流：逐 macro Adam 梯度求解、一次合并与资源统计。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # perf_counter 阶段计时
import warnings  # 学习率超限的风险提示（不改合法配置集合）
from dataclasses import asdict  # 记录序列化
from decimal import Decimal  # nm→DBU 精确换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # result NPZ 数组载体
import psutil  # summary 的 RSS 峰值采样
import torch  # CUDA 峰值统计（显式设备）

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.io import atomic_write_json, atomic_write_npz  # 原子写出
from common.runtime import resolve_device  # 设备解析
from common.units import exact_dbu  # nm→DBU 精确换算
from lithography import ICCAD13Lithography  # 固定 ICCAD13 光刻模型
from main._macro_pipeline import (  # 共用 macro 生命周期
    merge_macro_results,
    prepare_problems,
    save_final_lithography,
    write_macro_gds,
)
from main.configuration import (  # 统一配置体系（gradient 路径所需）
    EdgeConfig,
    GradientConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    load_config,
)
from opc.input.edge import MacroProblem, reconstruct_region  # problem 与重建
from opc.iteration.mbopc import (  # gradient 求解器
    GradientMBOPCConfig,
    GradientMBOPCResult,
    TargetCanvasCache,
    optimize_gradient_macro,
)

_GRADIENT_RESULT_VERSION = 1  # 每 macro 梯度结果 NPZ 结构版本


def solve_gradient_macro(
        problem: MacroProblem,
        model: ICCAD13Lithography,
        config: GradientMBOPCConfig,
        target_cache: TargetCanvasCache,
        output_dir: Path,
        *,
        dbu_um: float,
        show_progress: bool,
        progress_position: int,
        leave_progress: bool,
) -> tuple[GradientMBOPCResult, Path]:
    """显示 tile 进度，让一个 macro 完成全部梯度状态并写出 best GDS。"""
    bar = None  # 进度条（show_progress=False 时保持 None）
    if show_progress:  # 局部导入：关闭进度或未安装 tqdm 时不受影响
        from tqdm import tqdm  # 进度显示库
        bar = tqdm(  # baseline 与每个更新后状态都要评价全部 tile
            total=(config.iterations + 1) * problem.macro.core_count,
            desc=f"macro {problem.macro.macro_id}", unit="tile",  # tile 单位
            position=progress_position, leave=leave_progress)  # 多层条位置
    on_tiles = None if bar is None else bar.update  # backward 且释放后回调
    try:  # 异常路径也要收尾进度条（finally 关闭，不留未结束的终端状态）
        result = optimize_gradient_macro(  # 独立完成 baseline 与全部梯度状态
            problem, model, config, target_cache, on_tiles_completed=on_tiles)
    finally:  # 提前停止按实际完成量收尾，不伪造 100%
        if bar is not None:
            bar.close()
    output_dir.mkdir(parents=True, exist_ok=True)  # macro 专属目录
    best_region = reconstruct_region(  # best 位移的最终候选几何
        problem, result.best_displacements)
    best_gds = write_macro_gds(  # 完整候选 GDS（RESULT Cell）
        problem, best_region, output_dir / "best.gds", dbu_um)
    return result, best_gds  # 结果与 GDS 路径


def run_gradient_mbopc(config_path: str | Path) -> dict:
    """准备并逐 macro 独立求解梯度 MB-OPC，全部完成后一次合并（任意 macro 数）。"""
    total_started = time.perf_counter()  # 全流程计时
    process = psutil.Process()  # RSS 采样进程句柄
    rss_start = process.memory_info().rss  # 起点 RSS
    layout, partition, litho, edge, gradient, output = load_config(  # 统一加载
        config_path, LayoutConfig, PartitionConfig, LithographyConfig,
        EdgeConfig, GradientConfig, OutputConfig)
    if gradient.learning_rate_nm > edge.max_displacement_nm:  # 超限仍合法只提示
        # Adam 首步更新尺度与 lr 同量级，超限会让大量段一步打到 ±上限 被
        # clamp，抬高 invalid_geometry/优化停滞风险；不改参数、不硬拒绝。
        warnings.warn(
            f"learning_rate_nm={gradient.learning_rate_nm} 超过 "
            f"max_displacement_nm={edge.max_displacement_nm}；"
            "Adam 更新可能在早期大量触发位移 clamp，"
            "增加 invalid_geometry 或优化停滞风险",
            UserWarning, stacklevel=2)
    if gradient.epe_distance_nm > partition.context_nm:  # 探针越上下文
        raise ValueError("epe_distance_nm 不得超过 context_nm")
    plan = prepare_problems(  # 阶段 0/1（共用生命周期）
        layout, partition, litho, edge, output)
    rss_after_prepare = process.memory_info().rss  # 准备后 RSS
    peak_rss = max(rss_start, rss_after_prepare)  # 峰值初值
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # DBU 的 nm 值
    # 学习率是连续 optimizer 步长：Decimal 相除后转 float，不走 exact_dbu
    # 整数契约（其余网格/pixel/segment/探针参数仍走整数 DBU 契约）。
    solver_config = GradientMBOPCConfig(  # nm→DBU 运行时派生（solver 输入包）
        iterations=gradient.iterations,
        learning_rate_dbu=float(gradient.learning_rate_nm / dbu_nm),
        weight_nominal_l2=gradient.weight_nominal_l2,
        weight_process_l2=gradient.weight_process_l2,
        weight_pvband=gradient.weight_pvband,
        epe_distance_dbu=float(exact_dbu(  # 探针距离仍走精确整数换算
            gradient.epe_distance_nm, dbu_nm, "epe_distance_nm")),
        batch_size=gradient.batch_size,
        target_cache_bytes=gradient.target_cache_mb * 1024 * 1024)
    device = resolve_device(litho.device)  # 设备解析（auto→实际）
    # CUDA 峰值统计设备必须显式指定：不传 device 时 PyTorch 统计当前设备
    # （默认 cuda:0），多卡下会量错卡；这里不改进程全局设备（不 set_device）。
    cuda_stats_device = (torch.device(device)
                         if device.startswith("cuda") else None)  # 统计目标
    model = ICCAD13Lithography(device=device)  # 固定 ICCAD13 模型
    if cuda_stats_device is not None:  # CUDA 峰值从模型加载后开始计量
        torch.cuda.reset_peak_memory_stats(cuda_stats_device)  # 显式统计设备
    macro_count = plan["macro_count"]  # macro 总数（本入口接受任意 ≥1）
    target_cache = TargetCanvasCache(solver_config.target_cache_bytes)  # 跨 macro 共享
    work_dir = output.work_dir  # 非 None 已由 prepare_problems 保证
    macros_dir = work_dir / "macros"  # 逐 macro 产物根目录
    macro_gds: dict[str, Path] = {}  # macro_id → best GDS（merge 显式映射）
    macro_summaries = []  # 逐 macro 摘要
    outer_bar = None  # 多 macro 外层进度条
    if macro_count > 1 and output.show_progress:
        from tqdm import tqdm  # 进度显示库
        outer_bar = tqdm(total=macro_count, desc="macros",  # 外层 macro 单位
                         unit="macro", position=0)  # 占第 0 行
    for entry in plan["macros"]:  # 稳定顺序逐 macro 独立求解
        macro_id = entry["macro_id"]  # macro 编号
        problem = MacroProblem.load(Path(entry["problem_file"]))  # 加载 problem
        started = time.perf_counter()  # 单 macro 计时
        result, best_gds = solve_gradient_macro(  # 全部状态 + best GDS
            problem, model, solver_config, target_cache,
            macros_dir / macro_id,  # 专属产物目录
            dbu_um=float(plan["dbu_um"]),  # GDS 写出需要源 DBU（NPZ 不含）
            show_progress=output.show_progress,
            progress_position=1 if outer_bar is not None else 0,  # 外层占 0
            leave_progress=outer_bar is None)  # 多 macro 内层条不留存
        elapsed = time.perf_counter() - started  # 单 macro 耗时
        peak_rss = max(peak_rss, process.memory_info().rss)  # 逐 macro 采峰
        macro_dir = macros_dir / macro_id  # 产物目录
        atomic_write_npz(  # 梯度结果 NPZ（独立于 simple 的 result.npz）
            macro_dir / "gradient_result.npz",
            format_version=np.array([_GRADIENT_RESULT_VERSION], np.int32),
            macro_id=np.array([macro_id]),
            best_state_index=np.array([result.best_state_index], np.int32),
            best_displacements=np.ascontiguousarray(
                result.best_displacements, dtype=np.float64),
            stop_reason=np.array([result.stop_reason]))
        atomic_write_json(macro_dir / "gradient_metrics.json", {  # 逐状态标量
            "macro_id": macro_id,
            "best_state_index": result.best_state_index,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "records": [asdict(record) for record in result.records]})
        best_record = result.records[result.best_state_index]  # 最佳状态指标
        macro_summaries.append({  # 摘要（全量记录在 gradient_metrics.json）
            "macro_id": macro_id,
            "best_state_index": result.best_state_index,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "state_count": len(result.records),
            "best_total_loss": best_record.total_loss,
            "best_l2": best_record.l2, "best_pvband": best_record.pvband,
            "best_epe": best_record.epe,
            "best_gds": str(best_gds),
            "result_npz": str(macro_dir / "gradient_result.npz"),
            "metrics_json": str(macro_dir / "gradient_metrics.json"),
            "elapsed_seconds": elapsed})
        macro_gds[macro_id] = best_gds  # 记录显式映射
        if outer_bar is not None:  # 外层条按完成 macro 计数
            outer_bar.update(1)
        del problem, result  # 释放当前 macro 再处理下一个
    if outer_bar is not None:  # 外层条收尾
        outer_bar.close()
    # 全部 macro 完成后只合并一次（独立 macro 策略，不做逐轮全局合并）。
    merge_started = time.perf_counter()  # 合并计时
    final_path = merge_macro_results(  # 统一 ownership 权威覆盖写出
        plan, macro_gds, output.final_layout,
        cell_mode=output.final_cell_mode)
    merge_seconds = time.perf_counter() - merge_started  # 合并耗时
    manifest = None  # 最终光刻留档
    if output.save_final_lithography:  # 只对最终合并 GDS 运行一次
        manifest = save_final_lithography(  # 逐 tile 流式 PNG
            plan, final_path, model, gradient.batch_size,
            work_dir / "final_lithography")
    peak_rss = max(peak_rss, process.memory_info().rss)  # 收尾前采峰
    cuda_peak = (int(torch.cuda.max_memory_allocated(cuda_stats_device))  # 同卡峰值
                 if cuda_stats_device is not None else None)  # CPU 时为 None
    summary = {  # 完整摘要
        "method": "gradient_mbopc",
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "segment_count_sum": plan["segment_count_sum"],
        "device": str(model.device),
        "iterations": solver_config.iterations,
        "loss_weights": {"nominal_l2": solver_config.weight_nominal_l2,
                         "process_l2": solver_config.weight_process_l2,
                         "pvband": solver_config.weight_pvband},
        "macros": macro_summaries,
        "final_layout": str(final_path),
        "final_cell_mode": output.final_cell_mode,
        "merge_seconds": merge_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "rss_start_bytes": rss_start,
        "rss_after_prepare_bytes": rss_after_prepare,
        "peak_rss_bytes": peak_rss,
        "cuda_peak_bytes": cuda_peak,
        "final_lithography_tiles": None if manifest is None else manifest["tile_count"]}
    atomic_write_json(work_dir / "summary.json", summary)  # 落盘
    return summary  # 返回摘要


