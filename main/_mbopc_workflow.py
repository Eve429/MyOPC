"""MB-OPC 公共工作流：方法无关的求解生命周期，算法差异经 MBOPCMethod 注入。"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import psutil
import torch

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import atomic_write_json
from common.runtime import resolve_device
from lithography import ICCAD13Lithography

# 共用 macro 生命周期
from main._macro_pipeline import (
    merge_macro_results,
    prepare_problems,
    save_final_lithography,
    save_source_lithography,
    write_macro_gds,
)

# 统一配置体系（方法无关五段）
from main.configuration import (
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    load_config,
)

# problem 加载与 best 几何重建
from opc.input.edge import (
    MacroProblem,
    reconstruct_region,
)
from opc.iteration.mbopc import TargetCanvasCache


@dataclass(frozen=True, slots=True)
class MBOPCMethod:
    """保存一个 MB-OPC 方法注入公共生命周期的全部差异点。"""

    method_name: str  # summary["method"] 方法标识
    algo_config_type: type  # load_config 请求的算法段 Config
    build_solver_config: Callable  # (algo, partition, edge, dbu_nm) -> 配置
    optimize_macro: Callable  # (problem, model, cfg, cache, *, on_tiles_completed) -> result
    save_macro_result: Callable  # (macro_dir, macro_id, result)：NPZ+JSON
    macro_summary: Callable  # (macro_id, macro_dir, result, best_gds, elapsed) -> 条目
    summary_extras: Callable  # (solver_config) -> 顶层附加摘要键


def _solve_macro(
    method: MBOPCMethod,
    problem: MacroProblem,
    model,
    solver_config,
    target_cache,
    output_dir: Path,
    *,
    dbu_um: float,
    show_progress: bool,
    progress_position: int,
    leave_progress: bool,
) -> tuple[object, Path]:
    """显示 tile 进度，让一个 macro 完成全部求解并写出 best GDS。"""
    bar = None
    if show_progress:
        from tqdm import tqdm

        # baseline 与每个更新后状态都要评价全部 tile
        bar = tqdm(
            total=(solver_config.iterations + 1) * problem.macro.core_count,
            desc=f"macro {problem.macro.macro_id}",
            unit="tile",
            position=progress_position,
            leave=leave_progress,
        )
    on_tiles = None if bar is None else bar.update
    try:
        result = method.optimize_macro(problem, model, solver_config, target_cache, on_tiles_completed=on_tiles)
    finally:
        if bar is not None:
            bar.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    # best 位移的最终候选几何
    best_region = reconstruct_region(problem, result.best_displacements)
    # 完整候选 GDS（RESULT Cell）
    best_gds = write_macro_gds(problem.layer, best_region, output_dir / "best.gds", dbu_um)
    return result, best_gds


def run_mbopc_workflow(method: MBOPCMethod, config_path: str | Path) -> dict:
    """按 method 注入的算法差异逐 macro 独立求解，全部完成后一次合并。"""
    total_started = time.perf_counter()
    process = psutil.Process()
    rss_start = process.memory_info().rss
    layout, partition, litho, edge, algo, output = load_config(
        config_path, LayoutConfig, PartitionConfig, LithographyConfig, EdgeConfig, method.algo_config_type, OutputConfig
    )
    # 准备 problem（共用生命周期）
    plan = prepare_problems(layout, partition, litho, edge, output)
    rss_after_prepare = process.memory_info().rss
    peak_rss = max(rss_start, rss_after_prepare)
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # DBU 的 nm 值
    # 算法差异：跨段校验+nm→DBU
    solver_config = method.build_solver_config(algo, partition, edge, dbu_nm)
    device = resolve_device(litho.device)
    # CUDA 峰值统计设备必须显式指定：不传 device 时 PyTorch 统计当前设备
    cuda_stats_device = torch.device(device) if device.startswith("cuda") else None
    model = ICCAD13Lithography(device=device)
    if cuda_stats_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_stats_device)
    macro_count = plan["macro_count"]
    target_cache = TargetCanvasCache(solver_config.target_cache_bytes)  # 跨 macro 共享
    work_dir = output.work_dir
    macros_dir = work_dir / "macros"
    macro_gds: dict[str, Path] = {}
    macro_summaries = []
    outer_bar = None
    if macro_count > 1 and output.show_progress:
        from tqdm import tqdm

        outer_bar = tqdm(total=macro_count, desc="macros", unit="macro", position=0)
    try:
        for entry in plan["macros"]:
            macro_id = entry["macro_id"]
            problem = MacroProblem.load(Path(entry["problem_file"]))
            started = time.perf_counter()
            result, best_gds = _solve_macro(
                method,
                problem,
                model,
                solver_config,
                target_cache,
                macros_dir / macro_id,
                dbu_um=float(plan["dbu_um"]),
                show_progress=output.show_progress,
                progress_position=1 if outer_bar is not None else 0,
                leave_progress=outer_bar is None,
            )
            elapsed = time.perf_counter() - started
            peak_rss = max(peak_rss, process.memory_info().rss)
            macro_dir = macros_dir / macro_id
            method.save_macro_result(macro_dir, macro_id, result)
            macro_summaries.append(method.macro_summary(macro_id, macro_dir, result, best_gds, elapsed))
            macro_gds[macro_id] = best_gds
            if outer_bar is not None:
                outer_bar.update(1)
            del problem, result
    finally:
        if outer_bar is not None:
            outer_bar.close()
    # 全部 macro 完成后只合并一次（独立 macro 策略，不做逐轮全局合并）。
    merge_started = time.perf_counter()
    final_path = merge_macro_results(plan, macro_gds, output.final_layout, cell_mode=output.final_cell_mode)
    merge_seconds = time.perf_counter() - merge_started
    manifest = None
    source_manifest = None
    if output.save_final_lithography:
        # 逐 tile 流式 PNG（最终合并 GDS）
        manifest = save_final_lithography(
            plan, final_path, model, solver_config.batch_size, work_dir / "final_lithography"
        )
        # 源版图对照：同一模型同一网格参数（收尾前向次数翻倍的既定代价）
        source_manifest = save_source_lithography(
            plan, Path(plan["layout"]), model, solver_config.batch_size, work_dir / "final_lithography_source"
        )
    peak_rss = max(peak_rss, process.memory_info().rss)
    cuda_peak = int(torch.cuda.max_memory_allocated(cuda_stats_device)) if cuda_stats_device is not None else None
    summary = {
        "method": method.method_name,
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "segment_count_sum": plan["segment_count_sum"],
        "work_dir": str(work_dir),
        "save_metric_trends": output.save_metric_trends,
        "device": str(model.device),
        "iterations": solver_config.iterations,
        # 趋势字段属于用户输出配置，不复制进 solver config，避免算法状态与留档配置耦合。
        "metric_trend_fields": list(algo.metric_trend_fields),
        **method.summary_extras(solver_config),
        "macros": macro_summaries,
        "final_layout": str(final_path),
        "final_cell_mode": output.final_cell_mode,
        "merge_seconds": merge_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "rss_start_bytes": rss_start,
        "rss_after_prepare_bytes": rss_after_prepare,
        "peak_rss_bytes": peak_rss,
        "cuda_peak_bytes": cuda_peak,
        "final_lithography_tiles": None if manifest is None else manifest["tile_count"],
        "source_lithography_tiles": (None if source_manifest is None else source_manifest["tile_count"]),
    }
    atomic_write_json(work_dir / "summary.json", summary)
    return summary
