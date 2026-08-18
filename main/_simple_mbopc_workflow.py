"""simple MB-OPC 工作流：逐 macro 离散 EPE 求解、一次合并与摘要产出。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # perf_counter 阶段计时
from dataclasses import asdict  # 记录序列化
from decimal import Decimal  # nm→DBU 精确换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # result NPZ 数组载体

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
from main.configuration import (  # 统一配置体系（simple 路径所需）
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    MBOPCConfig,
    OutputConfig,
    PartitionConfig,
    load_config,
)
from opc.input.edge import MacroProblem, reconstruct_region  # problem 与重建
from opc.iteration.mbopc import (  # simple 求解器
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    TargetCanvasCache,
    optimize_macro,
)

_RESULT_FORMAT_VERSION = 1  # 每 macro result NPZ 结构版本


def solve_macro(
        problem: MacroProblem,
        model: ICCAD13Lithography,
        config: SimpleMBOPCConfig,
        target_cache: TargetCanvasCache,
        output_dir: Path,
        *,
        dbu_um: float,
        show_progress: bool,
        progress_position: int,
        leave_progress: bool,
) -> tuple[SimpleMBOPCResult, Path]:
    """显示 tile 进度，让一个 macro 完成全部迭代并写出 best GDS。"""
    bar = None  # 进度条（show_progress=False 时保持 None）
    if show_progress:  # 局部导入：关闭进度或未安装 tqdm 时不受影响
        from tqdm import tqdm  # 进度显示库
        bar = tqdm(  # baseline 与每个移动后状态都要评价全部 tile
            total=(config.iterations + 1) * problem.macro.core_count,
            desc=f"macro {problem.macro.macro_id}", unit="tile",  # tile 单位
            position=progress_position, leave=leave_progress)  # 多层条位置
    on_tiles = None if bar is None else bar.update  # 批完成且张量已释放后回调
    try:  # 异常路径也要收尾进度条（finally 关闭，不留未结束的终端状态）
        result = optimize_macro(  # 独立完成 baseline 与全部离散 EPE 轮次
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


def run_mbopc(config_path: str | Path) -> dict:
    """按 config 实际网格逐 macro 独立求解 simple MB-OPC，一次合并产出。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    total_started = time.perf_counter()  # 全流程计时
    layout, partition, litho, edge, mbopc, output = load_config(  # 统一加载
        config_path, LayoutConfig, PartitionConfig, LithographyConfig,
        EdgeConfig, MBOPCConfig, OutputConfig)
    # 跨 Config 契约（单一 Config 内业务校验已在各自 __post_init__）。
    if mbopc.initial_step_nm > edge.max_displacement_nm:  # 步长超位移上限
        raise ValueError("initial_step_nm 不得超过 max_displacement_nm")
    if mbopc.epe_distance_nm > partition.context_nm:  # 探针越上下文
        raise ValueError("epe_distance_nm 不得超过 context_nm")
    plan = prepare_problems(  # 阶段 0/1（work_dir 在此查 None）
        layout, partition, litho, edge, output)
    macro_count = plan["macro_count"]  # macro 总数
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # DBU 的 nm 值
    solver_config = SimpleMBOPCConfig(  # nm→DBU 运行时派生（solver 输入包）
        iterations=mbopc.iterations,
        initial_step_dbu=float(exact_dbu(
            mbopc.initial_step_nm, dbu_nm, "initial_step_nm")),
        decay_every=mbopc.decay_every,
        epe_distance_dbu=float(exact_dbu(
            mbopc.epe_distance_nm, dbu_nm, "epe_distance_nm")),
        batch_size=mbopc.batch_size,
        target_cache_bytes=mbopc.target_cache_mb * 1024 * 1024)
    device = resolve_device(litho.device)  # 设备解析（auto→实际）
    model = ICCAD13Lithography(device=device)  # 固定 ICCAD13 模型
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
        result, best_gds = solve_macro(  # 全部迭代 + best GDS
            problem, model, solver_config, target_cache,
            macros_dir / macro_id,  # 专属产物目录
            dbu_um=float(plan["dbu_um"]),  # GDS 写出需要源 DBU（NPZ 不含）
            show_progress=output.show_progress,
            progress_position=1 if outer_bar is not None else 0,  # 外层占 0
            leave_progress=outer_bar is None)  # 多 macro 内层条不留存
        elapsed = time.perf_counter() - started  # 单 macro 耗时
        macro_dir = macros_dir / macro_id  # 产物目录
        atomic_write_npz(  # result NPZ（位移与停止信息）
            macro_dir / "result.npz",
            format_version=np.array([_RESULT_FORMAT_VERSION], np.int32),
            macro_id=np.array([macro_id]),
            best_round=np.array([result.best_round], np.int32),
            best_displacements=np.ascontiguousarray(
                result.best_displacements, dtype=np.float64),
            stop_reason=np.array([result.stop_reason]))
        atomic_write_json(macro_dir / "metrics.json", {  # 逐轮标量与原因
            "macro_id": macro_id,
            "best_round": result.best_round,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "records": [asdict(record) for record in result.records]})
        macro_gds[macro_id] = best_gds  # 记录显式映射
        best_record = result.records[result.best_round]  # 最佳轮指标
        macro_summaries.append({  # 摘要（全量记录在 metrics.json）
            "macro_id": macro_id,
            "best_round": result.best_round,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "round_count": len(result.records),
            "best_epe": best_record.epe, "best_l2": best_record.l2,
            "best_pvband": best_record.pvband,
            "best_gds": str(best_gds),
            "elapsed_seconds": elapsed})
        if outer_bar is not None:  # 外层条按完成 macro 计数
            outer_bar.update(1)
        del problem  # 释放当前 macro 再处理下一个
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
            plan, final_path, model, mbopc.batch_size,
            work_dir / "final_lithography")
    summary = {  # 完整摘要
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "segment_count_sum": plan["segment_count_sum"],
        "device": str(model.device),
        "iterations": solver_config.iterations,
        "macros": macro_summaries,
        "final_layout": str(final_path),
        "final_cell_mode": output.final_cell_mode,
        "merge_seconds": merge_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "final_lithography_tiles": None if manifest is None else manifest["tile_count"]}
    atomic_write_json(work_dir / "summary.json", summary)  # 落盘
    return summary  # 返回摘要


