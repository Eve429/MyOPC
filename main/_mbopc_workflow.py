"""MB-OPC 公共工作流：方法无关的求解生命周期，算法差异经 MBOPCMethod 注入。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # perf_counter 阶段计时
from collections.abc import Callable  # 适配器钩子类型
from dataclasses import dataclass  # MBOPCMethod 打包
from decimal import Decimal  # nm→DBU 精确换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import psutil  # summary 的 RSS 峰值采样
import torch  # CUDA 峰值统计（显式设备）

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.io import atomic_write_json  # JSON 原子写出
from common.runtime import resolve_device  # 设备解析
from lithography import ICCAD13Lithography  # 固定 ICCAD13 光刻模型
from main._macro_pipeline import (  # 共用 macro 生命周期
    merge_macro_results,
    prepare_problems,
    save_final_lithography,
)
from main.configuration import (  # 统一配置体系（方法无关五段）
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    load_config,
)
from opc.input.edge import MacroProblem  # problem 加载
from opc.iteration.mbopc import TargetCanvasCache  # target uint8 LRU 缓存


@dataclass(frozen=True, slots=True)
class MBOPCMethod:
    """保存一个 MB-OPC 方法注入公共生命周期的全部差异点。

    solver_config 对本层不透明，但其 DBU 配置（Simple/Gradient 均然）必须
    暴露 target_cache_bytes/batch_size/iterations 三属性——缓存容量、留档
    批大小与摘要迭代上限消费的鸭子契约，新增方法须满足。
    """

    method_name: str               # summary["method"] 方法标识
    algo_config_type: type         # load_config 请求的算法段 Config
    build_solver_config: Callable  # (algo, partition, edge, dbu_nm) -> 配置
    solve_macro: Callable          # 单 macro 求解（tqdm+optimize+best GDS）
    save_macro_result: Callable    # (macro_dir, macro_id, result)：NPZ+JSON
    macro_summary: Callable        # (macro_id, macro_dir, result, best_gds, elapsed) -> 条目
    summary_extras: Callable       # (solver_config) -> 顶层附加摘要键


def run_mbopc_workflow(method: MBOPCMethod, config_path: str | Path) -> dict:
    """按 method 注入的算法差异逐 macro 独立求解，全部完成后一次合并。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    total_started = time.perf_counter()  # 全流程计时
    process = psutil.Process()  # RSS 采样进程句柄
    rss_start = process.memory_info().rss  # 起点 RSS
    layout, partition, litho, edge, algo, output = load_config(  # 统一加载
        config_path, LayoutConfig, PartitionConfig, LithographyConfig,
        EdgeConfig, method.algo_config_type, OutputConfig)
    plan = prepare_problems(  # 阶段 0/1（共用生命周期）
        layout, partition, litho, edge, output)
    rss_after_prepare = process.memory_info().rss  # 准备后 RSS
    peak_rss = max(rss_start, rss_after_prepare)  # 峰值初值
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # DBU 的 nm 值
    solver_config = method.build_solver_config(  # 算法差异：跨段校验+nm→DBU
        algo, partition, edge, dbu_nm)
    device = resolve_device(litho.device)  # 设备解析（auto→实际）
    # CUDA 峰值统计设备必须显式指定：不传 device 时 PyTorch 统计当前设备
    # （默认 cuda:0），多卡下会量错卡；这里不改进程全局设备（不 set_device）。
    cuda_stats_device = (torch.device(device)
                         if device.startswith("cuda") else None)  # 统计目标
    model = ICCAD13Lithography(device=device)  # 固定 ICCAD13 模型
    if cuda_stats_device is not None:  # CUDA 峰值从模型加载后开始计量
        torch.cuda.reset_peak_memory_stats(cuda_stats_device)  # 显式统计设备
    macro_count = plan["macro_count"]  # macro 总数（任意 ≥1）
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
        result, best_gds = method.solve_macro(  # 全部迭代 + best GDS
            problem, model, solver_config, target_cache,
            macros_dir / macro_id,  # 专属产物目录
            dbu_um=float(plan["dbu_um"]),  # GDS 写出需要源 DBU（NPZ 不含）
            show_progress=output.show_progress,
            progress_position=1 if outer_bar is not None else 0,  # 外层占 0
            leave_progress=outer_bar is None)  # 多 macro 内层条不留存
        elapsed = time.perf_counter() - started  # 单 macro 耗时
        peak_rss = max(peak_rss, process.memory_info().rss)  # 逐 macro 采峰
        macro_dir = macros_dir / macro_id  # 产物目录
        method.save_macro_result(  # 算法差异：NPZ + metrics.json
            macro_dir, macro_id, result)
        macro_summaries.append(method.macro_summary(  # 算法差异：摘要条目
            macro_id, macro_dir, result, best_gds, elapsed))
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
            plan, final_path, model, solver_config.batch_size,
            work_dir / "final_lithography")
    peak_rss = max(peak_rss, process.memory_info().rss)  # 收尾前采峰
    cuda_peak = (int(torch.cuda.max_memory_allocated(cuda_stats_device))  # 同卡峰值
                 if cuda_stats_device is not None else None)  # CPU 时为 None
    summary = {  # 完整摘要
        "method": method.method_name,
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "segment_count_sum": plan["segment_count_sum"],
        "device": str(model.device),
        "iterations": solver_config.iterations,
        **method.summary_extras(solver_config),  # 算法差异：附加顶层键
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
