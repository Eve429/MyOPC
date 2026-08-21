"""像素 ILT 公共工作流：方法无关的 macro 生命周期（准备/求解/终评/合并）。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # perf_counter 阶段计时
from collections.abc import Callable  # 方法钩子类型
from dataclasses import asdict, dataclass  # 方法描述与记录序列化
from decimal import Decimal  # nm→DBU 精确换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # 二值终评画布组装
import psutil  # summary 的 RSS 峰值采样
import torch  # CUDA 峰值统计（显式设备）

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.io import atomic_write_json, atomic_write_npz  # 原子写出
from common.runtime import resolve_device  # 设备解析
from evaluation import evaluate_binary_l2, evaluate_pvband  # 二值终评指标
from layout import LayerSpec, LayoutDB  # 版图打开与层规格
from lithography import ICCAD13Lithography  # 固定 ICCAD13 光刻模型

# 共用 macro 生命周期（merge/最终光刻留档/GDS 写出）
from main._macro_pipeline import (
    merge_macro_results,
    resolve_field_bounds,
    save_final_lithography,
    write_macro_gds,
)

# 统一配置体系（像素 ILT 不读取 [edge]）
from main.configuration import (
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    load_config,
    resolve_grid_config,
)

# 像素输入与网格
from opc.input import plan_macros
from opc.input.pixel import (
    PixelMacroProblem,
    prepare_pixel_macro_problem,
    reconstruct_pixel_region,
)

_ILT_PLAN_FORMAT_VERSION = 1  # ilt_plan.json 结构版本
_ILT_RESULT_FORMAT_VERSION = 1  # 每 macro 结果 NPZ 结构版本


@dataclass(frozen=True, slots=True)
class ILTMethod:
    """注入公共像素 ILT 生命周期的最小方法差异点（仅当前真实调用方）。

    optimize_macro 返回的 ILTMacroResult 对本层只读消费 best/binary/记录；
    config_type 必须可被 load_config 直接注册（get_type_hints 解析），且
    鸭子契约只须暴露 batch_size（终评分批）；固定 context 的 transmission
    定义由 build_fixed_context_canvas 策略注入，本层不读任何方法数学字段。
    """

    method_name: str  # summary/产物稳定标识
    config_type: type  # load_config 请求的算法段 Config
    optimize_macro: Callable  # (problem, model, config, *, on_tiles_completed)
    evaluated_states: Callable  # (config) -> 每 macro 评价状态数（进度 total）
    build_fixed_context_canvas: Callable
    # (problem, core_index, config) -> 固定 context 画布（方法数学所在地）


def prepare_pixel_problems(
    layout: LayoutConfig, partition: PartitionConfig, litho: LithographyConfig, output: OutputConfig
) -> dict:
    """像素 ILT 阶段 0/1：逐 macro 一次栅格化并写出 ilt_plan.json。"""
    if output.work_dir is None:  # 本流程要求工作目录
        raise ValueError("此流程要求 [output].work_dir")
    layer = LayerSpec(layout.layer, layout.datatype)  # 目标层规格
    started = time.perf_counter()  # 阶段计时起点
    process = psutil.Process()  # RSS 采样进程对象
    peak_rss = process.memory_info().rss  # 峰值初值
    with LayoutDB.open(layout.layout, layout.top_cell) as database:  # 打开并自动关闭
        top_cell_name = database.top_cell_name  # 在库存活期内捕获顶层名
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        layer_bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生，不物化）
        if layer_bounds is None:  # 目标层无图形
            raise ValueError(f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")
        # 处理框（field_box/field_size）：未配置时即 layer bbox，零行为变化；
        # 环带（field − layer bbox）transmission 由极性背景外推给出
        bounds = resolve_field_bounds(layout, layer_bounds, dbu_nm)
        # 网格换算不含边段参数；像素整除与画布容量在 plan_macros 内校验
        grid = resolve_grid_config(partition, litho, dbu_nm)
        macros = plan_macros(
            bounds,
            macro_grid=partition.macro_grid,
            macro_size_dbu=grid.macro_size_dbu,
            core_size_dbu=grid.core_dbu,
            context_dbu=grid.context_dbu,
            pixel_dbu=grid.pixel_dbu,
            canvas_pixels=litho.canvas_pixels,
        )
        # ownership 复核：面积和恰等于父框即无正面积重叠
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")
        problems_dir = output.work_dir / "pixel_problems"  # problem 存放目录
        problems_dir.mkdir(parents=True, exist_ok=True)  # 创建目录结构
        entries = []  # 逐 macro 计划条目
        pixel_count_sum = 0  # macro ownership 像素总数（summary 规模键）
        for macro in macros:  # 行优先顺序逐 macro 准备
            # 完整相交物化一次；实际 box 整像素校验在 problem 构造内前置
            batch = database.query([layer], macro.query_box).materialize_intersecting()
            problem = prepare_pixel_macro_problem(batch, layer, layout.polarity, macro, layout_bounds=bounds)
            problem_path = problem.save(problems_dir / f"{macro.macro_id}.npz")
            pixel_count_sum += int(problem.ownership_shape[0] * problem.ownership_shape[1])
            entries.append(
                {
                    "macro_id": macro.macro_id,
                    "ownership_box": [
                        macro.ownership_box.left,
                        macro.ownership_box.bottom,
                        macro.ownership_box.right,
                        macro.ownership_box.top,
                    ],
                    "core_count": macro.core_count,
                    "problem_file": str(problem_path),
                    "problem_bytes": problem_path.stat().st_size,
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)  # 采样峰值
            del batch, problem  # 立即释放当前 macro 大对象
    # 全部 problem 成功且 LayoutDB 已关闭才写出"准备完成"的 plan；
    # 键集与 merge_macro_results/save_final_lithography 消费面保持一致。
    prepare_seconds = time.perf_counter() - started
    plan = {
        "format_version": _ILT_PLAN_FORMAT_VERSION,
        "layout": str(layout.layout),
        "top_cell": top_cell_name,
        "dbu_um": float(dbu_nm / 1000),
        "layer": [layer.layer, layer.datatype],
        "polarity": layout.polarity.value,
        "core_size_dbu": grid.core_dbu,
        "context_dbu": grid.context_dbu,
        "pixel_dbu": grid.pixel_dbu,
        "canvas_pixels": litho.canvas_pixels,
        "macro_count": len(macros),
        "core_count": sum(macro.core_count for macro in macros),
        "pixel_count_sum": pixel_count_sum,
        "work_dir": str(output.work_dir),
        "final_layout": str(output.final_layout),
        "final_cell_mode": output.final_cell_mode,
        "macros": entries,
        "prepare_seconds": prepare_seconds,
        "prepare_peak_rss_bytes": peak_rss,
    }
    atomic_write_json(output.work_dir / "ilt_plan.json", plan)
    return plan


def _binary_canvas(
    problem: PixelMacroProblem, binary_mask: np.ndarray, core_index: int, build_context: Callable, config
) -> np.ndarray:
    """组装终评画布：trainable 像素取 best 二值，context 由方法策略提供。

    本函数不含任何方法数学：σ(β(2T−1))、hard target 等固定 context 的
    transmission 定义全部在方法模块的 build_fixed_context_canvas 内。
    """
    trainable = problem.trainable_index_canvas(core_index)
    values = binary_mask.reshape(-1)[np.maximum(trainable, 0)]
    context = build_context(problem, core_index, config)
    return np.where(trainable >= 0, values, context).astype(np.float32)


def _evaluate_best_binary(
    problem: PixelMacroProblem, result, model: ICCAD13Lithography, config, conditions, build_context: Callable
) -> tuple[int, int]:
    """在 best 二值掩膜上执行最终前向并按 ownership 统计 L2/PVBand。"""
    core_count = problem.macro.core_count
    binary_l2 = 0  # 二值 L2 累计
    pvband = 0  # 二值 PVBand 累计
    with torch.no_grad():  # 纯推理终评
        for batch_start in range(0, core_count, config.batch_size):
            core_indices = list(range(batch_start, min(batch_start + config.batch_size, core_count)))
            masks = np.stack(
                [_binary_canvas(problem, result.binary_mask, c, build_context, config) for c in core_indices]
            )
            targets = np.stack([problem.target_canvas(c) for c in core_indices])
            ownerships = np.stack([problem.ownership_canvas(c) for c in core_indices])
            target_tensor = torch.from_numpy(targets).to(device=model.device, dtype=torch.float32).div_(255.0)
            ownership_tensor = torch.from_numpy(ownerships).to(model.device)
            mask_tensor = torch.from_numpy(masks).to(model.device)
            printed = model.forward_many(mask_tensor, conditions)
            binary_l2 += evaluate_binary_l2(
                target_tensor,
                printed["nominal"],
                threshold=float(model.config.print_threshold),
                ownership_mask=ownership_tensor,
            )
            pvband += evaluate_pvband(
                printed["dose_max"],
                printed["defocus_min"],
                threshold=float(model.config.print_threshold),
                ownership_mask=ownership_tensor,
            )
            # 释放：每批写完立即失去引用
            del printed, mask_tensor, target_tensor, ownership_tensor
    return binary_l2, pvband


def run_ilt_workflow(method: ILTMethod, config_path: str | Path) -> dict:
    """像素 ILT 公共生命周期：准备→逐 macro 求解/终评/产物→恰一次合并→summary。"""
    total_started = time.perf_counter()  # 全流程计时
    process = psutil.Process()  # RSS 采样进程句柄
    rss_start = process.memory_info().rss  # 起点 RSS
    layout, partition, litho, algo, output = load_config(
        config_path, LayoutConfig, PartitionConfig, LithographyConfig, method.config_type, OutputConfig
    )
    plan = prepare_pixel_problems(layout, partition, litho, output)
    rss_after_prepare = process.memory_info().rss  # 准备后 RSS
    peak_rss = max(rss_start, rss_after_prepare)  # 峰值初值
    device = resolve_device(litho.device)  # 设备解析（auto→实际）
    # CUDA 峰值统计设备必须显式指定（与 MB-OPC 工作流同款约束）
    cuda_stats_device = torch.device(device) if device.startswith("cuda") else None
    model = ICCAD13Lithography(device=device)  # 固定 ICCAD13 模型
    if cuda_stats_device is not None:  # 从模型加载后开始计量
        torch.cuda.reset_peak_memory_stats(cuda_stats_device)
    conditions = (model.condition("nominal"), model.condition("dose_max"), model.condition("defocus_min"))
    layer = LayerSpec(plan["layer"][0], plan["layer"][1])  # 目标层
    dbu_um = float(plan["dbu_um"])  # 源版图 DBU
    macro_count = plan["macro_count"]  # macro 总数
    work_dir = output.work_dir  # 非 None 已由 prepare 保证
    macros_dir = work_dir / "macros"  # 逐 macro 产物根目录
    macro_gds: dict[str, Path] = {}  # macro_id → best GDS（merge 显式映射）
    macro_summaries = []  # 逐 macro 摘要
    outer_bar = None  # 多 macro 外层进度条
    if macro_count > 1 and output.show_progress:
        from tqdm import tqdm  # 进度显示库

        outer_bar = tqdm(total=macro_count, desc="macros", unit="macro", position=0)
    try:  # 异常路径也要收尾外层进度条
        for entry in plan["macros"]:  # 稳定顺序逐 macro 独立求解
            macro_id = entry["macro_id"]  # macro 编号
            problem = PixelMacroProblem.load(Path(entry["problem_file"]))
            started = time.perf_counter()  # 单 macro 计时
            bar = None  # 内层 tile 进度条
            if output.show_progress:
                from tqdm import tqdm  # 进度显示库

                # 求解进度 = 评价状态数 × core 数（终评批次不计入求解 total）
                bar = tqdm(
                    total=(method.evaluated_states(algo) * problem.macro.core_count),
                    desc=f"macro {macro_id}",
                    unit="tile",
                    position=1 if outer_bar is not None else 0,
                    leave=outer_bar is None,
                )
            on_tiles = None if bar is None else bar.update  # 批完成回调
            try:  # 异常路径同样收尾内层条
                result = method.optimize_macro(problem, model, algo, on_tiles_completed=on_tiles)
            finally:
                if bar is not None:
                    bar.close()
            # best 二值终评（REQ-010：独立于训练的最终前向）
            binary_l2, binary_pvband = _evaluate_best_binary(
                problem, result, model, algo, conditions, method.build_fixed_context_canvas
            )
            macro_dir = macros_dir / macro_id  # 产物目录
            macro_dir.mkdir(parents=True, exist_ok=True)
            # 像素 → Region → best GDS（RESULT Cell，完整 macro 候选）
            region = reconstruct_pixel_region(problem, result.binary_mask)
            best_gds = write_macro_gds(layer, region, macro_dir / "best.gds", dbu_um)
            # 结果 NPZ（独立于 MB-OPC 命名空间）
            atomic_write_npz(
                macro_dir / f"{method.method_name}_result.npz",
                format_version=np.array([_ILT_RESULT_FORMAT_VERSION], np.int32),
                macro_id=np.array([macro_id]),
                ownership_box=np.array(entry["ownership_box"], np.int64),
                best_parameters=np.ascontiguousarray(result.best_parameters, dtype=np.float32),
                soft_mask=np.ascontiguousarray(result.soft_mask, dtype=np.float32),
                binary_mask=np.ascontiguousarray(result.binary_mask).astype(np.uint8),
                best_state_index=np.array([result.best_state_index], np.int32),
            )
            best_record = result.records[result.best_state_index]
            metrics = {
                "macro_id": macro_id,
                "best_state_index": result.best_state_index,
                "state_count": len(result.records),
                "records": [asdict(record) for record in result.records],
                "binary_l2": binary_l2,
                "binary_pvband": binary_pvband,
                "core_count": problem.macro.core_count,
            }
            atomic_write_json(macro_dir / "metrics.json", metrics)
            elapsed = time.perf_counter() - started  # 单 macro 耗时
            peak_rss = max(peak_rss, process.memory_info().rss)  # 逐 macro 采峰
            macro_summaries.append(
                {
                    "macro_id": macro_id,
                    "best_state_index": result.best_state_index,
                    "state_count": len(result.records),
                    "best_total_loss": best_record.total_loss,
                    "best_nominal_l2": best_record.nominal_l2,
                    "best_process_l2": best_record.process_l2,
                    "best_pvband_loss": best_record.pvband_loss,
                    "best_curvature_loss": best_record.curvature_loss,
                    "binary_l2": binary_l2,
                    "binary_pvband": binary_pvband,
                    "best_gds": str(best_gds),
                    "result_npz": str(macro_dir / f"{method.method_name}_result.npz"),
                    "metrics_json": str(macro_dir / "metrics.json"),
                    "elapsed_seconds": elapsed,
                }
            )
            macro_gds[macro_id] = best_gds  # 记录显式映射
            if outer_bar is not None:
                outer_bar.update(1)
            del problem, result  # 释放当前 macro 再处理下一个
    finally:  # 第 2 个 macro 抛异常时外层条同样收尾
        if outer_bar is not None:
            outer_bar.close()
    # 全部 macro 完成后只合并一次（独立 macro 策略）
    merge_started = time.perf_counter()
    final_path = merge_macro_results(plan, macro_gds, output.final_layout, cell_mode=output.final_cell_mode)
    merge_seconds = time.perf_counter() - merge_started
    manifest = None  # 最终光刻留档
    if output.save_final_lithography:  # 只对最终合并 GDS 运行一次
        manifest = save_final_lithography(plan, final_path, model, algo.batch_size, work_dir / "final_lithography")
    peak_rss = max(peak_rss, process.memory_info().rss)  # 收尾前采峰
    cuda_peak = int(torch.cuda.max_memory_allocated(cuda_stats_device)) if cuda_stats_device is not None else None
    summary = {
        "method": method.method_name,
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "pixel_count_sum": plan["pixel_count_sum"],
        "device": str(model.device),
        "iterations": getattr(algo, "iterations", None),
        "states_total": method.evaluated_states(algo),  # 多尺度方法无 iterations 字段时的状态数事实源
        "macros": macro_summaries,
        "final_layout": str(final_path),
        "final_cell_mode": output.final_cell_mode,
        # 已知 seam 策略显式入档：macro 间不交换参数，context 恒为初始 target
        "seam_strategy": "macro_independent_fixed_context",
        "prepare_seconds": plan["prepare_seconds"],
        "merge_seconds": merge_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "rss_start_bytes": rss_start,
        "rss_after_prepare_bytes": rss_after_prepare,
        "peak_rss_bytes": peak_rss,
        "cuda_peak_bytes": cuda_peak,
        "final_lithography_tiles": (None if manifest is None else manifest["tile_count"]),
    }
    atomic_write_json(work_dir / "summary.json", summary)
    return summary
