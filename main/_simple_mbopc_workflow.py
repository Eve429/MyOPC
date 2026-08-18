"""simple MB-OPC 方法适配器：算法差异点注入公共工作流。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
from dataclasses import asdict  # 记录序列化
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # result NPZ 数组载体

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.io import atomic_write_json, atomic_write_npz  # 原子写出
from lithography import ICCAD13Lithography  # 模型类型注解
from main._macro_pipeline import write_macro_gds  # 单 macro 候选 GDS 写出
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow  # 公共生命周期
from main.configuration import (  # simple 配置解析
    MBOPCConfig,
    resolve_mbopc_config,
)
from opc.input.edge import MacroProblem, reconstruct_region  # problem 与重建
from opc.iteration.mbopc import (  # simple 求解器
    SimpleMBOPCConfig,  # solve_macro 的 DBU 配置类型注解
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


def save_macro_result(macro_dir: Path, macro_id: str,
                      result: SimpleMBOPCResult) -> None:
    """写出 simple 结果 NPZ 与逐轮 metrics（文件名独立于 gradient 产物）。"""
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


def macro_summary(macro_id: str, macro_dir: Path, result: SimpleMBOPCResult,
                  best_gds: Path, elapsed: float) -> dict:
    """构造公共循环消费的逐 macro 摘要条目。"""
    best_record = result.records[result.best_round]  # 最佳轮指标
    return {  # 摘要（全量记录在 metrics.json）
        "macro_id": macro_id,
        "best_round": result.best_round,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "round_count": len(result.records),
        "best_epe": best_record.epe, "best_l2": best_record.l2,
        "best_pvband": best_record.pvband,
        "best_gds": str(best_gds),
        "elapsed_seconds": elapsed}


def summary_extras(solver_config: SimpleMBOPCConfig) -> dict:
    """顶层附加摘要键：simple 无附加键（资源统计等公共键由公共层写）。"""
    return {}


SIMPLE_METHOD = MBOPCMethod(  # simple 方法适配器实例（公共生命周期消费）
    method_name="simple_mbopc",
    algo_config_type=MBOPCConfig,
    build_solver_config=resolve_mbopc_config,
    solve_macro=solve_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras)


def run_mbopc(config_path: str | Path) -> dict:
    """按 config 实际网格逐 macro 独立求解 simple MB-OPC，一次合并产出。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    return run_mbopc_workflow(SIMPLE_METHOD, config_path)  # 公共生命周期
