"""梯度 MB-OPC 方法适配器：算法差异点注入公共工作流。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
from dataclasses import asdict  # 记录序列化
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # result NPZ 数组载体

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.io import atomic_write_json, atomic_write_npz  # 原子写出
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow  # 公共生命周期

# 梯度配置解析
from main.configuration import (
    GradientConfig,
    resolve_gradient_config,
)

# gradient 求解器
from opc.iteration.mbopc import (
    GradientMBOPCConfig,
    GradientMBOPCResult,
    optimize_gradient_macro,
)

_GRADIENT_RESULT_VERSION = 1  # 每 macro 梯度结果 NPZ 结构版本


def save_macro_result(macro_dir: Path, macro_id: str,
                      result: GradientMBOPCResult) -> None:
    """写出梯度结果 NPZ 与逐状态 metrics（文件名独立于 simple 产物）。"""
    # 梯度结果 NPZ（独立于 simple 的 result.npz）
    atomic_write_npz(
        macro_dir / "gradient_result.npz",
        format_version=np.array([_GRADIENT_RESULT_VERSION], np.int32),
        macro_id=np.array([macro_id]),
        best_state_index=np.array([result.best_state_index], np.int32),
        best_displacements=np.ascontiguousarray(
            result.best_displacements, dtype=np.float64),
        stop_reason=np.array([result.stop_reason]))
    # 逐状态标量
    atomic_write_json(macro_dir / "gradient_metrics.json", {
        "macro_id": macro_id,
        "best_state_index": result.best_state_index,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "records": [asdict(record) for record in result.records]})


def macro_summary(macro_id: str, macro_dir: Path, result: GradientMBOPCResult,
                  best_gds: Path, elapsed: float) -> dict:
    """构造公共循环消费的逐 macro 摘要条目。"""
    best_record = result.records[result.best_state_index]  # 最佳状态指标
    # 摘要（全量记录在 gradient_metrics.json）
    return {
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
        "elapsed_seconds": elapsed}


def summary_extras(solver_config: GradientMBOPCConfig) -> dict:
    """顶层附加摘要键：三 loss 权重（方法标识等公共键由公共层写）。"""
    return {"loss_weights": {"nominal_l2": solver_config.weight_nominal_l2,
                             "process_l2": solver_config.weight_process_l2,
                             "pvband": solver_config.weight_pvband}}


# 梯度方法适配器实例（公共生命周期消费）
GRADIENT_METHOD = MBOPCMethod(
    method_name="gradient_mbopc",
    algo_config_type=GradientConfig,
    build_solver_config=resolve_gradient_config,
    optimize_macro=optimize_gradient_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras)


def run_gradient_mbopc(config_path: str | Path) -> dict:
    """准备并逐 macro 独立求解梯度 MB-OPC，全部完成后一次合并（任意 macro 数）。"""
    return run_mbopc_workflow(GRADIENT_METHOD, config_path)  # 公共生命周期
