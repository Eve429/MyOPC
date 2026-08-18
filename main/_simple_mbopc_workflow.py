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
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow  # 公共生命周期
from main.configuration import (  # simple 配置解析
    MBOPCConfig,
    resolve_mbopc_config,
)
from opc.iteration.mbopc import (  # simple 求解器
    SimpleMBOPCConfig,  # summary_extras 的 DBU 配置类型注解
    SimpleMBOPCResult,
    optimize_macro,
)

_RESULT_FORMAT_VERSION = 1  # 每 macro result NPZ 结构版本


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
    optimize_macro=optimize_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras)


def run_mbopc(config_path: str | Path) -> dict:
    """按 config 实际网格逐 macro 独立求解 simple MB-OPC，一次合并产出。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    return run_mbopc_workflow(SIMPLE_METHOD, config_path)  # 公共生命周期
