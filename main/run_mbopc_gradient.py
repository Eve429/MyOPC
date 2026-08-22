"""梯度 MB-OPC 直接运行入口：方法适配器 + CLI 摘要一体（单/多 macro 通用）。"""

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import atomic_write_json, atomic_write_npz
from common.metric_trends import save_metric_trends
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow

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


def save_macro_result(macro_dir: Path, macro_id: str, result: GradientMBOPCResult) -> None:
    """写出梯度结果 NPZ 与逐状态 metrics（文件名独立于 simple 产物）。"""
    # 梯度结果 NPZ（独立于 simple 的 result.npz）
    atomic_write_npz(
        macro_dir / "gradient_result.npz",
        format_version=np.array([_GRADIENT_RESULT_VERSION], np.int32),
        macro_id=np.array([macro_id]),
        best_state_index=np.array([result.best_state_index], np.int32),
        best_displacements=np.ascontiguousarray(result.best_displacements, dtype=np.float64),
        stop_reason=np.array([result.stop_reason]),
    )
    # 逐状态标量
    atomic_write_json(
        macro_dir / "gradient_metrics.json",
        {
            "macro_id": macro_id,
            "best_state_index": result.best_state_index,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "records": [asdict(record) for record in result.records],
        },
    )


def macro_summary(macro_id: str, macro_dir: Path, result: GradientMBOPCResult, best_gds: Path, elapsed: float) -> dict:
    """构造公共循环消费的逐 macro 摘要条目。"""
    best_record = result.records[result.best_state_index]
    # 摘要（全量记录在 gradient_metrics.json）
    return {
        "macro_id": macro_id,
        "best_state_index": result.best_state_index,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "state_count": len(result.records),
        "best_total_loss": best_record.total_loss,
        "best_l2": best_record.l2,
        "best_pvband": best_record.pvband,
        "best_epe": best_record.epe,
        "best_epe_loss": best_record.epe_loss,
        "best_gds": str(best_gds),
        "result_npz": str(macro_dir / "gradient_result.npz"),
        "metrics_json": str(macro_dir / "gradient_metrics.json"),
        "elapsed_seconds": elapsed,
    }


def summary_extras(solver_config: GradientMBOPCConfig) -> dict:
    """顶层附加摘要键：四 loss 权重与 EPE 陡度（公共键由公共层写）。"""
    return {
        "loss_weights": {
            "nominal_l2": solver_config.weight_nominal_l2,
            "process_l2": solver_config.weight_process_l2,
            "pvband": solver_config.weight_pvband,
            "epe": solver_config.weight_epe,
        },
        "epe_steepness": solver_config.epe_steepness,
    }


# 梯度方法适配器实例（公共生命周期消费）
GRADIENT_METHOD = MBOPCMethod(
    method_name="gradient_mbopc",
    algo_config_type=GradientConfig,
    build_solver_config=resolve_gradient_config,
    optimize_macro=optimize_gradient_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras,
)


def run_gradient_mbopc(
    config_path: str | Path,
    *,
    overview_mode: Literal["mean", "lines"] = "mean",
) -> dict:
    """准备并逐 macro 独立求解梯度 MB-OPC，全部完成后一次合并（任意 macro 数）。"""
    summary = run_mbopc_workflow(GRADIENT_METHOD, config_path)
    metrics_files = {macro["macro_id"]: Path(macro["metrics_json"]) for macro in summary["macros"]}
    best_states = {macro["macro_id"]: macro["best_state_index"] for macro in summary["macros"]}
    summary["metric_trends"] = (
        save_metric_trends(
            metrics_files,
            Path(summary["work_dir"]) / "metrics_trends",
            summary["metric_trend_fields"],
            best_state_indices=best_states,
            overview_mode=overview_mode,
        )
        if summary["save_metric_trends"]
        else None
    )
    atomic_write_json(Path(summary["work_dir"]) / "summary.json", summary)
    return summary


def main() -> int:
    """读取唯一位置参数 config，运行梯度 MB-OPC 流程并打印中文摘要。"""
    if len(sys.argv) != 2:
        print("用法：python main/run_mbopc_gradient.py <config.toml>", file=sys.stderr)
        return 2  # 参数错误退出码
    summary = run_gradient_mbopc(sys.argv[1])
    print("梯度 MB-OPC 执行完成：")
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")
    weights = summary["loss_weights"]
    # 目标函数
    print(
        f"  loss 权重：nominal={weights['nominal_l2']} "
        f"process={weights['process_l2']} pv={weights['pvband']} "
        f"epe={weights['epe']}(γ={summary['epe_steepness']})"
    )
    for macro in summary["macros"]:  # 逐 macro 摘要
        print(
            f"  {macro['macro_id']}：best_state={macro['best_state_index']} "
            f"loss={macro['best_total_loss']:.6f} "
            f"stop={macro['stop_reason']}"
        )
    # 耗时
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 {summary['total_seconds']:.2f}s")
    cuda_peak = summary["cuda_peak_bytes"]
    # CPU 运行无 CUDA 峰值
    cuda_text = "N/A" if cuda_peak is None else f"{cuda_peak / 1024 / 1024:.0f} MiB"
    # 资源
    print(f"  峰值 RSS：{summary['peak_rss_bytes'] / 1024 / 1024:.0f} MiB，CUDA 峰值：{cuda_text}")
    if summary["final_lithography_tiles"] is not None:
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")
    if summary["metric_trends"] is not None:
        print(f"  指标趋势图：{summary['metric_trends']['directory']}")
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")
    return 0  # 成功退出码


if __name__ == "__main__":
    raise SystemExit(main())
