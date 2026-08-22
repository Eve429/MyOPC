"""最简 MB-OPC 直接运行入口：方法适配器 + CLI 摘要一体（macro 数由网格决定）。"""

import json
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
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow

# simple 配置解析
from main.configuration import (
    MBOPCConfig,
    resolve_mbopc_config,
)

# simple 求解器
from opc.iteration.mbopc import (
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    optimize_simple_macro,
)

_RESULT_FORMAT_VERSION = 2  # 每 macro result NPZ 结构版本（v2：键改 state 词汇）
_TREND_FIELDS = ("epe", "l2", "pvband", "moved_segments")
_TREND_TITLES = {
    "epe": "EPE",
    "l2": "L2",
    "pvband": "PVBand",
    "moved_segments": "Moved Segments",
}


def save_macro_result(macro_dir: Path, macro_id: str, result: SimpleMBOPCResult) -> None:
    """写出 simple 结果 NPZ 与逐状态 metrics（文件名独立于 gradient 产物）。"""
    # result NPZ（位移与停止信息）
    atomic_write_npz(
        macro_dir / "result.npz",
        format_version=np.array([_RESULT_FORMAT_VERSION], np.int32),
        macro_id=np.array([macro_id]),
        best_state_index=np.array([result.best_state_index], np.int32),
        best_displacements=np.ascontiguousarray(result.best_displacements, dtype=np.float64),
        stop_reason=np.array([result.stop_reason]),
    )
    # 逐状态标量与原因
    atomic_write_json(
        macro_dir / "metrics.json",
        {
            "macro_id": macro_id,
            "best_state_index": result.best_state_index,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "records": [asdict(record) for record in result.records],
        },
    )


def macro_summary(macro_id: str, macro_dir: Path, result: SimpleMBOPCResult, best_gds: Path, elapsed: float) -> dict:
    """构造公共循环消费的逐 macro 摘要条目。"""
    best_record = result.records[result.best_state_index]
    # 摘要（全量记录在 metrics.json）
    return {
        "macro_id": macro_id,
        "best_state_index": result.best_state_index,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "state_count": len(result.records),
        "best_epe": best_record.epe,
        "best_l2": best_record.l2,
        "best_pvband": best_record.pvband,
        "best_gds": str(best_gds),
        "metrics_json": str(macro_dir / "metrics.json"),
        "elapsed_seconds": elapsed,
    }


def summary_extras(solver_config: SimpleMBOPCConfig) -> dict:
    """顶层附加摘要键：simple 无附加键（资源统计等公共键由公共层写）。"""
    return {}


def _read_trend_records(summary: dict) -> dict[str, list[dict]]:
    """读取公共 workflow 已写出的各 macro 指标记录。"""
    records_by_macro = {}
    for macro in summary["macros"]:
        metrics_path = Path(macro["metrics_json"])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records_by_macro[macro["macro_id"]] = payload["records"]
    return records_by_macro


def _plot_trend_panels(
    plt,
    records_by_macro: dict[str, list[dict]],
    output_path: Path,
    *,
    overview_mode: str,
    best_state_index: int | None = None,
) -> None:
    """绘制四项状态指标并保存 PNG，overview_mode 决定单 macro 或总览曲线。"""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    # 总览只在相同 state_index 的实际记录之间求均值，不对提前停止的 macro 插值。
    for axis, field in zip(axes.flat, _TREND_FIELDS):
        if overview_mode == "mean":
            state_values = {}
            for records in records_by_macro.values():
                for record in records:
                    state_values.setdefault(int(record["state_index"]), []).append(float(record[field]))
            states = sorted(state_values)
            values = [float(np.mean(state_values[state])) for state in states]
            axis.plot(states, values, marker="o", label="macro mean")
        elif overview_mode == "lines":
            for macro_id, records in records_by_macro.items():
                states = [int(record["state_index"]) for record in records]
                values = [float(record[field]) for record in records]
                axis.plot(states, values, marker="o", label=macro_id)
        elif overview_mode == "macro":
            records = next(iter(records_by_macro.values()))
            states = [int(record["state_index"]) for record in records]
            values = [float(record[field]) for record in records]
            axis.plot(states, values, marker="o")
            if best_state_index is not None:
                axis.axvline(best_state_index, color="tab:red", linestyle="--", alpha=0.6)
        else:
            raise ValueError("未知趋势图模式")
        axis.set_title(_TREND_TITLES[field])
        axis.set_xlabel("state_index")
        axis.grid(True, alpha=0.3)
    if overview_mode == "lines":
        axes.flat[0].legend(loc="best")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metric_trends(summary: dict, *, overview_mode: Literal["mean", "lines"] = "mean") -> dict:
    """读取 Simple 记录并保存每 macro 与总览趋势图。"""
    if overview_mode not in ("mean", "lines"):
        raise ValueError("overview_mode 必须是 'mean' 或 'lines'")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records_by_macro = _read_trend_records(summary)
    trend_dir = Path(summary["work_dir"]) / "metrics_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)
    macro_files = {}
    for macro_id, records in records_by_macro.items():
        output_path = trend_dir / f"macro_{macro_id}.png"
        _plot_trend_panels(
            plt,
            {macro_id: records},
            output_path,
            overview_mode="macro",
            best_state_index=next(
                item["best_state_index"] for item in summary["macros"] if item["macro_id"] == macro_id
            ),
        )
        macro_files[macro_id] = str(output_path)
    overview_path = trend_dir / f"overview_{overview_mode}.png"
    _plot_trend_panels(plt, records_by_macro, overview_path, overview_mode=overview_mode)
    return {
        "directory": str(trend_dir),
        "overview_mode": overview_mode,
        "overview_png": str(overview_path),
        "macro_pngs": macro_files,
    }


# simple 方法适配器实例（公共生命周期消费）
SIMPLE_METHOD = MBOPCMethod(
    method_name="simple_mbopc",
    algo_config_type=MBOPCConfig,
    build_solver_config=resolve_mbopc_config,
    optimize_macro=optimize_simple_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras,
)


def run_mbopc(config_path: str | Path, *, overview_mode: Literal["mean", "lines"] = "mean") -> dict:
    """按 config 实际网格逐 macro 独立求解 simple MB-OPC，一次合并产出。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    summary = run_mbopc_workflow(SIMPLE_METHOD, config_path)
    # 优化、合并和已有 metrics.json 全部成功后才绘图；绘图异常直接传播，
    # 不生成声称成功的趋势 metadata。
    summary["metric_trends"] = (
        save_metric_trends(summary, overview_mode=overview_mode) if summary["save_metric_trends"] else None
    )
    atomic_write_json(Path(summary["work_dir"]) / "summary.json", summary)
    return summary


def main() -> int:
    """读取唯一位置参数 config，运行 simple MB-OPC 流程并打印中文摘要。"""
    if len(sys.argv) != 2:
        print("用法：python main/run_mbopc.py <config.toml>", file=sys.stderr)
        return 2  # 参数错误退出码
    summary = run_mbopc(sys.argv[1])
    print("simple MB-OPC 执行完成：")
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")
    for macro in summary["macros"]:  # 逐 macro 摘要
        print(
            f"  {macro['macro_id']}：best_state={macro['best_state_index']} "
            f"best_epe={macro['best_epe']} stop={macro['stop_reason']}"
        )
    # 耗时
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 {summary['total_seconds']:.2f}s")
    cuda_peak = summary["cuda_peak_bytes"]
    # CPU 运行无 CUDA 峰值
    cuda_text = "N/A" if cuda_peak is None else f"{cuda_peak / 1024 / 1024:.0f} MiB"
    # 资源（与 gradient 入口同款，summary 键公共层本就提供）
    print(f"  峰值 RSS：{summary['peak_rss_bytes'] / 1024 / 1024:.0f} MiB，CUDA 峰值：{cuda_text}")
    if summary["final_lithography_tiles"] is not None:
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")
    if summary["metric_trends"] is not None:
        print(f"  指标趋势图：{summary['metric_trends']['directory']}")
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")
    return 0  # 成功退出码


if __name__ == "__main__":
    raise SystemExit(main())
