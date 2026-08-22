"""最简 MB-OPC 直接运行入口：方法适配器 + CLI 摘要一体（macro 数由网格决定）。"""

import sys  # 命令行参数与退出码
from dataclasses import asdict  # 记录序列化
from pathlib import Path  # 仓库根定位

import numpy as np  # result NPZ 数组载体

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_mbopc_workflow 与 opc 可导入

from common.io import atomic_write_json, atomic_write_npz  # 原子写出
from main._mbopc_workflow import MBOPCMethod, run_mbopc_workflow  # 公共生命周期

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


def save_macro_result(macro_dir: Path, macro_id: str,
                      result: SimpleMBOPCResult) -> None:
    """写出 simple 结果 NPZ 与逐状态 metrics（文件名独立于 gradient 产物）。"""
    # result NPZ（位移与停止信息）
    atomic_write_npz(
        macro_dir / "result.npz",
        format_version=np.array([_RESULT_FORMAT_VERSION], np.int32),
        macro_id=np.array([macro_id]),
        best_state_index=np.array([result.best_state_index], np.int32),
        best_displacements=np.ascontiguousarray(
            result.best_displacements, dtype=np.float64),
        stop_reason=np.array([result.stop_reason]))
    # 逐状态标量与原因
    atomic_write_json(macro_dir / "metrics.json", {
        "macro_id": macro_id,
        "best_state_index": result.best_state_index,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "records": [asdict(record) for record in result.records]})


def macro_summary(macro_id: str, macro_dir: Path, result: SimpleMBOPCResult,
                  best_gds: Path, elapsed: float) -> dict:
    """构造公共循环消费的逐 macro 摘要条目。"""
    best_record = result.records[result.best_state_index]  # 最佳状态指标
    # 摘要（全量记录在 metrics.json）
    return {
        "macro_id": macro_id,
        "best_state_index": result.best_state_index,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "state_count": len(result.records),
        "best_epe": best_record.epe, "best_l2": best_record.l2,
        "best_pvband": best_record.pvband,
        "best_gds": str(best_gds),
        "elapsed_seconds": elapsed}


def summary_extras(solver_config: SimpleMBOPCConfig) -> dict:
    """顶层附加摘要键：simple 无附加键（资源统计等公共键由公共层写）。"""
    return {}


# simple 方法适配器实例（公共生命周期消费）
SIMPLE_METHOD = MBOPCMethod(
    method_name="simple_mbopc",
    algo_config_type=MBOPCConfig,
    build_solver_config=resolve_mbopc_config,
    optimize_macro=optimize_simple_macro,
    save_macro_result=save_macro_result,
    macro_summary=macro_summary,
    summary_extras=summary_extras)


def run_mbopc(config_path: str | Path) -> dict:
    """按 config 实际网格逐 macro 独立求解 simple MB-OPC，一次合并产出。

    macro 数量不加人为约束：macro_grid/macro_size_nm 是几就按几求解。
    """
    return run_mbopc_workflow(SIMPLE_METHOD, config_path)  # 公共生命周期


def main() -> int:
    """读取唯一位置参数 config，运行 simple MB-OPC 流程并打印中文摘要。"""
    if len(sys.argv) != 2:  # 参数数量不符
        # 提示
        print("用法：python main/run_mbopc.py <config.toml>",
              file=sys.stderr)
        return 2  # 参数错误退出码
    summary = run_mbopc(sys.argv[1])  # 任意 macro 数的完整流程
    print("simple MB-OPC 执行完成：")  # 摘要标题
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")  # 运行环境
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")  # 网格规模
    for macro in summary["macros"]:  # 逐 macro 摘要
        # 关键指标
        print(f"  {macro['macro_id']}：best_state={macro['best_state_index']} "
              f"best_epe={macro['best_epe']} stop={macro['stop_reason']}")
    # 耗时
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 "
          f"{summary['total_seconds']:.2f}s")
    cuda_peak = summary["cuda_peak_bytes"]  # CUDA 峰值字节数
    # CPU 运行无 CUDA 峰值
    cuda_text = ("N/A" if cuda_peak is None
                 else f"{cuda_peak / 1024 / 1024:.0f} MiB")
    # 资源（与 gradient 入口同款，summary 键公共层本就提供）
    print(f"  峰值 RSS：{summary['peak_rss_bytes'] / 1024 / 1024:.0f} MiB，"
          f"CUDA 峰值：{cuda_text}")
    if summary["final_lithography_tiles"] is not None:  # 光刻留档
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")  # 数量
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")  # 输出
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
