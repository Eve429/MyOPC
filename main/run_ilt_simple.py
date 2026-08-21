"""Simple ILT 直接运行入口：方法适配器 + CLI 摘要一体。"""

import sys  # 命令行参数与退出码
from pathlib import Path  # 仓库根定位

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_ilt_workflow 与 opc 可导入

from main._ilt_workflow import ILTMethod, run_ilt_workflow  # 公共生命周期

# Simple ILT 求解器与终评 fixed-context helper
from opc.iteration.ilt import (
    SimpleILTConfig,
    build_simple_final_context_canvas,
    optimize_simple_macro,
)


def _evaluated_states(config: SimpleILTConfig) -> int:
    """Simple 每 macro 评价 iterations+1 个状态（N 次同步更新）。"""
    return config.iterations + 1


# Simple 方法适配器实例（公共生命周期消费）
SIMPLE_ILT_METHOD = ILTMethod(
    method_name="simple_ilt",
    config_type=SimpleILTConfig,
    optimize_macro=optimize_simple_macro,
    evaluated_states=_evaluated_states,
    build_fixed_context_canvas=build_simple_final_context_canvas,
)


def run_simple_ilt(config_path: str | Path) -> dict:
    """准备并逐 macro 独立求解 Simple ILT，全部完成后一次合并。"""
    return run_ilt_workflow(SIMPLE_ILT_METHOD, config_path)


def main() -> int:
    """读取可选位置参数 config（默认仓库内 config/simple_ilt.toml）并执行。"""
    if len(sys.argv) > 2:  # 参数数量不符
        print("用法：python main/run_simple_ilt.py [config.toml]", file=sys.stderr)
        return 2  # 参数错误退出码
    config = sys.argv[1] if len(sys.argv) == 2 else str(_REPO_ROOT / "config" / "simple_ilt.toml")
    summary = run_simple_ilt(config)  # 完整像素 ILT 流程
    print("Simple ILT 执行完成：")  # 摘要标题
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")
    print(
        f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}，"
        f"宏像素总数：{summary['pixel_count_sum']}"
    )
    for macro in summary["macros"]:  # 逐 macro 摘要
        print(
            f"  {macro['macro_id']}：best_state={macro['best_state_index']} "
            f"loss={macro['best_total_loss']:.6f} "
            f"binaryL2={macro['binary_l2']} stop_state={macro['state_count']}"
        )
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 {summary['total_seconds']:.2f}s")
    cuda_peak = summary["cuda_peak_bytes"]  # CUDA 峰值字节数
    cuda_text = "N/A" if cuda_peak is None else f"{cuda_peak / 1024 / 1024:.0f} MiB"
    print(f"  峰值 RSS：{summary['peak_rss_bytes'] / 1024 / 1024:.0f} MiB，CUDA 峰值：{cuda_text}")
    if summary["final_lithography_tiles"] is not None:  # 光刻留档
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")
    return 0  # 成功


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())
