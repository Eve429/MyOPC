"""梯度 MB-OPC 直接运行入口：单/多 macro 通用（macro 数任意 ≥1）。"""

import sys  # 命令行参数与退出码
from pathlib import Path  # 仓库根定位

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_gradient_mbopc_workflow 可导入

from main._gradient_mbopc_workflow import run_gradient_mbopc  # 梯度工作流


def main() -> int:
    """读取唯一位置参数 config，运行梯度 MB-OPC 流程并打印中文摘要。"""
    if len(sys.argv) != 2:  # 参数数量不符
        # 提示
        print("用法：python main/run_gradient_mbopc.py <config.toml>",
              file=sys.stderr)
        return 2  # 参数错误退出码
    summary = run_gradient_mbopc(sys.argv[1])  # 任意 macro 数的完整流程
    print("梯度 MB-OPC 执行完成：")  # 摘要标题
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")  # 运行环境
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")  # 网格规模
    weights = summary["loss_weights"]  # 四权重
    # 目标函数
    print(f"  loss 权重：nominal={weights['nominal_l2']} "
          f"process={weights['process_l2']} pv={weights['pvband']} "
          f"epe={weights['epe']}(γ={summary['epe_steepness']})")
    for macro in summary["macros"]:  # 逐 macro 摘要
        # 关键指标
        print(f"  {macro['macro_id']}：best_state={macro['best_state_index']} "
              f"loss={macro['best_total_loss']:.6f} "
              f"stop={macro['stop_reason']}")
    # 耗时
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 "
          f"{summary['total_seconds']:.2f}s")
    cuda_peak = summary["cuda_peak_bytes"]  # CUDA 峰值字节数
    # CPU 运行无 CUDA 峰值
    cuda_text = ("N/A" if cuda_peak is None
                 else f"{cuda_peak / 1024 / 1024:.0f} MiB")
    # 资源
    print(f"  峰值 RSS：{summary['peak_rss_bytes'] / 1024 / 1024:.0f} MiB，"
          f"CUDA 峰值：{cuda_text}")
    if summary["final_lithography_tiles"] is not None:  # 光刻留档
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")  # 数量
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")  # 输出
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
