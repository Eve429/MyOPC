"""最简 MB-OPC 直接运行入口：macro 数量由 config 网格决定（单/多通用）。"""

import sys  # 命令行参数与退出码
from pathlib import Path  # 仓库根定位

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_mbopc_workflow 可导入

from main._mbopc_workflow import run_mbopc  # 共享工作流


def main() -> int:
    """读取唯一位置参数 config，运行 simple MB-OPC 流程并打印中文摘要。"""
    if len(sys.argv) != 2:  # 参数数量不符
        print("用法：python main/run_mbopc.py <config.toml>",
              file=sys.stderr)  # 提示
        return 2  # 参数错误退出码
    summary = run_mbopc(sys.argv[1])  # 任意 macro 数的完整流程
    print("simple MB-OPC 执行完成：")  # 摘要标题
    print(f"  device：{summary['device']}，迭代上限：{summary['iterations']}")  # 运行环境
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")  # 网格规模
    for macro in summary["macros"]:  # 逐 macro 摘要
        print(f"  {macro['macro_id']}：best_round={macro['best_round']} "
              f"best_epe={macro['best_epe']} stop={macro['stop_reason']}")  # 关键指标
    print(f"  合并 {summary['merge_seconds']:.2f}s，总计 "
          f"{summary['total_seconds']:.2f}s")  # 耗时
    if summary["final_lithography_tiles"] is not None:  # 光刻留档
        print(f"  最终光刻 PNG：{summary['final_lithography_tiles']} 个 tile")  # 数量
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")  # 输出
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
