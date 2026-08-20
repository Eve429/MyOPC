"""Simple ILT 方法适配器：算法差异点注入公共像素工作流。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
from pathlib import Path  # 全部路径统一使用 Path 对象

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_ilt_workflow 可导入

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
    build_fixed_context_canvas=build_simple_final_context_canvas)


def run_simple_ilt(config_path: str | Path) -> dict:
    """准备并逐 macro 独立求解 Simple ILT，全部完成后一次合并。"""
    return run_ilt_workflow(SIMPLE_ILT_METHOD, config_path)
