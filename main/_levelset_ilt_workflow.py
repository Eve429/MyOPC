"""LevelSet ILT 方法适配器：算法差异点注入公共像素工作流。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
from functools import partial
from pathlib import Path  # 全部路径统一使用 Path 对象

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 main/_ilt_workflow 可导入

from main._ilt_workflow import ILTMethod, run_ilt_workflow  # 公共生命周期
from main.configuration import LithographyConfig, load_config

# LevelSet ILT 求解器与终评 fixed-context helper
from opc.iteration.ilt import (
    LevelSetILTConfig,
    build_levelset_final_context_canvas,
    optimize_levelset_macro,
)


def _evaluated_states(config: LevelSetILTConfig) -> int:
    """LevelSet 每 macro 评价 iterations+1 个状态（N 次 Adam 更新）。"""
    return config.iterations + 1


# LevelSet 方法适配器实例（公共生命周期消费）
LEVELSET_ILT_METHOD = ILTMethod(
    method_name="levelset_ilt",
    config_type=LevelSetILTConfig,
    optimize_macro=optimize_levelset_macro,
    evaluated_states=_evaluated_states,
    build_fixed_context_canvas=build_levelset_final_context_canvas)


def run_levelset_ilt(config_path: str | Path) -> dict:
    """准备并逐 macro 独立求解 LevelSet ILT，全部完成后一次合并。

    LevelSet 的 phi/step_size 使用物理 nm 单位；pixel_nm 的唯一事实源仍是
    [lithography]。适配层只把该运行时尺度绑定到 solver，不复制进算法配置。
    """
    litho, = load_config(config_path, LithographyConfig)
    physical_method = ILTMethod(
        method_name=LEVELSET_ILT_METHOD.method_name,
        config_type=LEVELSET_ILT_METHOD.config_type,
        optimize_macro=partial(
            LEVELSET_ILT_METHOD.optimize_macro, pixel_nm=float(litho.pixel_nm)),
        evaluated_states=LEVELSET_ILT_METHOD.evaluated_states,
        build_fixed_context_canvas=LEVELSET_ILT_METHOD.build_fixed_context_canvas)
    return run_ilt_workflow(physical_method, config_path)
