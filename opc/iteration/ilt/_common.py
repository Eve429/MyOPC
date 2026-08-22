"""ILT 方法共享的状态记录、宏结果与连续损失/曲率正则。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class ILTStateRecord:
    """保存一个完整已评价 macro 状态的多指标记录。

    stage/scale 是持久化 metrics 格式的通用坐标：Simple 恒写
    0/state_index/1；后续多尺度方法迁入时共用本记录而无需改格式。
    """

    state_index: int  # 全方法单调状态编号，0 起
    stage_index: int  # 多尺度 stage 坐标；Simple 恒 0
    stage_state_index: int  # stage 内状态序；Simple 等于 state_index
    scale: int  # 控制网格缩放比（像素比）；Simple 恒 1
    total_loss: float  # 全部 core ownership 加权损失之和
    nominal_l2: float  # ownership 连续 nominal L2
    process_l2: float  # ownership 全 process 条件对 target L2
    pvband_loss: float  # ownership process max-min 连续平方差
    curvature_loss: float  # ownership 卷积有效区曲率平方和
    elapsed_seconds: float  # 本 macro state 全部 core 墙钟耗时


@dataclass(frozen=True, slots=True)
class ILTMacroResult:
    """一个 macro 的最佳已评价参数、掩膜与全部状态记录（workflow 只读）。"""

    best_parameters: NDArray[np.float32]  # 唯一 macro best 参数 [Hm,Wm]
    soft_mask: NDArray[np.float32]  # best 对应 transmission [Hm,Wm]
    binary_mask: NDArray[np.bool_]  # 按方法定义二值化的 best 掩膜
    best_state_index: int  # 唯一 macro best state
    records: tuple[ILTStateRecord, ...]  # 恰 N+1 条已评价状态


def owned_continuous_losses(
    nominal: torch.Tensor,
    dose_max: torch.Tensor,
    defocus_min: torch.Tensor,
    target: torch.Tensor,
    ownership: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """三工艺角连续损失，只在 ownership 像素求和（每 core 各自调用）。

    数学与 OpenILT Simple 同源：nominal 对 target、process 全条件对
    target、PV 为 process 逐像素 max−min；context/padding 不计分。
    """
    nominal_l2 = ((nominal - target) ** 2 * ownership).sum()
    process_l2 = (((dose_max - target) ** 2 + (defocus_min - target) ** 2) * ownership).sum()
    pvband_loss = ((dose_max - defocus_min) ** 2 * ownership).sum()
    return nominal_l2, process_l2, pvband_loss


def curvature_loss(mask: torch.Tensor, ownership: torch.Tensor) -> torch.Tensor:
    """固定 3×3 零和核曲率，只在卷积有效区∩ownership 像素求和。

    与 00_PAST 同为无 padding valid 卷积（输出比输入小 2），ownership
    同步裁掉边缘一圈；context ≥ 1 像素时全部 ownership 像素都在有效区。
    """
    kernel = mask.new_tensor(
        (
            (-1.0 / 16.0, 5.0 / 16.0, -1.0 / 16.0),
            (5.0 / 16.0, -1.0, 5.0 / 16.0),
            (-1.0 / 16.0, 5.0 / 16.0, -1.0 / 16.0),
        )
    ).reshape(1, 1, 3, 3)
    curvature = functional.conv2d(mask[:, None], kernel)[:, 0]  # [B,H-2,W-2]
    return (curvature.square() * ownership[:, 1:-1, 1:-1]).sum()


def weighted_macro_loss(
    nominal_l2: torch.Tensor,
    process_l2: torch.Tensor,
    pvband_loss: torch.Tensor,
    curvature_value: torch.Tensor | float,
    *,
    weight_process_l2: float,
    weight_pvband: float,
    curvature_weight: float,
) -> torch.Tensor:
    """OpenILT Simple 同源的加权总损失；nominal 权重恒为 1。

    张量与浮点输入同式：批内用张量（保留 autograd），state 聚合用浮点。
    """
    return (
        nominal_l2 + weight_process_l2 * process_l2 + weight_pvband * pvband_loss + curvature_weight * curvature_value
    )


def resize_image(image: torch.Tensor, shape: tuple[int, int], mode: str) -> torch.Tensor:
    """保持 `[B,H,W]` 契约缩放目标、参数、wafer 或优化窗口。

    area 用于 stage 参考（保覆盖率），nearest 用于跨 stage warm-start 与
    控制网格上采样（不引入新灰度、最近邻路由可微分）；语义迁自 00_PAST。
    """
    return functional.interpolate(image[:, None], size=shape, mode=mode)[:, 0]


def smooth_sigmoid_mask(parameters: torch.Tensor, kernel: int, steepness: float, offset: float) -> torch.Tensor:
    """对连续参数执行固定均值平滑和带偏移 sigmoid，生成可微软掩膜。

    平滑在前（avg_pool 边缘补 k//2 零），σ(β(x−offset)) 在后：控制网格先
    获得空间连续性再生成透光倾向，是 CurvMulti 与逐像素 sigmoid（Simple）
    的参数化差异所在；梯度经 avg_pool/sigmoid 自然回传到控制参数。
    """
    pooled = functional.avg_pool2d(parameters[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    return torch.sigmoid(steepness * (pooled - offset))
