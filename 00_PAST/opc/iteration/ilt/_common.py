"""保存当前多种 ILT 求解器共同使用的张量操作。"""

from __future__ import annotations

import torch
from torch.nn import functional


def image_batch(image: torch.Tensor, name: str,
                device: torch.device) -> tuple[torch.Tensor, bool]:
    """把二维或三维图像统一成设备上的 float32 batch，并记录原维度。"""
    if image.ndim == 2:
        return image.unsqueeze(0).to(device=device, dtype=torch.float32), True
    if image.ndim == 3:
        return image.to(device=device, dtype=torch.float32), False
    raise ValueError(f"{name} 必须具有 [H,W] 或 [B,H,W] 形状")


def curvature_loss(mask: torch.Tensor) -> torch.Tensor:
    """用固定三乘三离散曲率核惩罚局部高频和孤立像素。"""
    kernel = mask.new_tensor((
        (-1.0 / 16.0, 5.0 / 16.0, -1.0 / 16.0),
        (5.0 / 16.0, -1.0, 5.0 / 16.0),
        (-1.0 / 16.0, 5.0 / 16.0, -1.0 / 16.0),
    )).reshape(1, 1, 3, 3)
    curvature = functional.conv2d(mask[:, None], kernel)[:, 0]
    return torch.sum(curvature.square())


def resize_image(image: torch.Tensor, shape: tuple[int, int], mode: str) -> torch.Tensor:
    """保持 `[B,H,W]` 契约缩放目标、参数、wafer 或优化窗口。"""
    return functional.interpolate(image[:, None], size=shape, mode=mode)[:, 0]


def smooth_sigmoid_mask(parameters: torch.Tensor, kernel: int,
                        steepness: float, offset: float) -> torch.Tensor:
    """对连续参数执行固定均值平滑和带偏移 sigmoid，生成可微软掩膜。"""
    pooled = functional.avg_pool2d(
        parameters[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    return torch.sigmoid(steepness * (pooled - offset))
