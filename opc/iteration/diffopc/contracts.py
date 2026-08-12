"""定义 DiffOPC 的配置、轮次记录和结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class DiffOPCConfig:
    """保存可微边段 OPC 的损失、步长和 tile 资源参数。"""

    iterations: int = 8
    learning_rate: float = 1.0
    soft_temperature: float = 1.0
    weight_l2: float = 1.0
    weight_pvband: float = 0.0
    weight_epe: float = 1.0
    max_displacement_dbu: float = 24.0
    pixel_dbu: int = 8
    canvas: int = 256
    batch_size: int = 8
    gradient_clip: float = 4.0

    def __post_init__(self) -> None:
        """在进入 GPU 计算前拒绝无效资源和优化参数。"""
        if self.iterations <= 0 or self.pixel_dbu <= 0 or self.canvas <= 0 or self.batch_size <= 0:
            raise ValueError("DiffOPC 迭代、pixel、canvas 和 batch 必须为正整数")
        values = (self.learning_rate, self.soft_temperature, self.weight_l2,
                  self.weight_pvband, self.weight_epe, self.max_displacement_dbu,
                  self.gradient_clip)
        if not all(isfinite(value) for value in values) or self.learning_rate <= 0.0 or self.soft_temperature <= 0.0:
            raise ValueError("DiffOPC 浮点配置必须有限且步长/温度为正")
        if any(value < 0.0 for value in values[2:]):
            raise ValueError("DiffOPC 权重、位移上限和梯度裁剪不能为负")


@dataclass(frozen=True, slots=True)
class DiffOPCIterationRecord:
    """保存一次全局梯度轮次的损失、位移和耗时。"""

    iteration: int
    total_loss: float
    l2: float
    pvband: float
    epe: float
    moved_segments: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DiffOPCResult:
    """保存最佳位移、最佳轮次和全部诊断记录。"""

    best_displacements: np.ndarray
    best_iteration: int
    records: tuple[DiffOPCIterationRecord, ...]
