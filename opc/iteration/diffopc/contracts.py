"""定义 DiffOPC 的配置、轮次记录和结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class DiffOPCConfig:
    """保存可微边段 OPC 的损失、探针、位移和 tile 资源参数。"""

    iterations: int = 8
    learning_rate: float = 1.0
    soft_temperature: float = 4.0
    weight_l2: float = 1.0
    weight_pvband: float = 0.0
    weight_epe: float = 1.0
    max_displacement_dbu: float = 24.0
    epe_distance_dbu: float = 16.0
    pixel_dbu: int = 8
    canvas: int = 256
    batch_size: int = 8
    gradient_clip: float = 4.0
    raster_chunk_size: int = 32
    target_cache_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        """在进入 GPU 计算前拒绝无效资源和优化参数。"""
        integers = (self.iterations, self.pixel_dbu, self.canvas,
                    self.batch_size, self.raster_chunk_size)
        if any(not isinstance(value, int) or value <= 0 for value in integers):
            raise ValueError("DiffOPC 迭代、pixel、canvas、batch 和 raster chunk 必须为正整数")
        if not isinstance(self.target_cache_bytes, int) or self.target_cache_bytes < 0:
            raise ValueError("DiffOPC target 缓存字节数必须是非负整数")
        values = (self.learning_rate, self.soft_temperature, self.weight_l2,
                  self.weight_pvband, self.weight_epe, self.max_displacement_dbu,
                  self.epe_distance_dbu, self.gradient_clip)
        if not all(isfinite(value) for value in values) or self.learning_rate <= 0.0 or self.soft_temperature <= 0.0:
            raise ValueError("DiffOPC 浮点配置必须有限且步长/温度为正")
        if any(value < 0.0 for value in values[2:]):
            raise ValueError("DiffOPC 权重、位移上限和梯度裁剪不能为负")
        if self.epe_distance_dbu <= 0.0:
            raise ValueError("DiffOPC EPE 探针距离必须为正")
        if self.weight_l2 + self.weight_pvband + self.weight_epe <= 0.0:
            raise ValueError("DiffOPC 至少需要一个正损失权重")


@dataclass(frozen=True, slots=True)
class DiffOPCIterationRecord:
    """保存一个被实际评价的全局状态、离散指标和耗时。"""

    iteration: int
    total_loss: float
    l2_loss: float
    pvband_loss: float
    epe_loss: float
    l2: int
    pvband: int
    epe: int
    valid_probes: int
    ambiguous_probes: int
    displaced_segments: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DiffOPCResult:
    """保存最佳位移、最佳轮次、停止原因和全部诊断记录。"""

    best_displacements: np.ndarray
    best_iteration: int
    records: tuple[DiffOPCIterationRecord, ...]
    stop_reason: str
