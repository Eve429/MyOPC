"""定义简单 MB-OPC 迭代器的配置、轮次记录和最终结果契约。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SimpleMBOPCConfig:
    """保存已经换算到 DBU/像素的固定迭代参数。"""

    iterations: int
    initial_step_dbu: float
    decay_every: int
    epe_distance_dbu: float
    pixel_dbu: int
    canvas: int = 256
    batch_size: int = 64
    target_cache_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        """拒绝会产生空轮次、无界位移或无效固定画布的配置。"""
        if (self.iterations <= 0 or self.decay_every <= 0 or self.pixel_dbu <= 0 or
                self.canvas <= 0 or self.batch_size <= 0 or self.target_cache_bytes < 0):
            raise ValueError("迭代次数、衰减周期、像素、画布、batch 和缓存必须有效")
        values = (self.initial_step_dbu, self.epe_distance_dbu)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("MB-OPC 浮点配置必须有限")
        if self.initial_step_dbu <= 0.0 or self.epe_distance_dbu <= 0.0:
            raise ValueError("MB-OPC 步长或探针距离无效")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """保存一个已评价状态的质量，以及由该状态提出的更新统计。"""

    iteration: int
    step_dbu: float
    epe: int
    l2: int
    pvband: int
    valid_probes: int
    ambiguous_probes: int
    moved_segments: int
    rejected_segments: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SimpleMBOPCResult:
    """保存最佳已评价位移、状态记录、最佳状态下标和停止原因。"""

    best_displacements: NDArray[np.float64]
    records: tuple[IterationRecord, ...]
    best_iteration: int
    stop_reason: str
