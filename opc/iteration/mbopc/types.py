"""简单 MB-OPC 的迭代配置、轮次记录和最终结果契约。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from geometry import ContourBatch


@dataclass(frozen=True, slots=True)
class SimpleMBOPCConfig:
    """保存已经换算到 DBU/像素的固定迭代参数。"""

    iterations: int
    initial_step_dbu: float
    decay_every: int
    max_displacement_dbu: float
    epe_distance_dbu: float
    pixel_dbu: int
    canvas: int = 256
    batch_size: int = 64
    target_cache_bytes: int = 512 * 1024 * 1024
    print_threshold: float = 0.499

    def __post_init__(self) -> None:
        """拒绝会产生空轮次、无界位移或无效固定画布的配置。"""
        if (self.iterations <= 0 or self.decay_every <= 0 or self.pixel_dbu <= 0 or
                self.canvas <= 0 or self.batch_size <= 0 or self.target_cache_bytes < 0):
            raise ValueError("迭代次数、衰减周期、像素、画布、batch 和缓存必须有效")
        values = (self.initial_step_dbu, self.max_displacement_dbu,
                  self.epe_distance_dbu, self.print_threshold)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("MB-OPC 浮点配置必须有限")
        if (self.initial_step_dbu <= 0.0 or self.max_displacement_dbu < 0.0 or
                self.epe_distance_dbu <= 0.0 or not 0.0 < self.print_threshold < 1.0):
            raise ValueError("MB-OPC 步长、位移、探针距离或阈值无效")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """保存一个已完成光刻评价状态的质量和更新统计。"""

    iteration: int
    step_dbu: float
    epe: int
    l2: float
    pvband: float
    valid_probes: int
    ambiguous_probes: int
    moved_segments: int
    rejected_segments: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SimpleMBOPCResult:
    """保存最佳位移、最佳轮廓、完整轮次记录和停止原因。"""

    best_displacements: NDArray[np.float64]
    best_contours: ContourBatch
    records: tuple[IterationRecord, ...]
    best_iteration: int
    stop_reason: str
