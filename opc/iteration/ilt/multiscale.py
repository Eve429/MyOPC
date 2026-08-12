"""实现复用统一光刻模型的 CurvMulti/Multilevel ILT 调度。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lithography import ICCAD13Lithography, ProcessCondition

from .levelset import LevelSetConfig, optimize_levelset
from .simple import SimpleILTResult


@dataclass(frozen=True, slots=True)
class MultiScaleILTConfig:
    """保存由粗到细阶段尺寸和每阶段水平集配置。"""

    scales: tuple[int, ...] = (1,)
    iterations: int = 10
    step_size: float = 0.2
    weight_process_l2: float = 1.0
    weight_pvband: float = 0.0
    curvature_weight: float = 0.0


def optimize_multiscale(target: torch.Tensor, model: ICCAD13Lithography,
                        config: MultiScaleILTConfig,
                        process_conditions: tuple[ProcessCondition, ...] | None = None
                        ) -> SimpleILTResult:
    """按 scale 从粗到细优化，并把上一阶段参数插值到下一阶段。"""
    if not config.scales or any(scale <= 0 for scale in config.scales):
        raise ValueError("scales 必须包含正整数")
    target_device = target.to(device=model.device, dtype=torch.float32)
    if target_device.ndim not in (2, 3):
        raise ValueError("target 必须是 [H,W] 或 [B,H,W]")
    base_shape = target_device.shape[-2:]
    parameters = None
    result: SimpleILTResult | None = None
    for scale in config.scales:
        shape = (max(1, base_shape[0] // scale), max(1, base_shape[1] // scale))
        stage_target = torch.nn.functional.interpolate(
            target_device[:, None] if target_device.ndim == 3 else target_device[None, None],
            size=shape, mode="area")[:, 0]
        if parameters is not None:
            parameters = torch.nn.functional.interpolate(parameters[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
        result = optimize_levelset(
            stage_target, model,
            LevelSetConfig(config.iterations, config.step_size, config.weight_process_l2,
                           config.weight_pvband, config.curvature_weight),
            initial_levelset=parameters, process_conditions=process_conditions)
        parameters = result.best_parameters if result.best_parameters.ndim == 3 else result.best_parameters[None]
    assert result is not None
    if target_device.ndim == 2 and result.binary_mask.ndim == 3:
        result = SimpleILTResult(
            result.best_parameters[0], result.soft_mask[0], result.binary_mask[0],
            result.best_iteration, result.records)
    return result
