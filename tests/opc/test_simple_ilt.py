"""SimpleILT 的梯度、独立工艺条件、优化窗口和真实模型集成测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lithography import ICCAD13Lithography, ProcessCondition
from opc.iteration.ilt import SimpleILTConfig, optimize


class _IdentityLithography:
    """用可微仿射曝光替代 FFT，以低成本检查 ILT 优化器自身逻辑。"""

    def __init__(self, canvas: int = 64) -> None:
        """建立 CPU 设备和最小 canvas 配置。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas)
        self.requested: list[tuple[str, ...]] = []

    def condition(self, name: str) -> ProcessCondition:
        """返回默认名称对应的独立伪工艺条件。"""
        kernel = "defocus" if name == "defocus_min" else "focus"
        return ProcessCondition(name, kernel, 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """记录条件组合并返回带微小条件偏置的连续曝光图。"""
        self.requested.append(tuple(condition.name for condition in conditions))
        offsets = {"nominal": 0.0, "high": 0.04, "low": -0.04,
                   "dose_max": 0.04, "defocus_min": -0.04}
        return {condition.name: torch.clamp(
            mask + offsets.get(condition.name, 0.0), 0.0, 1.0)
                for condition in conditions}


def test_simple_ilt_loss_decreases_and_returns_binary_mask() -> None:
    """连续参数优化应降低总损失，并返回同形的软掩膜和二值掩膜。"""
    target = torch.zeros((16, 16))
    target[4:12, 5:11] = 1.0
    initial = torch.zeros_like(target)
    result = optimize(
        target, _IdentityLithography(),
        SimpleILTConfig(8, 0.02, weight_process_l2=0.2),
        initial_parameters=initial)
    losses = [record.total_loss for record in result.records]
    assert losses[-1] < losses[0]
    assert result.soft_mask.shape == target.shape
    assert result.binary_mask.shape == target.shape
    assert result.binary_mask.dtype == torch.bool
    assert 0 <= result.best_iteration < 8


def test_simple_ilt_accepts_independent_conditions_and_fixed_area() -> None:
    """自定义条件不得被默认三工艺角替换，窗口外像素必须保持初始软值。"""
    model = _IdentityLithography()
    target = torch.zeros((12, 12))
    target[3:9, 3:9] = 1.0
    initial = torch.zeros_like(target)
    movable = torch.zeros_like(target)
    movable[2:10, 2:10] = 1.0
    nominal = ProcessCondition("nominal", "focus", 1.0)
    process = (
        ProcessCondition("high", "focus", 1.03),
        ProcessCondition("low", "defocus", 0.97),
    )
    result = optimize(
        target, model, SimpleILTConfig(2, 0.01), initial, movable,
        nominal, process)
    assert all(names == ("nominal", "high", "low") for names in model.requested)
    expected_fixed = torch.full_like(target, 0.5)
    assert torch.equal(result.soft_mask[movable == 0], expected_fixed[movable == 0])


def test_simple_ilt_supports_no_process_window_and_curvature() -> None:
    """调用方可只优化标称条件，曲率正则开启时必须产生有限损失。"""
    target = torch.zeros((1, 10, 10))
    target[:, 3:7, 3:7] = 1.0
    result = optimize(
        target, _IdentityLithography(),
        SimpleILTConfig(2, 0.01, curvature_weight=0.1),
        process_conditions=())
    assert all(record.process_l2 == 0.0 and record.pvband_loss == 0.0
               for record in result.records)
    assert all(torch.isfinite(torch.tensor(record.curvature_loss))
               for record in result.records)


def test_simple_ilt_rejects_misaligned_inputs_and_duplicate_names() -> None:
    """形状不一致、非法窗口和重复条件名称必须在首轮计算前拒绝。"""
    model = _IdentityLithography()
    target = torch.zeros((8, 8))
    config = SimpleILTConfig(1, 0.01)
    with pytest.raises(ValueError, match="形状一致"):
        optimize(target, model, config, initial_parameters=torch.zeros((7, 8)))
    invalid_mask = torch.ones_like(target)
    invalid_mask[0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        optimize(target, model, config, optimization_mask=invalid_mask)
    with pytest.raises(ValueError, match="边长不能小于 3"):
        optimize(torch.zeros((2, 2)), model,
                 SimpleILTConfig(1, 0.01, curvature_weight=0.1))
    duplicate = ProcessCondition("same", "focus", 1.0)
    with pytest.raises(ValueError, match="名称不能重复"):
        optimize(target, model, config, nominal_condition=duplicate,
                 process_conditions=(duplicate,))


def test_real_iccad13_model_completes_backward_and_update() -> None:
    """真实 Hopkins 模型应完成一轮 ILT backward，且参数保持有限。"""
    target = torch.zeros((32, 32))
    target[10:22, 11:21] = 1.0
    result = optimize(
        target, ICCAD13Lithography(device="cpu"),
        SimpleILTConfig(1, 1e-4, weight_process_l2=0.1))
    assert len(result.records) == 1
    assert torch.all(torch.isfinite(result.best_parameters))
    assert torch.all((result.soft_mask >= 0.0) & (result.soft_mask <= 1.0))
