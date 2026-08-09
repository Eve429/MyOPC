"""L2、PVBand 和边段 EPE 的向量化行为测试。"""

import pytest
import torch

from evaluation import evaluate_edge_probes, evaluate_process_window


def test_process_metrics_only_accumulate_owned_pixels() -> None:
    """halo 像素必须排除在全局 L2/PVBand 之外，避免跨 tile 重复累计。"""
    target = torch.zeros((1, 3, 4))
    nominal = target.clone()
    maximum = target.clone()
    minimum = target.clone()
    nominal[0, 1, 1] = 0.5
    nominal[0, 1, 3] = 1.0
    maximum[0, 0, 1] = 0.75
    minimum[0, 0, 1] = 0.25
    ownership = torch.zeros_like(target, dtype=torch.bool)
    ownership[:, :, :2] = True
    metrics = evaluate_process_window(target, nominal, maximum, minimum, ownership)
    assert metrics.pixel_count == 6
    assert metrics.l2 == pytest.approx(0.25)
    assert metrics.pvband == pytest.approx(0.25)


def test_probe_evaluation_distinguishes_inner_outer_and_ambiguity() -> None:
    """有效探针应分别生成外移、内移和冲突不移动三种结果。"""
    target = torch.zeros((1, 8, 8))
    target[:, 2:6, 2:6] = 1.0
    nominal = target.clone()
    nominal[0, 3, 3] = 0.0
    nominal[0, 4, 6] = 1.0
    nominal[0, 5, 3] = 0.0
    nominal[0, 5, 6] = 1.0
    inner = torch.tensor([[3.0, 3.0], [4.0, 4.0], [3.0, 5.0],
                          [1.0, 1.0], [3.0, 3.0]])
    outer = torch.tensor([[1.0, 3.0], [6.0, 4.0], [6.0, 5.0],
                          [0.0, 1.0], [3.0, 3.0]])
    result = evaluate_edge_probes(
        target, nominal, torch.zeros(5, dtype=torch.long), inner, outer)
    assert result.valid.tolist() == [True, True, True, False, False]
    assert result.inner_violations.tolist() == [True, False, True, False, False]
    assert result.outer_violations.tolist() == [False, True, True, False, False]
    assert result.ambiguous.tolist() == [False, False, True, False, False]
    assert result.directions.tolist() == [1, -1, 0, 0, 0]
    assert result.violation_count == 3


def test_out_of_bounds_and_narrow_feature_probes_are_invalid() -> None:
    """越界或 inner 未落在目标材料内的窄图形探针不得产生移动。"""
    target = torch.zeros((1, 8, 8))
    target[:, :, 3:5] = 1.0
    nominal = target.clone()
    inner = torch.tensor([[1.0, 4.0], [-1.0, 4.0], [4.0, 4.0]])
    outer = torch.tensor([[0.0, 4.0], [7.0, 4.0], [9.0, 4.0]])
    result = evaluate_edge_probes(
        target, nominal, torch.zeros(3, dtype=torch.long), inner, outer)
    assert not torch.any(result.valid)
    assert result.directions.tolist() == [0, 0, 0]


def test_metrics_reject_misaligned_inputs() -> None:
    """形状、设备或 ownership 不对齐必须明确拒绝，不能依赖隐式广播。"""
    image = torch.zeros((4, 4))
    with pytest.raises(ValueError, match="形状和设备"):
        evaluate_process_window(image, torch.zeros((3, 4)), image, image)
    with pytest.raises(ValueError, match="ownership_mask"):
        evaluate_process_window(image, image, image, image, torch.ones((3, 4)))
    with pytest.raises(ValueError, match="按边段对齐"):
        evaluate_edge_probes(image, image, torch.zeros(2), torch.zeros((1, 2)),
                             torch.zeros((1, 2)))
