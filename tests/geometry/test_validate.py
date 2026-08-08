"""退化数值轮廓数据的校验测试。"""

import numpy as np

from geometry import ContourBatch, validate_contours
from layout import LayerSpec


def test_valid_rectangle_has_no_issues() -> None:
    """简单四边外轮廓应通过轻量迭代前校验。"""
    contours = ContourBatch(
        LayerSpec(1, 0), np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        np.array([0, 4]), np.array([0]), np.array([False]))
    assert validate_contours(contours).is_valid


def test_zero_length_edge_and_zero_area_are_reported() -> None:
    """退化环应给出稳定的问题代码与索引，且不得隐式修复。"""
    contours = ContourBatch(
        LayerSpec(1, 0), np.array([[0, 0], [10, 0], [10, 0]]),
        np.array([0, 3]), np.array([0]), np.array([False]))
    report = validate_contours(contours)
    assert {issue.code for issue in report.issues} == {"zero_length_edge", "zero_area_ring"}
