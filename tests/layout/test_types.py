"""整数 DBU 与 Layer 数据契约单元测试。"""

import numpy as np
import pytest

from layout import DbuBox, LayerSpec


def test_dbu_box_geometry_and_numpy_integer_normalization() -> None:
    """数据库单位矩形保持精确整数，并将仅接触视为零面积重叠。"""
    box = DbuBox(np.int64(-10), 0, 10, 20)
    assert (box.width, box.height, box.area) == (20, 20, 400)
    assert box.expanded(5) == DbuBox(-15, -5, 15, 25)
    assert box.intersection(DbuBox(0, 10, 30, 40)) == DbuBox(0, 10, 10, 20)
    assert box.intersection(DbuBox(10, 0, 20, 20)) is None


@pytest.mark.parametrize("coords", [(0, 0, 0, 1), (0, 1, 1, 1), (2, 0, 1, 1)])
def test_dbu_box_rejects_non_positive_area(coords: tuple[int, int, int, int]) -> None:
    """空窗口和反向窗口必须在原生查询前失败。"""
    with pytest.raises(ValueError):
        DbuBox(*coords)


def test_layer_spec_is_orderable_and_validated() -> None:
    """图层描述可稳定排序，并拒绝负数标识。"""
    assert sorted([LayerSpec(2, 0), LayerSpec(1, 5)]) == [LayerSpec(1, 5), LayerSpec(2, 0)]
    with pytest.raises(ValueError):
        LayerSpec(-1, 0)
