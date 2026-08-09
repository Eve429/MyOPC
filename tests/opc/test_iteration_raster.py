"""OPC/ILT 共用固定画布栅格化和 core 像素归属测试。"""

import klayout.db as kdb
import numpy as np
import pytest

from layout import DbuBox
from opc.input.raster import ownership_canvas, rasterize_region_canvas


def test_rasterize_aligned_box_uses_bottom_left_coordinates() -> None:
    """整数像素对齐矩形应从画布左下角填充，且不执行图像上下翻转。"""
    region = kdb.Region(kdb.Box(0, 0, 20, 10))
    image = rasterize_region_canvas(region, DbuBox(0, 0, 40, 30), 10, 4)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, :2] = 1.0
    assert np.array_equal(image, expected)


def test_rasterize_hole_preserves_fractional_pixel_coverage() -> None:
    """中空图形应保留孔洞，并以像素面积比例表达非对齐边界。"""
    region = (kdb.Region(kdb.Box(0, 0, 50, 50)) -
              kdb.Region(kdb.Box(15, 15, 35, 35)))
    image = rasterize_region_canvas(region, DbuBox(0, 0, 50, 50), 10, 5)
    assert image[0, 0] == 1.0
    assert image[2, 2] == 0.0
    assert image[1, 1] == pytest.approx(0.75)
    assert image.sum() == pytest.approx(21.0)


def test_rasterize_rejects_context_larger_than_fixed_canvas() -> None:
    """context 像素范围超过光刻画布时必须在分配前给出明确异常。"""
    with pytest.raises(ValueError, match="超过"):
        rasterize_region_canvas(
            kdb.Region(kdb.Box(0, 0, 20, 20)), DbuBox(0, 0, 30, 20), 10, 2)


def test_ownership_canvas_counts_only_pixel_centers_inside_core() -> None:
    """halo 只读，落入半开 core 的像素中心应恰好形成唯一计分区。"""
    owned = ownership_canvas(DbuBox(0, 0, 50, 40), DbuBox(-10, -10, 60, 50), 10, 7)
    assert owned.shape == (7, 7)
    assert int(np.count_nonzero(owned)) == 20
    assert not np.any(owned[0])
    assert np.all(owned[1:5, 1:6])
