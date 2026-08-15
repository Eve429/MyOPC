"""验证 clear/opaque 统一透光语义、法向和输出几何。"""

import klayout.db as kdb
import numpy as np

from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.input import MaskPolarity
from opc.input.edge import FragmentationConfig, prepare_problem, reconstruct_region
from opc.input.raster import rasterize_mask_canvas


def _batch(region: kdb.Region) -> RegionBatch:
    """把单层 Region 包装为固定 100×100 DBU 测试批次。"""
    layer = LayerSpec(1, 0)
    return RegionBatch({layer: region}, DbuBox(0, 0, 100, 100), CellRef("TOP", 0))


def test_clear_and_opaque_raster_are_exact_complements_inside_field() -> None:
    """同一源多边形的两种极性应在显式处理框内逐像素互补。"""
    region = kdb.Region(kdb.Box(20, 20, 80, 80))
    field = DbuBox(0, 0, 100, 100)
    clear = rasterize_mask_canvas(region, field, 10, 12, polarity="clear")
    opaque = rasterize_mask_canvas(
        region, field, 10, 12, polarity="opaque", field_box=field)
    np.testing.assert_allclose(clear[:10, :10] + opaque[:10, :10], 1.0)
    assert not np.any(clear[10:, :]) and not np.any(opaque[10:, :])
    assert not np.any(clear[:, 10:]) and not np.any(opaque[:, 10:])


def test_opaque_context_outside_field_stays_dark() -> None:
    """opaque 反相不得把 halo 中处理框外的 padding 错置为透光。"""
    image = rasterize_mask_canvas(
        kdb.Region(kdb.Box(20, 20, 80, 80)), DbuBox(-20, -20, 120, 120), 10, 16,
        polarity=MaskPolarity.OPAQUE, field_box=DbuBox(0, 0, 100, 100))
    assert not np.any(image[:2, :]) and not np.any(image[12:, :])
    assert not np.any(image[:, :2]) and not np.any(image[:, 12:])
    assert image[2, 2] == 1.0 and image[4, 4] == 0.0


def test_opaque_normals_reverse_and_positive_move_expands_transmission() -> None:
    """opaque 法向应指向源多边形内部，正位移收缩不透光图形。"""
    batch = _batch(kdb.Region(kdb.Box(20, 20, 80, 80)))
    config = FragmentationConfig(5, 20, 10)
    clear = prepare_problem(batch, batch.layers[0], config, polarity="clear")
    opaque = prepare_problem(batch, batch.layers[0], config, polarity="opaque")
    np.testing.assert_allclose(opaque.segments.edge_normals, -clear.segments.edge_normals)
    moved = reconstruct_region(opaque, np.full(opaque.segments.segment_count, 2.0))
    assert moved.area() < opaque.physical_mask.region.area()
    # 重建始终返回源图形语义；处理框没有进入 contours，也不会多出四条虚假边。
    assert opaque.segments.contours.polygon_count == 1
    assert (reconstruct_region(opaque, np.zeros(opaque.segments.segment_count)) ^
            opaque.physical_mask.region).area() == 0


def test_opaque_raster_requires_explicit_field() -> None:
    """缺少处理框时必须失败，不能猜测无限背景范围。"""
    try:
        rasterize_mask_canvas(
            kdb.Region(), DbuBox(0, 0, 10, 10), 1, 10, polarity="opaque")
    except ValueError as exc:
        assert "field_box" in str(exc)
    else:
        raise AssertionError("opaque 缺少 field_box 时未失败")
