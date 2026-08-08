"""基于面积关系验证批量 Region 运算。"""

import klayout.db as kdb
import pytest

from geometry import CoordinateSystemError, GeometryEngine
from layout import CellRef, DbuBox, LayerSpec

from .helpers import region_batch


def test_boolean_combine_clip_offset_and_merge() -> None:
    """批量运算应得到预期集合面积，同时不修改输入。"""
    layer = LayerSpec(1, 0)
    left = region_batch({layer: kdb.Region(kdb.Box(0, 0, 100, 100))})
    right = region_batch({layer: kdb.Region(kdb.Box(50, 0, 150, 100))})
    engine = GeometryEngine()
    combined = engine.combine(left, right).region(layer)
    assert (combined.count(), combined.is_merged(), combined.area()) == (2, False, 15_000)
    assert engine.union(left, right).region(layer).area() == 15_000
    assert engine.intersection(left, right).region(layer).area() == 5_000
    assert engine.difference(left, right).region(layer).area() == 5_000
    assert engine.xor(left, right).region(layer).area() == 10_000
    assert engine.clip(left, DbuBox(25, 25, 75, 75)).region(layer).area() == 2_500
    assert engine.offset(left, 10).region(layer).area() == 14_400
    assert engine.merge(engine.combine(left, right)).region(layer).count() == 1
    assert left.region(layer).area() == 10_000


def test_binary_layer_semantics_and_coordinate_guard() -> None:
    """缺失 Layer 按明确集合语义处理，不同 Cell 坐标系不得混用。"""
    one, two = LayerSpec(1, 0), LayerSpec(2, 0)
    left = region_batch({one: kdb.Region(kdb.Box(0, 0, 10, 10))})
    right = region_batch({two: kdb.Region(kdb.Box(0, 0, 20, 20))})
    engine = GeometryEngine()
    assert engine.union(left, right).layers == (one, two)
    assert engine.intersection(left, right).layers == ()
    assert engine.difference(left, right).region(one).area() == 100
    foreign = region_batch({one: kdb.Region()}, cell=CellRef("OTHER", 1))
    with pytest.raises(CoordinateSystemError):
        engine.union(left, foreign)


def test_offset_requires_integer_dbu() -> None:
    """核心几何接口不接受隐式浮点微米距离。"""
    layer = LayerSpec(1, 0)
    batch = region_batch({layer: kdb.Region()})
    with pytest.raises(TypeError):
        GeometryEngine().offset(batch, 1.5)
