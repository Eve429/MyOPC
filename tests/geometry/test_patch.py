"""所有权与跨核心区域的几何补丁测试。"""

from pathlib import Path

import klayout.db as kdb
import pytest

from geometry import GeometryPatch, PatchConflictError, PatchSet, PatchWriter
from layout import DbuBox, LayerSpec, LayoutDB


def test_one_polygon_crossing_adjacent_cores_is_split_without_loss() -> None:
    """跨越 x=50 的图形应由两个相邻 core 各自唯一拥有对应部分。"""
    layer = LayerSpec(1, 0)
    crossing = kdb.Region(kdb.Box(25, 20, 75, 80))
    patches = PatchSet()
    left = patches.add(GeometryPatch("left", layer, crossing, DbuBox(0, 0, 50, 100)))
    right = patches.add(GeometryPatch("right", layer, crossing, DbuBox(50, 0, 100, 100)))
    assert left.region.area() == right.region.area() == 1_500
    assert (left.region & right.region).area() == 0
    assert (patches.region(layer) ^ crossing).area() == 0


def test_patch_conflicts_are_layer_aware_and_touching_is_allowed() -> None:
    """仅拒绝同一 Layer 上存在正面积的 ownership 重叠。"""
    one, two = LayerSpec(1, 0), LayerSpec(2, 0)
    region = kdb.Region(kdb.Box(0, 0, 100, 100))
    patches = PatchSet()
    patches.add(GeometryPatch("a", one, region, DbuBox(0, 0, 50, 50)))
    patches.add(GeometryPatch("b", one, region, DbuBox(50, 0, 100, 50)))
    patches.add(GeometryPatch("c", two, region, DbuBox(0, 0, 50, 50)))
    with pytest.raises(PatchConflictError, match="ownership overlap"):
        patches.add(GeometryPatch("d", one, region, DbuBox(25, 25, 75, 75)))
    with pytest.raises(PatchConflictError, match="duplicate patch_id"):
        patches.add(GeometryPatch("a", two, region, DbuBox(50, 50, 100, 100)))


@pytest.mark.parametrize("suffix", [".gds", ".oas"])
def test_patch_writer_round_trip(suffix: str, tmp_path: Path) -> None:
    """补丁输出重新读取后必须与所有权范围内的几何完全相同。"""
    layer = LayerSpec(17, 3)
    expected = kdb.Region(kdb.Box(-25, -10, 75, 40))
    patches = PatchSet()
    patches.add(GeometryPatch("p0", layer, expected, DbuBox(-100, -100, 100, 100)))
    output = PatchWriter.write(patches, tmp_path / f"patch{suffix}", 0.001)
    with LayoutDB.open(output) as db:
        box = db.bbox()
        assert box is not None
        actual = db.query([layer], box).materialize().region(layer)
        assert (actual ^ expected).area() == 0


def test_writer_rejects_unknown_format(tmp_path: Path) -> None:
    """输出格式必须明确，不允许错误猜测未知扩展名。"""
    with pytest.raises(ValueError):
        PatchWriter.write(PatchSet(), tmp_path / "patch.txt", 0.001)
