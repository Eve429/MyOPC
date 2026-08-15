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


def _macro_patches():
    """构造跨 x=50 宏边界、已按 ownership 裁剪的两个权威 patch。"""
    layer = LayerSpec(1, 0)
    crossing = kdb.Region(kdb.Box(25, 20, 75, 80))
    left_box, right_box = DbuBox(0, 0, 50, 100), DbuBox(50, 0, 100, 100)
    left = GeometryPatch("mr0c0", layer, crossing & kdb.Region(left_box.to_native()),
                         left_box)
    right = GeometryPatch("mr0c1", layer, crossing & kdb.Region(right_box.to_native()),
                          right_box)
    return layer, crossing, [left, right]


def _reload_region(path: Path, layer: LayerSpec) -> kdb.Region:
    """回读流文件并物化目标层全部图形。"""
    with LayoutDB.open(path) as db:
        box = db.bbox()
        assert box is not None
        return db.query([layer], box).materialize().region(layer)


def test_macro_results_single_cell_merges_cross_macro_polygon(tmp_path: Path) -> None:
    """single_cell 把跨界 polygon 全局 merge 成一个 Cell，不保留 seam。"""
    layer, crossing, patches = _macro_patches()
    output = PatchWriter.write_macro_results(
        patches, tmp_path / "final.gds", 0.001, cell_mode="single_cell")
    region = _reload_region(output, layer)
    assert (region ^ crossing).area() == 0
    assert region.count() == 1  # 两个半块被 merge 回同一个 polygon
    with LayoutDB.open(output) as db:
        assert all(not children for children in db.cell_hierarchy().values())


def test_macro_results_macro_cells_keeps_one_cell_per_macro(tmp_path: Path) -> None:
    """macro_cells 每个 macro 一个子 Cell，物理覆盖相同但表示 seam 保留。"""
    layer, crossing, patches = _macro_patches()
    output = PatchWriter.write_macro_results(
        patches, tmp_path / "final.gds", 0.001, cell_mode="macro_cells")
    region = _reload_region(output, layer)
    assert (region ^ crossing).area() == 0
    assert region.count() == 2  # 跨界 polygon 仍是两个半块
    with LayoutDB.open(output) as db:
        children = db.cell_hierarchy()["OPC_RESULT"]
    assert set(children) == {"mr0c0", "mr0c1"}


def test_macro_results_rejects_unknown_cell_mode(tmp_path: Path) -> None:
    """未知 cell_mode 直接失败。"""
    _, _, patches = _macro_patches()
    with pytest.raises(ValueError, match="cell_mode"):
        PatchWriter.write_macro_results(
            patches, tmp_path / "final.gds", 0.001, cell_mode="tiled")
