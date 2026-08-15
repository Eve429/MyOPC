"""原生层级 ROI 物化集成测试。"""

from pathlib import Path

import klayout.db as kdb

from layout import DbuBox, LayerSpec, LayoutDB
from tests.fixtures.layout_factory import write_advanced_layout


def test_generated_materialization_ignores_text_and_reports_it_on_demand(tmp_path: Path) -> None:
    """可转 Polygon 的图形保留在原生端，可选诊断能够识别 Text。"""
    layer = LayerSpec(1, 0)
    source = write_advanced_layout(tmp_path / "advanced.gds")
    with LayoutDB.open(source) as db:
        box = db.bbox()
        assert box is not None
        batch = db.query([layer], box).materialize(diagnostics=True)
        assert batch.counts()[layer] == 25
        assert batch.stats is not None
        assert batch.stats.shapes[layer].polygon_like == 25
        assert batch.stats.shapes[layer].text == 8


def test_existing_fixture_counts_and_top_transforms(reticle_dir: Path) -> None:
    """递归物化会把 SREF/AREF 变换统一应用到 cell1 坐标系。"""
    expected = {LayerSpec(1, 0): 13, LayerSpec(2, 0): 22, LayerSpec(3, 0): 1}
    with LayoutDB.open(reticle_dir / "test1.gds", top_cell="cell1") as db:
        box = db.bbox()
        assert box is not None
        batch = db.query(list(expected), box).materialize()
        assert batch.stats is None
        assert dict(batch.counts()) == expected
        assert str(batch.region(LayerSpec(1, 0)).bbox()) == "(-2300,-2800;7350,3600)"


def test_roi_query_does_not_return_distant_shapes(reticle_dir: Path) -> None:
    """小 ROI 使用 KLayout 感知层级的空间限制，不返回远处图形。"""
    layer = LayerSpec(1, 0)
    with LayoutDB.open(reticle_dir / "JustPoly.gds") as db:
        left = db.query([layer], DbuBox(-1350, -50, -950, 350)).materialize()
        right = db.query([layer], DbuBox(-950, -50, -550, 350)).materialize()
        assert left.counts()[layer] == 1
        assert right.counts()[layer] == 1


def test_roi_materialization_clips_cross_boundary_polygon(tmp_path: Path) -> None:
    """物化结果必须精确裁到 ROI，而不只是筛出与 ROI 相交的完整图形。"""
    source = tmp_path / "cross_boundary.gds"
    native = kdb.Layout()
    index = native.layer(kdb.LayerInfo(1, 0))
    top = native.create_cell("TOP")
    top.shapes(index).insert(kdb.Box(0, 0, 100, 100))
    native.write(str(source))
    layer, box = LayerSpec(1, 0), DbuBox(25, 20, 75, 80)
    with LayoutDB.open(source) as database:
        region = database.query([layer], box).materialize().region(layer)
    # RecursiveShapeIterator 只负责层级候选筛选；公共物化边界必须再做一次原生
    # Region 相交，使 MB-OPC、像素图和直接 CLI 无需各自补一套 ROI 裁剪规则。
    assert region.bbox() == box.to_native()
    assert region.area() == box.area


def test_preserve_properties_keeps_plain_and_tagged_geometry(tmp_path: Path) -> None:
    """启用属性导入不能过滤普通图形，且必须保留带属性图形的键值。"""
    source = tmp_path / "properties.gds"
    native = kdb.Layout()
    index = native.layer(kdb.LayerInfo(7, 0))
    top = native.create_cell("TOP")
    top.shapes(index).insert(kdb.Box(0, 0, 10, 10))
    tagged = top.shapes(index).insert(kdb.Box(20, 0, 30, 10))
    tagged.set_property(7, "tagged")
    native.write(str(source))
    # ROI 同时截断普通图形和带属性图形，确保精确裁剪不会改变集合或丢掉属性。
    layer, box = LayerSpec(7, 0), DbuBox(5, -1, 25, 11)
    with LayoutDB.open(source) as database:
        plain = database.query([layer], box).materialize().region(layer)
        preserved_batch = database.query(
            [layer], box, preserve_properties=True).materialize(diagnostics=True)
        preserved = preserved_batch.region(layer)
        assert plain.count() == preserved.count() == 2
        assert all(not polygon.properties() for polygon in plain.each())
        properties = {polygon.bbox().to_s(): polygon.properties() for polygon in preserved.each()}
        assert properties == {"(5,0;10,10)": {}, "(20,0;25,10)": {7: "tagged"}}
        assert preserved_batch.stats is not None
        assert preserved_batch.stats.shapes[layer].polygon_like == 2


def test_intersecting_materialization_keeps_full_crossing_shapes(tmp_path: Path) -> None:
    """未裁剪物化应保留完整相交图形，而普通物化继续精确服从 ROI。"""
    source = tmp_path / "crossing.gds"
    native = kdb.Layout(); native.dbu = 0.001
    index = native.layer(kdb.LayerInfo(1, 0)); top = native.create_cell("TOP")
    top.shapes(index).insert(kdb.Box(-50, -20, 150, 80))
    top.shapes(index).insert(kdb.Box(300, 300, 400, 400))
    native.write(str(source))
    layer, box = LayerSpec(1, 0), DbuBox(0, 0, 100, 50)
    with LayoutDB.open(source) as database:
        query = database.query([layer], box)
        clipped = query.materialize().region(layer)
        complete = query.materialize_intersecting().region(layer)
    assert clipped.bbox() == kdb.Box(0, 0, 100, 50)
    assert complete.bbox() == kdb.Box(-50, -20, 150, 80)
    assert complete.count() == 1


def test_intersecting_materialization_preserves_properties(tmp_path: Path) -> None:
    """未裁剪物化启用属性时应保留完整几何及其属性。"""
    source = tmp_path / "crossing-properties.gds"
    native = kdb.Layout(); index = native.layer(kdb.LayerInfo(7, 0))
    top = native.create_cell("TOP")
    tagged = top.shapes(index).insert(kdb.Box(-20, 0, 80, 40))
    tagged.set_property(7, "macro")
    native.write(str(source))
    layer, box = LayerSpec(7, 0), DbuBox(0, 10, 40, 30)
    with LayoutDB.open(source) as database:
        complete = database.query(
            [layer], box, preserve_properties=True).materialize_intersecting().region(layer)
    polygon = next(complete.each())
    assert polygon.bbox() == kdb.Box(-20, 0, 80, 40)
    assert polygon.properties() == {7: "macro"}
