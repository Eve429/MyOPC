"""版图单次加载、所有权和层级元数据回归测试。"""

import subprocess
import sys
from pathlib import Path

import klayout.db as kdb
import pytest

from layout import (
    AmbiguousTopCellError,
    CellNotFoundError,
    ClosedLayoutError,
    DbuBox,
    LayerNotFoundError,
    LayerSpec,
    LayoutDB,
    LayoutOpenError,
)
from tests.fixtures.layout_factory import write_advanced_layout


def _write_hierarchy_layout(path: Path) -> Path:
    """写出覆盖共享子 Cell、重复引用、AREF、多顶层和叶子节点的层级版图。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer = layout.layer(kdb.LayerInfo(1, 0))
    leaf = layout.create_cell("LEAF"); leaf.shapes(layer).insert(kdb.Box(0, 0, 10, 10))
    middle_a = layout.create_cell("MIDDLE_A")
    middle_a.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(0, 0)))
    middle_a.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(20, 0)))
    middle_a.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(0, 20),
                                     kdb.Vector(20, 0), kdb.Vector(0, 20), 100, 100))
    middle_b = layout.create_cell("MIDDLE_B")
    middle_b.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(0, 0)))
    top = layout.create_cell("TOP")
    top.insert(kdb.CellInstArray(middle_b.cell_index(), kdb.Trans(0, 0)))
    top.insert(kdb.CellInstArray(middle_a.cell_index(), kdb.Trans(0, 0)))
    independent = layout.create_cell("INDEPENDENT")
    independent.shapes(layer).insert(kdb.Box(100, 100, 110, 110))
    layout.write(str(path)); return path


def _write_two_tops(path: Path) -> Path:
    """写出互不引用的双顶层版图，替代用户回归数据驱动显式选择语义。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer = layout.layer(kdb.LayerInfo(1, 0))
    for name, offset in (("cell1", 0), ("test", 1000)):
        layout.create_cell(name).shapes(layer).insert(kdb.Box(offset, 0, offset + 100, 100))
    layout.write(str(path)); return path


def test_open_generated_layout_and_inspect_hierarchy(tmp_path: Path) -> None:
    """确定性版图应稳定给出数据库单位、顶层单元、图层、包围盒和子单元信息。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    with LayoutDB.open(source) as db:
        assert db.dbu_um == pytest.approx(0.001)
        assert db.top_cell_name == "TOP"
        assert db.layers() == (LayerSpec(1, 0), LayerSpec(2, 5))
        assert db.bbox() == DbuBox(-200, -200, 1000, 2700)
        assert db.cell_hierarchy() == {"LEAF": (), "TOP": ("LEAF",)}


def test_cell_hierarchy_returns_complete_dag_without_expanding_occurrences(tmp_path: Path) -> None:
    """层级邻接表应覆盖全部 Cell，且重复 SREF/AREF 不得按实例数量展开。"""
    source = _write_hierarchy_layout(tmp_path / "hierarchy.gds")
    with LayoutDB.open(source, top_cell="TOP") as db:
        hierarchy = db.cell_hierarchy()
    # 邻接表描述的是版图 DAG 而不是从已选顶层展开的树：共享 LEAF 只存一份，
    # 未被 TOP 引用的另一顶层仍保留；100×100 AREF 也不产生一万个条目。
    assert type(hierarchy) is dict
    assert hierarchy == {
        "LEAF": (), "MIDDLE_A": ("LEAF",), "MIDDLE_B": ("LEAF",),
        "TOP": ("MIDDLE_A", "MIDDLE_B"), "INDEPENDENT": (),
    }
    assert all(type(children) is tuple for children in hierarchy.values())


def test_cell_hierarchy_requires_open_database(tmp_path: Path) -> None:
    """关闭数据库后层级接口必须立即失败，不得返回可能过期的缓存。"""
    database = LayoutDB.open(write_advanced_layout(tmp_path / "closed.gds"))
    database.close()
    with pytest.raises(ClosedLayoutError):
        database.cell_hierarchy()


def test_multiple_top_requires_explicit_selection(tmp_path: Path) -> None:
    """多顶层版图不得依据 Cell 顺序静默选择掩模。"""
    source = _write_two_tops(tmp_path / "two_tops.gds")
    with pytest.raises(AmbiguousTopCellError, match="cell1.*test|test.*cell1"):
        LayoutDB.open(source)
    with LayoutDB.open(source, top_cell="cell1") as db:
        assert db.top_cell_name == "cell1"
        assert db.layers() == (LayerSpec(1, 0),)


def test_invalid_file_cell_and_layer_fail_clearly(tmp_path: Path) -> None:
    """非法调用参数应抛出领域异常，而不是泄漏原生调用栈。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    with pytest.raises(LayoutOpenError):
        LayoutDB.open(tmp_path / "missing.gds")
    with pytest.raises(CellNotFoundError):
        LayoutDB.open(source, top_cell="MISSING")
    with LayoutDB.open(source) as db, pytest.raises(LayerNotFoundError):
        db.query([LayerSpec(99, 0)], DbuBox(-10, -10, 10, 10))


def test_closed_database_invalidates_lazy_query(tmp_path: Path) -> None:
    """查询可以保留元数据，但不得继续使用已经释放的原生数据库。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    db = LayoutDB.open(source)
    query = db.query([LayerSpec(1, 0)], DbuBox(-200, -200, 1000, 2700))
    db.close()
    with pytest.raises(ClosedLayoutError):
        query.materialize()


def test_materialized_region_batch_survives_database_close(tmp_path: Path) -> None:
    """已物化 RegionBatch 应独立持有几何，关闭数据库后仍可读取和计算。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    db = LayoutDB.open(source)
    layer = LayerSpec(1, 0)
    batch = db.query([layer], DbuBox(-200, -200, 1000, 2700)).materialize()
    expected = (batch.region(layer).count(), batch.region(layer).area())
    db.close()
    # ShapeQuery 依赖数据库的层级迭代器，但 materialize 已把 ROI 结果复制进独立
    # KLayout Region；关闭源版图不得清空它，否则无法安全解耦输入准备和迭代阶段。
    assert (batch.region(layer).count(), batch.region(layer).area()) == expected
    assert expected[0] > 0 and expected[1] > 0


def test_recursive_polygon_scan_is_public_and_lifetime_bounded(tmp_path: Path) -> None:
    """公共容量扫描迭代器应只依赖打开的数据库，并能遍历 Polygon 类图形。"""
    source = write_advanced_layout(tmp_path / "scan.gds")
    database = LayoutDB.open(source)
    layer = LayerSpec(1, 0); box = database.bbox()
    assert box is not None
    assert sum(1 for _ in database.recursive_polygon_shapes(layer, box)) > 0
    database.close()
    with pytest.raises(ClosedLayoutError):
        database.recursive_polygon_shapes(layer, box)


def _write_layered_layout(path: Path) -> Path:
    """写出多层版图：各层图形位置错开，层 3 只出现在选定子树外的另一顶层。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    one = layout.layer(kdb.LayerInfo(1, 0)); two = layout.layer(kdb.LayerInfo(2, 5))
    leaf = layout.create_cell("LEAF"); leaf.shapes(one).insert(kdb.Box(10, 10, 30, 40))
    top = layout.create_cell("TOP")
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(0, 0)))
    top.shapes(two).insert(kdb.Box(-50, -20, 500, 80))
    other = layout.create_cell("OTHER")
    other.shapes(layout.layer(kdb.LayerInfo(3, 0))).insert(kdb.Box(0, 0, 5, 5))
    layout.write(str(path)); return path


def test_layer_bbox_filters_target_layer_and_subtree(tmp_path: Path) -> None:
    """layer_bbox 按层过滤并展开实例；层不在当前子树时返回 None。"""
    source = _write_layered_layout(tmp_path / "layered.gds")
    with LayoutDB.open(source, top_cell="TOP") as db:
        assert db.bbox() == DbuBox(-50, -20, 500, 80)
        assert db.layer_bbox(LayerSpec(1, 0)) == DbuBox(10, 10, 30, 40)
        assert db.layer_bbox(LayerSpec(2, 5)) == DbuBox(-50, -20, 500, 80)
        assert db.layer_bbox(LayerSpec(3, 0)) is None
    with LayoutDB.open(source, top_cell="TOP") as db, pytest.raises(LayerNotFoundError):
        db.layer_bbox(LayerSpec(99, 0))


def test_importing_layout_does_not_load_geometry() -> None:
    """基础版图层不得因公共导入而反向加载几何输出层。"""
    command = "import sys, layout; assert not any(n == 'geometry' or n.startswith('geometry.') for n in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", command], check=False, capture_output=True,
                               text=True)
    assert completed.returncode == 0, completed.stderr
