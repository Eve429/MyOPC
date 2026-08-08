"""版图单次加载、所有权和层级元数据回归测试。"""

from pathlib import Path

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


def test_open_simple_and_inspect_hierarchy(reticle_dir: Path) -> None:
    """simple 版图应稳定给出 DBU、顶层 Cell、Layer、bbox 和子 Cell 信息。"""
    with LayoutDB.open(reticle_dir / "simple.gds") as db:
        assert db.dbu_um == pytest.approx(0.001)
        assert db.top_cell.name == "TOP"
        assert db.layers() == (LayerSpec(1, 0),)
        assert db.bbox() == DbuBox(-2400, -500, 400, 1500)
        summary = db.hierarchy_summary()
        top = next(info for info in summary.cells if info.ref.name == "TOP")
        assert {child.name for child in top.child_cells} == {"CIRCLE", "TEXT"}
        assert (top.instance_records, top.logical_instances) == (2, 2)


def test_multiple_top_requires_explicit_selection(reticle_dir: Path) -> None:
    """多顶层版图不得依据 Cell 顺序静默选择掩模。"""
    with pytest.raises(AmbiguousTopCellError, match="cell1.*test|test.*cell1"):
        LayoutDB.open(reticle_dir / "test1.gds")
    with LayoutDB.open(reticle_dir / "test1.gds", top_cell="cell1") as db:
        assert db.top_cell.name == "cell1"
        assert db.layers() == (LayerSpec(1, 0), LayerSpec(2, 0), LayerSpec(3, 0))


def test_invalid_file_cell_and_layer_fail_clearly(reticle_dir: Path, tmp_path: Path) -> None:
    """非法调用参数应抛出领域异常，而不是泄漏原生调用栈。"""
    with pytest.raises(LayoutOpenError):
        LayoutDB.open(tmp_path / "missing.gds")
    with pytest.raises(CellNotFoundError):
        LayoutDB.open(reticle_dir / "simple.gds", top_cell="MISSING")
    with LayoutDB.open(reticle_dir / "simple.gds") as db:
        with pytest.raises(LayerNotFoundError):
            db.query([LayerSpec(99, 0)], DbuBox(-10, -10, 10, 10))


def test_closed_database_invalidates_lazy_query(reticle_dir: Path) -> None:
    """查询可以保留元数据，但不得继续使用已经释放的原生数据库。"""
    db = LayoutDB.open(reticle_dir / "simple.gds")
    query = db.query([LayerSpec(1, 0)], DbuBox(-2500, -600, 500, 1600))
    db.close()
    with pytest.raises(ClosedLayoutError):
        query.materialize()
