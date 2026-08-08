"""Integration tests for native hierarchical ROI materialization."""

from pathlib import Path

from layout import DbuBox, LayerSpec, LayoutDB


def test_simple_materialization_ignores_text_and_reports_it_on_demand(reticle_dir: Path) -> None:
    """Polygon-like shapes stay native while optional diagnostics classify text."""
    layer = LayerSpec(1, 0)
    with LayoutDB.open(reticle_dir / "simple.gds") as db:
        box = db.bbox()
        assert box is not None
        batch = db.query([layer], box).materialize(diagnostics=True)
        assert batch.counts()[layer] == 10
        assert batch.stats is not None
        assert batch.stats.shapes[layer].polygon_like == 10
        assert batch.stats.shapes[layer].text == 1


def test_existing_fixture_counts_and_top_transforms(reticle_dir: Path) -> None:
    """Recursive materialization applies SREF/AREF transforms into cell1 coordinates."""
    expected = {LayerSpec(1, 0): 13, LayerSpec(2, 0): 22, LayerSpec(3, 0): 1}
    with LayoutDB.open(reticle_dir / "test1.gds", top_cell="cell1") as db:
        box = db.bbox()
        assert box is not None
        batch = db.query(list(expected), box).materialize()
        assert dict(batch.counts()) == expected
        assert str(batch.region(LayerSpec(1, 0)).bbox()) == "(-2300,-2800;7350,3600)"


def test_roi_query_does_not_return_distant_shapes(reticle_dir: Path) -> None:
    """A small ROI uses KLayout's hierarchy-aware spatial restriction."""
    layer = LayerSpec(1, 0)
    with LayoutDB.open(reticle_dir / "JustPoly.gds") as db:
        left = db.query([layer], DbuBox(-1350, -50, -950, 350)).materialize()
        right = db.query([layer], DbuBox(-950, -50, -550, 350)).materialize()
        assert left.counts()[layer] == 1
        assert right.counts()[layer] == 1
