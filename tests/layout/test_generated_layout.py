"""针对变换与图形类型策略的生成式流文件回归测试。"""

from pathlib import Path

import pytest

from layout import DbuBox, LayerSpec, LayoutDB
from tests.fixtures.layout_factory import write_advanced_layout


@pytest.mark.parametrize("suffix", [".gds", ".oas"])
def test_generated_multilayer_hierarchy_and_shape_policy(suffix: str, tmp_path: Path) -> None:
    """GDS/OASIS 应一致应用全部层级变换并在几何物化时忽略 Text。"""
    path = write_advanced_layout(tmp_path / f"advanced{suffix}")
    mask, auxiliary = LayerSpec(1, 0), LayerSpec(2, 5)
    with LayoutDB.open(path) as db:
        assert db.layers() == (mask, auxiliary)
        box = db.bbox()
        # Layout 层的 bbox 有意反映全部流文件对象，其中也包括 Text。
        assert box == DbuBox(-200, -200, 1000, 2700)
        batch = db.query([mask, auxiliary], box).materialize(diagnostics=True)
        # 8 个 LEAF 实例 × 每实例 3 个可转 Polygon 图形，再加 1 个顶层 Polygon。
        assert dict(batch.counts()) == {mask: 25, auxiliary: 1}
        assert batch.stats is not None
        assert batch.stats.shapes[mask].text == 8
        assert batch.region(mask).bbox().to_s() == "(-100,-50;1000,2650)"


def test_generated_small_roi_selects_only_rotated_instance(tmp_path: Path) -> None:
    """聚焦 ROI 应避开镜像与阵列实例，同时保留 R90 实例几何。"""
    path = write_advanced_layout(tmp_path / "advanced.gds")
    mask = LayerSpec(1, 0)
    with LayoutDB.open(path) as db:
        batch = db.query([mask], DbuBox(700, -10, 1010, 110)).materialize()
        assert batch.counts()[mask] == 3
