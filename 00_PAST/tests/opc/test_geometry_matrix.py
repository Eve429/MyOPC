"""MB-OPC 多图形不变性、随机组合和图集回归测试。"""

from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest
from PIL import Image

from layout import CellRef, DbuBox, LayerSpec, LayoutDB, RegionBatch
from opc.diagnostics import build_geometry_cases, run_geometry_suite
from opc.input.edge import (
    FragmentationConfig,
    prepare_problem,
    reconstruct_region,
)
from tests.fixtures.layout_factory import write_advanced_layout


def _batch(region: kdb.Region, layer: LayerSpec) -> RegionBatch:
    """把任意随机 Region 放入足够大的单层处理窗口。"""
    return RegionBatch({layer: region}, DbuBox(-600, -600, 600, 600),
                       CellRef("GEOMETRY_MATRIX", 0))


@pytest.mark.parametrize("name", tuple(build_geometry_cases()))
def test_deterministic_geometry_cases_are_exact_and_bounded(name: str) -> None:
    """每种确定性图形的零位移必须精确回建，且分段不超上限。"""
    layer = LayerSpec(1, 0)
    problem = prepare_problem(_batch(build_geometry_cases()[name], layer), layer,
                              FragmentationConfig(8, 25, 12))
    zero = np.zeros(problem.segments.segment_count)
    rebuilt = reconstruct_region(problem, zero)
    assert (rebuilt ^ problem.physical_mask.region).area() == 0
    geometry = problem.segments.materialize()
    assert np.linalg.norm(geometry.ends - geometry.starts, axis=1).max() <= 25 + 1e-12
    np.testing.assert_allclose(np.linalg.norm(problem.segments.edge_normals, axis=1), 1.0)


def test_seeded_manhattan_unions_keep_exact_physical_boundary() -> None:
    """固定种子的矩形并集覆盖重叠、凹口、切线和多连通分量。"""
    layer, generator = LayerSpec(1, 0), np.random.default_rng(20260809)
    config = FragmentationConfig(7, 23, 10)
    for _ in range(100):
        region = kdb.Region()
        for _ in range(int(generator.integers(2, 9))):
            left, bottom = generator.integers(-300, 250, size=2)
            width, height = generator.integers(8, 140, size=2)
            region.insert(kdb.Box(int(left), int(bottom), int(left + width),
                                  int(bottom + height)))
        problem = prepare_problem(_batch(region, layer), layer, config)
        rebuilt = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
        assert (rebuilt ^ problem.physical_mask.region).area() == 0


def test_geometry_suite_writes_readable_annotated_images(tmp_path: Path) -> None:
    """主程序使用的图集入口必须输出每个用例的可读 PNG。"""
    summary = run_geometry_suite(tmp_path)
    assert summary["case_count"] == 5
    assert summary["all_zero_displacement_exact"]
    for case in summary["cases"]:
        with Image.open(tmp_path / case["image"]) as image:
            assert image.format == "PNG"
            assert max(image.size) >= 800


def test_hierarchy_path_array_rotation_and_mirror_reach_mbopc(tmp_path: Path) -> None:
    """Path、SREF、AREF、R90 和镜像物化后必须可精确分段及回建。"""
    source, layer = write_advanced_layout(tmp_path / "hierarchy.gds"), LayerSpec(1, 0)
    with LayoutDB.open(source) as database:
        box = database.bbox()
        assert box is not None
        batch = database.query([layer], box).materialize()
        # Region 仍由打开的 KLayout 数据库持有时完成规范化；问题对象
        # 随后只依赖自己的 Region 副本和 NumPy 紧凑数组，可脱离源文件复用。
        problem = prepare_problem(batch, layer, FragmentationConfig(5, 20, 8))
    rebuilt = reconstruct_region(problem, np.zeros(problem.segments.segment_count))
    assert problem.segments.contours.polygon_count > 1
    assert (rebuilt ^ problem.physical_mask.region).area() == 0
