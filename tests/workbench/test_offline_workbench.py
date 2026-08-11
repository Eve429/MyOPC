"""验证离线像素/边段归档、安全限制以及两个独立运行入口。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest
from PIL import Image

from layout import DbuBox, LayerSpec, LayoutDB
from layout.query import ShapeQuery
from opc.input.edge import reconstruct_region
from opc.input.raster import rasterize_region_canvas
from tests.workbench.offline_inputs import (
    load_raster_input,
    load_segment_input,
    prepare_raster_input,
    prepare_segment_input,
)
from tests.workbench.run_lithography import run_lithography_test
from tests.workbench.run_mbopc_iteration import run_mbopc_iteration_test

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_workbench_layout(path: Path) -> Path:
    """生成含孔洞、斜边、层级引用和跨 core 长边的紧凑版图。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    layer = layout.layer(kdb.LayerInfo(1, 0))
    leaf = layout.create_cell("LEAF")
    leaf.shapes(layer).insert(kdb.Box(0, 0, 28, 24))
    top = layout.create_cell("TOP")
    # 长矩形和斜多边形都跨过 x=128 DBU 的 core 边界；孔洞壁宽 20 DBU，
    # 同时覆盖外/内环法向及后续重建关系。
    top.shapes(layer).insert(kdb.Box(16, 20, 240, 52))
    top.shapes(layer).insert(kdb.Polygon([
        kdb.Point(32, 72), kdb.Point(152, 60), kdb.Point(232, 108),
        kdb.Point(112, 116),
    ]))
    donut = kdb.Region(kdb.Box(264, 16, 376, 128)) - kdb.Region(
        kdb.Box(284, 36, 356, 108))
    top.shapes(layer).insert(donut)
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(180, 152)))
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(kdb.Trans.R90, 244, 180)))
    layout.write(str(path))
    return path


def _rewrite_npz(source: Path, target: Path, name: str,
                 value: np.ndarray | None) -> Path:
    """复制一个测试归档并替换或删除指定字段，用于验证损坏输入。"""
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files if key != name}
    if value is not None:
        arrays[name] = value
    with target.open("wb") as stream:
        np.savez(stream, **arrays)
    return target


def test_raster_round_trip_matches_current_rasterizer(tmp_path: Path) -> None:
    """像素归档加载值必须与当前公共栅格函数逐像素完全一致。"""
    source = _write_workbench_layout(tmp_path / "input.gds")
    box, layer = DbuBox(0, 0, 512, 256), LayerSpec(1, 0)
    archive = prepare_raster_input(
        source, tmp_path / "raster.npz", layer=layer, box=box,
        pixel_nm=4.0, canvas=256)
    mask, metadata = load_raster_input(archive)
    with LayoutDB.open(source) as database:
        batch = database.query([layer], box).materialize()
        expected = rasterize_region_canvas(batch.region(layer), box, 4, 256)
    assert mask.dtype == np.float32
    assert np.array_equal(mask, expected)
    assert metadata["orientation"] == "bottom_left"
    assert metadata["active_width"] == 128
    assert metadata["active_height"] == 64


def test_raster_size_guard_runs_before_public_materialization(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """超出单 canvas 的 ROI 必须在进入 LayoutDB 物化路径前失败。"""
    source = _write_workbench_layout(tmp_path / "large.gds")

    def forbidden_materialize(*args: object, **kwargs: object) -> None:
        """如果像素尺寸拒绝后仍物化完整 Region，就让测试明确失败。"""
        raise AssertionError("超限 ROI 不应进入 ShapeQuery.materialize")

    monkeypatch.setattr(ShapeQuery, "materialize", forbidden_materialize)
    with pytest.raises(ValueError, match="超过 256x256 canvas"):
        prepare_raster_input(
            source, tmp_path / "too_large.npz", layer=LayerSpec(1, 0),
            box=DbuBox(0, 0, 4096, 4096), pixel_nm=4.0, canvas=256)


def test_strict_shape_guard_runs_before_region_materialization(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """层级展开图形数超限时必须在构造 KLayout Region 前停止。"""
    source = _write_workbench_layout(tmp_path / "complex.gds")

    def forbidden_materialize(*args: object, **kwargs: object) -> None:
        """预检超限后禁止构造完整 Region。"""
        raise AssertionError("复杂度超限不应进入 ShapeQuery.materialize")

    monkeypatch.setattr(ShapeQuery, "materialize", forbidden_materialize)
    with pytest.raises(ValueError, match="图形数超过上限"):
        prepare_segment_input(
            source, tmp_path / "too_complex.npz", layer=LayerSpec(1, 0),
            box=DbuBox(0, 0, 512, 256), tile_size_nm=128.0,
            halo_nm=64.0, max_shape_occurrences=1)


def test_segment_round_trip_preserves_cross_core_topology(tmp_path: Path) -> None:
    """跨 core、孔洞、斜边和引用在保存加载后必须保持全局拓扑与归属。"""
    source = _write_workbench_layout(tmp_path / "segments.gds")
    archive = prepare_segment_input(
        source, tmp_path / "segments.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 512, 256), tile_size_nm=128.0,
        halo_nm=64.0, corner_nm=8.0, segment_nm=16.0,
        max_displacement_nm=16.0)
    problem, metadata = load_segment_input(archive)
    zero = np.zeros(problem.segments.segment_count, dtype=np.float64)
    reconstructed = reconstruct_region(problem, zero)
    counts = metadata["counts"]
    assert counts["segments"] == problem.segments.segment_count
    assert counts["cores"] == 8
    assert counts["memberships"] > counts["segments"]
    assert np.unique(problem.owner_indices).size > 1
    assert int((reconstructed ^ problem.physical_mask.region).area()) == 0
    assert reconstructed.has_valid_polygons()


def test_segment_loader_rejects_missing_and_out_of_range_members(
        tmp_path: Path) -> None:
    """边段加载器必须拒绝缺字段和越界 membership，而不是延迟到迭代崩溃。"""
    source = _write_workbench_layout(tmp_path / "corrupt_source.gds")
    archive = prepare_segment_input(
        source, tmp_path / "valid.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), tile_size_nm=128.0,
        halo_nm=64.0, corner_nm=8.0, segment_nm=16.0)
    missing = _rewrite_npz(archive, tmp_path / "missing.npz", "edge_next_ids", None)
    with pytest.raises(ValueError, match="缺少字段"):
        load_segment_input(missing)
    with np.load(archive, allow_pickle=False) as data:
        members = np.array(data["member_segment_indices"], copy=True)
        segment_count = len(data["segment_edge_ids"])
    members[0] = segment_count
    corrupt = _rewrite_npz(
        archive, tmp_path / "corrupt.npz", "member_segment_indices", members)
    with pytest.raises(ValueError, match="超出 segment 范围"):
        load_segment_input(corrupt)


def test_segment_loader_rejects_v1_with_regeneration_message(tmp_path: Path) -> None:
    """旧 v1 生成物必须明确提示重新生成，不保留隐式兼容转换分支。"""
    source = _write_workbench_layout(tmp_path / "version_source.gds")
    archive = prepare_segment_input(
        source, tmp_path / "version_v2.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), tile_size_nm=128.0,
        halo_nm=64.0, corner_nm=8.0, segment_nm=16.0)
    missing_v2_field = _rewrite_npz(
        archive, tmp_path / "version_v1_fields.npz", "edge_next_ids", None)
    legacy = _rewrite_npz(
        missing_v2_field, tmp_path / "version_v1.npz", "format_version",
        np.array(1, dtype=np.int32))
    with pytest.raises(ValueError, match="重新生成离线边段输入"):
        load_segment_input(legacy)


def test_segment_loader_normalizes_invalid_count_metadata(tmp_path: Path) -> None:
    """损坏的 counts 对象必须转换为统一 ValueError，不能让 KeyError 泄漏到 CLI。"""
    source = _write_workbench_layout(tmp_path / "metadata_source.gds")
    archive = prepare_segment_input(
        source, tmp_path / "metadata_valid.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), tile_size_nm=128.0,
        halo_nm=64.0, corner_nm=8.0, segment_nm=16.0)
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    metadata["counts"] = {"segments": 1}
    corrupt = _rewrite_npz(
        archive, tmp_path / "metadata_corrupt.npz", "metadata_json",
        np.array(json.dumps(metadata, ensure_ascii=False)))
    with pytest.raises(ValueError, match="缺少有效计数"):
        load_segment_input(corrupt)


def test_raster_loader_enforces_archive_memory_limit(tmp_path: Path) -> None:
    """读取端必须在 NumPy 分配成员数组前执行归档总量上限。"""
    source = _write_workbench_layout(tmp_path / "limit.gds")
    archive = prepare_raster_input(
        source, tmp_path / "limit.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), pixel_nm=4.0)
    with pytest.raises(ValueError, match="超过读取上限"):
        load_raster_input(archive, max_archive_gib=1e-9)


def test_lithography_runner_saves_numeric_and_optional_png_results(
        tmp_path: Path) -> None:
    """光刻入口必须只消费像素归档并保存三个工艺角的数值和可视结果。"""
    source = _write_workbench_layout(tmp_path / "litho.gds")
    archive = prepare_raster_input(
        source, tmp_path / "litho_input.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), pixel_nm=8.0)
    output = tmp_path / "lithography"
    result = run_lithography_test(archive, output, device="cpu", save_png=True)
    with np.load(output / "lithography_result.npz", allow_pickle=False) as data:
        assert np.array_equal(data["nominal"], result.nominal.detach().cpu().numpy())
        assert data["nominal"].shape == (256, 256)
    for name in ("mask", "nominal", "maximum", "minimum"):
        with Image.open(output / f"{name}.png") as image:
            assert image.size == (256, 256)
    assert (output / "summary.json").is_file()


def test_mbopc_runner_optimizes_loaded_cross_core_problem(tmp_path: Path) -> None:
    """OPC 入口必须从离线跨 core 问题完成一轮优化并输出全部分析产物。"""
    source = _write_workbench_layout(tmp_path / "opc.gds")
    archive = prepare_segment_input(
        source, tmp_path / "opc_input.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 256, 128), tile_size_nm=128.0,
        halo_nm=64.0, corner_nm=8.0, segment_nm=16.0,
        max_displacement_nm=16.0)
    output = tmp_path / "mbopc"
    result = run_mbopc_iteration_test(
        archive, output, iterations=1, step_nm=4.0,
        epe_distance_nm=8.0, pixel_nm=8.0, batch_size=2,
        target_cache_mb=1, device="cpu", save_preview=True)
    with np.load(output / "mbopc_result.npz", allow_pickle=False) as data:
        assert np.array_equal(data["best_displacements"], result.best_displacements)
        assert int(data["best_iteration"]) == result.best_iteration
    assert len(result.records) == 1
    assert (output / "mbopc_result.gds").is_file()
    assert (output / "mbopc_result.png").is_file()
    assert (output / "summary.json").is_file()


def test_workbench_scripts_show_help_outside_repository(tmp_path: Path) -> None:
    """三个深层脚本必须能在仓库外直接启动且不依赖项目安装。"""
    scripts = (
        PROJECT_ROOT / "tests" / "workbench" / "offline_inputs.py",
        PROJECT_ROOT / "tests" / "workbench" / "run_lithography.py",
        PROJECT_ROOT / "tests" / "workbench" / "run_mbopc_iteration.py",
    )
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=tmp_path,
            capture_output=True, text=True, timeout=30, check=False)
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
