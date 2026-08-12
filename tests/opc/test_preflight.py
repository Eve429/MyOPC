"""验证大规模 MB-OPC 输入在完整物化前的容量估算和内存统计。"""

from __future__ import annotations

from pathlib import Path

import klayout.db as kdb
import numpy as np

from layout import DbuBox, LayerSpec, LayoutDB
from opc.input import RectilinearCoreGrid, preflight_layout, process_memory_snapshot


def _write_rectangle(path: Path, right: int = 1000, top: int = 1000) -> Path:
    """写出一个坐标范围可控的单矩形 GDS 供预检使用。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    layer = layout.layer(1, 0)
    cell = layout.create_cell("TOP")
    cell.shapes(layer).insert(kdb.Box(0, 0, right, top))
    layout.write(str(path))
    return path


def test_preflight_accepts_small_layout_and_reports_all_capacity_fields(tmp_path: Path) -> None:
    """小版图应完成扫描，并报告准备、求解、索引和预算判断。"""
    source = _write_rectangle(tmp_path / "small.gds")
    box = DbuBox(0, 0, 1000, 1000)
    grid = RectilinearCoreGrid(np.array([0, 500, 1000]), np.array([0, 1000]), 100)
    with LayoutDB.open(source) as database:
        result = preflight_layout(
            database, layer=LayerSpec(1, 0), box=box,
            corner_dbu=8.0, maximum_segment_dbu=32.0, grid=grid,
            memory_budget_bytes=1024 ** 3)
    assert result["accepted"] is True
    assert result["scan_complete"] is True
    assert result["counts_are_lower_bounds"] is False
    assert result["estimated_segments"] > 0
    assert result["estimated_memberships"] >= result["estimated_segments"]
    assert result["estimated_prepare_peak_bytes"] > 0
    assert result["estimated_solver_peak_bytes"] > 0
    assert 0 < result["max_memberships"] <= np.iinfo(np.int32).max
    assert result["recommended_mode"] == "in_memory"


def test_preflight_rejects_ten_billion_segments_without_segment_allocation(
        tmp_path: Path) -> None:
    """百亿级切分估算应命中 int32/内存保护，而不构造 SegmentBatch。"""
    extent = 2_000_000_000
    source = _write_rectangle(tmp_path / "huge-count.gds", extent, 1)
    box = DbuBox(0, 0, extent, 1)
    grid = RectilinearCoreGrid(np.array([0, extent]), np.array([0, 1]))
    with LayoutDB.open(source) as database:
        result = preflight_layout(
            database, layer=LayerSpec(1, 0), box=box,
            corner_dbu=0.1, maximum_segment_dbu=0.2, grid=grid,
            memory_budget_bytes=64 * 1024 ** 3)
    assert result["accepted"] is False
    assert result["int32_capacity_ok"] is False
    assert result["estimated_segments"] >= 10_000_000_000
    assert result["scan_complete"] is False
    assert result["counts_are_lower_bounds"] is True
    assert result["recommended_mode"] == "sharded_required"


def test_process_memory_snapshot_uses_stable_nonnegative_fields() -> None:
    """进程内存检查点应覆盖 RSS、私有内存、峰值工作集和系统可用量。"""
    snapshot = process_memory_snapshot()
    assert set(snapshot) == {
        "rss_bytes", "uss_bytes", "private_bytes", "peak_working_set_bytes",
        "system_available_bytes",
    }
    assert all(isinstance(value, int) and value >= 0 for value in snapshot.values())
