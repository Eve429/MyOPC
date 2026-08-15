"""验证 CPU macro 未裁剪候选、真实边和 tile 稀疏归属。"""

import klayout.db as kdb
import numpy as np
import pytest

from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.input import RectilinearCoreGrid, macro_boxes
from opc.input.edge import FragmentationConfig, prepare_macro


def _macro_batch(region: kdb.Region, context: DbuBox) -> tuple[RegionBatch, LayerSpec]:
    """构造已经保留完整相交图形的 macro 输入批次。"""
    layer = LayerSpec(1, 0)
    return RegionBatch({layer: region}, context, CellRef("MACRO", 0)), layer


def test_macro_crossing_shape_does_not_create_query_box_edges() -> None:
    """跨 context 的完整矩形只能保留原边，不能沿查询框生成虚假竖边。"""
    region = kdb.Region(kdb.Box(-20, 10, 120, 90))
    context, ownership = DbuBox(0, 0, 100, 100), DbuBox(0, 0, 100, 100)
    batch, layer = _macro_batch(region, context)
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100]), np.array([0, 50, 100]), 10)
    prepared = prepare_macro(
        batch, layer, FragmentationConfig(5, 20, 4), grid, ownership)
    geometry = prepared.segments.materialize()
    active = prepared.active_segment_indices
    # 原矩形两条水平边穿过 context；位于 x=-20/120 的两条真实竖边在 context
    # 外。若物化曾裁剪，active 中会错误出现 x=0/100 的竖边。
    vectors = geometry.ends[active] - geometry.starts[active]
    assert len(active) > 0
    assert np.allclose(vectors[:, 1], 0.0)
    assert not np.any(np.isin(geometry.starts[active, 0], (0.0, 100.0)) &
                      np.isclose(vectors[:, 0], 0.0))


def test_macro_tile_membership_uses_global_owner_and_local_context() -> None:
    """2×3 tile 中每个活跃段必须有一个全局 owner，halo 只增加只读 membership。"""
    region = kdb.Region(kdb.Box(10, 10, 140, 90))
    batch, layer = _macro_batch(region, DbuBox(0, 0, 150, 100))
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100, 150]), np.array([0, 50, 100]), 15)
    prepared = prepare_macro(
        batch, layer, FragmentationConfig(5, 20, 4), grid,
        DbuBox(0, 0, 150, 100))
    assert np.all(prepared.active_owner_indices >= 0)
    assert len(prepared.active_owner_indices) == len(prepared.active_segment_indices)
    assert len(np.unique(prepared.owned_segments())) == len(prepared.owned_segments())
    for local_core, global_core in enumerate(prepared.core_indices):
        members = prepared.segments_for_core(local_core)
        owned = prepared.active_segment_indices[
            prepared.active_owner_indices == global_core]
        assert set(owned).issubset(set(members))


def test_macro_keeps_outside_roi_edges_as_read_only_halo_context() -> None:
    """处理框外真实边进入 tile halo 时应保留 membership，但不能获得写 owner。"""
    region = kdb.Region(kdb.Box(-8, 10, 40, 40))
    batch, layer = _macro_batch(region, DbuBox(-10, 0, 50, 50))
    grid = RectilinearCoreGrid(np.array([0, 50]), np.array([0, 50]), 10)
    prepared = prepare_macro(
        batch, layer, FragmentationConfig(5, 20, 4), grid, DbuBox(0, 0, 50, 50))
    geometry = prepared.segments.materialize()
    outside_vertical = np.flatnonzero(
        np.isclose(geometry.starts[:, 0], -8.0) &
        np.isclose(geometry.ends[:, 0], -8.0))
    members = prepared.segments_for_core(0)
    assert len(outside_vertical) > 0
    assert set(outside_vertical).issubset(set(members))
    owner_by_segment = dict(zip(
        prepared.active_segment_indices.tolist(), prepared.active_owner_indices.tolist()))
    assert all(owner_by_segment[int(index)] == -1 for index in outside_vertical)
    assert not set(outside_vertical).intersection(set(prepared.owned_segments()))


def test_grid_core_matches_bulk_order_and_rejects_invalid_index() -> None:
    """按需 core 必须与既有批量顺序一致，并严格拒绝无效索引。"""
    grid = RectilinearCoreGrid(
        np.array([0, 40, 100]), np.array([0, 30, 80]), 10)
    assert tuple(grid.core(index) for index in range(grid.core_count)) == grid.cores()
    with pytest.raises(IndexError):
        grid.core(-1)
    with pytest.raises(IndexError):
        grid.core(grid.core_count)


def test_macro_boxes_group_whole_tiles_without_gaps() -> None:
    """macro planner 只能选现有 tile 切线，末端短 tile 也必须完整覆盖。"""
    grid = RectilinearCoreGrid(
        np.array([0, 40, 80, 105]), np.array([0, 30, 60]), 10)
    boxes = macro_boxes(grid, 70)
    assert boxes == (
        DbuBox(0, 0, 40, 60), DbuBox(40, 0, 105, 60))
    assert sum(box.area for box in boxes) == grid.bounds.area


def test_macro_membership_limit_rejects_before_csr_allocation() -> None:
    """布尔合并后 membership 超过预检上限时必须在最终数组分配前拒绝。"""
    region = kdb.Region(kdb.Box(10, 10, 140, 90))
    batch, layer = _macro_batch(region, DbuBox(0, 0, 150, 100))
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100, 150]), np.array([0, 50, 100]), 15)
    with pytest.raises(MemoryError, match="macro membership"):
        prepare_macro(
            batch, layer, FragmentationConfig(5, 20, 4), grid,
            DbuBox(0, 0, 150, 100), max_memberships=1)


@pytest.mark.parametrize("region", [
    kdb.Region(kdb.Polygon([
        kdb.Point(-20, 20), kdb.Point(220, 60),
        kdb.Point(180, 180), kdb.Point(-10, 140)])),
    kdb.Region(kdb.Box(-20, 10, 220, 190)) -
    kdb.Region(kdb.Box(30, 40, 170, 160)),
    (kdb.Region(kdb.Box(-20, 10, 220, 190)) -
     kdb.Region(kdb.Box(-10, 20, 210, 180))),
    kdb.Region(kdb.Box(-20, 20, 130, 120)) +
    kdb.Region(kdb.Box(70, 80, 220, 180)),
])
def test_macro_geometry_matrix_keeps_only_real_complete_shape_edges(
        region: kdb.Region) -> None:
    """斜边、孔洞、窄环和重叠图形跨 macro 时都不得产生查询框轮廓。"""
    context, ownership = DbuBox(0, 0, 120, 200), DbuBox(0, 0, 100, 200)
    batch, layer = _macro_batch(region, context)
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100, 150, 200]), np.array([0, 100, 200]), 20)
    prepared = prepare_macro(
        batch, layer, FragmentationConfig(5, 20, 4), grid, ownership)
    geometry = prepared.segments.materialize()
    # 图形坐标刻意避开查询框四边；若 context 参与裁剪，以下任一整段命中都说明
    # 裁剪框被误当成真实边。斜边仅在交点接触框线，不会触发整段判定。
    starts, ends = geometry.starts, geometry.ends
    on_query_boundary = (
        (np.isclose(starts[:, 0], context.left) & np.isclose(ends[:, 0], context.left)) |
        (np.isclose(starts[:, 0], context.right) & np.isclose(ends[:, 0], context.right)) |
        (np.isclose(starts[:, 1], context.bottom) & np.isclose(ends[:, 1], context.bottom)) |
        (np.isclose(starts[:, 1], context.top) & np.isclose(ends[:, 1], context.top)))
    assert not np.any(on_query_boundary)
    assert len(np.unique(prepared.owned_segments())) == len(prepared.owned_segments())


def test_macro_owned_segments_match_single_full_roi_preparation() -> None:
    """小版图逐 macro 发布的边段集合必须与一次性全 ROI 准备完全一致。"""
    layer, full_box = LayerSpec(1, 0), DbuBox(0, 0, 200, 200)
    region = (kdb.Region(kdb.Box(-20, 20, 220, 180)) -
              kdb.Region(kdb.Box(40, 60, 160, 140)))
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100, 150, 200]), np.array([0, 100, 200]), 20)
    config = FragmentationConfig(5, 20, 4)
    full_batch = RegionBatch({layer: region}, full_box, CellRef("FULL", 0))
    full = prepare_macro(full_batch, layer, config, grid, full_box)
    full_geometry = full.segments.materialize()
    expected = {
        tuple(np.concatenate((full_geometry.starts[index], full_geometry.ends[index],
                              full_geometry.normals[index])).tolist())
        for index in full.owned_segments()}
    actual: set[tuple[float, ...]] = set()
    for ownership in macro_boxes(grid, 100):
        context = ownership.expanded(24)
        # 合成测试直接保留与 context 相交的完整 Region；生产入口由
        # materialize_intersecting 在 KLayout 空间索引侧完成同一候选筛选。
        batch = RegionBatch({layer: region}, context, CellRef("MACRO", 0))
        prepared = prepare_macro(batch, layer, config, grid, ownership)
        geometry = prepared.segments.materialize()
        actual.update(
            tuple(np.concatenate((geometry.starts[index], geometry.ends[index],
                                  geometry.normals[index])).tolist())
            for index in prepared.owned_segments())
    assert actual == expected
