"""MacroProblem 准备、ownership 切线分段与 NPZ 持久化的生成式测试。"""

import zipfile

import klayout.db as kdb
import numpy as np
import pytest

from geometry import extract_contour
from layout import DbuBox, LayerSpec, LayoutDB, RegionBatch
from opc.input import normalize_mask, plan_macros
from opc.input.edge import (
    MacroProblem,
    prepare_macro_problem,
    reconstruct_region,
)
from opc.input.edge.fragmentation import FragmentationConfig

LAYER = LayerSpec(1, 0)
# 廉价网格契约：160² 版图、2×2 macro、core 40、context 20、pixel 4。
BOUNDS = DbuBox(0, 0, 160, 160)
CFG = FragmentationConfig(corner_length_dbu=8.0, max_segment_length_dbu=16.0,
                          max_displacement_dbu=10.0, miter_limit=4.0)


def _macros():
    """返回 2×2 macro 网格。"""
    return plan_macros(BOUNDS, macro_grid=(2, 2), core_size_dbu=40,
                       context_dbu=20, pixel_dbu=4, canvas_pixels=256)


def _problem(region, macro):
    """把原生 Region 直接包装为 RegionBatch 并生成 macro problem。"""
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_macro_problem(batch, LAYER, "clear", CFG, macro)


def _assert_problem_invariants(problem, source):
    """对单个 problem 复核设计文档 §15.3 的全部几何不变量。"""
    zeros = np.zeros(problem.segments.segment_count)
    # 零位移重建与源 Region XOR 为零：查询/上下文框的四条边没有成为物理边段。
    assert int((reconstruct_region(problem, zeros) ^ source).area()) == 0
    # owner 取值域：每段要么 -1（只读 context）要么合法局部 core 编号。
    assert np.all((problem.owner_indices >= -1) &
                  (problem.owner_indices < problem.macro.core_count))
    geometry = problem.segments.materialize(None)
    owned = np.flatnonzero(problem.owner_indices >= 0)
    # 可写段开区间不跨任何 ownership 切线（端点恰在切线上按半开约定归右/上）。
    lo = np.minimum(geometry.starts[owned], geometry.ends[owned])
    hi = np.maximum(geometry.starts[owned], geometry.ends[owned])
    for cuts, axis in ((problem.macro.x_cuts, 0), (problem.macro.y_cuts, 1)):
        for cut in cuts[1:-1]:
            assert not np.any((lo[:, axis] < cut) & (hi[:, axis] > cut))
    # 段中点归属与 owner 一致（中点定唯一 owner 的构造契约）。
    midpoints = (geometry.starts[owned] + geometry.ends[owned]) * 0.5
    assert np.array_equal(problem.macro.locate_owned_points(midpoints),
                          problem.owner_indices[owned])
    # own ⊆ membership：owner 段必然出现在其 owner 的 CSR 区间内。
    for core_index in range(problem.macro.core_count):
        members = set(problem.segments_for_core(core_index).tolist())
        owned_here = {int(s) for s in
                      np.flatnonzero(problem.owner_indices == core_index)}
        assert owned_here <= members


def _reference_ring_count(region, macro):
    """返回与准备管线同路径（normalize+extract）的参考 ring 数。"""
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return extract_contour(normalize_mask(batch, LAYER)).ring_count


def _edge_split_params(problem, first, second):
    """返回指定数学边（按两端点整数坐标匹配）上的分裂参数集。"""
    vertices = problem.segments.contours.vertices
    start = vertices[problem.segments.edge_ids]
    end = vertices[problem.segments.edge_next_ids[problem.segments.edge_ids]]
    hit = (((start[:, 0] == first[0]) & (start[:, 1] == first[1]) &
            (end[:, 0] == second[0]) & (end[:, 1] == second[1])) |
           ((start[:, 0] == second[0]) & (start[:, 1] == second[1]) &
            (end[:, 0] == first[0]) & (end[:, 1] == first[1])))
    params = np.concatenate((problem.segments.t0[hit], problem.segments.t1[hit]))
    return np.unique(np.round(params, 12))


class TestOwnershipSplit:
    """ownership 切线分段与 owner 唯一性。"""

    def test_owned_segment_never_crosses_two_owners(self):
        """可写段内部不跨任何 ownership 切线，中点归属即唯一 owner。"""
        problem = _problem(kdb.Region(kdb.Box(10, 10, 70, 50)), _macros()[0])
        geometry = problem.segments.materialize(None)
        owned = np.flatnonzero(problem.owner_indices >= 0)
        assert len(owned)
        lo = np.minimum(geometry.starts[owned], geometry.ends[owned])
        hi = np.maximum(geometry.starts[owned], geometry.ends[owned])
        # 内部穿越判定：切线严格落在段开区间 (lo, hi) 内。端点恰在切线上时按
        # 半开约定归右/上，与中点 owner 不同是预期行为，不算跨越。
        for cuts, axis in ((problem.macro.x_cuts, 0), (problem.macro.y_cuts, 1)):
            for cut in cuts[1:-1]:
                crossing = (lo[:, axis] < cut) & (hi[:, axis] > cut)
                assert not np.any(crossing)
        midpoints = (geometry.starts[owned] + geometry.ends[owned]) * 0.5
        assert np.array_equal(problem.macro.locate_owned_points(midpoints),
                              problem.owner_indices[owned])

    def test_segments_outside_macro_ownership_are_readonly_context(self):
        """伸入邻居 macro 的部分段 owner 为 -1，且数量非零。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), _macros()[0])
        owners = problem.owner_indices
        assert np.any(owners < 0)
        # 参考中点完全落在 macro ownership 右侧（x>80）的段必然是只读 context。
        geometry = problem.segments.materialize(None)
        midpoints = (geometry.starts + geometry.ends) * 0.5
        right_of_macro = midpoints[:, 0] > 80
        assert np.any(right_of_macro)
        assert np.all(owners[right_of_macro] == -1)

    def test_context_segment_may_appear_in_multiple_cores(self):
        """context 段可同时出现在多个 core 的 CSR 中，但 owner 唯一。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), _macros()[0])
        appear = np.zeros(problem.segments.segment_count, dtype=np.int64)
        for core_index in range(problem.macro.core_count):
            appear[problem.segments_for_core(core_index)] += 1
        assert np.any(appear >= 2)
        # 唯一可写 owner 的正确断言：每段的 owner 落在 [-1, C) 内恰一个编号上。
        owners = problem.owner_indices
        assert np.all((owners >= -1) & (owners < problem.macro.core_count))

    def test_owner_segments_always_inside_owner_membership(self):
        """owner 段必然属于其 owner 的 membership（构造期不变量复验）。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), _macros()[0])
        for core_index in range(problem.macro.core_count):
            members = set(problem.segments_for_core(core_index).tolist())
            for segment in np.flatnonzero(problem.owner_indices == core_index):
                assert int(segment) in members

    def test_query_box_edges_never_become_physical_segments(self):
        """查询/上下文框不进入物理边段：零位移重建与源 Region XOR 为零。"""
        region = kdb.Region(kdb.Box(30, 30, 50, 50))
        problem = _problem(region, _macros()[0])
        zeros = np.zeros(problem.segments.segment_count)
        assert int((reconstruct_region(problem, zeros) ^ region).area()) == 0


class TestSharedDiagonal:
    """共享 macro 边界两侧的斜边分裂一致性。"""

    def test_shared_macro_boundary_split_points_are_identical(self):
        """两侧 macro 对同一条斜边得到完全相同的分裂参数（无 33/34 分歧）。"""
        macros = _macros()
        region = kdb.Region(kdb.Polygon([
            kdb.Point(60, 100), kdb.Point(100, 100), kdb.Point(100, 20)]))
        left = _problem(region, macros[0])
        right = _problem(region, macros[1])
        # 共享切线 x=80 的分裂参数 t=(80-60)/40=0.5 必须逐位一致。
        left_params = _edge_split_params(left, (60, 100), (100, 20))
        right_params = _edge_split_params(right, (60, 100), (100, 20))
        assert np.array_equal(left_params, right_params)
        assert np.any(np.isclose(left_params, 0.5))


class TestGeometryMatrix:
    """§15.3 复杂几何矩阵：每类图形一个独立、可定位失败原因的测试函数。"""

    def test_concave_polygon_crossing_horizontal_macro_boundary(self):
        """凹多边形（L 形）跨横向 macro 边界 y=80 与多条 core 切线。"""
        region = kdb.Region(kdb.Polygon([
            kdb.Point(30, 60), kdb.Point(90, 60), kdb.Point(90, 140),
            kdb.Point(60, 140), kdb.Point(60, 90), kdb.Point(30, 90)]))
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            # 分段/分裂不得改变 ring 拓扑：与纯提取路径的 ring 数一致。
            assert (problem.segments.contours.ring_count ==
                    _reference_ring_count(region, macro))

    def test_two_dbu_narrow_ring_crossing_cuts(self):
        """2 DBU 宽窄环（外框−内孔差 2）跨 core 切线与 macro 边界。"""
        narrow = (kdb.Region(kdb.Box(20, 70, 60, 110)) -
                  kdb.Region(kdb.Box(22, 72, 58, 108)))
        for macro in _macros():
            problem = _problem(narrow, macro)
            _assert_problem_invariants(problem, narrow)
            assert problem.segments.contours.ring_count == 2

    def test_slope_crossing_vertical_macro_boundary(self):
        """斜率 1 的斜边跨竖向切线 x=80：两侧分裂参数逐位一致。"""
        region = kdb.Region(kdb.Polygon([
            kdb.Point(60, 20), kdb.Point(120, 20), kdb.Point(90, 50)]))
        first, second = (60, 20), (90, 50)
        left = _problem(region, _macros()[0])
        right = _problem(region, _macros()[1])
        _assert_problem_invariants(left, region)
        _assert_problem_invariants(right, region)
        assert np.array_equal(_edge_split_params(left, first, second),
                              _edge_split_params(right, first, second))

    def test_steep_slope_crossing_horizontal_macro_boundary(self):
        """陡斜边（斜率 8/3）跨横向切线 y=80：上下两侧分裂参数逐位一致。"""
        region = kdb.Region(kdb.Polygon([
            kdb.Point(20, 60), kdb.Point(60, 60), kdb.Point(35, 100)]))
        first, second = (20, 60), (35, 100)
        lower = _problem(region, _macros()[0])
        upper = _problem(region, _macros()[2])
        _assert_problem_invariants(lower, region)
        _assert_problem_invariants(upper, region)
        assert np.array_equal(_edge_split_params(lower, first, second),
                              _edge_split_params(upper, first, second))

    def test_slope_through_macro_corner_point(self):
        """斜边精确穿过 macro 角点 (80,80)：四个 macro 的分裂参数全部一致。"""
        region = kdb.Region(kdb.Polygon([
            kdb.Point(60, 60), kdb.Point(100, 60), kdb.Point(60, 100)]))
        first, second = (100, 60), (60, 100)
        params = []
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            params.append(_edge_split_params(problem, first, second))
        assert all(np.array_equal(params[0], other) for other in params[1:])
        # 角点 (80,80) 对应参数 t=0.5，必须出现在每个 macro 的分裂点中。
        assert all(np.any(np.isclose(p, 0.5)) for p in params)

    def test_edge_touching_pair_across_core_cut(self):
        """共边相接的一对矩形（相接边恰为 core 切线 x=40）合并后无残留边。"""
        region = kdb.Region(kdb.Box(10, 10, 40, 40))
        region.insert(kdb.Box(40, 10, 80, 40))
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            assert (problem.segments.contours.ring_count ==
                    _reference_ring_count(region, macro))

    def test_overlapping_pair_merges_before_extraction(self):
        """部分重叠的一对矩形在提边前合并，重叠区不产生内部边段。"""
        region = kdb.Region(kdb.Box(20, 20, 60, 60))
        region.insert(kdb.Box(40, 40, 100, 80))
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            assert (problem.segments.contours.ring_count ==
                    _reference_ring_count(region, macro))

    def test_contained_pair_collapses_to_outer(self):
        """完全包含的一对矩形坍缩为外框，内框不产生任何边段。"""
        region = kdb.Region(kdb.Box(20, 20, 120, 120))
        region.insert(kdb.Box(60, 60, 90, 90))
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            assert problem.segments.contours.ring_count == 1

    def test_single_sref_crossing_macro_boundary(self, tmp_path):
        """单个 SREF 的图形跨 macro 边界 x=80，展开后不变量保持。"""
        layout = kdb.Layout()
        layout.dbu = 0.001
        unit = layout.create_cell("UNIT")
        unit.shapes(layout.layer(1, 0)).insert(kdb.Box(60, 30, 100, 50))
        top = layout.create_cell("TOP")
        top.insert(kdb.CellInstArray(unit.cell_index(), kdb.Trans(0, 0)))
        path = tmp_path / "sref.gds"
        layout.write(str(path))
        for macro in _macros():
            with LayoutDB.open(path) as database:
                batch = database.query([LAYER], macro.query_box).materialize_intersecting()
            problem = prepare_macro_problem(batch, LAYER, "clear", CFG, macro)
            _assert_problem_invariants(problem, batch.region(LAYER))

    def test_two_by_two_aref_spreads_across_macros(self, tmp_path):
        """2×2 AREF 的四个 occurrence 分别落在四个不同 macro。"""
        layout = kdb.Layout()
        layout.dbu = 0.001
        unit = layout.create_cell("UNIT")
        unit.shapes(layout.layer(1, 0)).insert(kdb.Box(60, 30, 100, 50))
        top = layout.create_cell("TOP")
        top.insert(kdb.CellInstArray(unit.cell_index(), kdb.Trans(0, 0),
                                     kdb.Vector(40, 0), kdb.Vector(0, 60), 2, 2))
        path = tmp_path / "aref.gds"
        layout.write(str(path))
        owners_per_macro = {}
        for macro in _macros():
            with LayoutDB.open(path) as database:
                batch = database.query([LAYER], macro.query_box).materialize_intersecting()
            problem = prepare_macro_problem(batch, LAYER, "clear", CFG, macro)
            _assert_problem_invariants(problem, batch.region(LAYER))
            owners_per_macro[macro.macro_id] = int((problem.owner_indices >= 0).sum())
        # occurrence 中心分别位于 (80,40)、(120,40)、(80,100)、(120,100)：
        # 四个 macro 各自拥有属于自己的一组边段。
        assert all(count > 0 for count in owners_per_macro.values())

    def test_long_bar_crosses_at_least_three_cores(self):
        """长条连续跨越至少三个 core（横向贯穿两个 macro 的四列 core）。"""
        region = kdb.Region(kdb.Box(10, 30, 150, 50))
        columns = set()
        for macro in _macros():
            problem = _problem(region, macro)
            _assert_problem_invariants(problem, region)
            owned = problem.owner_indices[problem.owner_indices >= 0]
            columns |= {(macro.macro_id, int(o) % macro.column_count) for o in owned}
        assert len(columns) >= 3


class TestTopologyPreservation:
    """hole/ring 拓扑与跨 macro 零位移覆盖。"""

    def test_donut_rings_survive_ownership_split(self):
        """带孔图形分裂后 ring 拓扑保持且零位移 XOR 为零。"""
        donut = (kdb.Region(kdb.Box(10, 10, 150, 150)) -
                 kdb.Region(kdb.Box(60, 60, 100, 100)))
        problem = _problem(donut, _macros()[0])
        assert problem.segments.contours.ring_count >= 2
        zeros = np.zeros(problem.segments.segment_count)
        assert int((reconstruct_region(problem, zeros) ^ donut).area()) == 0

    def test_zero_displacement_owner_coverage_merges_to_input(self):
        """全部 macro 零位移结果按 ownership 汇总并 merge 后 XOR 为零。"""
        region = kdb.Region(kdb.Box(20, 20, 140, 130))
        merged = kdb.Region()
        for macro in _macros():
            problem = _problem(region, macro)
            zeros = np.zeros(problem.segments.segment_count)
            merged += reconstruct_region(problem, zeros) & kdb.Region(
                macro.ownership_box.to_native())
        assert int((merged.merged() ^ region).area()) == 0

    def test_array_references_materialize_into_problem(self, tmp_path):
        """SREF/AREF 展开后的图形可正常生成 problem 并保持不变量。"""
        layout = kdb.Layout()
        layout.dbu = 0.001
        unit = layout.create_cell("UNIT")
        unit.shapes(layout.layer(1, 0)).insert(kdb.Box(0, 0, 50, 20))
        top = layout.create_cell("TOP")
        top.insert(kdb.CellInstArray(unit.cell_index(), kdb.Trans(10, 10),
                                     kdb.Vector(60, 0), kdb.Vector(0, 60), 3, 3))
        path = tmp_path / "aref.gds"
        layout.write(str(path))
        macro = _macros()[0]
        with LayoutDB.open(path) as database:
            batch = database.query([LAYER], macro.query_box).materialize_intersecting()
        problem = prepare_macro_problem(batch, LAYER, "clear", CFG, macro)
        assert problem.segments.segment_count > 0
        zeros = np.zeros(problem.segments.segment_count)
        assert np.all(problem.owner_indices >= -1)
        expected = batch.region(LAYER)
        assert int((reconstruct_region(problem, zeros) ^ expected).area()) == 0


class TestPreparationValidation:
    """准备入口的显式契约校验。"""

    def test_query_box_mismatch_rejected(self):
        """batch.query_box 与 macro.query_box 不一致时拒绝。"""
        batch = RegionBatch({LAYER: kdb.Region(kdb.Box(0, 0, 10, 10))},
                            DbuBox(0, 0, 20, 20))
        with pytest.raises(ValueError, match="query_box"):
            prepare_macro_problem(batch, LAYER, "clear", CFG, _macros()[0])

    def test_unknown_polarity_rejected(self):
        """未知极性字符串在准备入口即失败。"""
        region = kdb.Region(kdb.Box(0, 0, 10, 10))
        macro = _macros()[0]
        batch = RegionBatch({LAYER: region}, macro.query_box)
        with pytest.raises(ValueError, match="极性"):
            prepare_macro_problem(batch, LAYER, "reverse", CFG, macro)

    def test_own_missing_from_owner_membership_rejected(self):
        """owner 不在该 core membership 时构造期拒绝（own⊆membership）。"""
        macro = _macros()[0]
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), macro)
        appear = np.zeros(problem.segments.segment_count, dtype=np.int64)
        for c in range(macro.core_count):
            appear[problem.segments_for_core(c)] += 1
        owned = np.flatnonzero(problem.owner_indices >= 0)
        segment = int(owned[np.argmin(appear[owned])])
        assert appear[segment] < macro.core_count
        member_cores = {c for c in range(macro.core_count)
                        if segment in set(problem.segments_for_core(c).tolist())}
        outsider = next(c for c in range(macro.core_count) if c not in member_cores)
        broken = problem.owner_indices.copy()
        broken[segment] = outsider
        with pytest.raises(ValueError, match="membership"):
            MacroProblem(macro, problem.layer, problem.polarity,
                         problem.fragmentation, problem.segments,
                         owner_indices=broken,
                         core_offsets=problem.core_offsets,
                         member_segment_indices=problem.member_segment_indices)

    def test_owned_segments_with_empty_membership_rejected(self):
        """有 owner 却完全无 membership 的损坏对象必须被拒绝（空 CSR 漏洞回归）。"""
        macro = _macros()[0]
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), macro)
        owners = problem.owner_indices.copy()
        # 复现审查清单描述的状态：全部 owned 段指向 core 0，但 CSR 为空。
        owners[owners >= 0] = 0
        with pytest.raises(ValueError, match="membership"):
            MacroProblem(macro, problem.layer, problem.polarity,
                         problem.fragmentation, problem.segments,
                         owner_indices=owners,
                         core_offsets=np.zeros(macro.core_count + 1, dtype=np.int64),
                         member_segment_indices=np.empty(0, dtype=np.int32))

    def test_all_context_problem_with_empty_membership_is_legal(self):
        """全部段均为只读 context（owner=-1）且 membership 为空时构造合法。"""
        macro = _macros()[0]
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), macro)
        empty = MacroProblem(macro, problem.layer, problem.polarity,
                             problem.fragmentation, problem.segments,
                             owner_indices=np.full(
                                 problem.segments.segment_count, -1, dtype=np.int32),
                             core_offsets=np.zeros(macro.core_count + 1, dtype=np.int64),
                             member_segment_indices=np.empty(0, dtype=np.int32))
        assert empty.segments.segment_count == problem.segments.segment_count


class TestPersistence:
    """NPZ 持久化与失败路径。"""

    def _saved(self, tmp_path):
        """生成并保存一个标准 problem，返回 (path, problem)。"""
        problem = _problem(kdb.Region(kdb.Box(20, 20, 140, 60)), _macros()[0])
        path = tmp_path / "problem.npz"
        problem.save(path)
        return path, problem

    def test_save_load_roundtrip_is_exact(self, tmp_path):
        """save/load 后全部标量与数组逐项相等。"""
        path, problem = self._saved(tmp_path)
        loaded = MacroProblem.load(path)
        assert loaded.macro.macro_id == problem.macro.macro_id
        assert loaded.macro.ownership_box == problem.macro.ownership_box
        assert np.array_equal(loaded.macro.x_cuts, problem.macro.x_cuts)
        assert np.array_equal(loaded.macro.y_cuts, problem.macro.y_cuts)
        assert loaded.layer == problem.layer and loaded.polarity == problem.polarity
        assert loaded.fragmentation == problem.fragmentation
        for name in ("vertices", "ring_offsets", "polygon_ring_offsets"):
            assert np.array_equal(getattr(loaded.segments.contours, name),
                                  getattr(problem.segments.contours, name))
        for name in ("edge_ids", "edge_next_ids", "edge_polygon_ids",
                     "edge_normals", "ring_segment_offsets", "t0", "t1"):
            assert np.array_equal(getattr(loaded.segments, name),
                                  getattr(problem.segments, name))
        assert np.array_equal(loaded.owner_indices, problem.owner_indices)
        assert np.array_equal(loaded.core_offsets, problem.core_offsets)
        assert np.array_equal(loaded.member_segment_indices,
                              problem.member_segment_indices)

    def test_npz_is_loadable_without_pickle(self, tmp_path):
        """NPZ 不含 object dtype，allow_pickle=False 可完整读取。"""
        path, _ = self._saved(tmp_path)
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                assert data[name].dtype != np.dtype(object)

    def test_wrong_format_version_rejected(self, tmp_path):
        """格式版本不符的 NPZ 直接失败。"""
        path, _ = self._saved(tmp_path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["format_version"] = np.array([99], dtype=np.int32)
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="format version"):
            MacroProblem.load(path)

    def test_truncated_file_rejected(self, tmp_path):
        """截断文件读取失败而不是返回半截数据。"""
        path, _ = self._saved(tmp_path)
        raw = path.read_bytes()
        (tmp_path / "broken.npz").write_bytes(raw[: len(raw) // 2])
        with pytest.raises(zipfile.BadZipFile):
            MacroProblem.load(tmp_path / "broken.npz")

    def test_wrong_displacement_length_rejected(self, tmp_path):
        """位移向量长度与段数不符时重建拒绝。"""
        _, problem = self._saved(tmp_path)
        with pytest.raises(ValueError, match="segment count"):
            reconstruct_region(problem, np.zeros(problem.segments.segment_count + 1))

    def test_atomic_save_failure_keeps_previous_file(self, tmp_path, monkeypatch):
        """保存中途失败不替换旧完整文件。"""
        path, problem = self._saved(tmp_path)
        before = path.read_bytes()
        import opc.input.edge.problem as problem_module

        def _boom(*args, **kwargs):
            """模拟写盘失败，验证原子保存不替换旧文件。"""
            raise OSError("disk full")
        monkeypatch.setattr(problem_module.np, "savez", _boom)
        with pytest.raises(OSError, match="disk full"):
            problem.save(path)
        assert path.read_bytes() == before
        monkeypatch.undo()
        MacroProblem.load(path)
