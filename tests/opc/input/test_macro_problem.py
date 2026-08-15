"""MacroProblem 准备、ownership 切线分段与 NPZ 持久化的生成式测试。"""

import zipfile

import klayout.db as kdb
import numpy as np
import pytest

from layout import DbuBox, LayerSpec, LayoutDB, RegionBatch
from opc.input import plan_macros
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

    def _diagonal_segment_params(self, macro):
        """构造三角带斜边图形并返回斜边上的分裂参数集。"""
        region = kdb.Region(kdb.Polygon([
            kdb.Point(60, 100), kdb.Point(100, 100), kdb.Point(100, 20)]))
        problem = _problem(region, macro)
        vertices = problem.segments.contours.vertices
        start = vertices[problem.segments.edge_ids]
        end = vertices[problem.segments.edge_next_ids[problem.segments.edge_ids]]
        diagonal = ((start[:, 0] == 60) & (start[:, 1] == 100)) | (
            (end[:, 0] == 60) & (end[:, 1] == 100))
        params = np.concatenate((problem.segments.t0[diagonal],
                                 problem.segments.t1[diagonal]))
        return np.unique(np.round(params, 12)), problem

    def test_shared_macro_boundary_split_points_are_identical(self):
        """两侧 macro 对同一条斜边得到完全相同的分裂参数（无 33/34 分歧）。"""
        macros = _macros()
        left_params, _ = self._diagonal_segment_params(macros[0])
        right_params, _ = self._diagonal_segment_params(macros[1])
        # 共享切线 x=80 的分裂参数 t=(80-60)/40=0.5 必须逐位一致。
        assert np.array_equal(left_params, right_params)
        assert np.any(np.isclose(left_params, 0.5))


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
            raise OSError("disk full")
        monkeypatch.setattr(problem_module.np, "savez", _boom)
        with pytest.raises(OSError, match="disk full"):
            problem.save(path)
        assert path.read_bytes() == before
        monkeypatch.undo()
        MacroProblem.load(path)
