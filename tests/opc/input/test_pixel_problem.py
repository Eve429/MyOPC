"""像素宏问题的一次栅格化、core 映射、极性回写与持久化生成式测试。"""

import klayout.db as kdb
import numpy as np
import pytest

import opc.input.edge  # edge 零调用守卫的补丁宿主
import opc.input.pixel.problem as pixel_problem  # spy 补丁宿主
from layout import DbuBox, LayerSpec, RegionBatch
from opc.input import plan_macros, rasterize_mask_canvas
from opc.input.pixel import (
    PixelMacroProblem,
    prepare_pixel_macro_problem,
    reconstruct_pixel_region,
)
from opc.input.raster import _center_padding, rasterize_region_window

LAYER = LayerSpec(1, 0)
# 廉价契约：80² 版图、单 macro 2×2 core、context 20、pixel 4（全整像素）。
BOUNDS = DbuBox(0, 0, 80, 80)
CANVAS = 256


def _macro(**overrides):
    """返回单 macro 规划（默认 80² 版图 2×2 core）。"""
    values = {"macro_grid": (1, 1), "core_size_dbu": 40, "context_dbu": 20,
              "pixel_dbu": 4, "canvas_pixels": 256}
    values.update(overrides)
    return plan_macros(BOUNDS, **values)[0]


def _prepare(region, macro=None, polarity="clear", layout_bounds=BOUNDS):
    """把原生 Region 包装为 RegionBatch 并生成像素 macro 问题。"""
    macro = macro if macro is not None else _macro()
    batch = RegionBatch({LAYER: region}, macro.query_box)
    return prepare_pixel_macro_problem(
        batch, LAYER, polarity, macro, layout_bounds=layout_bounds)


def _binary_over_ownership(problem):
    """按 0.5 阈值取 macro ownership 内的二值 transmission。"""
    pixel = problem.macro.pixel_dbu
    query = problem.macro.query_box
    box = problem.macro.ownership_box
    hm, wm = problem.ownership_shape
    r0 = (box.bottom - query.bottom) // pixel
    c0 = (box.left - query.left) // pixel
    return problem.target_u8[r0:r0 + hm, c0:c0 + wm] >= 128


def _canvas_cells(problem, core_index):
    """返回 core 画布位置 → query 栅格单元的换算所需 (r0,c0,low_y,low_x)。"""
    _, r0, r1, c0, c1 = problem._context_window(core_index)
    low_y, _, low_x, _ = _center_padding(r1 - r0, c1 - c0, CANVAS)
    return r0, c0, low_y, low_x


class TestPrepareAndPersistence:
    """一次栅格化、极性统一、edge 零依赖与 NPZ 往返（TEST-001）。"""

    def test_rasterize_once_and_no_edge_dependency(self, monkeypatch):
        """每个 macro 恰一次栅格；全程不触碰 edge 提边 API。"""
        counts = {"raster": 0}
        real = pixel_problem.rasterize_region_window

        def spy(*args, **kwargs):
            """计数栅格化调用并透传。"""
            counts["raster"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(pixel_problem, "rasterize_region_window", spy)

        def _forbidden(*args, **kwargs):
            """像素路径不得调用边段语义。"""
            raise AssertionError("像素路径不得调用 edge 提边/边段构造")

        monkeypatch.setattr(opc.input.edge, "fragment_edges", _forbidden)
        monkeypatch.setattr(opc.input.edge, "prepare_macro_problem", _forbidden)
        problem = _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)))
        assert counts["raster"] == 1  # 恰一次
        assert problem.ownership_shape == (20, 20)  # 80/4
        assert problem.query_shape == (30, 30)  # (80+2×20)/4

    def test_clear_transmission_values(self):
        """clear：图形内 255、外 0；非整像素边界保留分数覆盖率。"""
        problem = _prepare(kdb.Region(kdb.Box(8, 8, 41, 48)))
        target = problem.target_u8
        # query 左下为 (−20,−20)：图形内整像素
        assert target[8, 8] == 255  # 单元 [12,16)² 完全在图形内
        assert target[0, 0] == 0  # query 角落无图形
        # x=41 落在 [40,44) 单元：覆盖率 1/4 → rint(63.75)=64
        assert target[7, 15] == 64

    def test_opaque_transmission_inverted(self):
        """opaque：field−coverage；材料内 0、bbox 内背景 255、bbox 外 0。"""
        problem = _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)), polarity="opaque")
        target = problem.target_u8
        assert target[8, 8] == 0  # 材料不透光
        assert target[20, 20] == 255  # bbox 内无材料背景透光
        assert target[0, 0] == 0  # query 角落落在版图 bbox 外：恒不透光

    def test_npz_roundtrip_identical(self, tmp_path):
        """NPZ 往返：网格、层、极性与栅格逐值一致。"""
        problem = _prepare(kdb.Region(kdb.Box(4, 4, 76, 76)))
        path = problem.save(tmp_path / "p.npz")
        loaded = PixelMacroProblem.load(path)
        assert loaded.macro.macro_id == problem.macro.macro_id
        assert loaded.macro.ownership_box == problem.macro.ownership_box
        assert np.array_equal(loaded.macro.x_cuts, problem.macro.x_cuts)
        assert np.array_equal(loaded.macro.y_cuts, problem.macro.y_cuts)
        assert loaded.layer == problem.layer
        assert loaded.polarity == problem.polarity
        assert np.array_equal(loaded.target_u8, problem.target_u8)


class TestCoreMapping:
    """core 画布映射：计分唯一、trainable 索引跨 core 一致（TEST-002）。"""

    def test_ownership_exactly_once(self):
        """每个 macro 像素在全部 core 计分画布中恰出现一次。"""
        problem = _prepare(kdb.Region(kdb.Box(4, 4, 76, 76)))
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        hm, wm = problem.ownership_shape
        mrow0 = (box.bottom - query.bottom) // pixel
        mcol0 = (box.left - query.left) // pixel
        seen = np.zeros((hm, wm), dtype=np.int32)  # macro 像素被计分次数
        for core_index in range(problem.macro.core_count):
            r0, c0, low_y, low_x = _canvas_cells(problem, core_index)
            rows, cols = np.nonzero(problem.ownership_canvas(core_index))
            gy = rows - low_y + r0  # query 栅格行
            gx = cols - low_x + c0  # query 栅格列
            np.add.at(seen, (gy - mrow0, gx - mcol0), 1)
        assert int(seen.max()) == 1  # 无重复计分
        assert int(seen.sum()) == hm * wm  # 无遗漏

    def test_trainable_index_global_consistency(self):
        """trainable ≥0 即 macro ownership 像素；索引为行主序扁平且与 target 对齐。"""
        problem = _prepare(kdb.Region(kdb.Box(4, 4, 76, 76)))
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        hm, wm = problem.ownership_shape
        mrow0 = (box.bottom - query.bottom) // pixel
        mcol0 = (box.left - query.left) // pixel
        collected = set()  # 出现过的扁平索引全集
        for core_index in range(problem.macro.core_count):
            r0, c0, low_y, low_x = _canvas_cells(problem, core_index)
            canvas = problem.trainable_index_canvas(core_index)
            rows, cols = np.nonzero(canvas >= 0)
            gy = rows - low_y + r0
            gx = cols - low_x + c0
            expected = (gy - mrow0) * wm + (gx - mcol0)  # 行主序扁平
            assert np.array_equal(canvas[rows, cols], expected)
            # trainable 像素的 target 画布值与 query 栅格逐位一致（两画布同布局）
            target_canvas = problem.target_canvas(core_index)
            assert np.array_equal(
                target_canvas[rows, cols], problem.target_u8[gy, gx])
            collected.update(int(v) for v in expected)
        assert collected == set(range(hm * wm))  # 每个 macro 像素至少可见一次

    def test_ownership_matches_public_raster(self):
        """core 计分画布与其自身 ownership 框的公共栅格逐位一致。"""
        problem = _prepare(kdb.Region(kdb.Box(4, 4, 76, 76)))
        pixel = problem.macro.pixel_dbu
        for core_index in range(problem.macro.core_count):
            spec = problem.macro.core(core_index)
            # 独立路径：该 core 自己的 ownership 框栅格化（对齐框 → 二值覆盖）
            region = kdb.Region(spec.ownership_box.to_native())
            mask = rasterize_mask_canvas(
                region, spec.context_box, pixel, CANVAS, polarity="clear")
            assert np.array_equal(
                problem.ownership_canvas(core_index), mask >= 0.5)


class TestLayoutBounds:
    """版图 bbox 外恒不透光（00_PAST field_box 契约的迁移等价）。"""

    def test_opaque_outside_bounds_stays_dark(self):
        """opaque：query 超出 bbox 的环带全 0，不得反相成虚假透光。"""
        # 默认规划：query=(−20,−20,100,100)、bbox=[0,80)²、pixel 4
        # → 栅格 [5,25)² 之内是 bbox，四条外环带必须恒 0（对应旧
        # test_opaque_context_outside_field_stays_dark 的行为规格）。
        problem = _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)), polarity="opaque")
        target = problem.target_u8
        assert not target[:5].any() and not target[25:].any()
        assert not target[:, :5].any() and not target[:, 25:].any()
        assert target[20, 20] == 255  # bbox 内背景仍透光
        assert target[8, 8] == 0  # bbox 内材料不透光

    def test_clear_zeroing_is_bitwise_noop(self):
        """clear：bbox 外无几何，置零前后逐位一致（防误伤）。"""
        region = kdb.Region(kdb.Box(8, 8, 40, 48))
        bounded = _prepare(region)
        # bounds 完全盖住 query 时不置零任何像素，作为无干预参照
        reference = _prepare(region, layout_bounds=_macro().query_box)
        assert np.array_equal(bounded.target_u8, reference.target_u8)

    def test_interior_macro_not_zeroed(self):
        """内部 macro（query 全在 bbox 内）：不做任何置零。"""
        bounds = DbuBox(0, 0, 240, 240)
        macro = plan_macros(bounds, macro_grid=(3, 3), core_size_dbu=40,
                            context_dbu=20, pixel_dbu=4,
                            canvas_pixels=CANVAS)[4]  # 行优先中心 mr1c1
        assert macro.query_box == DbuBox(60, 60, 180, 180)  # 全在 bounds 内
        region = kdb.Region(kdb.Box(90, 90, 150, 150))
        problem = _prepare(region, macro=macro, layout_bounds=bounds)
        reference = _prepare(region, macro=macro, layout_bounds=macro.query_box)
        assert np.array_equal(problem.target_u8, reference.target_u8)

    def test_ownership_pixels_never_modified(self):
        """两极性下 ownership 像素与无干预参照逐位一致。"""
        region = kdb.Region(kdb.Box(8, 8, 40, 48))
        for polarity in ("clear", "opaque"):
            problem = _prepare(region, polarity=polarity)
            reference = _prepare(
                region, polarity=polarity, layout_bounds=_macro().query_box)
            query = problem.macro.query_box
            box = problem.macro.ownership_box
            r0 = (box.bottom - query.bottom) // 4
            c0 = (box.left - query.left) // 4
            block = problem.target_u8[r0:r0 + 20, c0:c0 + 20]
            other = reference.target_u8[r0:r0 + 20, c0:c0 + 20]
            assert np.array_equal(block, other)

    def test_ownership_not_within_bounds_rejected(self):
        """bounds 未四向包含 ownership：显式失败，不猜测场边界。"""
        with pytest.raises(ValueError, match="包含"):
            _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)),
                     layout_bounds=DbuBox(0, 0, 40, 80))

    def test_bounds_off_pixel_grid_rejected(self):
        """bounds 交叠边非整像素：显式失败，不静默取整。"""
        with pytest.raises(ValueError, match="整像素"):
            _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)),
                     layout_bounds=DbuBox(0, 0, 83, 80))


class TestPixelAlignment:
    """实际 box 整像素契约的合法/非法边界（TEST-002）。"""

    def test_shortened_final_core_allowed(self):
        """末端 core 缩短但整像素：合法（core 1 宽 20 = 5 像素）。"""
        macro = plan_macros(DbuBox(0, 0, 100, 80), macro_grid=(1, 1),
                            core_size_dbu=40, context_dbu=20, pixel_dbu=4,
                            canvas_pixels=CANVAS)[0]
        problem = _prepare(kdb.Region(kdb.Box(10, 10, 90, 70)), macro=macro,
                           layout_bounds=DbuBox(0, 0, 100, 80))
        assert problem.ownership_shape == (20, 25)  # 100/4 × 80/4

    def test_nonaligned_bbox_rejected_before_raster(self, monkeypatch):
        """bbox 宽度非整像素：栅格化前 ValueError，不产 partial 像素。"""
        macro = plan_macros(DbuBox(0, 0, 83, 80), macro_grid=(1, 1),
                            core_size_dbu=40, context_dbu=20, pixel_dbu=4,
                            canvas_pixels=CANVAS)[0]

        def _forbidden(*args, **kwargs):
            """非整像素必须在栅格化前失败。"""
            raise AssertionError("非整像素时不得执行栅格化")

        monkeypatch.setattr(pixel_problem, "rasterize_region_window", _forbidden)
        with pytest.raises(ValueError, match="整数倍"):
            _prepare(kdb.Region(kdb.Box(10, 10, 70, 70)), macro=macro)


class TestReconstruct:
    """pixel→Region 极性逆变换与几何矩阵（TEST-003）。"""

    def _roundtrip(self, region, polarity="clear"):
        """构造问题→二值→回写→重栅格，返回 (binary, 回栅格覆盖, problem)。"""
        problem = _prepare(region, polarity=polarity)
        binary = _binary_over_ownership(problem)
        rebuilt = reconstruct_pixel_region(problem, binary)
        coverage = rasterize_region_window(
            rebuilt, problem.macro.ownership_box, problem.macro.pixel_dbu)
        return binary, coverage >= 0.5, problem, rebuilt

    def _assert_roundtrip(self, region, polarity="clear"):
        """二值回写再栅格与输入二值逐位一致，面积与外框守恒。"""
        binary, coverage, problem, rebuilt = self._roundtrip(region, polarity)
        material = binary if polarity == "clear" else ~binary
        assert np.array_equal(coverage, material)
        pixel2 = problem.macro.pixel_dbu ** 2
        assert int(rebuilt.area()) == int(material.sum()) * pixel2
        box = problem.macro.ownership_box
        bbox = rebuilt.bbox()
        assert (bbox.left >= box.left and bbox.bottom >= box.bottom
                and bbox.right <= box.right and bbox.top <= box.top)

    def test_rectangle(self):
        """实心矩形：clear/opaque 双向一致。"""
        self._assert_roundtrip(kdb.Region(kdb.Box(8, 8, 40, 48)))
        self._assert_roundtrip(kdb.Region(kdb.Box(8, 8, 40, 48)), "opaque")

    def test_hole(self):
        """带孔图形：孔内不透光（clear）/孔外材料（opaque）。"""
        region = kdb.Region(kdb.Box(4, 4, 76, 76)) - kdb.Region(
            kdb.Box(32, 32, 48, 48))
        self._assert_roundtrip(region)
        self._assert_roundtrip(region, "opaque")

    def test_concave_and_diagonal(self):
        """L 形与 45° 斜边（阶梯化后回写一致）。"""
        concave = kdb.Region(kdb.Polygon([
            kdb.Point(8, 8), kdb.Point(48, 8), kdb.Point(48, 24),
            kdb.Point(24, 24), kdb.Point(24, 48), kdb.Point(8, 48)]))
        self._assert_roundtrip(concave)
        diagonal = kdb.Region(kdb.Polygon([
            kdb.Point(8, 8), kdb.Point(48, 8), kdb.Point(8, 48)]))
        self._assert_roundtrip(diagonal)

    def test_multi_island(self):
        """多岛图形：每岛独立游程合并。"""
        region = kdb.Region(kdb.Box(8, 8, 24, 24))
        region.insert(kdb.Box(48, 48, 72, 72))
        self._assert_roundtrip(region)

    def test_all_zero_and_all_one(self):
        """全 0（无图形）与全 1（整框覆盖）两个极端。"""
        binary, coverage, _, rebuilt = self._roundtrip(kdb.Region())
        assert not binary.any() and not coverage.any()
        assert rebuilt.is_empty()  # 空 target 回写为空 Region
        full = kdb.Region(_macro().query_box.to_native())
        self._assert_roundtrip(full)

    def test_wrong_binary_shape_rejected(self):
        """binary 形状/dtype 不符时显式失败。"""
        problem = _prepare(kdb.Region(kdb.Box(8, 8, 40, 48)))
        with pytest.raises(ValueError, match="布尔数组"):
            reconstruct_pixel_region(problem, np.zeros((5, 5), dtype=np.bool_))
        with pytest.raises(ValueError, match="布尔数组"):
            reconstruct_pixel_region(
                problem, np.zeros(problem.ownership_shape, dtype=np.uint8))


class TestCorruption:
    """problem 持久化损坏的显式失败（TEST-004）。"""

    def _saved(self, tmp_path):
        """保存一个合法 problem 并返回路径。"""
        return _prepare(kdb.Region(kdb.Box(8, 8, 40, 48))).save(
            tmp_path / "p.npz")

    @staticmethod
    def _retamper(path, tmp_path, name, **replacements):
        """读出全部数组、替换指定键后另存。"""
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays.update(replacements)
        np.savez(tmp_path / name, **arrays)
        return tmp_path / name

    def test_wrong_format_name_rejected(self, tmp_path):
        """格式名不符明确失败。"""
        bad = self._retamper(
            self._saved(tmp_path), tmp_path, "bad.npz",
            format=np.array(["something.else"]))
        with pytest.raises(ValueError, match="format name"):
            PixelMacroProblem.load(bad)

    def test_wrong_version_rejected(self, tmp_path):
        """版本号不符明确失败。"""
        bad = self._retamper(
            self._saved(tmp_path), tmp_path, "bad.npz",
            format_version=np.array([99], dtype=np.int32))
        with pytest.raises(ValueError, match="format version"):
            PixelMacroProblem.load(bad)

    def test_tampered_dtype_rejected(self, tmp_path):
        """target_u8 被换成 float：构造即校验失败。"""
        bad = self._retamper(
            self._saved(tmp_path), tmp_path, "bad.npz",
            target_u8=np.full((30, 30), 255, dtype=np.float32))
        with pytest.raises(ValueError, match="uint8"):
            PixelMacroProblem.load(bad)

    def test_tampered_shape_rejected(self, tmp_path):
        """target_u8 被截断：形状与网格整像素一致性失败。"""
        bad = self._retamper(
            self._saved(tmp_path), tmp_path, "bad.npz",
            target_u8=np.zeros((10, 10), dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            PixelMacroProblem.load(bad)

    def test_corrupted_cuts_rejected(self, tmp_path):
        """macro 切线损坏：MacroSpec 契约在加载期传播失败。"""
        bad = self._retamper(
            self._saved(tmp_path), tmp_path, "bad.npz",
            macro_x_cuts=np.array([0], dtype=np.int64))
        with pytest.raises(ValueError):
            PixelMacroProblem.load(bad)
