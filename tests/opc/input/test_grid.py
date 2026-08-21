"""两级 macro-core 网格规划与居中光刻画布的生成式测试。"""

import klayout.db as kdb
import numpy as np
import pytest

from layout import DbuBox
from opc.input import (
    MacroSpec,
    ownership_canvas,
    plan_macros,
    points_to_canvas,
    rasterize_mask_canvas,
    rasterize_region_window,
)

# 廉价默认参数：全部数值契约满足的最小网格组合，供多数用例复用。
BASE = {"core_size_dbu": 1, "context_dbu": 0, "pixel_dbu": 1, "canvas_pixels": 256}


def _widths(cuts):
    """按切线数组返回各区间宽度。"""
    return np.diff(cuts).tolist()


class TestMacroPlanningBySize:
    """macro_size_dbu 模式的切分与校验。"""

    def test_size_mode_splits_axis_with_shortened_last_macro(self):
        """21 按 11 切分得到 11+10，不做 10.5+10.5 硬等分。"""
        macros = plan_macros(DbuBox(0, 0, 21, 17), macro_size_dbu=11, **BASE)
        assert len(macros) == 4
        x_macro_widths = sorted({int(m.ownership_box.width) for m in macros})
        assert x_macro_widths == [10, 11]

    def test_size_mode_shortened_core_only_in_outermost_macro(self):
        """轴长 215、macro 110、core 10 时只有最外侧 macro 出现 5 宽缩短 core。"""
        macros = plan_macros(DbuBox(0, 0, 215, 110), macro_size_dbu=110,
                             core_size_dbu=10, context_dbu=0,
                             pixel_dbu=1, canvas_pixels=256)
        by_id = {m.macro_id: m for m in macros}
        inner = _widths(by_id["mr0c0"].x_cuts)
        outer = _widths(by_id["mr0c1"].x_cuts)
        assert set(inner) == {10}
        assert outer[:-1] == [10] * (len(outer) - 1) and outer[-1] == 5

    def test_size_mode_rejects_macro_not_multiple_of_core(self):
        """名义 macro 不是 core 整数倍时直接失败。"""
        with pytest.raises(ValueError, match="whole multiple"):
            plan_macros(DbuBox(0, 0, 21, 17), macro_size_dbu=11,
                        core_size_dbu=10, context_dbu=0,
                        pixel_dbu=1, canvas_pixels=256)

    def test_size_mode_rejects_macro_not_exceeding_core(self):
        """macro 等于或小于 core 时失败：两级网格不得退化为单级 core 网格。"""
        common = {"core_size_dbu": 40, "context_dbu": 0,
                  "pixel_dbu": 1, "canvas_pixels": 256}
        with pytest.raises(ValueError, match="exceed core size"):
            plan_macros(DbuBox(0, 0, 80, 40), macro_size_dbu=40, **common)
        with pytest.raises(ValueError, match="exceed core size"):
            plan_macros(DbuBox(0, 0, 80, 40), macro_size_dbu=20, **common)

    def test_size_mode_accepts_macro_above_core_multiple(self):
        """macro 严格大于 core 且整除时成功（80/40 → 单 macro 双 core）。"""
        macros = plan_macros(DbuBox(0, 0, 80, 40), macro_size_dbu=80,
                             core_size_dbu=40, context_dbu=0,
                             pixel_dbu=1, canvas_pixels=256)
        assert len(macros) == 1
        assert macros[0].core_count == 2


class TestMacroPlanningByCount:
    """macro_grid 模式的均衡分配与校验。"""

    def test_count_mode_balances_core_units(self):
        """21×17、core 1、2×2 得到横向 11+10、纵向 9+8。"""
        macros = plan_macros(DbuBox(0, 0, 21, 17), macro_grid=(2, 2), **BASE)
        by_id = {m.macro_id: m for m in macros}
        assert int(by_id["mr0c0"].ownership_box.width) == 11
        assert int(by_id["mr0c1"].ownership_box.width) == 10
        assert int(by_id["mr0c0"].ownership_box.height) == 9
        assert int(by_id["mr1c0"].ownership_box.height) == 8
        assert list(by_id) == ["mr0c0", "mr0c1", "mr1c0", "mr1c1"]

    def test_exactly_one_macro_entry_required(self):
        """macro_size_dbu 与 macro_grid 同时出现或同时缺失都失败。"""
        bounds = DbuBox(0, 0, 10, 10)
        with pytest.raises(ValueError, match="exactly one"):
            plan_macros(bounds, macro_size_dbu=5, macro_grid=(2, 1), **BASE)
        with pytest.raises(ValueError, match="exactly one"):
            plan_macros(bounds, **BASE)

    def test_count_mode_rejects_more_macros_than_core_units(self):
        """macro 数超过该轴 core 单元数时失败，不允许空 macro。"""
        with pytest.raises(ValueError, match="exceeds core unit count"):
            plan_macros(DbuBox(0, 0, 21, 10), macro_grid=(4, 1),
                        core_size_dbu=10, context_dbu=0,
                        pixel_dbu=1, canvas_pixels=256)


class TestMacroOwnershipContract:
    """macro/core ownership 的互斥与全覆盖契约。"""

    def test_macro_and_core_tiles_partition_parent_exactly(self):
        """全部 core ownership 面积和等于父框面积且无正面积重叠。"""
        bounds = DbuBox(0, 0, 21, 17)
        macros = plan_macros(bounds, macro_grid=(2, 2), **BASE)
        area = 0
        for macro in macros:
            assert int(macro.ownership_box.width) * int(macro.ownership_box.height) == sum(
                int(macro.core(i).ownership_box.area) for i in range(macro.core_count))
            area += int(macro.ownership_box.area)
        assert area == int(bounds.area)

    def test_negative_coordinate_bounds_supported(self):
        """负坐标 bbox 的切线与行优先编号保持全局 DBU 语义。"""
        macros = plan_macros(DbuBox(-100, -50, 100, 50), macro_grid=(2, 2),
                             core_size_dbu=10, context_dbu=0,
                             pixel_dbu=1, canvas_pixels=256)
        assert macros[0].ownership_box == DbuBox(-100, -50, 0, 0)
        assert macros[1].ownership_box == DbuBox(0, -50, 100, 0)
        assert macros[2].ownership_box == DbuBox(-100, 0, 0, 50)
        assert macros[3].ownership_box == DbuBox(0, 0, 100, 50)
        assert [m.macro_id for m in macros] == ["mr0c0", "mr0c1", "mr1c0", "mr1c1"]

    def test_context_expands_query_without_touching_ownership(self):
        """context 扩张 query/context 框，但绝不改变 ownership 切线。"""
        bounds = DbuBox(0, 0, 40, 40)
        plain = plan_macros(bounds, macro_grid=(1, 1), core_size_dbu=10,
                            context_dbu=0, pixel_dbu=1, canvas_pixels=256)
        wide = plan_macros(bounds, macro_grid=(1, 1), core_size_dbu=10,
                           context_dbu=5, pixel_dbu=1, canvas_pixels=256)
        assert plain[0].x_cuts.tolist() == wide[0].x_cuts.tolist()
        assert wide[0].query_box == DbuBox(-5, -5, 45, 45)
        assert wide[0].core(0).context_box == DbuBox(-5, -5, 15, 15)

    def test_locate_owned_points_returns_minus_one_outside_macro(self):
        """ownership 外的点返回 -1，共享边界归右/上，外沿归末行/列。"""
        macro = plan_macros(DbuBox(0, 0, 20, 20), macro_grid=(1, 1),
                            core_size_dbu=10, context_dbu=0,
                            pixel_dbu=1, canvas_pixels=256)[0]
        owners = macro.locate_owned_points(
            [[5, 5], [15, 5], [5, 15], [10, 10], [20, 20], [-1, 5], [25, 25]])
        assert owners.tolist() == [0, 1, 2, 3, 3, -1, -1]


class TestGridValidation:
    """像素整除与画布容量校验。"""

    def test_core_or_context_not_pixel_multiple_fails(self):
        """core 或 context 不是 pixel 整数倍时失败。"""
        with pytest.raises(ValueError, match="whole multiples of a pixel"):
            plan_macros(DbuBox(0, 0, 30, 30), macro_grid=(1, 1),
                        core_size_dbu=10, context_dbu=5,
                        pixel_dbu=4, canvas_pixels=256)

    def test_canvas_capacity_boundary_is_inclusive(self):
        """core+2context 恰满 256×pixel 合法，超出 1 pixel 即失败。"""
        common = {"macro_grid": (1, 1), "context_dbu": 400, "pixel_dbu": 8}
        exact = plan_macros(DbuBox(0, 0, 1248, 1248),
                            core_size_dbu=1248, canvas_pixels=256, **common)
        assert len(exact) == 1
        with pytest.raises(ValueError, match="exceeds the fixed canvas"):
            plan_macros(DbuBox(0, 0, 1256, 1256),
                        core_size_dbu=1256, canvas_pixels=256, **common)

    def test_canvas_must_be_256(self):
        """canvas_pixels 不是 256 时在规划层即失败。"""
        with pytest.raises(ValueError, match="frozen"):
            plan_macros(DbuBox(0, 0, 10, 10), macro_grid=(1, 1), **{
                **BASE, "canvas_pixels": 255})
        with pytest.raises(ValueError, match="frozen"):
            MacroSpec("mr0c0", DbuBox(0, 0, 1, 1),
                      np.array([0, 1]), np.array([0, 1]), 0, 1, 255)


class TestCenteredCanvas:
    """局部窗口居中入 canvas 的 padding 与极性契约。"""

    def test_local_window_centered_with_equal_zero_padding(self):
        """228×228 local mask 在 256 canvas 四边各补 14 个零像素。"""
        context = DbuBox(0, 0, 1824, 1824)
        region = kdb.Region(kdb.Box(0, 0, 912, 1824))
        canvas = rasterize_mask_canvas(region, context, 8, 256, polarity="clear")
        assert np.all(canvas[:14] == 0) and np.all(canvas[:, :14] == 0)
        assert np.all(canvas[242:] == 0) and np.all(canvas[:, 242:] == 0)
        # 左半覆盖：local 列 0..113 为 1（中心 x=912-4=908<912），列 114 起为 0。
        assert canvas[14, 14 + 113] == 1.0
        assert canvas[14, 14 + 114] == 0.0

    def test_odd_padding_remainder_goes_to_high_side(self):
        """奇数差值时低侧 floor，高侧接收余量（14 低 + 15 高）。"""
        context = DbuBox(0, 0, 1816, 1816)
        region = kdb.Region(kdb.Box(0, 0, 1816, 1816))
        canvas = rasterize_mask_canvas(region, context, 8, 256, polarity="clear")
        assert np.all(canvas[:14] == 0)
        assert np.all(canvas[241:] == 0)
        assert np.all(canvas[14:241, 14:241] == 1.0)

    def test_opaque_background_is_one_but_padding_stays_zero(self):
        """opaque 极性局部背景为 1 减 coverage，canvas 外围 padding 仍为 0。"""
        context = DbuBox(0, 0, 1824, 1824)
        region = kdb.Region(kdb.Box(456, 456, 1368, 1368))
        canvas = rasterize_mask_canvas(region, context, 8, 256, polarity="opaque")
        # 覆盖中心区域透光率为 0，未覆盖的局部窗口背景为 1，padding 为 0。
        assert canvas[128, 128] == 0.0
        assert canvas[14, 14] == 1.0
        assert np.all(canvas[:14] == 0) and np.all(canvas[:, :14] == 0)


    # 2026-08-22 起 rasterize_mask_canvas 不再有 dark_box 参数：环带暗场
    # 由 prepare 阶段的负板补铬几何保证（锚点移至 test_macro_problem 的
    # TestOpaqueSurround 与像素/runner 级场边界用例）。

    def test_clear_polarity_keeps_one_as_transmission(self):
        """clear 极性下 coverage 直接就是透光率。"""
        context = DbuBox(0, 0, 1824, 1824)
        region = kdb.Region(kdb.Box(0, 0, 1824, 912))
        canvas = rasterize_mask_canvas(region, context, 8, 256, polarity="clear")
        assert canvas[14, 14] == 1.0
        assert canvas[241, 14] == 0.0

    def test_ownership_canvas_aligns_with_centered_mask_canvas(self):
        """ownership canvas 与居中 mask canvas 同 shape、同偏移。"""
        context = DbuBox(0, 0, 1824, 1824)
        owned = ownership_canvas(DbuBox(400, 400, 1424, 1424), context, 8, 256)
        assert owned.shape == (256, 256)
        # ownership 1024nm=128px，居中偏移 14：计分像素恰为 128×128。
        assert int(owned.sum()) == 128 * 128
        assert not owned[14, 14] and owned[64, 64]
        assert not owned[64, 63]
        assert not owned[13, 13]

    def test_global_dbu_probe_mapping_includes_low_padding(self):
        """全局 DBU 坐标按公式映射到 canvas 索引时必须计入低侧 padding。"""
        context = DbuBox(0, 0, 1824, 1824)
        ownership = DbuBox(400, 400, 1424, 1424)
        owned = ownership_canvas(ownership, context, 8, 256)
        low_x = low_y = 14
        # x_canvas = (x_dbu - context.left)/pixel - 0.5 + low_x
        x_dbu, y_dbu = 404.0, 404.0
        x_canvas = x_dbu / 8 - 0.5 + low_x
        y_canvas = y_dbu / 8 - 0.5 + low_y
        assert owned[int(y_canvas), int(x_canvas)]
        # 反向验证：ownership 占 local 列 50..177（canvas 列 64..191），
        # 右边界 1424 落在 191/192 像素交界——191 是最后一个计分像素，192 起不计分。
        x_edge = 1424.0
        x_index = int(x_edge / 8 - 0.5 + low_x)
        assert x_index == 191 and owned[64, 191] and not owned[64, 192]

    def test_full_canvas_local_needs_no_padding(self):
        """满 256 local 输入不再添加 padding，canvas 等于局部窗口。"""
        context = DbuBox(0, 0, 2048, 2048)
        region = kdb.Region(kdb.Box(0, 0, 1024, 2048))
        canvas = rasterize_mask_canvas(region, context, 8, 256, polarity="clear")
        assert canvas[0, 0] == 1.0 and canvas[255, 255] == 0.0
        assert canvas[0, 127] == 1.0 and canvas[0, 128] == 0.0

    def test_local_window_exceeding_canvas_fails_before_allocation(self):
        """超过 256 像素的局部窗口在栅格化前即失败。"""
        context = DbuBox(0, 0, 2056, 2056)
        region = kdb.Region(kdb.Box(0, 0, 100, 100))
        with pytest.raises(ValueError, match="固定画布"):
            rasterize_mask_canvas(region, context, 8, 256, polarity="clear")

    def test_region_window_keeps_left_bottom_origin_without_padding(self):
        """最小窗口栅格不添加 canvas padding，行 0 对应最低 Y。"""
        window = rasterize_region_window(
            kdb.Region(kdb.Box(0, 0, 16, 8)), DbuBox(0, 0, 16, 16), 8)
        assert window.shape == (2, 2)
        assert window.tolist() == [[1.0, 1.0], [0.0, 0.0]]


class TestPointsToCanvas:
    """DBU 点到居中 canvas 连续坐标换算与 ownership 对齐。"""

    def test_full_canvas_has_no_padding_offset(self):
        """满 256 窗口 low=0：第一像素中心 (4,4)DBU 映射到 (0,0)。"""
        context = DbuBox(0, 0, 2048, 2048)
        out = points_to_canvas([[4.0, 4.0]], context, 8, 256)
        assert out.shape == (1, 2)
        assert out[0].tolist() == [0.0, 0.0]

    def test_even_padding_shifts_by_low_side(self):
        """228 像素窗口低侧 padding 14：像素中心仍映射回自身索引。"""
        context = DbuBox(0, 0, 1824, 1824)
        out = points_to_canvas([[4.0, 4.0]], context, 8, 256)
        assert out[0].tolist() == [14.0, 14.0]  # 局部 (0,0) 像素中心

    def test_odd_remainder_keeps_low_floor(self):
        """227 像素窗口 14 低 15 高：窗口首尾像素中心映射正确。"""
        context = DbuBox(0, 0, 1816, 1816)
        out = points_to_canvas([[4.0, 4.0], [1812.0, 1812.0]], context, 8, 256)
        # 局部首像素中心 4DBU → 局部索引 0 → canvas 14；
        # 末像素中心 1816-4=1812DBU → 局部索引 226 → canvas 240。
        assert out[0].tolist() == [14.0, 14.0]
        assert out[1].tolist() == [240.0, 240.0]

    def test_nonzero_context_origin(self):
        """context 原点非零时先减原点再换算，与全局坐标无关。"""
        context = DbuBox(1000, 2000, 2824, 3824)
        out = points_to_canvas([[1004.0, 2004.0]], context, 8, 256)
        assert out[0].tolist() == [14.0, 14.0]

    def test_all_ownership_pixels_round_trip_to_integers(self):
        """ownership 计分像素中心经换算恰落回自身整数索引（批量对齐）。"""
        context = DbuBox(0, 0, 1824, 1824)
        ownership = DbuBox(400, 400, 1424, 1424)
        owned = ownership_canvas(ownership, context, 8, 256)
        rows, columns = np.nonzero(owned)
        # 正向公式来自 ownership_canvas：中心 = 原点 + (索引 - low + 0.5)×pixel。
        centers = np.stack(
            (context.left + (columns - 14 + 0.5) * 8,
             context.bottom + (rows - 14 + 0.5) * 8), axis=1)
        out = points_to_canvas(centers, context, 8, 256)
        np.testing.assert_array_equal(out[:, 0], columns.astype(np.float64))
        np.testing.assert_array_equal(out[:, 1], rows.astype(np.float64))

    def test_x_and_y_are_not_swapped(self):
        """x 进列索引、y 进行索引，两轴互不交换。"""
        context = DbuBox(0, 0, 1824, 1824)
        out = points_to_canvas([[12.0, 4.0], [4.0, 12.0]], context, 8, 256)
        assert out[0, 0] == pytest.approx(15.0)  # x=12 → 列 15
        assert out[0, 1] == pytest.approx(14.0)  # y=4 → 行 14
        assert out[1, 0] == pytest.approx(14.0)
        assert out[1, 1] == pytest.approx(15.0)

    def test_fractional_coordinates_stay_continuous(self):
        """非像素中心的连续 DBU 坐标保留小数，不取整不裁剪。"""
        context = DbuBox(0, 0, 1824, 1824)
        out = points_to_canvas([[9.0, 5.0]], context, 8, 256)
        assert out[0].tolist() == [14.625, 14.125]  # (9/8-0.5+14, 5/8-0.5+14)

    def test_out_of_window_points_are_not_clipped(self):
        """context 外的点照样换算（可能为负或超界），裁剪留给评价层。"""
        context = DbuBox(0, 0, 1824, 1824)
        out = points_to_canvas([[-200.0, 2200.0]], context, 8, 256)
        assert out[0, 0] < 0.0  # 左侧 200DBU → canvas -11.5，保留负值
        assert out[0, 1] > 255.0  # 上方 376DBU → canvas 288.5，保留超界值

    def test_non_pair_points_fail(self):
        """[N,3] 或一维输入不是 (x,y) 点集，失败。"""
        context = DbuBox(0, 0, 1824, 1824)
        with pytest.raises(ValueError, match="N,2"):
            points_to_canvas(np.zeros((2, 3)), context, 8, 256)
        with pytest.raises(ValueError, match="N,2"):
            points_to_canvas(np.zeros(4), context, 8, 256)
