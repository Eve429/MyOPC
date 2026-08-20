"""LevelSet ILT 物理单位：nm-SDF 与像素尺寸不变量。"""

import klayout.db as kdb
import numpy as np
import pytest

from opc.iteration.ilt import (
    macro_gradient_magnitude,
    signed_distance_initialization,
)
from tests.opc.iteration.test_simple_ilt import _problem


class TestPhysicalSDFUnits:
    """pixel_nm 只改变 phi 的物理尺度，不改变 SDF 几何语义。"""

    def test_sdf_scales_linearly_with_pixel_nm(self):
        target = np.zeros((7, 9), np.uint8)
        target[2:5, 3:7] = 255
        unit = signed_distance_initialization(target, pixel_nm=1.0)
        physical = signed_distance_initialization(target, pixel_nm=4.0)
        np.testing.assert_allclose(physical, unit * 4.0, rtol=1e-6, atol=1e-6)
        assert np.array_equal(physical < 0.0, unit < 0.0)

    def test_constant_fields_use_physical_nm(self):
        foreground = signed_distance_initialization(
            np.full((4, 9), 255, np.uint8), pixel_nm=2.5)
        background = signed_distance_initialization(
            np.zeros((4, 9), np.uint8), pixel_nm=2.5)
        np.testing.assert_array_equal(
            foreground, np.full((4, 9), -22.5, np.float32))
        np.testing.assert_array_equal(
            background, np.full((4, 9), 22.5, np.float32))

    @pytest.mark.parametrize("pixel_nm", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_pixel_nm_rejected(self, pixel_nm):
        with pytest.raises(ValueError, match="pixel_nm"):
            signed_distance_initialization(
                np.zeros((3, 3), np.uint8), pixel_nm=pixel_nm)


class TestPhysicalGradientUnits:
    """phi 与空间间距同时按 nm 缩放后，|grad(phi)| 应保持无量纲不变。"""

    def test_gradient_magnitude_invariant_to_unit_scaling(self):
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        phi1 = signed_distance_initialization(problem.target_u8, pixel_nm=1.0)
        phi4 = signed_distance_initialization(problem.target_u8, pixel_nm=4.0)
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        hm, wm = problem.ownership_shape
        r0 = (box.bottom - query.bottom) // pixel
        c0 = (box.left - query.left) // pixel
        crop1 = phi1[r0:r0 + hm, c0:c0 + wm].copy()
        crop4 = phi4[r0:r0 + hm, c0:c0 + wm].copy()
        mag1 = macro_gradient_magnitude(
            problem, phi1, crop1, pixel_nm=1.0)
        mag4 = macro_gradient_magnitude(
            problem, phi4, crop4, pixel_nm=4.0)
        np.testing.assert_allclose(mag4, mag1, rtol=1e-6, atol=1e-6)

    def test_non_sdf_snapshot_scales_consistently(self):
        problem = _problem(kdb.Region(kdb.Box(8, 8, 41, 48)))
        phi1 = signed_distance_initialization(problem.target_u8, pixel_nm=1.0)
        phi2 = signed_distance_initialization(problem.target_u8, pixel_nm=2.0)
        pixel = problem.macro.pixel_dbu
        query = problem.macro.query_box
        box = problem.macro.ownership_box
        hm, wm = problem.ownership_shape
        r0 = (box.bottom - query.bottom) // pixel
        c0 = (box.left - query.left) // pixel
        crop1 = phi1[r0:r0 + hm, c0:c0 + wm].copy()
        crop1[::2, ::3] += np.float32(0.25)
        crop2 = crop1 * np.float32(2.0)
        mag1 = macro_gradient_magnitude(
            problem, phi1, crop1, pixel_nm=1.0)
        mag2 = macro_gradient_magnitude(
            problem, phi2, crop2, pixel_nm=2.0)
        np.testing.assert_allclose(mag2, mag1, rtol=1e-6, atol=1e-6)
