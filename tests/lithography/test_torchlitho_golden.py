"""TorchLitho 迁移一致性 golden 测试（批 B：多图案两级判定，用户点名交付物）。

golden 数据由一次性脚本调用外部 TorchLitho-2.0 原库生成（步骤全文见
doc/changes/completed/CHG-20260823-torchlitho/test_report.md），本文件只消费
golden/ 下的 .pt 对照，不在测试内依赖外部库。两级判定：
- point 源（原库默认 σ=0.05 时源盘只含 DC，两方法物理同一模型）：逐位一致；
- disk 源：Abbe 修正版与原库缺陷版的差异量化（R2）+ 与原库 TCC 机器逐位；
- resize 分支：2048nm 视场（纯零嵌入）逐位、4096nm 视场（真插值）容差。
"""

from pathlib import Path

import pytest
import torch

from lithography.torchlitho import TorchLithoConfig, TorchLithoLithography

# golden 生成与对照共用参数（与原库 example/rect.py 同参数族）。
CANVAS, PIXEL_NM = 64, 8.0
NA, WAVELENGTH, SIGMA_POINT, DEFOCUS = 1.35, 193.0, 0.05, 40.0
GOLDEN_DIR = Path(__file__).parent / "golden"


def _patterns() -> dict[str, torch.Tensor]:
    """八种 64×64 合成图案（固定定义，golden 生成脚本与本测试共用）。"""
    canvas = CANVAS

    def blank() -> torch.Tensor:
        """全暗画布。"""
        return torch.zeros(canvas, canvas)

    single_rect = blank()
    single_rect[16:48, 16:48] = 1.0

    dense_lines = blank()
    for start in range(4, 64, 8):
        dense_lines[:, start : start + 4] = 1.0

    double_lines = blank()
    double_lines[16:20, :] = 1.0
    double_lines[44:48, :] = 1.0

    l_shape = blank()
    l_shape[12:52, 12:20] = 1.0
    l_shape[44:52, 12:52] = 1.0

    isolated_hole = blank()
    isolated_hole[24:40, 24:40] = 1.0

    bridge = blank()
    bridge[12:52, 12:20] = 1.0
    bridge[12:52, 44:52] = 1.0
    bridge[12:20, 20:44] = 1.0

    generator = torch.Generator().manual_seed(20260823)
    coarse = (torch.rand(8, 8, generator=generator) > 0.5).float()
    random_blobs = coarse.repeat_interleave(8, 0).repeat_interleave(8, 1)

    checkerboard = blank()
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                checkerboard[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = 1.0

    return {
        "single_rect": single_rect,
        "dense_lines": dense_lines,
        "double_lines": double_lines,
        "l_shape": l_shape,
        "isolated_hole": isolated_hole,
        "bridge": bridge,
        "random_blobs": random_blobs,
        "checkerboard": checkerboard,
    }


def _load(name: str) -> dict:
    """加载一份 golden 数据包。"""
    return torch.load(GOLDEN_DIR / name, weights_only=False)


def _model(method: str, source_shape: str = "point", sigma: float = SIGMA_POINT, **kwargs) -> TorchLithoLithography:
    """按 golden 参数族构造待测模型。"""
    return TorchLithoLithography(
        TorchLithoConfig(method=method, source_shape=source_shape, sigma=sigma, **kwargs),
        CANVAS,
        PIXEL_NM,
        device="cpu",
    )


class TestGoldenPoint:
    """点源全图案逐位一致：迁移前后数值等价的第一层证据。"""

    @pytest.mark.parametrize("pattern_name", list(_patterns()))
    @pytest.mark.parametrize("method", ["abbe", "hopkins"])
    @pytest.mark.parametrize("defocus_key", ["focus", "defocus"])
    def test_matches_original_bitwise(self, method, defocus_key, pattern_name):
        """8 图案 × 2 方法 × 2 离焦对 golden 逐点一致。

        abbe 数值链与原库完全同构，rtol=1e-6 下逐位；hopkins 原库前向多一步
        恒等 interpolate 且求和顺序不同，实测差异全部在 float32 舍入包络内
        （相对 ≤4.8e-7），判定容差取 1e-5，另有 rank-1 互证兜底。
        """
        golden = _load("golden_point.pt")
        index = golden["names"].index(pattern_name)
        mask = golden["masks"][index]
        model = _model(method)
        prepared = model._prepare_mask(mask)[0]
        defocus_nm = 0.0 if defocus_key == "focus" else DEFOCUS
        aerial = (
            model._abbe_aerial(prepared, defocus_nm)
            if method == "abbe"
            else model._hopkins_aerial(prepared, defocus_nm)
        )
        reference = golden[method][defocus_key][index]
        tolerance = 1e-6 if method == "abbe" else 1e-5
        assert torch.allclose(aerial, reference, rtol=tolerance, atol=1e-8), (
            f"{method}/{defocus_key}/{pattern_name} 最大绝对差 {float((aerial - reference).abs().max()):.3e}"
        )


class TestGoldenRank1:
    """点源 TCC 的 rank-1 解析事实：对迁移 SVD 机器的独立校验（不依赖原库）。"""

    def test_point_source_tcc_has_single_kernel(self):
        """点源 TCC 阈值过滤后恰 1 个核（其余奇异值低于 1e-6）。"""
        model = _model("hopkins")
        model(torch.zeros(CANVAS, CANVAS), model.condition("nominal"))  # 触发 TCC 构造
        kernels, weights = model._tcc_cache[0.0]
        assert kernels.shape[0] == 1
        assert weights[0] > 1e-3

    def test_point_source_aerial_matches_abbe(self):
        """点源下两方法 aerial 一致（Hopkins 退化相干成像 = Abbe DC 源）。"""
        golden = _load("golden_point.pt")
        masks = golden["masks"]
        abbe, hopkins = _model("abbe"), _model("hopkins")
        for index in range(masks.shape[0]):
            prepared = abbe._prepare_mask(masks[index])[0]
            aerial_a = abbe._abbe_aerial(prepared, 0.0)
            aerial_h = hopkins._hopkins_aerial(prepared, 0.0)
            assert torch.allclose(aerial_a, aerial_h, rtol=1e-4, atol=1e-7), index


class TestGoldenDisk:
    """盘源两级判定：与原库 TCC 机器逐位 + R2 修正差异量化。"""

    DISK_PATTERNS = ("single_rect", "dense_lines", "l_shape", "random_blobs")

    @pytest.mark.parametrize("pattern_name", DISK_PATTERNS)
    def test_hopkins_disk_matches_original_bitwise(self, pattern_name):
        """disk 源 Hopkins 与原库 TCC 数值链逐位（compute_tcc 迁移零漂移）。"""
        golden = _load("golden_disk.pt")
        index = golden["names"].index(pattern_name)
        mask = golden["masks"][index]
        model = _model("hopkins", source_shape="disk", sigma=0.3)
        aerial = model._hopkins_aerial(model._prepare_mask(mask)[0], 0.0)
        reference = golden["hopkins_disk"][index]
        assert torch.allclose(aerial, reference, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("pattern_name", DISK_PATTERNS)
    def test_abbe_shift_fix_deviation_bounded(self, pattern_name):
        """R2 修正版 Abbe 与原库同心放大版的差异在记录的观测界内（防回归恶化）。"""
        golden = _load("golden_disk.pt")
        index = golden["names"].index(pattern_name)
        mask = golden["masks"][index]
        model = _model("abbe", source_shape="disk", sigma=0.3)
        aerial = model._abbe_aerial(model._prepare_mask(mask)[0], 0.0)
        reference = golden["abbe_orig_disk"][index]
        relative = float((aerial - reference).norm() / reference.norm())
        assert relative < 0.6, f"{pattern_name} 相对差异 {relative:.3f} 超出记录上界"


class TestResizePath:
    """resize 分支：2048nm 纯零嵌入逐位、4096nm 真插值容差。"""

    RESIZE_PATTERNS = ("single_rect", "bridge", "checkerboard")

    @pytest.mark.parametrize("pattern_name", RESIZE_PATTERNS)
    def test_padding_only_path_bitwise(self, pattern_name):
        """2048nm 视场（n=64、padding=4、无插值）与原库 cv2 恒等路径逐位。"""
        golden = _load("golden_resize.pt")
        index = golden["names"].index(pattern_name)
        mask = golden["masks_2048"][index]
        model = TorchLithoLithography(TorchLithoConfig(method="hopkins"), 256, 8.0, device="cpu")
        aerial = model._hopkins_aerial(model._prepare_mask(mask)[0], 0.0)
        assert torch.allclose(aerial, golden["hopkins_2048"][index], rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("pattern_name", RESIZE_PATTERNS)
    def test_interpolate_path_tolerance(self, pattern_name):
        """4096nm 视场（resize=2 真插值）torch 双线性与原库 cv2 差异在容差内。"""
        golden = _load("golden_resize.pt")
        index = golden["names"].index(pattern_name)
        mask = golden["masks_2048"][index]  # 两视场共用同一 256 网格图案
        model = TorchLithoLithography(TorchLithoConfig(method="hopkins"), 256, 16.0, device="cpu")
        aerial = model._hopkins_aerial(model._prepare_mask(mask)[0], 0.0)
        reference = golden["hopkins_4096"][index]
        relative = float((aerial - reference).norm() / reference.norm())
        assert relative < 0.05, f"{pattern_name} 插值相对差异 {relative:.4f} 超出容差"


class TestDiscretization:
    """单位约定保护：频率格距与瞳/源覆盖格数的数值断言。"""

    def test_frequency_quantities(self):
        """256 画布 8nm 像素：格距 1/2048、瞳半径约 14.3 格、σ=0.3 盘约 4.3 格。"""
        model = TorchLithoLithography(TorchLithoConfig(), 256, 8.0, device="cpu")
        spacing = float(model.freq_x[0, 1] - model.freq_x[0, 0])
        assert spacing == pytest.approx(1.0 / 2048.0, rel=1e-6)
        pupil_radius_cells = NA / WAVELENGTH / spacing
        assert 14.0 < pupil_radius_cells < 15.0
        disk_radius_cells = 0.3 * NA / WAVELENGTH / spacing
        assert 4.0 < disk_radius_cells < 5.0
