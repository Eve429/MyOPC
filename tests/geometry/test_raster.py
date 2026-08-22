"""规划器局部区域灰度栅格化与 PNG 输出测试。"""

from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest
from PIL import Image

import geometry.raster as raster_module
from geometry import RasterizationError, render_layout_region, render_region_batch
from layout import DbuBox, LayerSpec, LayoutDB
from tests.fixtures.layout_factory import write_advanced_layout

from .helpers import region_batch


def test_partial_coverage_and_y_axis_are_exact() -> None:
    """部分覆盖像素应为灰度，返回数组第 0 行统一对应版图最低 Y。"""
    layer = LayerSpec(1, 0)
    region = kdb.Region(kdb.Box(0, 0, 15, 10))
    batch = region_batch({layer: region}, DbuBox(0, 0, 20, 20))
    pixels = render_region_batch(batch, layer, 0.001, pixel_size_nm=10)
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[255, 128], [0, 0]]


def test_hole_overlap_merge_and_forced_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """孔洞保持为空，重叠不重复增亮，二维分块不改变最终像素。"""
    layer = LayerSpec(2, 0)
    outer = kdb.Region(kdb.Box(0, 0, 20, 20))
    donut = outer - kdb.Region(kdb.Box(5, 5, 15, 15))
    donut += kdb.Region(kdb.Box(0, 0, 5, 5))
    batch = region_batch({layer: donut}, DbuBox(0, 0, 20, 20))
    monkeypatch.setattr(raster_module, "_MAX_STRIPE_PIXELS", 3)
    pixels = render_region_batch(batch, layer, 0.001, pixel_size_nm=5)
    assert pixels.tolist() == [
        [255, 255, 255, 255],
        [255, 0, 0, 255],
        [255, 0, 0, 255],
        [255, 255, 255, 255],
    ]


def test_non_multiple_box_pads_right_and_top_with_partial_coverage() -> None:
    """查询框无法整除像素尺寸时，右侧和顶部只保留真实覆盖比例。"""
    layer = LayerSpec(1, 0)
    box = DbuBox(0, 0, 12, 7)
    batch = region_batch({layer: kdb.Region(box.to_native())}, box)
    pixels = render_region_batch(batch, layer, 0.001, pixel_size_nm=5)
    assert pixels.tolist() == [[255, 255, 102], [102, 102, 41]]


def test_cross_core_images_reassemble_without_loss_or_overlap() -> None:
    """跨 core 图形的两个局部像素图应能无损拼回完整区域。"""
    layer = LayerSpec(1, 0)
    crossing = kdb.Region(kdb.Box(25, 20, 75, 80))
    left_box, right_box = DbuBox(0, 0, 50, 100), DbuBox(50, 0, 100, 100)
    full_box = DbuBox(0, 0, 100, 100)
    left = render_region_batch(region_batch({layer: crossing}, left_box), layer, 0.001, 10)
    right = render_region_batch(region_batch({layer: crossing}, right_box), layer, 0.001, 10)
    full = render_region_batch(region_batch({layer: crossing}, full_box), layer, 0.001, 10)
    assert np.array_equal(np.column_stack((left, right)), full)


def test_png_save_and_optional_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """返回值保持模型方向，PNG 与查看器只在显示边界翻为顶部原点。"""
    layer = LayerSpec(3, 1)
    # 只填充 ROI 下半部，确保返回数组和保存图片的上下方向可被真实区分。
    batch = region_batch({layer: kdb.Region(kdb.Box(0, 0, 10, 10))}, DbuBox(0, 0, 10, 20))
    shown: list[str | None] = []
    shown_pixels: list[np.ndarray] = []

    def capture_show(image: Image.Image, title: str | None = None) -> None:
        """记录查看器收到的图片方向，避免测试依赖桌面环境。"""
        shown.append(title)
        shown_pixels.append(np.asarray(image).copy())

    monkeypatch.setattr(Image.Image, "show", capture_show)
    output = tmp_path / "roi.png"
    expected = render_region_batch(batch, layer, 0.001, pixel_size_nm=5, output_path=output, show=True)
    with Image.open(output) as image:
        assert image.mode == "L"
        assert np.array_equal(np.asarray(image), np.flipud(expected))
    assert shown == ["Layer 3/1"]
    assert np.array_equal(shown_pixels[0], np.flipud(expected))


def test_layout_convenience_function_uses_existing_database(tmp_path: Path) -> None:
    """高层函数应复用已打开数据库，并完成层级 ROI 提取与栅格化。"""
    source = write_advanced_layout(tmp_path / "advanced.gds")
    layer = LayerSpec(1, 0)
    with LayoutDB.open(source) as database:
        pixels = render_layout_region(database, DbuBox(-100, -50, 100, 50), layer, pixel_size_nm=10)
    assert pixels.shape == (10, 20)
    assert np.all(pixels == 255)


@pytest.mark.parametrize(
    ("dbu_um", "pixel_nm", "max_pixels"),
    [
        (0.0, 5.0, 100),
        (0.001, 0.0, 100),
        (0.001, 2.5, 100),
        (0.001, 5.0, 0),
    ],
)
def test_invalid_sampling_parameters_are_rejected(dbu_um: float, pixel_nm: float, max_pixels: int) -> None:
    """非法物理采样或内存上限必须在分配像素数组前失败。"""
    layer = LayerSpec(1, 0)
    batch = region_batch({layer: kdb.Region()}, DbuBox(0, 0, 10, 10))
    with pytest.raises(RasterizationError):
        render_region_batch(batch, layer, dbu_um, pixel_nm, max_pixels=max_pixels)


def test_missing_layer_size_guard_and_output_rules(tmp_path: Path) -> None:
    """缺失 Layer、超限图片和错误输出位置应返回清晰领域错误。"""
    layer = LayerSpec(1, 0)
    batch = region_batch({layer: kdb.Region()}, DbuBox(0, 0, 100, 100))
    with pytest.raises(RasterizationError, match="不包含图层"):
        render_region_batch(batch, LayerSpec(2, 0), 0.001)
    with pytest.raises(RasterizationError, match="超过上限"):
        render_region_batch(batch, layer, 0.001, max_pixels=100)
    with pytest.raises(RasterizationError, match="扩展名"):
        render_region_batch(batch, layer, 0.001, output_path=tmp_path / "roi.jpg")
    with pytest.raises(FileNotFoundError):
        render_region_batch(batch, layer, 0.001, output_path=tmp_path / "missing" / "roi.png")
