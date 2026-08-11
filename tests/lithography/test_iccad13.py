"""ICCAD13 Hopkins 光刻模型的配置、数值和直接运行测试。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lithography import ICCAD13Config, ICCAD13Lithography


def _model_mask(height: int = 200, width: int = 150) -> torch.Tensor:
    """构造同时包含实心矩形和孔洞的确定性 batch mask。"""
    mask = torch.zeros((2, height, width), dtype=torch.float32)
    mask[0, 40:160, 40:110] = 1.0
    mask[1, 20:180, 20:130] = 1.0
    mask[1, 70:130, 60:90] = 0.0
    return mask


def _legacy_aerial(model: ICCAD13Lithography, mask: torch.Tensor, dose: float,
                   kernels: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """按优化前每个工艺角独立 FFT 的公式计算数值参考。"""
    spectrum = torch.fft.fft2((dose * mask).to(torch.complex64).unsqueeze(1), norm="forward")
    fields = torch.fft.ifft2(
        model._kernel_multiply(kernels, spectrum, model.config.kernel_count), norm="forward")
    weights = scales[:model.config.kernel_count][None, :, None, None]
    return torch.sum(weights * torch.abs(fields).square(), dim=1)


def test_config_parser_validates_required_fields(tmp_path: Path) -> None:
    """配置解析必须接受标准文件并拒绝缺字段或错误剂量顺序。"""
    source = Path(__file__).resolve().parents[2] / "lithography" / "config" / "iccad13.txt"
    config = ICCAD13Config.from_file(source)
    assert (config.kernel_count, config.canvas, config.resolution) == (24, 256, 256)
    missing = tmp_path / "missing.txt"
    missing.write_text("KernelNum 24\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少字段"):
        ICCAD13Config.from_file(missing)
    invalid = tmp_path / "invalid.txt"
    invalid.write_text(source.read_text(encoding="utf-8").replace(
        "DoseMin 0.98", "DoseMin 1.03"), encoding="utf-8")
    with pytest.raises(ValueError, match="剂量顺序"):
        ICCAD13Config.from_file(invalid)


def test_openilt_assets_have_fixed_hashes() -> None:
    """迁移的四个 MIT 资产必须保持与已验证 OpenILT 文件逐字节一致。"""
    asset_dir = Path(__file__).resolve().parents[2] / "lithography" / "assets" / "iccad13"
    expected = {
        "focus.pt": "204bee39d8225c6d3cda52ea2d13b7c6f6cf4e4244de2ce960576d1bc741438f",
        "defocus.pt": "df624de9e17485d819e488ccada7edff133690cfa01370bfabec8f9e7cb8d532",
        "focus_scale.pt": "4e6f6136d419bdf0b56e9b461471c72d97ab3ba582fa19fe65ffaf25d188dab6",
        "defocus_scale.pt": "4ce70debf23593594c2fda1bd0cadf427abb0c132b16a047f6589f51148c8dc8",
    }
    actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in asset_dir.glob("*.pt")}
    assert actual == expected


def test_cpu_forward_supports_batch_and_matches_single_images() -> None:
    """CPU batch 输出应保持输入尺寸、连续范围并与逐张仿真一致。"""
    model = ICCAD13Lithography(device="cpu")
    mask = _model_mask()
    with torch.no_grad():
        batch = model(mask)
        first = model(mask[0])
    assert batch.nominal.shape == mask.shape
    assert torch.all((batch.minimum >= 0.0) & (batch.maximum <= 1.0))
    torch.testing.assert_close(batch.nominal[0], first.nominal, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(batch.maximum[0], first.maximum, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(batch.minimum[0], first.minimum, rtol=0.0, atol=1e-6)
    # 这些基线来自同一资产上的 OpenILT `opc/iccad13.py`，200×150 输入避开其
    # 满 canvas 时错误返回原 mask 的 unpad 分支；三工艺角逐像素最大差实测为 0。
    expected_sums = torch.tensor([25802.533203125, 26009.16796875, 25675.23828125])
    actual_sums = torch.stack((batch.nominal.sum(), batch.maximum.sum(), batch.minimum.sum()))
    torch.testing.assert_close(actual_sums, expected_sums, rtol=0.0, atol=0.05)


def test_full_canvas_runs_lithography_instead_of_returning_input() -> None:
    """满 canvas 输入仍必须返回连续曝光图，回归 OpenILT 原实现的 unpad 错误。"""
    model = ICCAD13Lithography(device="cpu")
    mask = torch.zeros((256, 256), dtype=torch.float32)
    mask[80:176, 96:160] = 1.0
    with torch.no_grad():
        result = model(mask)
    assert not torch.equal(result.nominal, mask)
    assert torch.any((result.nominal > 0.0) & (result.nominal < 1.0))


def test_shared_spectrum_matches_independent_fft_reference() -> None:
    """共享频谱的三工艺角结果与原独立 FFT 公式逐像素误差不得超过 5e-6。"""
    model = ICCAD13Lithography(device="cpu")
    mask = _model_mask(72, 64)[:1]
    prepared, padding = model._prepare_mask(mask)
    steepness, density = model.config.print_steepness, model.config.target_density
    with torch.no_grad():
        result = model(mask)
        aerials = (
            _legacy_aerial(model, prepared, model.config.dose_nominal,
                           model.focus_kernels, model.focus_scales),
            _legacy_aerial(model, prepared, model.config.dose_max,
                           model.focus_kernels, model.focus_scales),
            _legacy_aerial(model, prepared, model.config.dose_min,
                           model.defocus_kernels, model.defocus_scales),
        )
        reference = tuple(model._restore_size(
            torch.sigmoid(steepness * (aerial - density)), padding) for aerial in aerials)
    for actual, expected in zip(
            (result.nominal, result.maximum, result.minimum), reference, strict=True):
        assert torch.max(torch.abs(actual - expected)).item() <= 5e-6


def test_shared_spectrum_preserves_mask_gradient() -> None:
    """共享中间量后标称与工艺窗损失仍应向输入 mask 传播有限非零梯度。"""
    model = ICCAD13Lithography(device="cpu")
    mask = _model_mask(64, 64)[:1].clone().requires_grad_()
    result = model(mask)
    loss = result.nominal.mean() + result.maximum.mean() - result.minimum.mean()
    loss.backward()
    assert mask.grad is not None
    assert torch.all(torch.isfinite(mask.grad))
    assert torch.count_nonzero(mask.grad).item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有 CUDA")
def test_direct_environment_python_loads_cuda_runtime(project_root: Path,
                                                       tmp_path: Path) -> None:
    """直接调用环境 Python 必须找到 NVRTC DLL，不依赖 conda run 或项目安装。"""
    code = (
        "import json,torch; from lithography import ICCAD13Lithography; "
        "m=ICCAD13Lithography(device='cuda'); x=torch.zeros((1,64,64),device='cuda'); "
        "y=m(x); torch.cuda.synchronize(); "
        "print(json.dumps({'shape':list(y.nominal.shape),'device':str(y.nominal.device)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=project_root, capture_output=True,
        text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"shape": [1, 64, 64], "device": "cuda:0"}


def test_model_rejects_invalid_shape_and_oversized_mask() -> None:
    """非图像张量和超过固定 canvas 的输入必须在分配频域中间量前拒绝。"""
    model = ICCAD13Lithography(device="cpu")
    with pytest.raises(ValueError, match="形状"):
        model(torch.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="超过 canvas"):
        model(torch.zeros((257, 256)))
