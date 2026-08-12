"""验证 DiffOPC 的软边数学、流式归属、拓扑屏障和正式入口。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import klayout.db as kdb
import numpy as np
import pytest
import torch

from layout import LayerSpec
from lithography import ProcessCondition
from main.run_diffopc import run_diffopc
from opc.errors import ReconstructionError
from opc.input import RectilinearCoreGrid
from opc.input.edge import FragmentationConfig, prepare_problem, reconstruct_region
from opc.iteration.diffopc import DiffOPCConfig, optimize
from opc.iteration.diffopc.rasterizer import rasterize_soft_edges
from opc.iteration.diffopc.solver import _sample_probe

from .test_common import _batch


class _ScaledIdentityModel:
    """用可微缩放代替 Hopkins 传播，以隔离边段、归属和流式反传语义。"""

    def __init__(self, canvas: int = 128, scale: float = 0.8) -> None:
        """保存 CPU 设备、画布、阈值及确定性缩放系数。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas, print_threshold=0.5)
        self.scale = scale

    def condition(self, name: str) -> ProcessCondition:
        """返回满足求解器命名约定的独立伪工艺条件。"""
        return ProcessCondition(name, "focus", 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """对所有工艺角返回同一可微缩放图，令 PVBand 严格为零。"""
        return {condition.name: mask * self.scale for condition in conditions}


def _problem(region: kdb.Region, grid: RectilinearCoreGrid,
             maximum: float = 8.0) -> object:
    """用紧凑分段参数构造专项测试共享的边段问题。"""
    batch = _batch(region)
    return prepare_problem(
        batch, batch.layers[0], FragmentationConfig(4, 16, maximum), grid)


def _config(*, iterations: int = 2, batch_size: int = 1) -> DiffOPCConfig:
    """返回适合 128² CPU 假模型的确定性 DiffOPC 配置。"""
    return DiffOPCConfig(
        iterations=iterations, learning_rate=0.5, soft_temperature=1.5,
        weight_l2=1.0, weight_pvband=0.0, weight_epe=0.0,
        max_displacement_dbu=8.0, epe_distance_dbu=4.0,
        pixel_dbu=1, canvas=128, batch_size=batch_size,
        raster_chunk_size=4, target_cache_bytes=1 << 20)


def test_soft_raster_zero_is_exact_and_gradient_matches_finite_difference() -> None:
    """零位移必须逐像素等于参考图，且自动梯度应匹配中心有限差分。"""
    reference = np.zeros((24, 24), dtype=np.float32)
    reference[:, :12] = 1.0
    starts = np.array([[12.0, 2.0]])
    ends = np.array([[12.0, 22.0]])
    normals = np.array([[1.0, 0.0]])
    displacement = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    result = rasterize_soft_edges(
        reference, starts, ends, normals, displacement,
        pixel_dbu=1, temperature=1.2, origin_dbu=(0, 0), chunk_size=1)
    np.testing.assert_array_equal(result.detach().numpy(), reference)
    displacement = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    result = rasterize_soft_edges(
        reference, starts, ends, normals, displacement,
        pixel_dbu=1, temperature=1.2, origin_dbu=(0, 0), chunk_size=1)
    weights = torch.linspace(-0.4, 0.7, result.numel()).reshape_as(result)
    loss = torch.sum(result * weights)
    loss.backward()
    epsilon = 1e-3

    def value(offset: float) -> float:
        """返回给定位移下同一加权像素目标，供有限差分对照。"""
        moved = rasterize_soft_edges(
            reference, starts, ends, normals,
            torch.tensor([offset], dtype=torch.float64),
            pixel_dbu=1, temperature=1.2, origin_dbu=(0, 0), chunk_size=1)
        return float(torch.sum(moved * weights).item())

    numeric = (value(0.6 + epsilon) - value(0.6 - epsilon)) / (2.0 * epsilon)
    assert float(displacement.grad.item()) == pytest.approx(numeric, rel=3e-3, abs=3e-3)


def test_soft_raster_outward_motion_adds_outer_occupancy_for_hull_and_hole() -> None:
    """外轮廓与孔洞法向相反时，正位移都应沿各自外法向增加占据。"""
    reference = np.zeros((32, 32), dtype=np.float32)
    hull = rasterize_soft_edges(
        reference, [[16.0, 4.0]], [[16.0, 28.0]], [[1.0, 0.0]],
        torch.tensor([2.0]), pixel_dbu=1, temperature=1.0,
        origin_dbu=(0, 0), chunk_size=1)
    hole = rasterize_soft_edges(
        reference, [[16.0, 4.0]], [[16.0, 28.0]], [[-1.0, 0.0]],
        torch.tensor([2.0]), pixel_dbu=1, temperature=1.0,
        origin_dbu=(0, 0), chunk_size=1)
    # y=16 位于端帽内部；外法向翻转后，增加 occupancy 的一侧也必须翻转。
    assert hull[16, 17] > hull[16, 14]
    assert hole[16, 14] > hole[16, 17]


def test_probe_sampling_uses_raster_pixel_centers() -> None:
    """位于第一个像素中心的 DBU probe 必须精确采到 [0,0] 像素。"""
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    sampled = _sample_probe(
        image, np.array([[11.0, 21.0], [13.0, 23.0]]),
        torch.tensor([10.0, 20.0]), 2)
    assert sampled.tolist() == pytest.approx([0.0, 5.0])


def test_streaming_batch_size_does_not_change_optimized_state() -> None:
    """逐 batch 反传释放图后，batch=1 与 batch=2 应发布相同全局位移。"""
    region = kdb.Region(kdb.Box(10, 10, 90, 70))
    grid = RectilinearCoreGrid(
        np.array([0, 50, 100]), np.array([0, 80]), 16)
    problem = _problem(region, grid)
    first = optimize(problem, _ScaledIdentityModel(), _config(batch_size=1))
    second = optimize(problem, _ScaledIdentityModel(), _config(batch_size=2))
    np.testing.assert_allclose(
        first.best_displacements, second.best_displacements, atol=1e-6, rtol=0.0)
    assert [record.total_loss for record in first.records] == pytest.approx(
        [record.total_loss for record in second.records], rel=1e-6, abs=1e-7)


def test_owner_scoring_is_invariant_to_one_or_two_core_partition() -> None:
    """同一物理像素只计分一次，1-core/2-core 切分不得改变全局更新。"""
    region = kdb.Region(kdb.Box(10, 10, 90, 70))
    one_core = _problem(
        region, RectilinearCoreGrid(
            np.array([0, 100]), np.array([0, 80]), 0))
    two_cores = _problem(
        region, RectilinearCoreGrid(
            np.array([0, 50, 100]), np.array([0, 80]), 20))
    first = optimize(one_core, _ScaledIdentityModel(), _config(batch_size=1))
    second = optimize(two_cores, _ScaledIdentityModel(), _config(batch_size=1))
    # 两个问题由相同参考轮廓和 fragmentation 生成，全局 segment 顺序一致；
    # 变化的只有 owner 与 halo membership，正好隔离 tile 划分不变量。
    assert np.array_equal(
        one_core.segments.edge_ids, two_cores.segments.edge_ids)
    np.testing.assert_allclose(
        first.best_displacements, second.best_displacements, atol=2e-6, rtol=0.0)
    assert [record.total_loss for record in first.records] == pytest.approx(
        [record.total_loss for record in second.records], rel=2e-5, abs=2e-7)


def test_best_displacements_match_same_recorded_snapshot() -> None:
    """最佳记录若来自初态，保存位移也必须是初态而非其后的 Adam 候选。"""
    problem = _problem(
        kdb.Region(kdb.Box(10, 10, 90, 70)),
        RectilinearCoreGrid(np.array([0, 100]), np.array([0, 80])))
    result = optimize(problem, _ScaledIdentityModel(), _config(iterations=2))
    assert len(result.records) == 2
    assert result.records[0].displaced_segments == 0
    assert result.records[1].displaced_segments > 0
    expected = min(result.records, key=lambda record: record.total_loss)
    assert result.best_iteration == expected.iteration
    assert np.count_nonzero(result.best_displacements) == expected.displaced_segments


def test_hole_diagonal_and_cross_core_candidates_reconstruct_validly() -> None:
    """孔洞、斜边和跨 core segment 的小位移必须保持统一全局合法几何。"""
    diagonal = kdb.Polygon([
        kdb.Point(8, 8), kdb.Point(90, 18),
        kdb.Point(82, 86), kdb.Point(18, 76)])
    region = (kdb.Region(diagonal) - kdb.Region(kdb.Box(35, 35, 60, 58)))
    grid = RectilinearCoreGrid(
        np.array([0, 48, 100]), np.array([0, 50, 100]), 20)
    problem = _problem(region, grid, maximum=4.0)
    values = np.full(problem.segments.segment_count, 0.5)
    reconstructed = reconstruct_region(problem, values)
    assert reconstructed.has_valid_polygons()
    crossing = [index for index in range(problem.segments.segment_count)
                if np.count_nonzero(problem.member_segment_indices == index) > 1]
    assert crossing
    assert all(0 <= problem.owner_indices[index] < problem.core_count
               for index in crossing)


def test_invalid_candidate_is_rolled_back_before_publication(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """全局重建拒绝候选时应停止并保留最后一个已评价合法状态。"""
    import opc.iteration.diffopc.solver as solver

    problem = _problem(
        kdb.Region(kdb.Box(10, 10, 90, 70)),
        RectilinearCoreGrid(np.array([0, 100]), np.array([0, 80])))

    def reject(*args: object, **kwargs: object) -> None:
        """模拟对边穿越或孔洞越界产生的全局重建失败。"""
        raise ReconstructionError("synthetic invalid candidate")

    monkeypatch.setattr(solver, "reconstruct_region", reject)
    result = solver.optimize(problem, _ScaledIdentityModel(), _config(iterations=3))
    assert result.stop_reason == "invalid_geometry"
    assert len(result.records) == 1
    assert np.count_nonzero(result.best_displacements) == 0


@pytest.mark.parametrize("changes", [
    {"iterations": 0}, {"epe_distance_dbu": 0.0},
    {"raster_chunk_size": 0}, {"target_cache_bytes": -1},
    {"weight_l2": 0.0, "weight_pvband": 0.0, "weight_epe": 0.0},
])
def test_diffopc_config_rejects_invalid_resources(changes: dict[str, object]) -> None:
    """无效轮数、探针、chunk、缓存及全零损失必须在求解前拒绝。"""
    values = dict(
        iterations=2, epe_distance_dbu=4.0, raster_chunk_size=4,
        target_cache_bytes=1024, weight_l2=1.0,
        weight_pvband=0.0, weight_epe=0.0)
    values.update(changes)
    with pytest.raises(ValueError):
        DiffOPCConfig(**values)


def test_direct_gds_runner_saves_geometry_metrics_and_final_lithography(
        tmp_path: Path) -> None:
    """正式入口应直接消费 GDS，并保存位移、GDS、JSON 与最终光刻 tile。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top, layer_index = layout.create_cell("TOP"), layout.layer(1, 0)
    top.shapes(layer_index).insert(kdb.Box(16, 16, 96, 80))
    source = tmp_path / "direct.gds"
    layout.write(str(source))
    output = tmp_path / "output"
    summary = run_diffopc(
        source, output, iterations=1, pixel_nm=1.0,
        tile_size_nm=128.0, halo_nm=32.0, corner_nm=8.0,
        segment_nm=16.0, max_displacement_nm=8.0,
        epe_distance_nm=4.0, batch_size=1, raster_chunk_size=4,
        device="cpu", layer=LayerSpec(1, 0), box=(0, 0, 128, 128),
        save_preview=False, save_final_lithography_png=False)
    assert summary["status"] == "completed"
    assert summary["verification"] == {"reconstructed_valid": True}
    assert summary["optimization"]["records"][0]["l2"] >= 0
    assert summary["optimization"]["records"][0]["pvband"] >= 0
    assert summary["optimization"]["records"][0]["epe"] >= 0
    assert Path(summary["artifacts"]["summary"]).is_file()
    assert Path(summary["artifacts"]["result_npz"]).is_file()
    assert Path(summary["artifacts"]["gds"]).is_file()
    manifest = summary["artifacts"]["final_lithography"]
    assert Path(manifest["manifest"]).is_file()
    assert manifest["tile_count"] == 1
