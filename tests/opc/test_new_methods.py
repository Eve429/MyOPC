"""验证新增 LevelSet、多尺度 ILT 与 DiffOPC 基础契约。"""

from pathlib import Path

import numpy as np
import torch

from layout import DbuBox, LayerSpec
from lithography import ICCAD13Lithography
from main.offline_inputs import prepare_segment_input
from opc.iteration.diffopc import DiffOPCConfig
from opc.iteration.diffopc import optimize as optimize_diffopc
from opc.iteration.diffopc.rasterizer import rasterize_soft_edges
from opc.iteration.ilt import (
    CurvMultiConfig,
    LevelSetConfig,
    optimize_curvmulti,
    optimize_levelset,
)


def test_soft_edge_rasterizer_has_finite_gradient() -> None:
    """解析软边段栅格器必须能对位移反向且梯度有限。"""
    displacement = torch.zeros(1, requires_grad=True)
    result = rasterize_soft_edges(
        np.zeros((8, 8), dtype=np.float32), [[4.0, 0.0]], [[4.0, 8.0]],
        [[1.0, 0.0]], displacement, pixel_dbu=1, temperature=1.0,
        origin_dbu=(0, 0))
    result.sum().backward()
    assert torch.isfinite(displacement.grad)


def test_levelset_and_curvmulti_return_common_result() -> None:
    """LevelSet 与 CurvMultiILT 应返回统一结果字段。"""
    model = ICCAD13Lithography(device="cpu")
    target = torch.zeros((16, 16), dtype=torch.float32)
    levelset = optimize_levelset(target, model, LevelSetConfig(iterations=1, step_size=0.1))
    curvmulti = optimize_curvmulti(
        target, model, CurvMultiConfig((2, 1), 1, 0.1, curvature_weight=0.0))
    for result in (levelset, curvmulti):
        assert result.binary_mask.shape == target.shape
    assert len(levelset.records) == 1
    assert len(curvmulti.records) == 2


def test_diffopc_runs_on_saved_problem(tmp_path: Path) -> None:
    """DiffOPC 应能消费现有 segment archive 并保持精确 Region 合法。"""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.001
    layer = layout.layer(1, 0)
    top = layout.create_cell("TOP")
    top.shapes(layer).insert(kdb.Box(16, 16, 96, 96))
    source = tmp_path / "diff.gds"
    layout.write(str(source))
    archive = prepare_segment_input(
        source, tmp_path / "input.npz", layer=LayerSpec(1, 0),
        box=DbuBox(0, 0, 128, 128), tile_size_nm=128, halo_nm=0,
        corner_nm=8, segment_nm=16, max_displacement_nm=8)
    from main.offline_inputs import load_segment_input
    problem, _ = load_segment_input(archive)
    result = optimize_diffopc(problem, ICCAD13Lithography(device="cpu"),
                               DiffOPCConfig(
                                   iterations=1, pixel_dbu=1, canvas=128,
                                   max_displacement_dbu=8))
    assert result.best_displacements.shape == (problem.segments.segment_count,)
