"""二值 L2、PVBand、边段 EPE 与光刻契约的生成式测试。"""

import pytest
import torch

from evaluation import (
    EPEEvaluation,
    evaluate_binary_l2,
    evaluate_edge_probes,
    evaluate_pvband,
)
from lithography import ICCAD13Lithography, LithographyModel
from lithography.contracts import LithographyConfigView


class TestBinaryL2:
    """evaluate_binary_l2 的计数、阈值与 ownership 屏蔽。"""

    def test_single_image_counts_mismatched_pixels(self):
        """[H,W] 输入统计二值不一致像素数，阈值语义为大于等于。"""
        target = torch.zeros((4, 5))
        target[1:3, 1:4] = 1.0  # 目标图形
        nominal = target.clone()
        nominal[2, 2] = 0.0  # 缺一个像素
        nominal[3, 0] = 1.0  # 多一个像素
        assert evaluate_binary_l2(target, nominal) == 2  # 两处不一致

    def test_threshold_semantics_is_inclusive(self):
        """恰好等于阈值的值按已打印处理（>= 而不是 >）。"""
        target = torch.full((2, 2), 0.5)  # 恰在阈值
        nominal = torch.full((2, 2), 0.4999)  # 阈值之下
        # target>=0.5 为 True，nominal 为 False → 全部不一致
        assert evaluate_binary_l2(target, nominal, threshold=0.5) == 4

    def test_batch_counts_across_images(self):
        """[B,H,W] 输入跨 batch 累计不一致数。"""
        target = torch.zeros((3, 4, 4))
        nominal = target.clone()
        nominal[0, 0, 0] = 1.0  # 第一张 1 处
        nominal[1, 1, 1] = 1.0  # 第二张 1 处
        nominal[1, 2, 2] = 1.0  # 第二张再 1 处
        assert evaluate_binary_l2(target, nominal) == 3  # 1+2 累计

    def test_identical_images_count_zero(self):
        """完全一致的连续值图计 0（含同侧中间值）。"""
        image = torch.rand((2, 6, 6))
        assert evaluate_binary_l2(image, image.clone()) == 0

    def test_ownership_mask_excludes_non_owned_pixels(self):
        """ownership 为 False 区域的不一致不参与计数。"""
        target = torch.zeros((4, 4))
        nominal = torch.ones((4, 4))  # 全图不一致
        ownership = torch.ones((4, 4), dtype=torch.bool)
        ownership[:, 2:] = False  # 右半不属于本 core
        assert evaluate_binary_l2(target, nominal,
                                  ownership_mask=ownership) == 8  # 左半

    def test_2d_ownership_accepts_single_image_batch_only(self):
        """[H,W] ownership 只与单图 batch 对齐；B>1 时必须显式给 [B,H,W]。"""
        target = torch.zeros((1, 4, 4))
        nominal = torch.ones((1, 4, 4))
        ownership = torch.zeros((4, 4), dtype=torch.bool)
        ownership[0, 0] = True  # 只统计 1 个像素
        assert evaluate_binary_l2(target, nominal,
                                  ownership_mask=ownership) == 1
        with pytest.raises(ValueError, match="ownership_mask"):
            evaluate_binary_l2(  # 两张图不能共用一张 2D ownership
                torch.zeros((2, 4, 4)), torch.zeros((2, 4, 4)),
                ownership_mask=ownership)

    def test_shape_mismatch_fails(self):
        """target 与 nominal 形状不一致时失败。"""
        with pytest.raises(ValueError, match="形状和设备必须一致"):
            evaluate_binary_l2(torch.zeros((4, 5)), torch.zeros((4, 6)))

    def test_wrong_ndim_fails(self):
        """四维输入不是 [H,W]/[B,H,W]，失败。"""
        with pytest.raises(ValueError, match="形状"):
            evaluate_binary_l2(torch.zeros((2, 3, 4, 5)),
                               torch.zeros((2, 3, 4, 5)))

    def test_ownership_shape_mismatch_fails(self):
        """ownership 形状与评价图不一致时失败。"""
        with pytest.raises(ValueError, match="ownership_mask"):
            evaluate_binary_l2(torch.zeros((1, 4, 5)),
                               torch.zeros((1, 4, 5)),
                               ownership_mask=torch.ones((3, 4), dtype=torch.bool))


class TestPVBand:
    """evaluate_pvband 的工艺带计数与 ownership 屏蔽。"""

    def test_band_counts_condition_disagreement(self):
        """最大/最小工艺条件二值不一致的像素计入带。"""
        maximum = torch.zeros((4, 4))
        minimum = torch.zeros((4, 4))
        maximum[0:2, :] = 1.0  # 大剂量多印两行
        assert evaluate_pvband(maximum, minimum) == 8

    def test_identical_conditions_count_zero(self):
        """两条件结果一致时带为 0。"""
        image = torch.rand((5, 5))
        assert evaluate_pvband(image, image.clone()) == 0

    def test_ownership_excludes_outside(self):
        """ownership 外的带差异不计。"""
        maximum = torch.ones((4, 4))
        minimum = torch.zeros((4, 4))
        ownership = torch.ones((4, 4), dtype=torch.bool)
        ownership[1:, :] = False  # 只统计首行
        assert evaluate_pvband(maximum, minimum,
                               ownership_mask=ownership) == 4

    def test_shape_mismatch_fails(self):
        """maximum 与 minimum 形状不一致时失败。"""
        with pytest.raises(ValueError, match="形状和设备必须一致"):
            evaluate_pvband(torch.zeros((3, 3)), torch.zeros((3, 4)))


class TestEdgeProbes:
    """evaluate_edge_probes 的方向、有效性与批量评价。"""

    def _canvas_pair(self, height=8, width=8):
        """构造全零 target/nominal 对，测试内直接摆放探针值。"""
        target = torch.zeros((1, height, width))
        nominal = torch.zeros((1, height, width))
        return target, nominal

    def test_four_direction_cases(self):
        """四种 nominal 情形分别产生 +1/-1/0(ambiguous)/0(无违规)。"""
        target, nominal = self._canvas_pair()
        # 四个位置对：inner 在透光侧、outer 在不透光侧（target 语义有效）
        pairs = [((2, 2), (3, 2)), ((2, 5), (3, 5)),
                 ((5, 2), (6, 2)), ((5, 5), (6, 5))]
        for inner, outer in pairs:
            target[0, inner[1], inner[0]] = 1.0
        # 段 0：印刷不足（inner/outer 都未打印）→ +1 外移（全零默认即是）
        # 段 1：印刷过量（都打印）→ -1 内移
        nominal[0, 5, 2] = 1.0  # 段 1 的 inner(x=2,y=5) 已打印
        nominal[0, 5, 3] = 1.0  # 段 1 的 outer(x=3,y=5) 也打印
        # 段 2：双向冲突（inner 未打印、outer 打印）→ 0 歧义
        nominal[0, 2, 6] = 1.0
        # 段 3：无违规（inner 打印、outer 未打印）→ 0
        nominal[0, 5, 5] = 1.0
        batches = torch.zeros(4, dtype=torch.long)
        inner_xy = torch.tensor([[p[0][0], p[0][1]] for p in pairs],
                                dtype=torch.float64)
        outer_xy = torch.tensor([[p[1][0], p[1][1]] for p in pairs],
                                dtype=torch.float64)
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert isinstance(result, EPEEvaluation)  # 返回评价结构
        assert torch.all(result.valid)  # 四段全部有效
        assert result.directions.tolist() == [1, -1, 0, 0]  # 方向表
        assert result.ambiguous.tolist() == [False, False, True, False]
        assert result.violation_count == 3  # 歧义段也计入违规数

    def test_out_of_bounds_probe_is_invalid(self):
        """round 后越界的探针坐标无效且方向为 0。"""
        target, nominal = self._canvas_pair()
        target[0, 2, 2] = 1.0  # 唯一透光点
        batches = torch.tensor([0, 0])
        inner_xy = torch.tensor([[-0.6, 2.0], [2.0, 2.0]])  # x round 到 -1
        outer_xy = torch.tensor([[3.0, 2.0], [8.4, 2.0]])  # 第二段 x=8 越界
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert result.valid.tolist() == [False, False]
        assert result.directions.tolist() == [0, 0]  # 无效不产生方向

    def test_batch_index_out_of_range_is_invalid(self):
        """batch 索引越界的探针无效。"""
        target, nominal = self._canvas_pair()
        target[0, 2, 2] = 1.0
        batches = torch.tensor([1])  # 只有 1 张图
        inner_xy = torch.tensor([[2.0, 2.0]])
        outer_xy = torch.tensor([[3.0, 2.0]])
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert not bool(result.valid[0])

    def test_coincident_probes_are_invalid(self):
        """inner 与 outer 像素相同时无效（distinct 检查）。"""
        target, nominal = self._canvas_pair()
        target[0, 2, 2] = 1.0
        batches = torch.tensor([0])
        inner_xy = torch.tensor([[2.4, 2.4]])  # round 到 (2,2)
        outer_xy = torch.tensor([[2.4, 2.4]])  # 同一像素
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert not bool(result.valid[0])

    def test_target_semantics_must_hold(self):
        """target 上 inner 不透光或 outer 透光的探针无效。"""
        target, nominal = self._canvas_pair()
        batches = torch.tensor([0, 0])
        # 段 0：inner 处 target=0（不透光）→ 无效
        inner_xy = torch.tensor([[2.0, 2.0], [5.0, 5.0]])
        outer_xy = torch.tensor([[3.0, 2.0], [6.0, 5.0]])
        target[0, 5, 5] = 1.0  # 只给段 1 的 inner 透光
        target[0, 5, 6] = 1.0  # 段 1 的 outer 也透光 → 无效
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert result.valid.tolist() == [False, False]
        assert result.directions.tolist() == [0, 0]

    def test_coordinates_round_to_nearest_pixel(self):
        """连续坐标 (x,y) 四舍五入到最近像素，x 是列、y 是行。"""
        target, nominal = self._canvas_pair()
        target[0, 4, 2] = 1.0  # (x=2, y=4) 透光
        nominal[0, 4, 2] = 1.0  # 已正确打印
        batches = torch.tensor([0])
        inner_xy = torch.tensor([[2.4, 3.6]])  # round → (2,4)
        outer_xy = torch.tensor([[5.4, 3.6]])  # round → (5,4)
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert bool(result.valid[0])  # 取到了正确的像素
        assert result.violation_count == 0  # 无违规

    def test_default_threshold_is_half(self):
        """默认阈值 0.5（与 L2/PVBand 统一）：0.4995 视为未打印，显式 0.499 下已打印。"""
        target, nominal = self._canvas_pair()
        target[0, 2, 2] = 1.0
        nominal[0, 2, 2] = 0.4995  # 介于两个阈值之间
        batches = torch.tensor([0])
        inner_xy = torch.tensor([[2.0, 2.0]])
        outer_xy = torch.tensor([[3.0, 2.0]])
        default = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert default.directions.tolist() == [1]  # 0.5 下未打印 → 外移
        legacy = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy, threshold=0.499)
        assert legacy.directions.tolist() == [0]  # 0.499 下已打印 → 无违规

    def test_batch_indices_select_images(self):
        """同一探针坐标按 batch_indices 取不同图的评价。"""
        target = torch.zeros((2, 8, 8))
        nominal = torch.zeros((2, 8, 8))
        target[0, 2, 2] = target[1, 2, 2] = 1.0  # 两张同 target
        nominal[0, 2, 2] = 1.0  # 第 0 张正确打印
        # 第 1 张 inner 未打印 → 印刷不足
        batches = torch.tensor([0, 1])
        inner_xy = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        outer_xy = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
        result = evaluate_edge_probes(
            target, nominal, batches, inner_xy, outer_xy)
        assert result.directions.tolist() == [0, 1]

    def test_misaligned_probe_shapes_fail(self):
        """探针数量或坐标列数不对齐时失败。"""
        target, nominal = self._canvas_pair()
        with pytest.raises(ValueError, match="对齐"):
            evaluate_edge_probes(
                target, nominal, torch.tensor([0, 0]),
                torch.zeros((1, 2)), torch.zeros((1, 2)))  # 数量 2 对 1
        with pytest.raises(ValueError, match="对齐"):
            evaluate_edge_probes(
                target, nominal, torch.tensor([0]),
                torch.zeros((1, 3)), torch.zeros((1, 3)))  # 3 列坐标

    def test_device_mismatch_fails(self):
        """target 与 nominal 设备不一致时失败（需要 CUDA）。"""
        if not torch.cuda.is_available():
            pytest.skip("当前环境没有 CUDA")
        with pytest.raises(ValueError, match="设备必须一致"):
            evaluate_edge_probes(
                torch.zeros((1, 4, 4)), torch.zeros((1, 4, 4)).cuda(),
                torch.tensor([0]), torch.zeros((1, 2)), torch.zeros((1, 2)))


class TestLithographyContract:
    """ICCAD13 实现满足求解器消费的 LithographyModel 契约。"""

    def test_iccad13_satisfies_model_protocol(self):
        """ICCAD13Lithography 可作为 LithographyModel 结构化通过。"""
        model = ICCAD13Lithography(device="cpu")
        assert isinstance(model, LithographyModel)  # 运行时结构检查

    def test_iccad13_config_satisfies_view_protocol(self):
        """ICCAD13Config 暴露契约要求的 canvas 与 print_threshold。"""
        model = ICCAD13Lithography(device="cpu")
        assert isinstance(model.config, LithographyConfigView)
        assert model.config.canvas == 256  # 冻结画布
        assert model.config.print_threshold == pytest.approx(0.5)
