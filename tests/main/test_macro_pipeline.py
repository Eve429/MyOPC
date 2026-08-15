"""双轮 macro-core 迭代管线的配置校验与端到端生成式测试。"""

import shutil

import klayout.db as kdb
import numpy as np
import pytest

import main.run_macro_pipeline as pipeline
from layout import DbuBox, LayoutDB
from opc.input.edge import MacroProblem, reconstruct_region

# 测试版图：DBU=1nm，bar 图形使目标层 bbox 为 (20,20)-(140,60)，
# 2×2 macro、core 30、context 10、pixel 1 恰好满足全部网格契约。


def _write_gds(tmp_path):
    """生成单层 bar 图形的 GDS 并返回路径。"""
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU，配置数值直接等于 DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(20, 20, 140, 60))  # 目标层 bar
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_config(tmp_path, layout_path, **overrides):
    """按默认契约生成 TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值全部满足网格与分段契约
        "macro_grid": "[2, 2]", "core_size_nm": 30, "context_nm": 10,
        "pixel_nm": 1, "corner_nm": 4, "segment_nm": 10,
        "max_displacement_nm": 8, "miter_limit": 4.0, "deltas": "[2, -2]"}
    values.update(overrides)  # 应用覆盖
    text = f"""  # 组装 TOML 文本
[input]
layout = "{layout_path.as_posix()}"
top_cell = "TOP"
layer = 1
datatype = 0
polarity = "clear"

[grid]
macro_grid = {values["macro_grid"]}
core_size_nm = {values["core_size_nm"]}
context_nm = {values["context_nm"]}

[lithography]
pixel_nm = {values["pixel_nm"]}
canvas_pixels = 256

[edge]
corner_nm = {values["corner_nm"]}
segment_nm = {values["segment_nm"]}
max_displacement_nm = {values["max_displacement_nm"]}
miter_limit = {values["miter_limit"]}

[iteration]
round_deltas_nm = {values["deltas"]}

[output]
work_dir = "{(tmp_path / "work").as_posix()}"
final_layout = "{(tmp_path / "final.gds").as_posix()}"
final_cell_mode = "single_cell"
"""
    config_path = tmp_path / "pipeline.toml"  # 配置路径
    config_path.write_text(text, encoding="utf-8")  # 写盘
    return config_path  # 返回路径


@pytest.fixture
def prepared(tmp_path):
    """生成版图与配置并执行阶段 0/1，返回 (config, plan)。"""
    gds = _write_gds(tmp_path)  # 生成 GDS
    config = pipeline.load_config(_write_config(tmp_path, gds))  # 严格加载
    plan = pipeline.prepare_problems(config)  # 阶段 0/1
    return config, plan  # 返回两件套


class TestConfigValidation:
    """load_config 与 exact_dbu 的显式校验。"""

    def test_unknown_key_rejected(self, tmp_path):
        """未知键直接失败，防止拼写错误被静默忽略。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds)  # 生成配置
        text = path.read_text(encoding="utf-8").replace(  # 注入未知键
            "[grid]", "[grid]\nbogus = 1")  # 在 grid 段加键
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="未知键"):  # 必须报错
            pipeline.load_config(path)  # 加载

    def test_macro_entry_exclusivity(self, tmp_path):
        """macro_grid 与 macro_size_nm 同现或同缺都失败。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        both = _write_config(  # 同时给出两种入口
            tmp_path, gds, macro_grid="[2, 2]")  # 数量模式
        text = both.read_text(encoding="utf-8").replace(  # 再加尺寸模式
            "macro_grid = [2, 2]", "macro_grid = [2, 2]\nmacro_size_nm = 60")  # 注入
        both.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="恰好填写一个"):  # 必须报错
            pipeline.load_config(both)  # 加载
        neither = _write_config(tmp_path, gds)  # 数量模式配置
        text = neither.read_text(encoding="utf-8").replace(  # 删除唯一入口
            "macro_grid = [2, 2]", "")  # 移除
        neither.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="恰好填写一个"):  # 必须报错
            pipeline.load_config(neither)  # 加载

    def test_exact_dbu_rejects_off_grid_value(self, tmp_path):
        """不能精确转换 DBU 的 nm 参数失败并写明参数名。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds, context_nm=10.5)  # 0.5nm 无法落 1nm 格点
        config = pipeline.load_config(path)  # 加载本身不做换算
        with pytest.raises(ValueError, match="context_nm"):  # 报错含参数名
            pipeline.prepare_problems(config)  # 精确换算发生在阶段 0

    def test_context_below_max_displacement_rejected(self, tmp_path):
        """context 小于最大位移时阶段 0 校验失败。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        config = pipeline.load_config(  # context=2 < max_disp=8
            _write_config(tmp_path, gds, context_nm=2))  # 配置
        with pytest.raises(ValueError, match="max_displacement_nm"):  # 必须报错
            pipeline.prepare_problems(config)  # 执行


class TestPrepareProblems:
    """阶段 0/1 的 plan 与 problem 产物。"""

    def test_plan_and_problems_written(self, prepared, tmp_path):
        """plan.json 与全部 problem NPZ 在全部成功后落盘。"""
        _, plan = prepared  # 解包
        work = tmp_path / "work"  # 工作目录
        assert (work / "plan.json").is_file()  # 计划存在
        assert len(list((work / "problems").glob("*.npz"))) == 4  # 2×2 恰 4 个 problem
        assert plan["macro_count"] == 4  # macro 计数
        assert plan["segment_count_sum"] > 0  # 段数非零
        assert plan["maximum_problem_bytes"] > 0  # 字节统计非零

    def test_macro_ownership_partitions_layer_bbox(self, prepared):
        """全部 macro ownership 面积和恰等于目标层 bbox 面积。"""
        _, plan = prepared  # 解包
        area = sum((box[2] - box[0]) * (box[3] - box[1])  # 逐 macro 面积
                   for box in (entry["ownership_box"]  # 取四元组
                               for entry in plan["macros"]))  # 遍历条目
        assert area == 120 * 40  # 目标层 bbox 面积 (20,20)-(140,60)


class TestTwoRounds:
    """双轮迭代的状态机行为。"""

    def _problem(self, plan, entry):
        """按计划条目加载 problem。"""
        return MacroProblem.load(entry["problem_file"])  # 直接用计划内路径

    def test_round_one_moves_all_owned_segments(self, prepared):
        """第一轮全部 owner 段位移恰为 +2 nm，context 段保持零。"""
        _, plan = prepared  # 解包
        summary = pipeline.run_round(plan, 1, plan["round_deltas_dbu"][0])  # 第一轮 +2
        assert summary["macro_gds_count"] == 4  # 每轮 4 个 macro GDS
        for entry in plan["macros"]:  # 逐 macro 检查
            problem = self._problem(plan, entry)  # 加载 problem
            owned = problem.owner_indices >= 0  # owner 掩码
            with np.load(entry["problem_file"].replace(  # problem 路径换算到 result 路径
                    "problems", "round_001/results"), allow_pickle=False) as data:  # 打开 result
                displacements = data["segment_displacements"]  # 位移状态
                written = int(data["written_owner_count"][0])  # 写入计数
            assert np.all(displacements[owned] == 2.0)  # owner 恰 +2
            assert np.all(displacements[~owned] == 0.0)  # context 恰 0
            assert written == int(owned.sum())  # 每段恰写一次

    def test_round_two_reads_previous_state(self, prepared, tmp_path):
        """第二轮从第一轮位移续读，不从零重启。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 正常第一轮
        work = tmp_path / "work"  # 工作目录
        target = work / "round_001" / "results" / "mr0c0.npz"  # 篡改目标
        with np.load(target, allow_pickle=False) as data:  # 读取
            arrays = {name: data[name] for name in data.files}  # 全量复制
        problem = MacroProblem.load(  # 加载同 macro problem
            work / "problems" / "mr0c0.npz")  # 路径
        arrays["segment_displacements"][problem.owner_indices >= 0] = 5.0  # owner 置 5
        np.savez(target, **arrays)  # 写回
        pipeline.run_round(plan, 2, -2)  # 第二轮 -2
        with np.load(work / "round_002" / "results" / "mr0c0.npz",  # 读第二轮
                     allow_pickle=False) as data:  # 打开
            displacements = data["segment_displacements"]  # 位移
        # 续读证明：owner = 5 - 2 = 3，而不是从零重启的 -2。
        assert np.all(displacements[problem.owner_indices >= 0] == 3.0)  # 断言

    def test_round_two_returns_to_exact_zero(self, prepared):
        """第二轮结束后全部 owner 位移精确回到 0。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        for entry in plan["macros"]:  # 逐 macro
            with np.load(entry["problem_file"].replace(  # 定位第二轮 result
                    "problems", "round_002/results"), allow_pickle=False) as data:  # 打开
                displacements = data["segment_displacements"]  # 位移
            assert np.all(displacements == 0.0)  # 全部精确为零

    def test_each_round_writes_one_npz_and_gds_per_macro(self, prepared, tmp_path):
        """每 macro 每轮恰一个 result NPZ 和一个 GDS，两轮共 8 个 GDS。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        work = tmp_path / "work"  # 工作目录
        total_gds = 0  # GDS 总数
        for round_name in ("round_001", "round_002"):  # 两轮
            results = list((work / round_name / "results").glob("*.npz"))  # result 列表
            gds = list((work / round_name / "gds").glob("*.gds"))  # GDS 列表
            assert len(results) == len(gds) == 4  # 每轮 4+4
            total_gds += len(gds)  # 累计
        assert total_gds == 8  # 两轮共 8

    def test_transmission_sums_finite_for_every_core(self, prepared):
        """两轮每个 core 都产出有限 transmission sum。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        for entry in plan["macros"]:  # 逐 macro
            problem = self._problem(plan, entry)  # 加载
            for round_name in ("round_001", "round_002"):  # 两轮
                with np.load(entry["problem_file"].replace(  # 定位 result
                        "problems", f"{round_name}/results"),  # 路径
                        allow_pickle=False) as data:  # 打开
                    sums = data["core_transmission_sums"]  # 读总和
                assert len(sums) == problem.macro.core_count  # 每 core 一项
                assert np.all(np.isfinite(sums))  # 全部有限
                assert sums.sum() > 0  # 非空版图必然有透光像素

    def test_round_two_gds_equals_zero_displacement_reference(self, prepared):
        """第二轮每个 macro GDS 等于其零位移参考（回零即回形）。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        for entry in plan["macros"]:  # 逐 macro
            problem = self._problem(plan, entry)  # 加载
            zeros = np.zeros(problem.segments.segment_count)  # 零位移
            reference = reconstruct_region(problem, zeros)  # 参考重建
            gds_path = entry["problem_file"].replace(  # 定位 GDS
                "problems", "round_002/gds").replace(".npz", ".gds")  # 后缀
            with LayoutDB.open(gds_path) as database:  # 回读
                batch = database.query(  # 全框查询
                    [(problem.layer.layer, problem.layer.datatype)],
                    DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)).materialize()  # 物化
            assert int((batch.region(problem.layer) ^ reference).area()) == 0  # XOR 零

    def test_iteration_order_does_not_change_results(self, prepared, tmp_path):
        """macro 正序、逆序执行的两轮状态完全一致。"""
        _, plan = prepared  # 解包
        pipeline.run_round(plan, 1, 2)  # 正序第一轮
        work = tmp_path / "work"  # 工作目录
        reference_dir = work / "round_001" / "results"  # 正序产物目录
        kept = {path.name: np.load(path, allow_pickle=False)[  # 保存正序位移
            "segment_displacements"].copy() for path in reference_dir.glob("*.npz")}  # 复制
        shutil.rmtree(reference_dir)  # 清空后重跑
        reversed_plan = dict(plan)  # 浅拷贝计划
        reversed_plan["macros"] = list(reversed(plan["macros"]))  # 逆序条目
        pipeline.run_round(reversed_plan, 1, 2)  # 逆序第一轮
        for name, expected in kept.items():  # 逐 macro 比较
            with np.load(reference_dir / name, allow_pickle=False) as data:  # 打开
                assert np.array_equal(data["segment_displacements"], expected)  # 一致

    def test_stage_two_never_repeats_stage_one(self, prepared, monkeypatch):
        """阶段二不调用 LayoutDB 物化或 problem 准备（调用计数证明）。"""
        _, plan = prepared  # 解包
        calls = {"prepare": 0, "materialize": 0}  # 计数器
        real_prepare = pipeline.prepare_macro_problem  # 原函数

        def _counting_prepare(*args, **kwargs):  # 计数包装
            calls["prepare"] += 1  # 计数
            return real_prepare(*args, **kwargs)  # 透传
        monkeypatch.setattr(pipeline, "prepare_macro_problem", _counting_prepare)  # 替换入口
        from layout.query import ShapeQuery  # 物化入口类
        real_materialize = ShapeQuery.materialize_intersecting  # 原方法

        def _counting_materialize(self, *args, **kwargs):  # 计数包装
            calls["materialize"] += 1  # 计数
            return real_materialize(self, *args, **kwargs)  # 透传
        monkeypatch.setattr(ShapeQuery, "materialize_intersecting",  # 替换方法
                            _counting_materialize)  # 完成
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        assert calls == {"prepare": 0, "materialize": 0}  # 零调用
