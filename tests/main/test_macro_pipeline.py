"""双轮 macro-core 迭代管线的配置校验与端到端生成式测试。"""

from pathlib import Path

import klayout.db as kdb
import numpy as np
import pytest

import main.run_macro_pipeline as pipeline
from layout import DbuBox, LayerSpec, LayoutDB
from opc.input.edge import MacroProblem, reconstruct_region

# 测试版图：DBU=1nm，bar 图形使目标层 bbox 为 (20,20)-(140,60)，
# 2×2 macro、core 30、context 10、pixel 1 恰好满足全部网格契约。


def _write_gds(tmp_path):
    """生成上下锚框加中间内条的双层 GDS 并返回路径。

    锚框只负责把层 bbox 撑到 (20,20)-(140,60)；内条与锚框不重叠且完全在
    bbox 内部——这是位移有效性的关键：铺满 bbox 的图形外扩会全部落在
    macro ownership 之外被正确裁掉，第一轮合并将看不到任何变化。
    2/0 层是「未处理层不得复制」验证的对照层。
    """
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU，配置数值直接等于 DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(20, 20, 140, 28))  # 下锚框
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(20, 52, 140, 60))  # 上锚框
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(60, 30, 100, 50))  # 内条（跨 macro 切线）
    top.shapes(layout.layer(2, 0)).insert(kdb.Box(30, 32, 130, 38))  # 非目标对照层
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_config(tmp_path, layout_path, **overrides):
    """按默认契约生成 TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值全部满足网格与分段契约
        "macro_grid": "[2, 2]", "core_size_nm": 30, "context_nm": 10,
        "pixel_nm": 1, "corner_nm": 4, "segment_nm": 10,
        "max_displacement_nm": 8, "miter_limit": 4.0, "deltas": "[2, -2]",
        "final_cell_mode": "single_cell"}
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
final_cell_mode = "{values["final_cell_mode"]}"
"""
    config_path = tmp_path / "pipeline.toml"  # 配置路径
    config_path.write_text(text, encoding="utf-8")  # 写盘
    return config_path  # 返回路径


def _load(config_path):
    """加载宏管线配置并放行验证专属 [iteration] 段。"""
    return pipeline.load_macro_config(  # 共用六段加载器
        config_path, extra_sections=("iteration",))  # 白名单放行验证段


@pytest.fixture
def prepared(tmp_path):
    """生成版图与配置并执行阶段 0/1，返回 (config, plan)。"""
    gds = _write_gds(tmp_path)  # 生成 GDS
    config = _load(_write_config(tmp_path, gds))  # 严格加载
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
            _load(path)  # 加载

    def test_macro_entry_exclusivity(self, tmp_path):
        """macro_grid 与 macro_size_nm 同现或同缺都失败。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        both = _write_config(  # 同时给出两种入口
            tmp_path, gds, macro_grid="[2, 2]")  # 数量模式
        text = both.read_text(encoding="utf-8").replace(  # 再加尺寸模式
            "macro_grid = [2, 2]", "macro_grid = [2, 2]\nmacro_size_nm = 60")  # 注入
        both.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="恰好填写一个"):  # 必须报错
            _load(both)  # 加载
        neither = _write_config(tmp_path, gds)  # 数量模式配置
        text = neither.read_text(encoding="utf-8").replace(  # 删除唯一入口
            "macro_grid = [2, 2]", "")  # 移除
        neither.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="恰好填写一个"):  # 必须报错
            _load(neither)  # 加载

    def test_exact_dbu_rejects_off_grid_value(self, tmp_path):
        """不能精确转换 DBU 的 nm 参数失败并写明参数名。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds, context_nm=10.5)  # 0.5nm 无法落 1nm 格点
        config = _load(path)  # 加载本身不做换算
        with pytest.raises(ValueError, match="context_nm"):  # 报错含参数名
            pipeline.prepare_problems(config)  # 精确换算发生在阶段 0

    def test_context_below_max_displacement_rejected(self, tmp_path):
        """context 小于最大位移时阶段 0 校验失败。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        config = _load(  # context=2 < max_disp=8
            _write_config(tmp_path, gds, context_nm=2))  # 配置
        with pytest.raises(ValueError, match="max_displacement_nm"):  # 必须报错
            pipeline.prepare_problems(config)  # 执行

    def test_round_deltas_frozen_to_two_nm(self, tmp_path):
        """双轮位移冻结为 [+2nm,-2nm]：其他幅值或不对称序列一律失败。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        for deltas in ("[3, -3]", "[2, -1]"):  # 幅值不符 / 不对称
            config_path = _write_config(tmp_path, gds, deltas=deltas)  # 覆盖序列
            with pytest.raises(ValueError, match="冻结"):  # 必须报错
                pipeline.load_validation_deltas(config_path)  # 验证段解析即冻结检查

    def test_two_nm_not_representable_fails_with_parameter_name(self, tmp_path):
        """2 nm 无法精确换算到当前 DBU 时失败并保留参数名。"""
        layout = kdb.Layout()  # 独立原生版图
        layout.dbu = 0.0025  # 2.5 nm/DBU：2 nm 不是其整数倍
        top = layout.create_cell("TOP")  # 顶层
        top.shapes(layout.layer(1, 0)).insert(kdb.Box(0, 0, 30, 30))  # 目标层图形
        gds = tmp_path / "odd_dbu.gds"  # 输出路径
        layout.write(str(gds))  # 写盘
        config_path = _write_config(  # 全部参数取 2.5 的倍数
            tmp_path, gds, pixel_nm=2.5, corner_nm=5, segment_nm=10,
            max_displacement_nm=5)  # 只有 2nm 常量无法落格点
        with pytest.raises(ValueError, match="round_deltas_nm"):  # 报错含参数名
            pipeline.run(config_path)  # 落格点换算在 run 内用实际 dbu 执行

    @pytest.mark.parametrize(
        ("old", "new", "field"),
        [("layer = 1", "layer = 1.5", "layer"),
         ("layer = 1", "layer = true", "layer"),
         ("canvas_pixels = 256", "canvas_pixels = 256.0", "canvas_pixels"),
         ("canvas_pixels = 256", "canvas_pixels = true", "canvas_pixels")],
        ids=["layer=1.5", "layer=true", "canvas=256.0", "canvas=true"])
    def test_non_integer_layer_or_canvas_rejected(self, tmp_path, old, new, field):
        """浮点或布尔的 layer/canvas_pixels 被严格拒绝（审查 P1.3 回归）。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        path = _write_config(tmp_path, gds)  # 基准配置
        text = path.read_text(encoding="utf-8").replace(old, new)  # 注入类型错误
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match=field):  # 必须报字段名
            _load(path)  # 宏管线配置加载


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
        summary = pipeline.run_round(plan, 1, 2)  # 第一轮 +2（冻结步长）
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

    def test_iteration_order_does_not_change_results(self, tmp_path, monkeypatch):
        """macro 正序与逆序执行的两轮位移与最终覆盖完全一致。"""
        shared = tmp_path / "shared"  # 共用源版图目录
        shared.mkdir()  # 创建
        gds = _write_gds(shared)  # 生成一份源版图
        finals = {}  # 顺序 → 最终版图路径
        plans = {}  # 顺序 → 计划
        import main._macro_pipeline as macro_pipeline  # plan_macros 现居共享模块
        real_plan_macros = macro_pipeline.plan_macros  # 循环外捕获，闭包不绑定循环变量

        def _reversed_macros(*args, **kwargs):
            """反转 macro 规划顺序，验证迭代顺序无关性。"""
            return tuple(reversed(real_plan_macros(*args, **kwargs)))
        for tag, reverse in (("forward", False), ("reverse", True)):  # 两种顺序
            base = tmp_path / tag  # 独立工作目录
            base.mkdir()  # 创建
            config = _load(_write_config(base, gds))  # 各自配置
            if reverse:  # 逆序只在准备期反转 macro 规划顺序
                monkeypatch.setattr(macro_pipeline, "plan_macros", _reversed_macros)  # 替换
                plans[tag] = pipeline.prepare_problems(config)  # 阶段 0/1
                monkeypatch.undo()  # 立即恢复，后续阶段不受影响
            else:  # 正序
                plans[tag] = pipeline.prepare_problems(config)  # 阶段 0/1
            plan = plans[tag]  # 当前计划
            pipeline.run_round(plan, 1, 2)  # 第一轮 +2（冻结步长）
            pipeline.run_round(plan, 2, -2)  # 第二轮 -2
            finals[tag] = pipeline.merge_macro_results(  # 合并
                plan, pipeline.collect_round_macro_gds(plan, 2),  # 校验并显式映射
                base / "final.gds", cell_mode="single_cell")  # 固定模式
        # ① 两轮每 macro 的位移数组逐位一致（按 macro_id 配对，顺序不同）。
        reverse_by_id = {e["macro_id"]: e for e in plans["reverse"]["macros"]}  # 索引
        for entry_f in plans["forward"]["macros"]:  # 正序条目
            entry_r = reverse_by_id[entry_f["macro_id"]]  # 同 macro 逆序条目
            for round_name in ("round_001", "round_002"):  # 两轮
                path_f = Path(entry_f["problem_file"].replace(  # 正序 result
                    "problems", f"{round_name}/results"))  # 换算
                path_r = Path(entry_r["problem_file"].replace(  # 逆序 result
                    "problems", f"{round_name}/results"))  # 换算
                with np.load(path_f, allow_pickle=False) as df, np.load(  # 同时打开
                        path_r, allow_pickle=False) as dr:  # 打开
                    assert np.array_equal(df["segment_displacements"],  # 比较
                                          dr["segment_displacements"])  # 一致
        # ② 最终顶层物理覆盖 XOR 为零（比较覆盖，不比较 GDS 字节）。
        layer = LayerSpec(1, 0)  # 目标层
        coverage = {}  # 顺序 → 覆盖 Region
        for tag, path in finals.items():  # 两个最终版图
            with LayoutDB.open(path) as database:  # 回读
                coverage[tag] = database.query(  # 全框查询
                    [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
                ).materialize().region(layer)  # 物化
        assert (coverage["forward"] ^ coverage["reverse"]).area() == 0  # XOR 零

    def test_stage_two_never_repeats_stage_one(self, prepared, monkeypatch):
        """阶段二不调用 LayoutDB 物化或 problem 准备（调用计数证明）。"""
        _, plan = prepared  # 解包
        calls = {"prepare": 0, "materialize": 0}  # 计数器
        import main._macro_pipeline as macro_pipeline  # 准备函数现居共享模块
        real_prepare = macro_pipeline.prepare_macro_problem  # 原函数

        def _counting_prepare(*args, **kwargs):
            """计数 problem 准备调用并透传，证明阶段二零调用。"""
            calls["prepare"] += 1  # 计数
            return real_prepare(*args, **kwargs)  # 透传
        monkeypatch.setattr(macro_pipeline, "prepare_macro_problem", _counting_prepare)  # 替换入口
        from layout.query import ShapeQuery  # 物化入口类
        real_materialize = ShapeQuery.materialize_intersecting  # 原方法

        def _counting_materialize(self, *args, **kwargs):
            """计数物化调用并透传，证明阶段二零调用。"""
            calls["materialize"] += 1  # 计数
            return real_materialize(self, *args, **kwargs)  # 透传
        monkeypatch.setattr(ShapeQuery, "materialize_intersecting",  # 替换方法
                            _counting_materialize)  # 完成
        pipeline.run_round(plan, 1, 2)  # 第一轮
        pipeline.run_round(plan, 2, -2)  # 第二轮
        assert calls == {"prepare": 0, "materialize": 0}  # 零调用

    def test_prepare_never_iterates_shapes_in_python(self, tmp_path, monkeypatch):
        """阶段 0 取层 bbox 走原生路径：全程不触碰公共 shape 迭代器。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        config = _load(_write_config(tmp_path, gds))  # 配置
        from layout import LayoutDB as _LayoutDB  # 迭代器宿主类
        real_iterate = _LayoutDB.recursive_polygon_shapes  # 原方法
        calls = {"iterate": 0}  # 计数器

        def _counting_iterate(self, *args, **kwargs):
            """计数 shape 迭代器调用并透传，证明阶段 0 零逐 shape 遍历。"""
            calls["iterate"] += 1  # 计数
            return real_iterate(self, *args, **kwargs)  # 透传
        monkeypatch.setattr(_LayoutDB, "recursive_polygon_shapes",  # 替换方法
                            _counting_iterate)  # 完成
        pipeline.prepare_problems(config)  # 执行阶段 0/1
        assert calls["iterate"] == 0  # 层 bbox 全程来自原生 bbox_per_layer


@pytest.fixture
def full(tmp_path):
    """执行完整流程（准备 + 双轮 + 最终合并），返回 (config_path, summary)。"""
    gds = _write_gds(tmp_path)  # 生成 GDS
    config_path = _write_config(tmp_path, gds)  # 生成配置
    summary = pipeline.run(config_path)  # 完整流程
    return config_path, summary  # 返回两件套


class TestFinalMerge:
    """最终权威覆盖合并与两种 Cell 输出。"""

    def test_single_cell_has_one_result_cell_without_seam(self, full, tmp_path):
        """single_cell 只有一个结果 Cell，跨 macro polygon merge 后无 seam。"""
        _, summary = full  # 解包
        layer = LayerSpec(1, 0)  # 目标层
        with LayoutDB.open(summary["final_layout"]) as db:  # 回读最终版图
            region = db.query([layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
                              ).materialize().region(layer)  # 物化
            hierarchy = db.cell_hierarchy()  # Cell 结构
        assert summary["final_xor_area"] == 0  # 回零 XOR 为零
        assert region.count() == 3  # 两锚框与内条各一个 polygon，跨界处无 seam
        assert all(not children for children in hierarchy.values())  # 无 macro 子 Cell

    def test_macro_cells_contains_expected_child_cells(self, tmp_path):
        """macro_cells 输出含预期数量的 macro 子 Cell。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        config_path = _write_config(  # macro_cells 模式
            tmp_path, gds, final_cell_mode="macro_cells")  # 覆盖模式
        summary = pipeline.run(config_path)  # 完整流程
        assert summary["final_cell_mode"] == "macro_cells"  # 模式确认
        with LayoutDB.open(summary["final_layout"]) as db:  # 回读
            children = db.cell_hierarchy()["OPC_RESULT"]  # 顶层子 Cell
        assert len(children) == 4  # 2×2 恰 4 个子 Cell

    def test_both_cell_modes_have_identical_coverage(self, tmp_path):
        """两种模式的顶层物理覆盖 XOR 为零。"""
        layer = LayerSpec(1, 0)  # 目标层
        gds = _write_gds(tmp_path)  # 生成 GDS
        coverage = {}  # 模式 → 覆盖 Region
        for mode in ("single_cell", "macro_cells"):  # 两种模式
            base = tmp_path / mode  # 独立目录
            base.mkdir()  # 创建
            config_path = _write_config(  # 各自配置
                base, gds, final_cell_mode=mode)  # 覆盖模式
            summary = pipeline.run(config_path)  # 完整流程
            with LayoutDB.open(summary["final_layout"]) as db:  # 回读
                coverage[mode] = db.query(  # 全框查询
                    [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
                ).materialize().region(layer)  # 物化
        assert (coverage["single_cell"] ^ coverage["macro_cells"]).area() == 0  # XOR 零

    def test_merge_does_not_change_coverage_area(self, full):
        """merge/normalize 不改变最终覆盖面积。"""
        _, summary = full  # 解包
        assert summary["final_xor_area"] == 0  # 与原始层一致
        # bar 面积 = 120×40 = 4800；面积若被 normalize 改变，run 会在内部失败。

    def test_round_one_merge_differs_from_reference(self, tmp_path):
        """第一轮合并结果相对参考有非零变化，证明 +2 确实生效。"""
        gds = _write_gds(tmp_path)  # 生成 GDS
        config = _load(_write_config(tmp_path, gds))  # 加载
        plan = pipeline.prepare_problems(config)  # 阶段 0/1
        pipeline.run_round(plan, 1, 2)  # 仅第一轮
        round_one_final = tmp_path / "round1.gds"  # 第一轮合并输出
        pipeline.merge_macro_results(  # 合并第一轮
            plan, pipeline.collect_round_macro_gds(plan, 1),  # 第一轮显式映射
            round_one_final, cell_mode=config.final_cell_mode)  # 配置模式
        with LayoutDB.open(gds) as db:  # 原始版图
            reference = db.query(  # 全框查询
                [LayerSpec(1, 0)], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
            ).materialize_intersecting().region(LayerSpec(1, 0))  # 完整原图形
        with LayoutDB.open(round_one_final) as db:  # 第一轮结果
            moved = db.query(  # 全框查询
                [LayerSpec(1, 0)], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
            ).materialize().region(LayerSpec(1, 0))  # 物化
        assert int((moved ^ reference).area()) > 0  # 外扩 2nm 必然产生非零 XOR

    def test_missing_macro_gds_fails_merge(self, full, tmp_path):
        """缺失 macro GDS 时合并明确失败。"""
        _, _ = full  # 解包
        target = tmp_path / "work" / "round_002" / "gds" / "mr0c0.gds"  # 待删文件
        target.unlink()  # 删除
        config = _load(  # 重新加载配置以重建 plan 路径
            tmp_path / "pipeline.toml")  # 配置
        plan = pipeline.prepare_problems(config)  # 重新准备（problems 目录已存在会被覆盖）
        with pytest.raises(FileNotFoundError, match="macro GDS"):  # 明确失败
            pipeline.merge_macro_results(  # 合并（缺文件在映射检查后失败）
                plan, pipeline.collect_round_macro_gds(plan, 2),  # 第二轮显式映射
                tmp_path / "final2.gds", cell_mode="single_cell")  # 固定模式

    def test_duplicate_macro_id_fails_merge(self, full, tmp_path):
        """重复 macro ID 时合并明确失败。"""
        _, _ = full  # 解包
        config = _load(tmp_path / "pipeline.toml")  # 配置
        plan = pipeline.prepare_problems(config)  # 重新准备
        plan["macros"].append(dict(plan["macros"][0]))  # 注入重复条目
        with pytest.raises(ValueError, match="重复 macro ID"):  # 明确失败
            pipeline.merge_macro_results(  # 合并（计划条目重复即失败）
                plan, pipeline.collect_round_macro_gds(plan, 2),  # 第二轮显式映射
                tmp_path / "final3.gds", cell_mode="single_cell")  # 固定模式

    def test_result_round_mismatch_fails_merge(self, full, tmp_path):
        """result 轮次与合并轮次不一致时明确失败。"""
        _, _ = full  # 解包
        target = tmp_path / "work" / "round_002" / "results" / "mr0c0.npz"  # 篡改目标
        with np.load(target, allow_pickle=False) as data:  # 读取
            arrays = {name: data[name] for name in data.files}  # 复制
        arrays["round_index"] = np.array([1], np.int32)  # 改成错误轮次
        np.savez(target, **arrays)  # 写回
        config = _load(tmp_path / "pipeline.toml")  # 配置
        plan = pipeline.prepare_problems(config)  # 重新准备
        with pytest.raises(ValueError, match="轮次"):  # 明确失败
            pipeline.merge_macro_results(  # 轮次校验在收集映射时失败
                plan, pipeline.collect_round_macro_gds(plan, 2),  # 篡改后被拒
                tmp_path / "final4.gds", cell_mode="single_cell")  # 固定模式

    def test_unprocessed_layers_are_not_copied(self, full, tmp_path):
        """源含 1/0 与 2/0 两层时，最终版图只保留处理过的目标层 1/0。"""
        _, summary = full  # 解包
        with LayoutDB.open(tmp_path / "reticle.gds") as db:  # 回读源版图
            assert db.layers() == (LayerSpec(1, 0), LayerSpec(2, 0))  # 源确有两层
        with LayoutDB.open(summary["final_layout"]) as db:  # 回读最终版图
            assert db.layers() == (LayerSpec(1, 0),)  # 只有目标层被复制
