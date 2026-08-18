"""单/多 macro MB-OPC 入口的端到端生成式测试。"""

import json
import subprocess
import sys

import klayout.db as kdb
import numpy as np
import pytest

import main._simple_mbopc_workflow as workflow
from layout import DbuBox, LayerSpec, LayoutDB

_TARGET_LAYER = LayerSpec(1, 0)  # 生成式版图唯一目标层


def _write_gds(tmp_path):
    """生成矩形 + hole + 斜边三角形的混合 GDS 并返回路径。

    图形全部落在 (10,10)-(150,90) 内：160×90 的 ROI 按 core 40 切出
    足够的 tile；斜边覆盖非 Manhattan 方向；hole 验证带孔拓扑。
    """
    layout = kdb.Layout()  # 独立原生版图
    layout.dbu = 0.001  # 1 nm/DBU，配置数值即 DBU
    top = layout.create_cell("TOP")  # 唯一顶层
    shapes = top.shapes(layout.layer(1, 0))  # 目标层
    shapes.insert(kdb.Box(10, 10, 90, 80))  # 左侧实心矩形
    hole_region = (kdb.Region(kdb.Box(100, 10, 150, 80)) -  # 右侧外框
                   kdb.Region(kdb.Box(115, 25, 135, 65)))  # 中心 hole
    hole_region.insert_into(layout, top.cell_index(),
                            layout.layer(1, 0))  # 插入带孔图形
    top.shapes(layout.layer(1, 0)).insert(  # 底部斜边三角形
        kdb.Polygon([(20, 82), (60, 82), (20, 88)]))
    path = tmp_path / "reticle.gds"  # 输出路径
    layout.write(str(path))  # 写盘
    return path  # 返回路径


def _write_config(tmp_path, layout_path, macro_grid="[1, 1]", **overrides):
    """按默认契约生成 MB-OPC TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值满足全部网格与迭代契约
        "macro_grid": macro_grid, "core_size_nm": 40, "context_nm": 20,
        "pixel_nm": 4, "corner_nm": 8, "segment_nm": 16,
        "max_displacement_nm": 10, "miter_limit": 4.0,
        "iterations": 2, "initial_step_nm": 2, "decay_every": 2,
        "epe_distance_nm": 4, "batch_size": 4, "target_cache_mb": 16,
        "device": "cpu", "save_final_lithography": "true",
        "show_progress": "false", "final_cell_mode": "single_cell"}
    values.update(overrides)  # 应用覆盖
    text = f"""  # 组装 TOML 文本
[layout]
layout = "{layout_path.as_posix()}"
top_cell = "TOP"
layer = 1
datatype = 0
polarity = "clear"

[partition]
macro_grid = {values["macro_grid"]}
core_size_nm = {values["core_size_nm"]}
context_nm = {values["context_nm"]}

[lithography]
pixel_nm = {values["pixel_nm"]}
canvas_pixels = 256
device = "{values["device"]}"

[edge]
corner_nm = {values["corner_nm"]}
segment_nm = {values["segment_nm"]}
max_displacement_nm = {values["max_displacement_nm"]}
miter_limit = {values["miter_limit"]}

[mbopc]
iterations = {values["iterations"]}
initial_step_nm = {values["initial_step_nm"]}
decay_every = {values["decay_every"]}
epe_distance_nm = {values["epe_distance_nm"]}
batch_size = {values["batch_size"]}
target_cache_mb = {values["target_cache_mb"]}

[output]
work_dir = "{(tmp_path / "work").as_posix()}"
save_final_lithography = {values["save_final_lithography"]}
show_progress = {values["show_progress"]}
final_layout = "{(tmp_path / "final.gds").as_posix()}"
final_cell_mode = "{values["final_cell_mode"]}"
"""
    config_path = tmp_path / "mbopc.toml"  # 配置路径
    config_path.write_text(text, encoding="utf-8")  # 写盘
    return config_path  # 返回路径


def _coverage(path, layer=_TARGET_LAYER):
    """回读 GDS 目标层全框覆盖 Region。"""
    with LayoutDB.open(path) as database:  # 打开
        return database.query(  # 全框物化
            [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
        ).materialize().region(layer)  # 覆盖


class TestSingleMacroRunner:
    """单 macro 入口的产物与约束。"""

    @pytest.fixture(scope="class")
    @classmethod
    def single(cls, tmp_path_factory):
        """执行一次单 macro 完整流程并返回 (tmp, summary)。"""
        tmp = tmp_path_factory.mktemp("single")  # 独立目录
        gds = _write_gds(tmp)  # 生成版图
        summary = workflow.run_mbopc(_write_config(tmp, gds))  # 完整流程
        return tmp, summary  # 两件套

    def test_summary_and_artifacts(self, single):
        """summary、每 macro result/best/metrics 与最终 GDS 全部落盘。"""
        tmp, summary = single  # 解包
        work = tmp / "work"  # 工作目录
        assert summary["macro_count"] == 1  # 恰一个 macro
        assert summary["method"] == "simple_mbopc"  # 方法标识（公共层写入）
        for key in ("rss_start_bytes", "rss_after_prepare_bytes",  # 资源字段
                    "peak_rss_bytes", "cuda_peak_bytes"):
            assert key in summary, key  # 资源统计上提公共层后的新增键
        assert summary["device"].startswith("cpu")  # 测试固定 CPU
        assert (work / "plan.json").is_file()  # 计划
        assert (work / "summary.json").is_file()  # 摘要
        macro_dir = work / "macros" / summary["macros"][0]["macro_id"]
        for name in ("result.npz", "best.gds", "metrics.json"):  # 三件产物
            assert (macro_dir / name).is_file(), name
        assert (tmp / "final.gds").is_file()  # 最终合并版图
        with np.load(macro_dir / "result.npz", allow_pickle=False) as data:
            assert set(data.files) == {  # §12.3 契约键集
                "format_version", "macro_id", "best_round",
                "best_displacements", "stop_reason"}
            assert str(data["macro_id"][0]) == summary["macros"][0]["macro_id"]

    def test_records_are_baseline_then_moved(self, single):
        """metrics.json 的 records[0] 是 baseline，records[1] 是移动后评价。"""
        tmp, summary = single  # 解包
        metrics_path = (tmp / "work" / "macros" /
                        summary["macros"][0]["macro_id"] / "metrics.json")
        records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
        assert [r["round_index"] for r in records] == [0, 1, 2]  # baseline+2 轮
        assert records[0]["step_dbu"] == 0.0  # baseline 无步长
        assert records[0]["moved_segments"] == 0  # baseline 无移动
        assert len(records) <= summary["iterations"] + 1  # 不超过迭代上限

    def test_final_lithography_manifest_and_pngs(self, single):
        """save_final_lithography 落盘 manifest 与逐 tile PNG。"""
        tmp, summary = single  # 解包
        out_dir = tmp / "work" / "final_lithography"  # 留档目录
        manifest = json.loads(  # 读清单
            (out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tile_count"] > 0  # 至少一个 tile
        assert manifest["threshold"] == pytest.approx(0.5)  # 模型阈值
        for tile in manifest["tiles"]:  # 逐 tile 检查
            assert (out_dir / tile["nominal_png"]).is_file()  # 连续 PNG
            assert (out_dir / tile["binary_png"]).is_file()  # 二值 PNG
            assert len(tile["ownership_box"]) == 4  # 计分框四元组
        assert summary["final_lithography_tiles"] == manifest["tile_count"]

class TestMultiMacroRunner:
    """多 macro 入口的独立迭代、一次合并与差异量化。"""

    @pytest.fixture(scope="class")
    @classmethod
    def multi(cls, tmp_path_factory):
        """执行一次多 macro 完整流程并返回 (tmp, summary)。"""
        tmp = tmp_path_factory.mktemp("multi")  # 独立目录
        gds = _write_gds(tmp)  # 生成版图（与单 macro 同一份几何）
        summary = workflow.run_mbopc(  # 2×2 macro
            _write_config(tmp, gds, macro_grid="[2, 2]"))
        return tmp, summary  # 两件套

    def test_every_macro_solved_independently(self, multi):
        """每个 macro 都有独立产物，全部完成后才有最终 GDS。"""
        tmp, summary = multi  # 解包
        assert summary["macro_count"] == 4  # 2×2
        ids = {macro["macro_id"] for macro in summary["macros"]}
        assert len(ids) == 4  # 四个不同 macro
        for macro in summary["macros"]:  # 逐 macro 三件产物
            macro_dir = tmp / "work" / "macros" / macro["macro_id"]
            for name in ("result.npz", "best.gds", "metrics.json"):
                assert (macro_dir / name).is_file(), name
        assert (tmp / "final.gds").is_file()  # 一次合并的最终版图

    def test_merge_called_exactly_once(self, monkeypatch, tmp_path):
        """multi 全流程调用 merge_macro_results 恰一次。"""
        gds = _write_gds(tmp_path)  # 生成版图
        config = _write_config(tmp_path, gds, macro_grid="[2, 2]")  # 配置
        import main._mbopc_workflow as workflow_host  # merge 调用宿主（公共循环）
        calls = []  # 调用计数
        real = workflow_host.merge_macro_results  # 原函数

        def _counting(*args, **kwargs):
            """计数合并调用并透传。"""
            calls.append(1)
            return real(*args, **kwargs)
        monkeypatch.setattr(workflow_host, "merge_macro_results", _counting)
        workflow.run_mbopc(config)  # 完整流程
        assert calls == [1]  # 恰一次

    def test_macro_order_does_not_change_coverage(self, tmp_path, monkeypatch):
        """macro 正逆序求解的最终物理覆盖 XOR 为零（独立 macro 性质）。"""
        import main._macro_pipeline as macro_pipeline  # plan_macros 宿主
        gds = _write_gds(tmp_path)  # 生成版图
        real_plan = macro_pipeline.plan_macros  # 原函数

        def _reversed(*args, **kwargs):
            """反转 macro 规划顺序。"""
            return tuple(reversed(real_plan(*args, **kwargs)))
        finals = {}  # 顺序 → 最终版图
        for tag, reverse in (("forward", False), ("reverse", True)):  # 两种顺序
            base = tmp_path / tag  # 独立目录
            base.mkdir()  # 创建
            config = _write_config(base, gds, macro_grid="[2, 2]")  # 配置
            if reverse:
                monkeypatch.setattr(macro_pipeline, "plan_macros", _reversed)
            finals[tag] = workflow.run_mbopc(config)["final_layout"]
            monkeypatch.undo()  # 立即恢复
        assert int((_coverage(finals["forward"]) ^  # 覆盖一致
                    _coverage(finals["reverse"])).area()) == 0

    def test_batch_size_does_not_change_best(self, tmp_path):
        """batch_size=2 与 4 的 best 位移逐位一致。"""
        gds = _write_gds(tmp_path)  # 生成版图
        best = {}  # batch → best 位移
        for batch_size in (2, 4):  # 两种批大小
            base = tmp_path / f"b{batch_size}"  # 独立目录
            base.mkdir()  # 创建
            config = _write_config(  # 仅批大小不同
                base, gds, macro_grid="[2, 2]", batch_size=batch_size)
            summary = workflow.run_mbopc(config)  # 完整流程
            path = (base / "work" / "macros" /
                    summary["macros"][0]["macro_id"] / "result.npz")
            with np.load(path, allow_pickle=False) as data:  # 读 best
                best[batch_size] = data["best_displacements"]
        np.testing.assert_array_equal(best[2], best[4])  # 批不变性

    def test_invalid_geometry_macro_keeps_best_and_continues(
            self, tmp_path, monkeypatch):
        """一个 macro 候选非法时保留其较早 best，其余 macro 继续完成。"""
        from opc.iteration.mbopc import simple  # 重建宿主
        gds = _write_gds(tmp_path)  # 生成版图
        real_reconstruct = simple.reconstruct_region  # 原函数
        poisoned = {"active": False, "hits": 0}  # 注入状态

        def _failing_for_second_macro(problem, displacements):
            """第二个 macro 的首个非零位移候选抛非法几何。"""
            if (problem.macro.macro_id == "mr0c1" and
                    np.any(displacements != 0.0)):  # 该 macro 的候选
                poisoned["hits"] += 1
                if poisoned["hits"] == 1:
                    from opc.errors import ReconstructionError  # 局部导入
                    raise ReconstructionError("hole escaped its hull")
            return real_reconstruct(problem, displacements)
        monkeypatch.setattr(simple, "reconstruct_region", _failing_for_second_macro)
        config = _write_config(tmp_path, gds, macro_grid="[2, 2]")  # 配置
        summary = workflow.run_mbopc(config)  # 完整流程
        by_id = {m["macro_id"]: m for m in summary["macros"]}  # 索引
        assert by_id["mr0c1"]["stop_reason"] == "invalid_geometry"  # 停止原因
        assert "hole escaped" in by_id["mr0c1"]["stop_detail"]  # 原因在案
        for macro_id in ("mr0c0", "mr1c0", "mr1c1"):  # 其余 macro 正常
            assert by_id[macro_id]["stop_reason"] != "invalid_geometry"
        assert (tmp_path / "final.gds").is_file()  # 最终合并照常完成


class TestSingleVersusMulti:
    """独立 macro 边界取舍的量化（不宣称等价于全局同步）。"""

    def test_difference_area_is_quantified(self, tmp_path):
        """单 macro 与多 macro 最终覆盖差异面积被计算（可为非零）。"""
        gds = _write_gds(tmp_path)  # 同一份几何
        single_base = tmp_path / "single"  # 独立目录
        multi_base = tmp_path / "multi"  # 独立目录
        single_base.mkdir()  # 创建
        multi_base.mkdir()  # 创建
        single = workflow.run_mbopc(  # 全 ROI 一个 macro
            _write_config(single_base, gds, macro_grid="[1, 1]"))
        multi = workflow.run_mbopc(  # 2×2 macro
            _write_config(multi_base, gds, macro_grid="[2, 2]"))
        difference = int(  # 差异面积（独立 context 取舍的量化结果）
            (_coverage(single["final_layout"]) ^
             _coverage(multi["final_layout"])).area())
        # 上界断言：差异面积不可能超过源图形总面积（防边界差异爆炸性回归），
        # 精确数值仍打印量化（固定参考 context 代价的观测值）。
        source_area = int(_coverage(gds).area())  # 源图形总面积
        assert 0 <= difference <= source_area  # 非恒真：拦截差异扩散
        print(f"single vs multi 差异面积：{difference} DBU²（源面积 {source_area}）")


class TestDirectExecution:
    """两个入口从仓库外直接运行与进度条行为。"""

    def _run(self, script, config_path, cwd, extra_env=None):
        """以子进程直跑入口脚本。"""
        env = {**__import__("os").environ, **(extra_env or {})}  # 环境变量
        return subprocess.run(  # 直跑（免安装）
            [sys.executable, str(script), str(config_path)], cwd=cwd,
            capture_output=True, text=True, timeout=600, check=False, env=env)

    @pytest.fixture(scope="class")
    @classmethod
    def outside_config(cls, tmp_path_factory, project_root):
        """仓库外目录 + 生成式配置（含仓库外 work_dir）。"""
        tmp = tmp_path_factory.mktemp("outside")  # 仓库外目录
        gds = _write_gds(tmp)  # 生成版图
        return tmp, _write_config(tmp, gds, macro_grid="[2, 2]")  # 多 macro

    def test_multi_runs_outside_repository(self, outside_config, project_root):
        """从仓库外直跑 multi 入口退出码 0 并产出全部关键标记。"""
        tmp, config = outside_config  # 解包
        script = project_root / "main" / "run_mbopc.py"  # 入口
        completed = self._run(script, config, tmp)  # cwd=仓库外
        assert completed.returncode == 0, completed.stderr  # 正常退出
        for marker in ("simple MB-OPC 执行完成", "device：",
                       "合并", "最终版图"):  # 摘要标记
            assert marker in completed.stdout, marker

    def test_single_runs_outside_repository(self, tmp_path, project_root):
        """从仓库外直跑 single 入口同样成功。"""
        gds = _write_gds(tmp_path)  # 生成版图
        config = _write_config(tmp_path, gds, macro_grid="[1, 1]")  # 单 macro
        script = project_root / "main" / "run_mbopc.py"  # 入口
        completed = self._run(script, config, tmp_path)  # cwd=仓库外
        assert completed.returncode == 0, completed.stderr  # 正常退出
        assert "simple MB-OPC 执行完成" in completed.stdout  # 标题

    def test_progress_output_only_when_enabled(self, tmp_path, project_root):
        """show_progress=true 时 stderr 有进度条，false 时完全静默。"""
        script = project_root / "main" / "run_mbopc.py"  # 入口
        quiet_dir = tmp_path / "quiet"  # 静默目录
        quiet_dir.mkdir()  # 创建
        verbose_dir = tmp_path / "verbose"  # 进度目录
        verbose_dir.mkdir()  # 创建
        quiet = self._run(  # show_progress=false
            script, _write_config(quiet_dir, _write_gds(quiet_dir),
                                  macro_grid="[2, 2]",
                                  show_progress="false"), quiet_dir)
        assert quiet.returncode == 0, quiet.stderr  # 正常退出
        assert "tile" not in quiet.stderr and "macro" not in quiet.stderr  # 静默
        verbose = self._run(  # show_progress=true
            script, _write_config(verbose_dir, _write_gds(verbose_dir),
                                  macro_grid="[2, 2]",
                                  show_progress="true"), verbose_dir)
        assert verbose.returncode == 0, verbose.stderr  # 正常退出
        assert ("tile" in verbose.stderr) or ("macro" in verbose.stderr)  # 有进度

    def test_missing_argument_returns_usage(self, project_root):
        """无参数运行打印用法并以退出码 2 结束。"""
        script = project_root / "main" / "run_mbopc.py"  # 入口
        completed = subprocess.run(  # 无参数直跑
            [sys.executable, str(script)], cwd=project_root,
            capture_output=True, text=True, timeout=60, check=False)
        assert completed.returncode == 2  # 参数错误退出码
        assert "用法" in completed.stderr  # 用法提示


class TestConfigValidation:
    """MB-OPC 配置层的专属校验。"""

    def _config_path(self, tmp_path, **overrides):
        """生成配置并返回路径。"""
        gds = _write_gds(tmp_path)  # 生成版图
        return _write_config(tmp_path, gds, **overrides)  # 带覆盖

    def test_unknown_mbopc_key_rejected(self, tmp_path):
        """[mbopc] 段未知键失败。"""
        path = self._config_path(tmp_path)  # 基准配置
        text = path.read_text(encoding="utf-8").replace(  # 注入未知键
            "[mbopc]", "[mbopc]\nbogus = 1")
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="未知键"):
            workflow.run_mbopc(path)  # 装配前置检查在准备前毫秒级抛出

    @pytest.mark.parametrize(
        "overrides, pattern",
        [({"iterations": 0}, "必须为正"),
         ({"initial_step_nm": 11}, "不得超过 max_displacement_nm"),
         ({"epe_distance_nm": 21}, "不得超过 context_nm"),
         ({"device": "gpu"}, "device")],
        ids=["iter=0", "step>max", "epe>context", "device=gpu"])
    def test_invalid_values_fail(self, tmp_path, overrides, pattern):
        """越界迭代参数、步长/探针越限与未知设备都失败。"""
        with pytest.raises(ValueError, match=pattern):
            workflow.run_mbopc(self._config_path(tmp_path, **overrides))

    @pytest.mark.parametrize(
        ("overrides", "field"),
        [({"iterations": 1.5}, "iterations"),
         ({"iterations": "true"}, "iterations"),
         ({"batch_size": 2.0}, "batch_size"),
         ({"target_cache_mb": "true"}, "target_cache_mb")],
        ids=["iter=1.5", "iter=true", "batch=2.0", "cache=true"])
    def test_non_integer_values_fail(self, tmp_path, overrides, field):
        """浮点或布尔的整数配置被严格拒绝，不静默截断（审查 P1.3 回归）。"""
        with pytest.raises(ValueError, match=field):
            workflow.run_mbopc(self._config_path(tmp_path, **overrides))
