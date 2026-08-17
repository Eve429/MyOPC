"""梯度 MB-OPC 入口的配置、产物、合并与进度资源生成式测试。"""

import json
import subprocess
import sys

import klayout.db as kdb
import numpy as np
import pytest
import torch

import main._mbopc_workflow as workflow
from layout import DbuBox, LayerSpec, LayoutDB
from main.configuration import load_config

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
    """按默认契约生成梯度 MB-OPC TOML，允许键值覆盖后返回路径。"""
    values = {  # 默认值满足全部网格与梯度契约
        "macro_grid": macro_grid, "core_size_nm": 40, "context_nm": 20,
        "pixel_nm": 4, "corner_nm": 8, "segment_nm": 16,
        "max_displacement_nm": 10, "miter_limit": 4.0,
        "iterations": 1, "learning_rate_nm": 1.0,
        "weight_nominal_l2": 1.0, "weight_process_l2": 0.5,
        "weight_pvband": 0.1, "epe_distance_nm": 4, "batch_size": 4,
        "target_cache_mb": 16, "device": "cpu",
        "save_final_lithography": "true",
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

[gradient]
iterations = {values["iterations"]}
learning_rate_nm = {values["learning_rate_nm"]}
weight_nominal_l2 = {values["weight_nominal_l2"]}
weight_process_l2 = {values["weight_process_l2"]}
weight_pvband = {values["weight_pvband"]}
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
    config_path = tmp_path / "gradient_mbopc.toml"  # 配置路径
    config_path.write_text(text, encoding="utf-8")  # 写盘
    return config_path  # 返回路径


def _coverage(path, layer=_TARGET_LAYER):
    """回读 GDS 目标层全框覆盖 Region。"""
    with LayoutDB.open(path) as database:  # 打开
        return database.query(  # 全框物化
            [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)
        ).materialize().region(layer)  # 覆盖


class TestGradientConfig:
    """[gradient] 配置段的专属校验（TEST-013）。"""

    def _config_path(self, tmp_path, **overrides):
        """生成配置并返回路径。"""
        gds = _write_gds(tmp_path)  # 生成版图
        return _write_config(tmp_path, gds, **overrides)  # 带覆盖

    def test_valid_config_loads(self, tmp_path):
        """合法配置解析出 Gradient/Lithography/Output 三 Config 字段。"""
        gds = _write_gds(tmp_path)  # 生成版图
        from main.configuration import (  # 按入口顺序请求六 Config
            EdgeConfig,
            GradientConfig,
            LayoutConfig,
            LithographyConfig,
            OutputConfig,
            PartitionConfig,
        )
        _, _, litho, _, gradient, output = load_config(
            _write_config(tmp_path, gds), LayoutConfig, PartitionConfig,
            LithographyConfig, EdgeConfig, GradientConfig, OutputConfig)
        assert gradient.iterations == 1  # 迭代
        assert str(gradient.learning_rate_nm) == "1.0"  # Decimal 学习率
        assert gradient.weight_pvband == pytest.approx(0.1)  # 权重
        assert litho.device == "cpu"  # 设备在光刻段
        assert output.save_final_lithography is True  # 留档开关在输出段

    def test_unknown_gradient_key_rejected(self, tmp_path):
        """[gradient] 段未知键失败。"""
        path = self._config_path(tmp_path)  # 基准配置
        text = path.read_text(encoding="utf-8").replace(  # 注入未知键
            "[gradient]", "[gradient]\nbogus = 1")
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="未知键"):
            workflow.run_gradient_mbopc(path)  # 统一加载在入口内

    def test_missing_required_key_rejected(self, tmp_path):
        """缺必填键失败。"""
        path = self._config_path(tmp_path)  # 基准配置
        text = path.read_text(encoding="utf-8").replace(  # 删除一行必填键
            "learning_rate_nm = 1.0\n", "")
        path.write_text(text, encoding="utf-8")  # 写回
        with pytest.raises(ValueError, match="缺少必填键"):
            workflow.run_gradient_mbopc(path)  # 统一加载在入口内

    @pytest.mark.parametrize(
        "overrides, pattern",
        [({"iterations": 1.5}, "iterations"),
         ({"iterations": "true"}, "iterations"),
         ({"batch_size": 2.0}, "batch_size"),
         ({"target_cache_mb": "true"}, "target_cache_mb"),
         ({"iterations": 0}, "必须为正"),
         ({"learning_rate_nm": 0}, "必须为正"),
         ({"learning_rate_nm": -1.5}, "必须为正"),
         ({"weight_nominal_l2": -0.1}, "非负"),
         ({"weight_nominal_l2": 0, "weight_process_l2": 0,
           "weight_pvband": 0}, "至少一个为正"),
         ({"weight_pvband": '"0.1"'}, "必须是数值"),
         ({"epe_distance_nm": 21}, "不得超过 context_nm"),
         ({"device": "gpu"}, "device")],
        ids=["iter=1.5", "iter=true", "batch=2.0", "cache=true", "iter=0",
             "lr=0", "lr<0", "w<0", "全零权重", "w=字符串", "epe>context",
             "device=gpu"])
    def test_invalid_values_fail(self, tmp_path, overrides, pattern):
        """类型、数值与语义非法的配置全部失败（ERR-001）。"""
        with pytest.raises(ValueError, match=pattern):
            workflow.run_gradient_mbopc(  # 类型错在加载层、跨段错在装配层
                self._config_path(tmp_path, **overrides))

    def test_float_epe_not_exact_dbu_fails(self, tmp_path):
        """探针距离不能精确换整数 DBU 时在运行准备期失败。"""
        gds = _write_gds(tmp_path)  # dbu=1nm，0.5nm 无法整除
        config = _write_config(tmp_path, gds, epe_distance_nm=4.5)
        with pytest.raises(ValueError, match="epe_distance_nm"):
            workflow.run_gradient_mbopc(config)  # exact_dbu 层拦截

    def test_oversized_learning_rate_warns_without_modifying(self, tmp_path):
        """lr 超过位移上限：装配处 UserWarning 且求解继续（不硬拒不改参）。"""
        gds = _write_gds(tmp_path)  # 生成版图（max_displacement_nm 默认 10）
        with pytest.warns(UserWarning, match="max_displacement"):  # 风险提示
            summary = workflow.run_gradient_mbopc(  # 完整小跑（CPU 生成式图）
                _write_config(tmp_path, gds, learning_rate_nm=20))
        assert summary["macro_count"] >= 1  # 流程照常完成（参数未被拒绝）

    def test_normal_learning_rate_does_not_warn(self, tmp_path):
        """lr 不超过位移上限时全程不产生该提示。"""
        import warnings as warnings_module  # 局部捕获
        gds = _write_gds(tmp_path)  # 生成版图
        for learning_rate_nm in (10, 1):  # 恰等于限与常规值
            with warnings_module.catch_warnings(record=True) as caught:
                warnings_module.simplefilter("always")  # 全部捕获
                workflow.run_gradient_mbopc(  # 完整小跑（无提示断言在下方）
                    _write_config(tmp_path, gds,
                                  learning_rate_nm=learning_rate_nm))
                assert not [w for w in caught  # 只过滤本提示
                            if "max_displacement" in str(w.message)]
            assert not [w for w in caught
                        if "max_displacement" in str(w.message)]  # 无提示


class TestGradientRunner:
    """梯度入口的产物与 summary 契约（TEST-014）。"""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory):
        """执行一次单 macro 完整梯度流程并返回 (tmp, summary)。"""
        tmp = tmp_path_factory.mktemp("gradient_single")  # 独立目录
        gds = _write_gds(tmp)  # 生成版图
        summary = workflow.run_gradient_mbopc(  # 完整流程
            _write_config(tmp, gds))
        return tmp, summary  # 两件套

    def test_summary_contract(self, run):
        """summary 含方法名、权重、资源字段（§8.2 键全集）。"""
        _, summary = run  # 解包
        assert summary["method"] == "gradient_mbopc"  # 方法标识
        assert summary["macro_count"] == 1  # 恰一个 macro
        assert summary["device"].startswith("cpu")  # 测试固定 CPU
        assert set(summary["loss_weights"]) == {  # 三权重
            "nominal_l2", "process_l2", "pvband"}
        for key in ("rss_start_bytes", "rss_after_prepare_bytes",  # 资源字段
                    "peak_rss_bytes", "cuda_peak_bytes",
                    "final_lithography_tiles", "merge_seconds",
                    "total_seconds"):
            assert key in summary, key
        assert summary["cuda_peak_bytes"] is None  # CPU 时无 CUDA 峰值
        assert summary["peak_rss_bytes"] >= summary["rss_start_bytes"] > 0
        assert summary["final_lithography_tiles"] > 0  # 配置开启了留档

    def test_gradient_artifacts(self, run):
        """每 macro 的 NPZ/JSON/best GDS 满足 §8.2 契约。"""
        tmp, summary = run  # 解包
        macro = summary["macros"][0]  # 唯一 macro 摘要
        macro_dir = tmp / "work" / "macros" / macro["macro_id"]
        for name in ("gradient_result.npz", "gradient_metrics.json",
                     "best.gds"):  # 三件梯度产物
            assert (macro_dir / name).is_file(), name
        assert not (macro_dir / "result.npz").exists()  # 不覆盖 simple 产物名
        with np.load(macro_dir / "gradient_result.npz",  # NPZ 契约
                     allow_pickle=False) as data:
            assert set(data.files) == {
                "format_version", "macro_id", "best_state_index",
                "best_displacements", "stop_reason"}
            assert data["format_version"].dtype == np.int32  # 版本类型
            assert str(data["macro_id"][0]) == macro["macro_id"]
            assert data["best_state_index"][0] == macro["best_state_index"]
            assert data["best_displacements"].dtype == np.float64  # 位移类型
        metrics = json.loads(  # JSON records 契约
            (macro_dir / "gradient_metrics.json").read_text(encoding="utf-8"))
        records = metrics["records"]
        assert [record["state_index"] for record in records] == [0, 1]  # 基线+1
        for field in ("total_loss", "nominal_l2_loss", "process_l2_loss",
                      "pvband_loss", "l2", "pvband", "epe",
                      "displaced_segments", "elapsed_seconds"):
            assert field in records[0], field  # 记录字段齐全
        assert metrics["best_state_index"] == macro["best_state_index"]

    def test_final_layout_and_lithography(self, run):
        """最终合并 GDS 与逐 tile 光刻 PNG/manifest 落盘。"""
        tmp, _ = run  # 解包
        assert (tmp / "final.gds").is_file()  # 一次合并的最终版图
        assert int(_coverage(tmp / "final.gds").area()) > 0  # 覆盖非空
        out_dir = tmp / "work" / "final_lithography"  # 留档目录
        manifest = json.loads(  # 读清单
            (out_dir / "manifest.json").read_text(encoding="utf-8"))
        for tile in manifest["tiles"]:  # 逐 tile 检查
            assert (out_dir / tile["nominal_png"]).is_file()  # 连续 PNG
            assert (out_dir / tile["binary_png"]).is_file()  # 二值 PNG


class TestCudaStatsDevice:
    """CUDA 峰值统计必须显式传入目标设备（审查问题 1）。"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_peak_stats_receive_explicit_device(self, tmp_path, monkeypatch):
        """reset/max 收到显式 torch.device（透传真实现的真实小跑）。"""
        gds = _write_gds(tmp_path)  # 生成版图
        received = {"reset": [], "max": []}  # 记录收到的 device 参数
        real_reset = torch.cuda.reset_peak_memory_stats  # 真实现
        real_max = torch.cuda.max_memory_allocated  # 真实现

        def _spy_reset(device=None):
            """记录并透传重置。"""
            received["reset"].append(device)
            return real_reset(device)

        def _spy_max(device=None):
            """记录并透传读取。"""
            received["max"].append(device)
            return real_max(device)

        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _spy_reset)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", _spy_max)
        config = _write_config(tmp_path, gds, device="cuda")  # 显式 CUDA
        summary = workflow.run_gradient_mbopc(config)  # 真实小跑
        assert summary["cuda_peak_bytes"] >= 0  # 正常读到峰值
        # 两次调用都收到显式 device 对象（非 None 即证明不再依赖当前设备）。
        assert received["reset"] == [torch.device("cuda")]
        assert received["max"] == [torch.device("cuda")]


class TestGradientMultiMacro:
    """多 macro 独立求解与一次合并（TEST-015）。"""

    def test_multi_macro_artifacts_and_merge(self, tmp_path, monkeypatch):
        """2×2 macro 各自独立产物、merge 恰一次、最终 GDS 落盘。"""
        gds = _write_gds(tmp_path)  # 生成版图
        config = _write_config(tmp_path, gds, macro_grid="[2, 2]")  # 配置
        calls = []  # 合并计数
        real = workflow.merge_macro_results  # 原函数

        def _counting(*args, **kwargs):
            """计数合并调用并透传。"""
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(workflow, "merge_macro_results", _counting)
        summary = workflow.run_gradient_mbopc(config)  # 完整流程
        assert calls == [1]  # 恰一次合并
        assert summary["macro_count"] == 4  # 2×2
        ids = {macro["macro_id"] for macro in summary["macros"]}
        assert len(ids) == 4  # 四个不同 macro
        for macro in summary["macros"]:  # 逐 macro 三件梯度产物
            macro_dir = tmp_path / "work" / "macros" / macro["macro_id"]
            for name in ("gradient_result.npz", "gradient_metrics.json",
                         "best.gds"):
                assert (macro_dir / name).is_file(), name
        assert (tmp_path / "final.gds").is_file()  # 最终版图

    def test_macro_order_does_not_change_coverage(self, tmp_path, monkeypatch):
        """macro 正逆序求解的最终物理覆盖 XOR 为零（独立 macro 性质）。"""
        import main._macro_pipeline as macro_pipeline  # plan_macros 宿主
        gds = _write_gds(tmp_path)  # 生成版图
        real_plan = macro_pipeline.plan_macros  # 原函数

        def _reversed(*args, **kwargs):
            """反转 macro 规划顺序。"""
            return tuple(reversed(real_plan(*args, **kwargs)))

        finals = {}  # 顺序 → 最终版图
        for tag, reverse in (("forward", False), ("reverse", True)):
            base = tmp_path / tag  # 独立目录
            base.mkdir()  # 创建
            config = _write_config(base, gds, macro_grid="[2, 2]")  # 配置
            if reverse:
                monkeypatch.setattr(macro_pipeline, "plan_macros", _reversed)
            finals[tag] = workflow.run_gradient_mbopc(config)["final_layout"]
            monkeypatch.undo()  # 立即恢复
        assert int((_coverage(finals["forward"]) ^  # 覆盖一致
                    _coverage(finals["reverse"])).area()) == 0


class TestGradientProgress:
    """进度计数、异常收尾与资源报告（TEST-016）。"""

    def test_progress_counts_completed_tiles(self, tmp_path, monkeypatch):
        """进度更新总数恰等于（iterations+1）×core 数。"""
        import tqdm as tqdm_module  # 进度库宿主
        gds = _write_gds(tmp_path)  # 生成版图
        totals = {"updates": 0, "closed": 0, "total": None}  # 观测
        real_tqdm = tqdm_module.tqdm  # 原类

        class _SpyBar:
            """记录 update/close 的进度条代理。"""

            def __init__(self, *args, **kwargs):
                self._bar = real_tqdm(*args, **kwargs)  # 真实条
                totals["total"] = kwargs.get("total")  # 声明的总数

            def update(self, value):
                """计数更新并透传。"""
                totals["updates"] += value
                return self._bar.update(value)

            def close(self):
                """计数关闭并透传。"""
                totals["closed"] += 1
                return self._bar.close()

        monkeypatch.setattr(tqdm_module, "tqdm", _SpyBar)
        config = _write_config(tmp_path, gds, show_progress="true")  # 开进度
        summary = workflow.run_gradient_mbopc(config)  # 完整流程
        core_count = summary["core_count"]  # tile 总数
        assert totals["total"] == (1 + 1) * core_count  # 声明总数正确
        assert totals["updates"] == totals["total"]  # 更新恰走满
        assert totals["closed"] >= 1  # 正常关闭

    def test_progress_bar_closes_on_error(self, tmp_path, monkeypatch):
        """求解异常时进度条仍被关闭，异常原样传播。"""
        import tqdm as tqdm_module  # 进度库宿主
        gds = _write_gds(tmp_path)  # 生成版图
        closed = {"n": 0}  # 关闭计数
        real_tqdm = tqdm_module.tqdm  # 原类

        class _SpyBar:
            """只记录 close 的进度条代理。"""

            def __init__(self, *args, **kwargs):
                self._bar = real_tqdm(*args, **kwargs)  # 真实条

            def update(self, value):
                """透传更新。"""
                return self._bar.update(value)

            def close(self):
                """计数关闭并透传。"""
                closed["n"] += 1
                return self._bar.close()

        monkeypatch.setattr(tqdm_module, "tqdm", _SpyBar)

        def exploding(*args, **kwargs):
            """模拟求解期未知程序异常。"""
            raise RuntimeError("求解崩溃")

        monkeypatch.setattr(workflow, "optimize_gradient_macro", exploding)
        config = _write_config(tmp_path, gds, show_progress="true")  # 开进度
        with pytest.raises(RuntimeError, match="求解崩溃"):  # 异常传播
            workflow.run_gradient_mbopc(config)
        assert closed["n"] >= 1  # finally 收尾了进度条
        assert not (tmp_path / "work" / "summary.json").exists()  # 无半份摘要


class TestGradientDirectExecution:
    """梯度入口从仓库外直接运行（TEST-013）。"""

    def _run(self, script, config_path, cwd):
        """以子进程直跑入口脚本。"""
        import os  # 环境变量
        env = {**os.environ}  # 继承环境
        return subprocess.run(  # 直跑（免安装）
            [sys.executable, str(script), str(config_path)], cwd=cwd,
            capture_output=True, text=True, timeout=600, check=False, env=env)

    def test_runs_outside_repository(self, tmp_path, project_root):
        """从仓库外直跑梯度入口退出码 0 并产出全部关键标记。"""
        gds = _write_gds(tmp_path)  # 生成版图
        config = _write_config(tmp_path, gds, macro_grid="[2, 2]")  # 多 macro
        script = project_root / "main" / "run_gradient_mbopc.py"  # 入口
        completed = self._run(script, config, tmp_path)  # cwd=仓库外
        assert completed.returncode == 0, completed.stderr  # 正常退出
        for marker in ("梯度 MB-OPC 执行完成", "device：",  # 摘要标记
                       "loss 权重", "合并", "最终版图"):
            assert marker in completed.stdout, marker

    def test_missing_argument_returns_usage(self, project_root):
        """无参数运行打印用法并以退出码 2 结束。"""
        script = project_root / "main" / "run_gradient_mbopc.py"  # 入口
        completed = subprocess.run(  # 无参数直跑
            [sys.executable, str(script)], cwd=project_root,
            capture_output=True, text=True, timeout=60, check=False)
        assert completed.returncode == 2  # 参数错误退出码
        assert "用法" in completed.stderr  # 用法提示
