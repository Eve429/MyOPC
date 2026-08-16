"""Macro–Core 两级网格双轮迭代管线的直接运行入口（阶段 0–3）。"""

import os  # 原子替换 result NPZ 的临时文件
import sys  # 把仓库根加入模块路径，保证免安装直接运行
import tempfile  # 创建与目标同目录的临时文件
import time  # perf_counter 阶段计时
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # 位移状态与统计数组的载体
import psutil  # 阶段 RSS 峰值测量；缺失时直接 ImportError 不降级
from tomllib import loads as toml_loads  # Python 3.12 标准库 TOML 解析

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from layout import LayerSpec, LayoutDB  # 回零验证的版图查询
from main._macro_pipeline import (  # 两个真实流程共用的 macro 生命周期
    atomic_write_json,
    exact_dbu,
    load_macro_config,
    merge_macro_results,
    prepare_problems,
    write_macro_gds,
)
from opc.input import rasterize_mask_canvas  # run_round 的居中光刻画布
from opc.input.edge import MacroProblem, reconstruct_region  # problem 加载与重建

_RESULT_FORMAT_VERSION = 1  # 每轮 result NPZ 结构版本


def load_validation_deltas(path: str | Path) -> tuple[Decimal, Decimal]:
    """解析验证专属 [iteration] 段并把双轮位移冻结为精确 [+2,-2] nm。"""
    config_path = Path(path).expanduser().resolve()  # 配置绝对路径
    raw = toml_loads(config_path.read_text(encoding="utf-8"))  # 解析 TOML 文本
    section = raw.get("iteration", {})  # 验证段
    unknown = set(section) - {"round_deltas_nm"}  # 段内未知键
    if unknown:  # 拒绝拼错键
        raise ValueError(f"[iteration] 含未知键：{sorted(unknown)}")
    if "round_deltas_nm" not in section:  # 缺必填键
        raise ValueError("[iteration] 缺少必填键：['round_deltas_nm']")
    deltas = section["round_deltas_nm"]  # 双轮位移列表
    if not isinstance(deltas, list) or len(deltas) != 2:  # 恰好两轮
        raise ValueError("round_deltas_nm 必须是恰好两个数值的列表")
    pair = (Decimal(str(deltas[0])), Decimal(str(deltas[1])))  # 十进制保存
    # 双轮位移冻结为 [+2nm,-2nm]（设计文档 §5.2）：只查和为零会放行 [3,-3] 等配置，
    # 让「回零验证」失去对固定步长的约束力；DBU 落格点检查在 run 内用实际 dbu 换算。
    if pair != (Decimal(2), Decimal(-2)):  # 值不符即拒绝
        raise ValueError(  # 报错列出冻结要求与实际值
            f"round_deltas_nm 当前冻结为 [+2nm, -2nm]，实际为 {list(pair)}")
    return pair  # 返回精确十进制对


def run_round(plan: dict, round_index: int, delta_dbu: int) -> dict:
    """执行一个全局轮次：逐 macro、逐 core 更新并保存状态与完整候选 GDS。"""
    started = time.perf_counter()  # 轮次计时
    process = psutil.Process()  # RSS 采样
    peak_rss = process.memory_info().rss  # 峰值初值
    work_dir = Path(plan["work_dir"])  # 工作目录
    round_dir = work_dir / f"round_{round_index:03d}"  # 本轮目录
    (round_dir / "results").mkdir(parents=True, exist_ok=True)  # result 目录
    (round_dir / "gds").mkdir(parents=True, exist_ok=True)  # GDS 目录
    dbu_um = float(plan["dbu_um"])  # 写 GDS 所需 DBU
    pixel_dbu = int(plan["pixel_dbu"])  # 栅格像素
    canvas_pixels = int(plan["canvas_pixels"])  # 画布尺寸
    previous_dir = (work_dir / f"round_{round_index - 1:03d}"  # 上一轮目录
                    if round_index > 1 else None)  # 首轮无上一轮
    for entry in plan["macros"]:  # 按计划顺序逐 macro
        macro_id = entry["macro_id"]  # macro 编号
        problem = MacroProblem.load(work_dir / "problems" / f"{macro_id}.npz")  # 加载 problem
        segment_count = problem.segments.segment_count  # 段数
        if previous_dir is None:  # 第一轮从全零位移开始
            current = np.zeros(segment_count, dtype=np.float64)  # 全零状态
        else:  # 后续轮从上一轮位移续读，不从零重启
            with np.load(previous_dir / "results" / f"{macro_id}.npz",  # 上一轮 result
                         allow_pickle=False) as previous:  # 只读打开
                current = np.ascontiguousarray(  # 拷贝为可写数组
                    previous["segment_displacements"], dtype=np.float64)  # 位移状态
            if len(current) != segment_count:  # 长度不符直接失败
                raise ValueError(f"{macro_id} 上一轮位移长度与 problem 段数不符")
        following = current.copy()  # 本轮新状态（读旧写新）
        written = np.zeros(segment_count, dtype=np.bool_)  # 每段恰写一次标记
        for core_index in range(problem.macro.core_count):  # 局部行优先逐 core
            owner_segments = problem.owner_segments_for_core(core_index)  # 唯一可写段
            if np.any(written[owner_segments]):  # 重复写立即失败
                raise RuntimeError(f"{macro_id} core{core_index} 出现重复写入")
            following[owner_segments] += float(delta_dbu)  # owner 段累计当前轮位移
            written[owner_segments] = True  # 标记已写
        # 轮末守卫：全部 owner 段恰写一次，context 段（owner=-1）从未被写。
        if not np.array_equal(written, problem.owner_indices >= 0):  # 写集核对
            raise RuntimeError(f"{macro_id} owner 段未全部写入或写入了 context 段")
        region = reconstruct_region(problem, following)  # 按新位移重建完整候选 Region
        core_count = problem.macro.core_count  # core 数
        transmission_sums = np.zeros(core_count, dtype=np.float64)  # 每 core 透光率和
        for core_index in range(core_count):  # 逐 core 构造居中画布
            spec = problem.macro.core(core_index)  # 即时构造 CoreSpec
            canvas = rasterize_mask_canvas(  # 居中 256×256 透光率画布
                region, spec.context_box, pixel_dbu, canvas_pixels,  # 几何与契约
                polarity=problem.polarity)  # 极性
            transmission_sums[core_index] = float(canvas.sum(dtype=np.float64))  # 记录总和
            del canvas  # 只保留当前一张，控制峰值内存
            if not np.isfinite(transmission_sums[core_index]):  # 每核必须产出有限值
                raise RuntimeError(f"{macro_id} core{core_index} transmission 非有限")
        result_path = round_dir / "results" / f"{macro_id}.npz"  # result 路径
        handle, temporary_name = tempfile.mkstemp(  # 同目录临时文件
            prefix=f".{macro_id}-", suffix=".npz", dir=result_path.parent)  # 命名
        os.close(handle)  # 关闭句柄
        temporary = Path(temporary_name)  # Path 化
        try:  # 写出 result NPZ
            with temporary.open("wb") as stream:  # 二进制写
                np.savez(stream,  # 不压缩 NPZ
                         format_version=np.array([_RESULT_FORMAT_VERSION], np.int32),  # 版本
                         macro_id=np.array([macro_id]),  # macro 编号
                         round_index=np.array([round_index], np.int32),  # 轮次
                         round_delta_dbu=np.array([float(delta_dbu)], np.float64),  # 本轮位移
                         segment_displacements=following,  # 累计位移状态
                         written_owner_count=np.array([int(written.sum())], np.int64),  # 写入计数
                         core_transmission_sums=transmission_sums)  # 每 core 总和
            os.replace(temporary, result_path)  # 原子替换
        finally:  # 清理
            if temporary.exists():  # 尚存
                temporary.unlink()  # 删除
        write_macro_gds(  # 完整候选 GDS（RESULT Cell，不裁 ownership）
            problem, region, round_dir / "gds" / f"{macro_id}.gds", dbu_um)  # 写盘
        peak_rss = max(peak_rss, process.memory_info().rss)  # 采样峰值
        del problem, region  # 释放当前 macro 再处理下一个
    return {  # 轮次摘要
        "round_index": round_index,  # 轮次号
        "round_delta_dbu": delta_dbu,  # 本轮位移
        "macro_result_count": len(plan["macros"]),  # result 数量
        "macro_gds_count": len(plan["macros"]),  # GDS 数量
        "round_seconds": time.perf_counter() - started,  # 轮次耗时
        "iteration_peak_rss_bytes": peak_rss}  # 轮次 RSS 峰值


def collect_round_macro_gds(plan: dict, round_index: int) -> dict[str, Path]:
    """校验指定轮次 result 一致后，返回 macro_id 到该轮 GDS 的显式映射。"""
    work_dir = Path(plan["work_dir"])  # 工作目录
    round_dir = work_dir / f"round_{round_index:03d}"  # 轮次目录
    mapping: dict[str, Path] = {}  # 显式映射
    for entry in plan["macros"]:  # 逐 macro
        macro_id = entry["macro_id"]  # macro 编号
        # result 轮次一致性：result NPZ 记录的轮次必须与本次合并轮次相同，
        # 防止把旧轮 GDS 当作最新状态交给合并。
        with np.load(round_dir / "results" / f"{macro_id}.npz",  # 读 result
                     allow_pickle=False) as data:  # 只读
            if int(data["round_index"][0]) != round_index:  # 轮次不符
                raise ValueError(f"{macro_id} result 轮次与合并轮次不一致")  # 失败
        mapping[macro_id] = round_dir / "gds" / f"{macro_id}.gds"  # 记录映射
    return mapping  # 返回显式映射


def run(config_path: str | Path) -> dict:
    """按准备、两轮迭代、最终合并、验证顺序执行完整流程并返回摘要。"""
    total_started = time.perf_counter()  # 全流程计时
    config = load_macro_config(  # 宏管线六段（放行验证专属 iteration 段）
        config_path, extra_sections=("iteration",))  # 严格加载
    deltas = load_validation_deltas(config_path)  # [+2,-2] nm 冻结检查
    plan = prepare_problems(config)  # 阶段 0/1（共用生命周期）
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # plan 内 DBU 的 nm 值
    deltas_dbu = tuple(  # 双轮位移精确换算（2nm 落不了格点在此失败）
        exact_dbu(value, dbu_nm, "round_deltas_nm") for value in deltas)
    round_one = run_round(plan, 1, deltas_dbu[0])  # 第一轮 +2 nm
    round_two = run_round(plan, 2, deltas_dbu[1])  # 第二轮 -2 nm
    macro_gds = collect_round_macro_gds(plan, 2)  # 校验轮次并显式映射
    merge_started = time.perf_counter()  # 合并计时
    final_path = merge_macro_results(  # 阶段 3 合并写出（共用生命周期）
        plan, macro_gds, config.final_layout,  # 显式映射与输出路径
        cell_mode=config.final_cell_mode)  # Cell 模式
    merge_seconds = time.perf_counter() - merge_started  # 合并耗时
    # 说明：单次采样无法回溯进程历史峰值，此处取合并完成后的即时 RSS 作为
    # 近似上界，具体口径在测试报告如实记录。
    merge_peak_rss = psutil.Process().memory_info().rss  # 合并后即时 RSS
    # 回零验证：第二轮位移精确为零后，最终覆盖与原始目标层 XOR 面积必须为零。
    layer = LayerSpec(plan["layer"][0], plan["layer"][1])  # 目标层
    with LayoutDB.open(final_path) as database:  # 回读最终版图
        final_bounds = database.layer_bbox(layer)  # 最终层真实包络
        if final_bounds is None:  # 空层即回零失败
            raise RuntimeError("最终版图目标层为空")
        final_batch = database.query(  # 层包络内查询物化
            [layer], final_bounds).materialize()  # 物化
    with LayoutDB.open(plan["layout"], plan["top_cell"]) as database:  # 原始版图
        source_bounds = database.layer_bbox(layer)  # 原始层真实包络
        if source_bounds is None:  # 空层无法比较
            raise RuntimeError("原始版图目标层为空")
        source_batch = database.query(  # 包络内完整相交物化
            [layer], source_bounds  #   （不引入查询框边；包络恰含全部图形）
        ).materialize_intersecting()  # 完整原图形
    final_xor_area = int(  # 回零 XOR 面积
        (final_batch.region(layer) ^ source_batch.region(layer)).area())  # 比较
    if final_xor_area != 0:  # 第二轮回零后 XOR 非零即为回零失败
        raise RuntimeError(f"第二轮回零后最终 XOR 面积非零：{final_xor_area}")  # 失败
    summary = {  # 完整摘要（§16 契约字段）
        "macro_count": plan["macro_count"],  # macro 总数
        "core_count": plan["core_count"],  # core 总数
        "problem_count": len(plan["macros"]),  # problem 总数
        "round_count": 2,  # 恰好两轮
        "round_001_macro_gds_count": round_one["macro_gds_count"],  # 第一轮 GDS 数
        "round_002_macro_gds_count": round_two["macro_gds_count"],  # 第二轮 GDS 数
        "segment_count_sum": plan["segment_count_sum"],  # 段数总计
        "membership_count_sum": plan["membership_count_sum"],  # membership 总计
        "maximum_problem_bytes": plan["maximum_problem_bytes"],  # 最大 problem
        "maximum_problem_macro_id": plan["maximum_problem_macro_id"],  # 最大 problem macro
        "prepare_seconds": plan["prepare_seconds"],  # 准备耗时
        "round_001_seconds": round_one["round_seconds"],  # 第一轮耗时
        "round_002_seconds": round_two["round_seconds"],  # 第二轮耗时
        "merge_seconds": merge_seconds,  # 合并耗时
        "total_seconds": time.perf_counter() - total_started,  # 总耗时
        "prepare_peak_rss_bytes": plan["prepare_peak_rss_bytes"],  # 准备 RSS 峰值
        "iteration_peak_rss_bytes": round_two["iteration_peak_rss_bytes"],  # 迭代 RSS 峰值
        "merge_peak_rss_bytes": merge_peak_rss,  # 合并 RSS 峰值
        "final_cell_mode": plan["final_cell_mode"],  # Cell 模式
        "final_layout": str(final_path),  # 最终版图
        "final_xor_area": final_xor_area}  # 回零 XOR 面积
    atomic_write_json(config.work_dir / "summary.json", summary)  # 落盘摘要
    return summary  # 返回摘要


def main() -> int:
    """读取唯一位置参数 config，运行流程并打印中文摘要。"""
    if len(sys.argv) != 2:  # 参数数量不符
        print("用法：python main/run_macro_pipeline.py <config.toml>", file=sys.stderr)  # 提示
        return 2  # 参数错误退出码
    summary = run(sys.argv[1])  # 执行完整流程
    print("Macro–Core 双轮迭代管线执行完成：")  # 摘要标题
    print(f"  macro 数：{summary['macro_count']}，core 数：{summary['core_count']}")  # 网格规模
    print(f"  段数总计：{summary['segment_count_sum']}，membership 总计："  # 规模
          f"{summary['membership_count_sum']}")  # 规模续
    print(f"  每轮 macro GDS：{summary['round_001_macro_gds_count']} × 2 轮")  # 产物数量
    print(f"  准备 {summary['prepare_seconds']:.2f}s，第一轮 "  # 耗时
          f"{summary['round_001_seconds']:.2f}s，第二轮 "  # 耗时续
          f"{summary['round_002_seconds']:.2f}s，合并 "  # 耗时续
          f"{summary['merge_seconds']:.2f}s，总计 {summary['total_seconds']:.2f}s")  # 耗时总
    print(f"  最终 XOR 面积：{summary['final_xor_area']}（应为 0）")  # 回零验证
    print(f"  最终版图：{summary['final_layout']}（{summary['final_cell_mode']}）")  # 输出位置
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
