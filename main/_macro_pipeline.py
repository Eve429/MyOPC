"""多个真实流程共用的 macro 生命周期：problem 准备、候选写出与最终合并。"""

import json  # 序列化 plan.json
import os  # 原子替换与文件系统操作
import sys  # 把仓库根加入模块路径，保证免安装直接运行
import tempfile  # 创建与目标同目录的临时文件
import time  # perf_counter 阶段计时
from collections.abc import Mapping  # merge 映射参数的只读类型
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象
from typing import Literal  # cell_mode 的字面量类型

import klayout.db as kdb  # 写出 macro GDS 与所有权裁剪的原生版图对象
import psutil  # 阶段 RSS 峰值测量；缺失时直接 ImportError 不降级

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from geometry import GeometryPatch, PatchWriter  # 权威 patch 与双模式最终写出
from layout import DbuBox, LayerSpec, LayoutDB  # 版图打开、层规格与坐标框
from main.configuration import (  # 统一配置体系（按业务划分的输入）
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
)
from opc.input import plan_macros  # 两级网格规划
from opc.input.edge import MacroProblem, prepare_macro_problem  # problem 构造
from opc.input.edge.fragmentation import FragmentationConfig  # 边段配置

_PLAN_FORMAT_VERSION = 1  # plan.json 结构版本


def exact_dbu(value_nm: Decimal, dbu_nm: Decimal, name: str) -> int:
    """把必须落在版图格点上的 nm 参数精确转换为整数 DBU。"""
    quotient = value_nm / dbu_nm  # 十进制除法，无二进制浮点误差
    if quotient != quotient.to_integral_value():  # 非整数倍即无法精确落格点
        raise ValueError(  # 报错必须写明参数名、nm 值与当前 dbu_nm
            f"{name}={value_nm} nm 无法精确换算为 {dbu_nm} nm/DBU 的整数倍")
    return int(quotient)  # 精确整数 DBU


def atomic_write_json(path: Path, payload: dict) -> Path:
    """把 JSON 载荷经同目录临时文件原子写出，避免留下半截 plan。"""
    handle, temporary_name = tempfile.mkstemp(  # 与目标同目录同卷
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent)  # 临时文件名
    os.close(handle)  # 只借用文件名，内容用文本模式重写
    temporary = Path(temporary_name)  # Path 化
    try:  # 写入并原子替换
        with temporary.open("w", encoding="utf-8") as stream:  # 文本写
            json.dump(payload, stream, ensure_ascii=False, indent=2)  # 中文可读输出
        os.replace(temporary, path)  # 原子替换目标
    finally:  # 无论成败清理临时文件
        if temporary.exists():  # 尚存即删除
            temporary.unlink()  # 删除
    return path  # 返回最终路径


def prepare_problems(layout: LayoutConfig, partition: PartitionConfig,
                     litho: LithographyConfig, edge: EdgeConfig,
                     output: OutputConfig) -> dict:
    """执行阶段 0/1，逐 macro 生成 problem，并写出 plan.json。"""
    if output.work_dir is None:  # 本流程要求工作目录（单遍等流程可不填）
        raise ValueError("此流程要求 [output].work_dir")  # 消费方显式报错
    layer = LayerSpec(layout.layer, layout.datatype)  # 目标层规格
    started = time.perf_counter()  # 阶段计时起点
    process = psutil.Process()  # RSS 采样进程对象
    peak_rss = process.memory_info().rss  # 峰值初值
    with LayoutDB.open(layout.layout, layout.top_cell) as database:  # 打开并自动关闭
        top_cell_name = database.top_cell_name  # 在库存活期内捕获顶层名
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生逐层，不物化）
        if bounds is None:  # 目标层在顶层子树内无图形
            raise ValueError(  # 空层无法规划网格
                f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")  # 报层号
        # 全部 nm 参数精确换算：不能整除直接失败，不四舍五入吸收误差。
        core_dbu = exact_dbu(partition.core_size_nm, dbu_nm, "core_size_nm")  # core
        context_dbu = exact_dbu(partition.context_nm, dbu_nm, "context_nm")  # context
        pixel_dbu = exact_dbu(litho.pixel_nm, dbu_nm, "pixel_nm")  # pixel
        corner_dbu = exact_dbu(edge.corner_nm, dbu_nm, "corner_nm")  # 拐角段
        segment_dbu = exact_dbu(edge.segment_nm, dbu_nm, "segment_nm")  # 中段
        max_displacement_dbu = exact_dbu(  # 位移上限
            edge.max_displacement_nm, dbu_nm, "max_displacement_nm")
        if max_displacement_dbu > context_dbu:  # context 必须覆盖最大位移
            raise ValueError("context_nm 必须不小于 max_displacement_nm")
        # 边段数值约束（正长度、segment≥2×corner、非负位移）由 FragmentationConfig
        # 构造统一校验，这里不重复检查。
        fragmentation = FragmentationConfig(  # DBU 级边段配置
            corner_length_dbu=float(corner_dbu),  # 拐角段
            max_segment_length_dbu=float(segment_dbu),  # 中段上限
            max_displacement_dbu=float(max_displacement_dbu),  # 位移上限
            miter_limit=edge.miter_limit)  # miter
        macros = plan_macros(  # 两级网格规划（内部完成像素/画布校验）
            bounds, macro_grid=partition.macro_grid, macro_size_dbu=(
                exact_dbu(partition.macro_size_nm, dbu_nm, "macro_size_nm")  # 尺寸模式换算
                if partition.macro_size_nm is not None else None),  # 数量模式为空
            core_size_dbu=core_dbu, context_dbu=context_dbu,  # core/context
            pixel_dbu=pixel_dbu, canvas_pixels=litho.canvas_pixels)  # 画布契约
        # 阶段 0 步骤 7：ownership 复核——面积和恰等于父框即无正面积重叠。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:  # 面积守恒
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")
        problems_dir = output.work_dir / "problems"  # problem 存放目录
        problems_dir.mkdir(parents=True, exist_ok=True)  # 创建目录结构
        entries = []  # 逐 macro 计划条目
        segment_count_sum = 0  # 段数累计
        membership_count_sum = 0  # membership 累计
        maximum_problem_bytes = 0  # 最大 problem 字节数
        maximum_problem_macro_id = ""  # 最大 problem 所属 macro
        for macro in macros:  # 按行优先顺序逐 macro 准备
            batch = database.query(  # 完整相交物化（不裁剪 occurrence）
                [layer], macro.query_box).materialize_intersecting()  # 惰性查询执行
            problem = prepare_macro_problem(  # 一次完成提边/分段/切线分裂/ownership
                batch, layer, layout.polarity, fragmentation, macro)  # 阶段 1 核心
            problem_path = problem.save(problems_dir / f"{macro.macro_id}.npz")  # 原子落盘
            problem_bytes = problem_path.stat().st_size  # 文件字节数即持久字节数
            segment_count_sum += problem.segments.segment_count  # 累计段数
            membership_count_sum += len(problem.member_segment_indices)  # 累计 membership
            if problem_bytes > maximum_problem_bytes:  # 更新最大 problem
                maximum_problem_bytes = problem_bytes  # 记录字节
                maximum_problem_macro_id = macro.macro_id  # 记录 macro
            entries.append({  # 单 macro 计划条目
                "macro_id": macro.macro_id,  # 行优先编号
                "ownership_box": [macro.ownership_box.left, macro.ownership_box.bottom,  # 左下
                                  macro.ownership_box.right, macro.ownership_box.top],  # 右上
                "core_count": macro.core_count,  # core 总数
                "segment_count": problem.segments.segment_count,  # 段数
                "membership_count": len(problem.member_segment_indices),  # membership 数
                "problem_file": str(problem_path),  # NPZ 路径
                "problem_bytes": problem_bytes})  # NPZ 字节
            peak_rss = max(peak_rss, process.memory_info().rss)  # 采样峰值
            del batch, problem  # 立即释放当前 macro 大对象再进入下一个
    # 全部 problem 成功且 LayoutDB 已关闭，才允许写出表示「准备完成」的 plan。
    prepare_seconds = time.perf_counter() - started  # 阶段耗时
    plan = {  # 完整计划（后续阶段唯一允许消费的产物）
        "format_version": _PLAN_FORMAT_VERSION,  # 计划版本
        "layout": str(layout.layout),  # 输入版图
        "top_cell": top_cell_name,  # 实际选定的顶层 Cell 名
        "dbu_um": float(dbu_nm / 1000),  # DBU 微米值（写出最终 GDS 用）
        "layer": [layer.layer, layer.datatype],  # 目标层
        "polarity": layout.polarity.value,  # 极性
        "core_size_dbu": core_dbu,  # core
        "context_dbu": context_dbu,  # context
        "pixel_dbu": pixel_dbu,  # pixel
        "canvas_pixels": litho.canvas_pixels,  # canvas
        "macro_count": len(macros),  # macro 总数
        "core_count": sum(macro.core_count for macro in macros),  # core 总数
        "fragmentation": {  # 边段配置
            "corner_length_dbu": float(corner_dbu),  # 拐角段
            "max_segment_length_dbu": float(segment_dbu),  # 中段上限
            "max_displacement_dbu": float(max_displacement_dbu),  # 位移上限
            "miter_limit": edge.miter_limit},  # miter
        "work_dir": str(output.work_dir),  # 工作目录
        "final_layout": str(output.final_layout),  # 最终版图
        "final_cell_mode": output.final_cell_mode,  # Cell 模式
        "macros": entries,  # 逐 macro 条目
        "segment_count_sum": segment_count_sum,  # 段数总计
        "membership_count_sum": membership_count_sum,  # membership 总计
        "maximum_problem_bytes": maximum_problem_bytes,  # 最大 problem 字节
        "maximum_problem_macro_id": maximum_problem_macro_id,  # 最大 problem macro
        "prepare_seconds": prepare_seconds,  # 准备耗时
        "prepare_peak_rss_bytes": peak_rss}  # 准备 RSS 峰值
    atomic_write_json(output.work_dir / "plan.json", plan)  # 原子写出计划
    return plan  # 返回内存计划供调用方直接消费


def write_macro_gds(problem: MacroProblem, region: kdb.Region, path: Path,
                    dbu_um: float) -> Path:
    """把单 macro 当前完整候选 Region 写入 RESULT Cell，供检查和最终合并。"""
    layout = kdb.Layout()  # 独立原生版图对象
    layout.dbu = dbu_um  # 与源版图一致的 DBU，整数坐标物理尺寸不变
    cell = layout.create_cell("RESULT")  # 固定结果 Cell 名
    index = layout.layer(kdb.LayerInfo(  # 目标层
        problem.layer.layer, problem.layer.datatype))  # 层号/ datatype
    region.insert_into(layout, cell.cell_index(), index)  # 插入完整候选 Region
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    handle, temporary_name = tempfile.mkstemp(  # 同目录临时文件
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)  # 命名
    os.close(handle)  # 关闭句柄
    temporary = Path(temporary_name)  # Path 化
    try:  # 写出并原子替换
        layout.write(str(temporary))  # KLayout 完整写出
        os.replace(temporary, path)  # 原子替换
    finally:  # 清理
        if temporary.exists():  # 尚存
            temporary.unlink()  # 删除
    return path  # 返回路径


def merge_macro_results(
        plan: dict, macro_gds_paths: Mapping[str, Path], output_path: Path, *,
        cell_mode: Literal["single_cell", "macro_cells"]) -> Path:
    """按 plan 选择各 macro ownership 权威覆盖并写出一个全局结果。"""
    if cell_mode not in ("single_cell", "macro_cells"):  # 模式枚举校验
        raise ValueError(f"未知 cell_mode：{cell_mode}")  # 拒绝拼错
    layer = LayerSpec(plan["layer"][0], plan["layer"][1])  # 目标层
    dbu_um = float(plan["dbu_um"])  # 源版图 DBU
    identifiers = set()  # macro ID 去重集合
    for entry in plan["macros"]:  # 先核对计划条目自身无重复
        macro_id = entry["macro_id"]  # macro 编号
        if macro_id in identifiers:  # 重复 macro ID
            raise ValueError(f"重复 macro ID：{macro_id}")  # 明确失败
        identifiers.add(macro_id)  # 记录
    # 映射完整性：调用方必须显式给出每个 macro 的 GDS，函数不猜任何路径。
    missing = sorted(identifiers - set(macro_gds_paths))  # 缺失映射
    if missing:  # 有 macro 没给 GDS
        raise ValueError(f"macro GDS 映射缺失：{missing}")  # 明确失败
    extra = sorted(set(macro_gds_paths) - identifiers)  # 多余映射
    if extra:  # 映射含计划外的 macro
        raise ValueError(f"macro GDS 映射多余：{extra}")  # 明确失败
    patches: list[GeometryPatch] = []  # 权威 patch 集合
    area_before = 0  # merge 前覆盖面积
    for entry in plan["macros"]:  # 按计划顺序逐 macro
        macro_id = entry["macro_id"]  # macro 编号
        gds_path = Path(macro_gds_paths[macro_id])  # 该 macro 的 GDS 路径
        if not gds_path.is_file():  # 缺失 macro GDS
            raise FileNotFoundError(f"缺失 macro GDS：{gds_path}")  # 明确失败
        with LayoutDB.open(gds_path) as database:  # 回读完整候选
            layer_bounds = database.layer_bbox(layer)  # 候选层真实包络
            if layer_bounds is None:  # 候选 GDS 目标层为空
                raise RuntimeError(f"{macro_id} 候选 GDS 目标层为空")
            batch = database.query(  # 层包络内查询物化（不用魔法框：
                [layer], layer_bounds).materialize()  #   ±2^30 只盖 int32 域一半)
        region = batch.region(layer)  # 候选 Region
        if not region.has_valid_polygons():  # 无效 polygon
            raise RuntimeError(f"{macro_id} 候选 Region 含无效 polygon")  # 明确失败
        ownership = DbuBox(*entry["ownership_box"])  # macro ownership 框
        # 权威覆盖选择：完整候选只贡献自身 ownership 内的部分，消除相邻 macro
        # context 的正面积重复；裁剪不是最终结果，seam 由写出端全局 merge 消除。
        clipped = region & kdb.Region(ownership.to_native())  # 精确相交
        area_before += int(clipped.area())  # 统计覆盖面积
        patches.append(GeometryPatch(macro_id, layer, clipped, ownership))  # 收集
    written = PatchWriter.write_macro_results(  # 按配置模式写出最终版图
        patches, output_path, dbu_um,  # patch 集合与 DBU
        cell_mode=cell_mode)  # single_cell 或 macro_cells
    # 回读验证：merge/normalize 只能改变表示方式，不得改变物理覆盖面积。
    # 逐 macro 在自身 ownership 窗口统计面积后累加——ownership 半开不重叠、
    # 最终图形不越出 bbox，分块求和与全量面积数学等价，且避免第二个全量
    # Region 常驻（每窗口 Region 用完即弃）；失败时可定位到具体 macro。
    area_after = 0  # 回读累计面积
    with LayoutDB.open(written) as database:  # 回读最终版图
        for entry in plan["macros"]:  # 逐 macro 窗口
            ownership = DbuBox(*entry["ownership_box"])  # 该 macro 计分框
            window = (database.query([layer], ownership)  # 窗口查询
                      .materialize_intersecting())  # 完整相交物化（不裁剪）
            # 完整相交会带入跨界 polygon 伸出窗口的部分，必须显式裁回
            # ownership（与主路径 clipped 同款），否则相邻窗口重复计数。
            region = (window.region(layer)  # 窗口内覆盖
                      & kdb.Region(ownership.to_native()))  # 精确裁剪
            if not region.has_valid_polygons():  # 无效 polygon
                raise RuntimeError(  # 明确失败并定位 macro
                    f"{entry['macro_id']} 窗口含无效 polygon")
            area_after += int(region.area())  # 累计窗口面积
    if area_after != area_before:  # 覆盖面积被 normalize 改变
        raise RuntimeError(  # 明确失败
            f"merge 前后覆盖面积改变：{area_before} -> {area_after}")  # 报数值
    return written  # 返回最终版图路径
