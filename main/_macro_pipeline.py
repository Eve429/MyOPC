"""两个真实流程（验证管线与 MB-OPC）共用的 Macro 配置、准备、写出与合并。"""

import json  # 序列化 plan.json
import os  # 原子替换与文件系统操作
import sys  # 把仓库根加入模块路径，保证免安装直接运行
import tempfile  # 创建与目标同目录的临时文件
import time  # perf_counter 阶段计时
from collections.abc import Mapping  # merge 映射参数的只读类型
from dataclasses import dataclass  # 定义唯一共享配置结构 MacroPipelineConfig
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象
from typing import Literal  # cell_mode 的字面量类型

import klayout.db as kdb  # 写出 macro GDS 与所有权裁剪的原生版图对象
import psutil  # 阶段 RSS 峰值测量；缺失时直接 ImportError 不降级
from tomllib import loads as toml_loads  # Python 3.12 标准库 TOML 解析

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from geometry import GeometryPatch, PatchWriter  # 权威 patch 与双模式最终写出
from layout import DbuBox, LayerSpec, LayoutDB  # 版图打开、层规格与坐标框
from opc.input import MaskPolarity, plan_macros  # 两级网格规划
from opc.input.edge import MacroProblem, prepare_macro_problem  # problem 构造
from opc.input.edge.fragmentation import FragmentationConfig  # 边段配置

# 宏管线自身六个段允许出现的键；未知键一律拒绝，防止拼写错误被静默忽略。
# 调用方流程专属段（如验证的 [iteration]、MB-OPC 的 [mbopc]）通过
# load_macro_config 的 extra_sections 显式放行，段内键由各流程自己校验。
_ALLOWED_KEYS = {  # 段名 → 允许键集合
    "input": {"layout", "top_cell", "layer", "datatype", "polarity"},  # 输入版图与目标层
    "grid": {"macro_grid", "macro_size_nm", "core_size_nm", "context_nm"},  # 两级网格参数
    "lithography": {"pixel_nm", "canvas_pixels"},  # 光刻采样契约
    "edge": {"corner_nm", "segment_nm", "max_displacement_nm", "miter_limit"},  # 边段配置
    "output": {"work_dir", "final_layout", "final_cell_mode"},  # 产物位置与最终 Cell 模式
}
_PLAN_FORMAT_VERSION = 1  # plan.json 结构版本


@dataclass(frozen=True, slots=True)
class MacroPipelineConfig:
    """保存两级网格、problem 准备和 macro 结果写出共同需要的全部显式配置。"""

    layout_path: Path                     # 输入 GDS/OASIS/GLP 的绝对路径
    top_cell: str | None                  # 显式顶层；None 表示要求版图只有一个顶层
    layer: LayerSpec                      # 本次处理的唯一目标 layer/datatype
    polarity: MaskPolarity                # 源 polygon 的 clear/opaque 极性
    macro_size_nm: Decimal | None         # 按 nm 切 macro；与 macro_grid 恰好一个非空
    macro_grid: tuple[int, int] | None    # 按 [列,行] 数量切 macro
    core_size_nm: Decimal                 # 名义 core 边长
    context_nm: Decimal                   # core 每侧通用上下文宽度
    pixel_nm: Decimal                     # 光刻采样像素尺寸
    canvas_pixels: int                    # 当前 ICCAD13 固定为 256
    corner_nm: Decimal                    # 拐角控制段长度
    segment_nm: Decimal                   # 普通控制段最大长度
    max_displacement_nm: Decimal          # 允许的绝对位移上限
    miter_limit: float                    # 拐角重建 miter 上限
    work_dir: Path                        # problem/result/macro GDS/summary 根目录
    final_layout: Path                    # 最终完整目标层版图路径
    final_cell_mode: Literal["single_cell", "macro_cells"]  # 最终 Cell 模式


def load_macro_config(path: str | Path, *,
                      extra_sections: tuple[str, ...] = ()) -> MacroPipelineConfig:
    """严格读取宏管线六段 TOML，解析相对路径并拒绝未知或互斥字段。"""
    config_path = Path(path).expanduser().resolve()  # 配置文件绝对路径
    raw = toml_loads(config_path.read_text(encoding="utf-8"))  # 解析 TOML 文本
    known_sections = set(_ALLOWED_KEYS) | set(extra_sections)  # 本流程允许的段全集
    unknown = set(raw) - known_sections  # 检查未知顶层段
    if unknown:  # 拒绝未知段（拼错的流程段进不了任何白名单）
        raise ValueError(f"未知配置段：{sorted(unknown)}")
    for section, allowed in _ALLOWED_KEYS.items():  # 逐段检查未知键
        keys = set(raw.get(section, {}))  # 该段实际出现的键
        if keys - allowed:  # 出现允许清单之外的键
            raise ValueError(f"[{section}] 含未知键：{sorted(keys - allowed)}")
    input_section = raw.get("input", {})  # 输入段
    grid_section = raw.get("grid", {})  # 网格段
    litho_section = raw.get("lithography", {})  # 光刻段
    edge_section = raw.get("edge", {})  # 边段段
    output_section = raw.get("output", {})  # 输出段
    # 必填键收集：缺失直接失败，不在 Python 侧补默认值。
    for section, required in (  # 各段必填键清单
            ("input", ("layout", "layer", "datatype", "polarity")),
            ("grid", ("core_size_nm", "context_nm")),
            ("lithography", ("pixel_nm", "canvas_pixels")),
            ("edge", ("corner_nm", "segment_nm", "max_displacement_nm", "miter_limit")),
            ("output", ("work_dir", "final_layout", "final_cell_mode"))):
        missing = [key for key in required if key not in raw.get(section, {})]  # 缺失键
        if missing:  # 显式报错
            raise ValueError(f"[{section}] 缺少必填键：{missing}")
    # 互斥检查：macro_grid 与 macro_size_nm 恰好出现一个。
    has_grid = "macro_grid" in grid_section  # 是否显式给定 macro 数量
    has_size = "macro_size_nm" in grid_section  # 是否显式给定 macro 尺寸
    if has_grid == has_size:  # 两者同真或同假都是配置意图不明
        raise ValueError("macro_grid 与 macro_size_nm 必须恰好填写一个")
    # 相对路径一律相对 TOML 文件目录解析。
    base = config_path.parent  # 路径解析基准目录
    layout_path = (base / str(input_section["layout"])).resolve()  # 输入版图绝对路径
    work_dir = (base / str(output_section["work_dir"])).resolve()  # 工作目录绝对路径
    final_layout = (base / str(output_section["final_layout"])).resolve()  # 最终版图绝对路径
    # 极性在配置层就归一化，后续阶段不再处理字符串。
    try:  # 尝试把极性字符串转为枚举
        polarity = MaskPolarity(str(input_section["polarity"]))  # clear/opaque
    except ValueError as exc:  # 未知极性
        raise ValueError(f"不支持的极性：{input_section['polarity']!r}") from exc
    final_cell_mode = str(output_section["final_cell_mode"])  # 最终 Cell 模式
    if final_cell_mode not in ("single_cell", "macro_cells"):  # 模式枚举校验
        raise ValueError(f"未知 final_cell_mode：{final_cell_mode}")
    canvas_pixels = int(litho_section["canvas_pixels"])  # 画布像素数
    if canvas_pixels != 256:  # ICCAD13 契约冻结为 256
        raise ValueError("canvas_pixels 当前固定为 256")
    # macro_grid 规范化：两项正整数 [列, 行]。
    if has_grid:  # 数量模式
        entries = grid_section["macro_grid"]  # 读取列表
        if (not isinstance(entries, list) or len(entries) != 2 or
                not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                        for v in entries)):  # 校验两项正整数
            raise ValueError("macro_grid 必须是两项正整数 [列, 行]")
        macro_grid: tuple[int, int] | None = (int(entries[0]), int(entries[1]))  # 归一化
        macro_size: Decimal | None = None  # 另一模式置空
    else:  # 尺寸模式
        macro_grid = None  # 置空
        macro_size = Decimal(str(grid_section["macro_size_nm"]))  # 十进制精确保存
    return MacroPipelineConfig(  # 组装冻结配置对象
        layout_path=layout_path,  # 输入路径
        top_cell=str(input_section["top_cell"]) if "top_cell" in input_section else None,  # 顶层
        layer=LayerSpec(int(input_section["layer"]), int(input_section["datatype"])),  # 目标层
        polarity=polarity,  # 极性
        macro_size_nm=macro_size,  # 尺寸模式
        macro_grid=macro_grid,  # 数量模式
        core_size_nm=Decimal(str(grid_section["core_size_nm"])),  # core 尺寸
        context_nm=Decimal(str(grid_section["context_nm"])),  # context 宽度
        pixel_nm=Decimal(str(litho_section["pixel_nm"])),  # 像素尺寸
        canvas_pixels=canvas_pixels,  # 画布
        corner_nm=Decimal(str(edge_section["corner_nm"])),  # 拐角段长
        segment_nm=Decimal(str(edge_section["segment_nm"])),  # 中段上限
        max_displacement_nm=Decimal(str(edge_section["max_displacement_nm"])),  # 位移上限
        miter_limit=float(edge_section["miter_limit"]),  # miter 上限
        work_dir=work_dir,  # 工作目录
        final_layout=final_layout,  # 最终版图
        final_cell_mode=final_cell_mode)  # Cell 模式


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


def prepare_problems(config: MacroPipelineConfig) -> dict:
    """执行阶段 0/1，逐 macro 生成 problem，并写出 plan.json。"""
    started = time.perf_counter()  # 阶段计时起点
    process = psutil.Process()  # RSS 采样进程对象
    peak_rss = process.memory_info().rss  # 峰值初值
    with LayoutDB.open(config.layout_path, config.top_cell) as database:  # 打开并自动关闭
        top_cell_name = database.top_cell_name  # 在库存活期内捕获顶层名
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        bounds = database.layer_bbox(config.layer)  # 目标层整体 bbox（原生逐层，不物化）
        if bounds is None:  # 目标层在顶层子树内无图形
            raise ValueError(  # 空层无法规划网格
                f"目标层 {config.layer.layer}/{config.layer.datatype} 不含任何图形")  # 报层号
        # 全部 nm 参数精确换算：不能整除直接失败，不四舍五入吸收误差。
        core_dbu = exact_dbu(config.core_size_nm, dbu_nm, "core_size_nm")  # core
        context_dbu = exact_dbu(config.context_nm, dbu_nm, "context_nm")  # context
        pixel_dbu = exact_dbu(config.pixel_nm, dbu_nm, "pixel_nm")  # pixel
        corner_dbu = exact_dbu(config.corner_nm, dbu_nm, "corner_nm")  # 拐角段
        segment_dbu = exact_dbu(config.segment_nm, dbu_nm, "segment_nm")  # 中段
        max_displacement_dbu = exact_dbu(  # 位移上限
            config.max_displacement_nm, dbu_nm, "max_displacement_nm")
        if max_displacement_dbu > context_dbu:  # context 必须覆盖最大位移
            raise ValueError("context_nm 必须不小于 max_displacement_nm")
        # 边段数值约束（正长度、segment≥2×corner、非负位移）由 FragmentationConfig
        # 构造统一校验，这里不重复检查。
        fragmentation = FragmentationConfig(  # DBU 级边段配置
            corner_length_dbu=float(corner_dbu),  # 拐角段
            max_segment_length_dbu=float(segment_dbu),  # 中段上限
            max_displacement_dbu=float(max_displacement_dbu),  # 位移上限
            miter_limit=config.miter_limit)  # miter
        macros = plan_macros(  # 两级网格规划（内部完成像素/画布校验）
            bounds, macro_grid=config.macro_grid, macro_size_dbu=(
                exact_dbu(config.macro_size_nm, dbu_nm, "macro_size_nm")  # 尺寸模式换算
                if config.macro_size_nm is not None else None),  # 数量模式为空
            core_size_dbu=core_dbu, context_dbu=context_dbu,  # core/context
            pixel_dbu=pixel_dbu, canvas_pixels=config.canvas_pixels)  # 画布契约
        # 阶段 0 步骤 7：ownership 复核——面积和恰等于父框即无正面积重叠。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:  # 面积守恒
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")
        problems_dir = config.work_dir / "problems"  # problem 存放目录
        problems_dir.mkdir(parents=True, exist_ok=True)  # 创建目录结构
        entries = []  # 逐 macro 计划条目
        segment_count_sum = 0  # 段数累计
        membership_count_sum = 0  # membership 累计
        maximum_problem_bytes = 0  # 最大 problem 字节数
        maximum_problem_macro_id = ""  # 最大 problem 所属 macro
        for macro in macros:  # 按行优先顺序逐 macro 准备
            batch = database.query(  # 完整相交物化（不裁剪 occurrence）
                [config.layer], macro.query_box).materialize_intersecting()  # 惰性查询执行
            problem = prepare_macro_problem(  # 一次完成提边/分段/切线分裂/ownership
                batch, config.layer, config.polarity, fragmentation, macro)  # 阶段 1 核心
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
        "layout": str(config.layout_path),  # 输入版图
        "top_cell": top_cell_name,  # 实际选定的顶层 Cell 名
        "dbu_um": float(dbu_nm / 1000),  # DBU 微米值（写出最终 GDS 用）
        "layer": [config.layer.layer, config.layer.datatype],  # 目标层
        "polarity": config.polarity.value,  # 极性
        "core_size_dbu": core_dbu,  # core
        "context_dbu": context_dbu,  # context
        "pixel_dbu": pixel_dbu,  # pixel
        "canvas_pixels": config.canvas_pixels,  # canvas
        "macro_count": len(macros),  # macro 总数
        "core_count": sum(macro.core_count for macro in macros),  # core 总数
        "fragmentation": {  # 边段配置
            "corner_length_dbu": float(corner_dbu),  # 拐角段
            "max_segment_length_dbu": float(segment_dbu),  # 中段上限
            "max_displacement_dbu": float(max_displacement_dbu),  # 位移上限
            "miter_limit": config.miter_limit},  # miter
        "work_dir": str(config.work_dir),  # 工作目录
        "final_layout": str(config.final_layout),  # 最终版图
        "final_cell_mode": config.final_cell_mode,  # Cell 模式
        "macros": entries,  # 逐 macro 条目
        "segment_count_sum": segment_count_sum,  # 段数总计
        "membership_count_sum": membership_count_sum,  # membership 总计
        "maximum_problem_bytes": maximum_problem_bytes,  # 最大 problem 字节
        "maximum_problem_macro_id": maximum_problem_macro_id,  # 最大 problem macro
        "prepare_seconds": prepare_seconds,  # 准备耗时
        "prepare_peak_rss_bytes": peak_rss}  # 准备 RSS 峰值
    atomic_write_json(config.work_dir / "plan.json", plan)  # 原子写出计划
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
            batch = database.query(  # 全框查询目标层
                [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)).materialize()  # 物化
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
    with LayoutDB.open(written) as database:  # 回读最终版图
        final_batch = database.query(  # 全框查询目标层
            [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)).materialize()  # 物化
    coverage = final_batch.region(layer)  # 最终覆盖
    if not coverage.has_valid_polygons():  # 无效 polygon
        raise RuntimeError("最终版图含无效 polygon")  # 明确失败
    if int(coverage.area()) != area_before:  # 覆盖面积被 normalize 改变
        raise RuntimeError(  # 明确失败
            f"merge 前后覆盖面积改变：{area_before} -> {coverage.area()}")  # 报数值
    return written  # 返回最终版图路径
