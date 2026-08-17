"""单遍偏置扩张管线：两级网格 + 单次位移 + 权威覆盖合并直出最终版图。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # 阶段计时（本入口保留的唯一统计）
from dataclasses import dataclass, fields  # 配置结构与公共层组装
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import klayout.db as kdb  # ownership 裁剪用的原生 Region
import numpy as np  # 单遍位移向量
from tomllib import loads as toml_loads  # Python 3.12 标准库 TOML 解析

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from geometry import GeometryPatch, PatchWriter  # 权威 patch 与双模式最终写出
from layout import LayoutDB  # 版图打开
from main._macro_pipeline import (  # 共享配置解析与精确换算（消除入口间依赖）
    MacroCommonConfig,
    exact_dbu,
    load_macro_common_config,
)
from opc.input import plan_macros  # 两级网格规划
from opc.input.edge import (  # problem 构造与位移重建
    prepare_macro_problem,
    reconstruct_region,
)
from opc.input.edge.fragmentation import FragmentationConfig  # 边段配置


@dataclass(frozen=True, slots=True)
class SinglePassConfig(MacroCommonConfig):
    """在公共 Macro 配置之上补充单遍位移参数。"""

    displacement_nm: Decimal              # 单遍位移；正=沿外法向，负=反向


def load_config(path: str | Path) -> SinglePassConfig:
    """严格读取单遍 TOML：公共段走共享解析器，专属段只查 displacement_nm。"""
    config_path = Path(path).expanduser().resolve()  # 配置文件绝对路径
    # 公共五段（input/grid/lithography/edge/output）全部由共享层校验解析；
    # 本入口 [output] 不放行 work_dir（出现即按未知键拒绝），产物契约不变。
    common = load_macro_common_config(
        config_path, extra_sections=("iteration",))  # 专属段白名单放行
    raw = toml_loads(config_path.read_text(encoding="utf-8"))  # 再读一次取专属段
    iter_section = raw.get("iteration", {})  # 迭代段
    unknown = set(iter_section) - {"displacement_nm"}  # 段内未知键
    if unknown:  # 拒绝拼错键
        raise ValueError(f"[iteration] 含未知键：{sorted(unknown)}")  # 报键名
    if "displacement_nm" not in iter_section:  # 必填显式报错
        raise ValueError("[iteration] 缺少必填键：['displacement_nm']")  # 报缺失
    displacement_nm = Decimal(str(iter_section["displacement_nm"]))  # 十进制精确保存
    # replace() 会按基类构造，专属字段必须显式以子类组装；公共字段按
    # fields 浅拷贝（asdict 会把嵌套的 LayerSpec 递归转 dict，不可用）。
    return SinglePassConfig(
        **{field.name: getattr(common, field.name)
           for field in fields(MacroCommonConfig)},
        displacement_nm=displacement_nm)  # 组装


def run_single_pass(config: SinglePassConfig) -> Path:
    """单遍执行两级网格偏置扩张并写出最终目标层版图。"""
    total_started = time.perf_counter()  # 总计时起点
    patches: list[GeometryPatch] = []  # 逐 macro 权威覆盖收集器
    segment_count_sum = 0  # 段数总计（摘要打印用）
    core_count_sum = 0  # core 总数（摘要打印用）
    with LayoutDB.open(config.layout_path, config.top_cell) as database:  # 打开并自动关闭
        # 阶段 0：元数据、精确换算与两级网格规划（与验证管线同一套契约）。
        prepare_started = time.perf_counter()  # 准备计时起点
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        layer = config.layer  # 目标层
        bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生逐层，不物化）
        if bounds is None:  # 目标层在顶层子树内无图形
            raise ValueError(  # 空层无法规划网格
                f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")  # 报层号
        core_dbu = exact_dbu(config.core_size_nm, dbu_nm, "core_size_nm")  # core
        context_dbu = exact_dbu(config.context_nm, dbu_nm, "context_nm")  # context
        pixel_dbu = exact_dbu(config.pixel_nm, dbu_nm, "pixel_nm")  # pixel
        corner_dbu = exact_dbu(config.corner_nm, dbu_nm, "corner_nm")  # 拐角段
        segment_dbu = exact_dbu(config.segment_nm, dbu_nm, "segment_nm")  # 中段
        max_displacement_dbu = exact_dbu(  # 位移上限
            config.max_displacement_nm, dbu_nm, "max_displacement_nm")  # 换算
        displacement_dbu = exact_dbu(  # 单遍位移（允许负值=沿法向反向）
            config.displacement_nm, dbu_nm, "displacement_nm")  # 换算
        # 位移契约：|d| ≤ max_displacement ≤ context；后者保证 context 能覆盖
        # 位移后的几何，邻居 macro 的副本仍落在可见范围内。
        if abs(displacement_dbu) > max_displacement_dbu:  # 超出配置上限
            raise ValueError("displacement_nm 的绝对值不得超过 max_displacement_nm")  # 报契约
        if max_displacement_dbu > context_dbu:  # context 不足以覆盖位移
            raise ValueError("context_nm 必须不小于 max_displacement_nm")  # 报契约
        fragmentation = FragmentationConfig(  # DBU 级边段配置（数值约束由构造校验）
            corner_length_dbu=float(corner_dbu),  # 拐角段
            max_segment_length_dbu=float(segment_dbu),  # 中段上限
            max_displacement_dbu=float(max_displacement_dbu),  # 位移上限
            miter_limit=config.miter_limit)  # miter
        macros = plan_macros(  # 两级网格规划（像素整除/画布容量在此校验）
            bounds, macro_grid=config.macro_grid, macro_size_dbu=(
                exact_dbu(config.macro_size_nm, dbu_nm, "macro_size_nm")  # 尺寸模式换算
                if config.macro_size_nm is not None else None),  # 数量模式为空
            core_size_dbu=core_dbu, context_dbu=context_dbu,  # core/context
            pixel_dbu=pixel_dbu, canvas_pixels=config.canvas_pixels)  # 画布契约
        # 规划复核：ownership 面积和恰等于父框即无正面积重叠（O(macro 数)）。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:  # 面积失守
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")  # 报规划错误
        prepare_seconds = time.perf_counter() - prepare_started  # 准备耗时
        # 执行阶段：逐 macro 全内存完成 prepare → 单遍位移 → 重建 → 权威裁剪。
        execute_started = time.perf_counter()  # 执行计时起点
        for macro in macros:  # 按行优先顺序逐 macro
            batch = database.query(  # 完整相交物化（不裁剪 occurrence，不引入假边）
                [layer], macro.query_box).materialize_intersecting()  # 惰性查询执行
            problem = prepare_macro_problem(  # 全内存 problem：提边/分段/切线分裂/ownership
                batch, layer, config.polarity, fragmentation, macro)  # 不落盘
            # 单遍位移：owner 段沿外法向统一移动 displacement，context 段保持零。
            # 法向约定为「材料指向空区」，因此带孔图形的孔壁法向指向孔内——
            # 统一正值位移自动实现「外环外扩、孔壁内收」的双向扩张。
            displacements = np.where(  # 一次构造整条位移向量
                problem.owner_indices >= 0, float(displacement_dbu), 0.0)  # owner=位移
            region = reconstruct_region(problem, displacements)  # 按位移重建完整候选
            ownership = kdb.Region(macro.ownership_box.to_native())  # macro 权威框
            clipped = region & ownership  # 权威覆盖选择：只保留自己负责的部分
            patches.append(GeometryPatch(  # 收集 patch（seam 由写出端全局 merge 消除）
                macro.macro_id, layer, clipped, macro.ownership_box))  # macro 编号即 patch_id
            segment_count_sum += problem.segments.segment_count  # 累计段数
            core_count_sum += macro.core_count  # 累计 core 数
            del batch, problem, region, clipped  # 释放当前 macro 再进入下一个
        execute_seconds = time.perf_counter() - execute_started  # 执行耗时
    # 写出阶段：双模式原子写出唯一产物；single_cell 全局 merge 消除表示层 seam。
    write_started = time.perf_counter()  # 写出计时起点
    config.final_layout.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    written = PatchWriter.write_macro_results(  # 权威 patch 集中写出
        patches, config.final_layout, float(dbu_nm / 1000),  # patch 集合与源版图 DBU
        cell_mode=config.final_cell_mode)  # single_cell 或 macro_cells
    write_seconds = time.perf_counter() - write_started  # 写出耗时
    total_seconds = time.perf_counter() - total_started  # 总耗时
    print("单遍偏置扩张完成：")  # 摘要标题
    print(f"  macro 数：{len(macros)}，core 数：{core_count_sum}，段数总计："  # 网格规模
          f"{segment_count_sum}")  # 规模续
    print(f"  位移：{config.displacement_nm} nm（沿外法向）")  # 本遍位移
    print(f"  准备 {prepare_seconds:.2f}s，执行 {execute_seconds:.2f}s，"  # 分段耗时
          f"写出 {write_seconds:.2f}s，总计 {total_seconds:.2f}s")  # 耗时总
    print(f"  最终版图：{written}（{config.final_cell_mode}）")  # 产物位置
    return written  # 返回最终版图路径


def main() -> int:
    """读取唯一位置参数 config，执行并打印中文摘要与耗时。"""
    if len(sys.argv) != 2:  # 参数数量不符
        print("用法：python main/run_single_pass.py <config.toml>", file=sys.stderr)  # 提示
        return 2  # 参数错误退出码
    run_single_pass(load_config(sys.argv[1]))  # 加载配置并执行
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
