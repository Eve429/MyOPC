"""单遍偏置扩张管线：两级网格 + 单次位移 + 权威覆盖合并直出最终版图。"""

import sys  # 把仓库根加入模块路径，保证免安装直接运行
import time  # 阶段计时（本入口保留的唯一统计）
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import klayout.db as kdb  # ownership 裁剪用的原生 Region
import numpy as np  # 单遍位移向量

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from common.units import exact_dbu  # 精确 nm→DBU 换算
from geometry import GeometryPatch, PatchWriter  # 权威 patch 与双模式最终写出
from layout import LayerSpec, LayoutDB  # 版图打开与层规格

# 统一配置体系
from main.configuration import (
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    SinglePassConfig,
    load_config,
    resolve_prepare_config,
)
from opc.input import plan_macros  # 两级网格规划

# problem 构造与位移重建
from opc.input.edge import (
    prepare_macro_problem,
    reconstruct_region,
)


def run_single_pass(layout: LayoutConfig, partition: PartitionConfig,
                    litho: LithographyConfig, edge: EdgeConfig,
                    single_pass: SinglePassConfig, output: OutputConfig) -> Path:
    """单遍执行两级网格偏置扩张并写出最终目标层版图。"""
    total_started = time.perf_counter()  # 总计时起点
    patches: list[GeometryPatch] = []  # 逐 macro 权威覆盖收集器
    segment_count_sum = 0  # 段数总计（摘要打印用）
    core_count_sum = 0  # core 总数（摘要打印用）
    with LayoutDB.open(layout.layout, layout.top_cell) as database:  # 打开并自动关闭
        # 元数据、精确换算与两级网格规划（与验证管线同一套契约）。
        prepare_started = time.perf_counter()  # 准备计时起点
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        layer = LayerSpec(layout.layer, layout.datatype)  # 目标层
        bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生逐层，不物化）
        if bounds is None:  # 目标层在顶层子树内无图形
            # 空层无法规划网格
            raise ValueError(
                f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")
        # nm→DBU 换算、context 契约与边段配置构造集中在 resolve_prepare_config。
        runtime = resolve_prepare_config(partition, litho, edge, dbu_nm)
        # 单遍位移（允许负值=沿法向反向）
        displacement_dbu = exact_dbu(
            single_pass.displacement_nm, dbu_nm, "displacement_nm")
        # 位移契约：|d| ≤ max_displacement（单遍专属；context 契约已在 resolve 内）。
        if abs(displacement_dbu) > runtime.fragmentation.max_displacement_dbu:  # 超上限
            raise ValueError("displacement_nm 的绝对值不得超过 max_displacement_nm")  # 报契约
        # 两级网格规划（像素整除/画布容量在此校验）
        macros = plan_macros(
            bounds, macro_grid=partition.macro_grid,
            macro_size_dbu=runtime.grid.macro_size_dbu,
            core_size_dbu=runtime.grid.core_dbu,
            context_dbu=runtime.grid.context_dbu,
            pixel_dbu=runtime.grid.pixel_dbu,
            canvas_pixels=litho.canvas_pixels)
        # 规划复核：ownership 面积和恰等于父框即无正面积重叠（O(macro 数)）。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:  # 面积失守
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")  # 报规划错误
        prepare_seconds = time.perf_counter() - prepare_started  # 准备耗时
        # 执行阶段：逐 macro 全内存完成 prepare → 单遍位移 → 重建 → 权威裁剪。
        execute_started = time.perf_counter()  # 执行计时起点
        for macro in macros:  # 按行优先顺序逐 macro
            # 完整相交物化（不裁剪 occurrence，不引入假边）
            batch = database.query(
                [layer], macro.query_box).materialize_intersecting()
            # 全内存 problem：提边/分段/切线分裂/ownership
            problem = prepare_macro_problem(
                batch, layer, layout.polarity, runtime.fragmentation, macro)
            # 单遍位移：owner 段沿外法向统一移动 displacement，context 段保持零。
            # 法向约定为「材料指向空区」，因此带孔图形的孔壁法向指向孔内——
            # 统一正值位移自动实现「外环外扩、孔壁内收」的双向扩张。
            displacements = np.where(
                problem.owner_indices >= 0, float(displacement_dbu), 0.0)
            region = reconstruct_region(problem, displacements)  # 按位移重建完整候选
            ownership = kdb.Region(macro.ownership_box.to_native())  # macro 权威框
            clipped = region & ownership  # 权威覆盖选择：只保留自己负责的部分
            # 收集 patch（seam 由写出端全局 merge 消除）
            patches.append(GeometryPatch(
                macro.macro_id, layer, clipped, macro.ownership_box))
            segment_count_sum += problem.segments.segment_count  # 累计段数
            core_count_sum += macro.core_count  # 累计 core 数
            del batch, problem, region, clipped  # 释放当前 macro 再进入下一个
        execute_seconds = time.perf_counter() - execute_started  # 执行耗时
    # 写出阶段：双模式原子写出唯一产物；single_cell 全局 merge 消除表示层 seam。
    write_started = time.perf_counter()  # 写出计时起点
    output.final_layout.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    # 权威 patch 集中写出
    written = PatchWriter.write_macro_results(
        patches, output.final_layout, float(dbu_nm / 1000),
        cell_mode=output.final_cell_mode)
    write_seconds = time.perf_counter() - write_started  # 写出耗时
    total_seconds = time.perf_counter() - total_started  # 总耗时
    print("单遍偏置扩张完成：")  # 摘要标题
    # 网格规模
    print(f"  macro 数：{len(macros)}，core 数：{core_count_sum}，段数总计："
          f"{segment_count_sum}")
    print(f"  位移：{single_pass.displacement_nm} nm（沿外法向）")  # 本遍位移
    # 分段耗时
    print(f"  准备 {prepare_seconds:.2f}s，执行 {execute_seconds:.2f}s，"
          f"写出 {write_seconds:.2f}s，总计 {total_seconds:.2f}s")
    print(f"  最终版图：{written}（{output.final_cell_mode}）")  # 产物位置
    return written  # 返回最终版图路径


def main() -> int:
    """读取唯一位置参数 config，执行并打印中文摘要与耗时。"""
    if len(sys.argv) != 2:  # 参数数量不符
        print("用法：python main/run_single_pass.py <config.toml>", file=sys.stderr)  # 提示
        return 2  # 参数错误退出码
    # 统一加载六 Config（单遍不使用 work_dir）
    configs = load_config(
        sys.argv[1], LayoutConfig, PartitionConfig, LithographyConfig,
        EdgeConfig, SinglePassConfig, OutputConfig)
    run_single_pass(*configs)  # 解包执行
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
