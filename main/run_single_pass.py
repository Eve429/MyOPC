"""单遍偏置扩张管线：两级网格 + 单次位移 + 权威覆盖合并直出最终版图。"""

import sys
import time
from decimal import Decimal
from pathlib import Path

import klayout.db as kdb
import numpy as np

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.units import exact_dbu
from geometry import GeometryPatch, PatchWriter
from layout import LayerSpec, LayoutDB

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
from opc.input import plan_macros

# problem 构造与位移重建
from opc.input.edge import (
    prepare_macro_problem,
    reconstruct_region,
)


def run_single_pass(
    layout: LayoutConfig,
    partition: PartitionConfig,
    litho: LithographyConfig,
    edge: EdgeConfig,
    single_pass: SinglePassConfig,
    output: OutputConfig,
) -> Path:
    """单遍执行两级网格偏置扩张并写出最终目标层版图。"""
    total_started = time.perf_counter()
    patches: list[GeometryPatch] = []  # 逐 macro 权威覆盖收集器
    segment_count_sum = 0  # 段数总计（摘要打印用）
    core_count_sum = 0  # core 总数（摘要打印用）
    with LayoutDB.open(layout.layout, layout.top_cell) as database:
        # 元数据、精确换算与两级网格规划（与验证管线同一套契约）。
        prepare_started = time.perf_counter()
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # 0.0001 µm/DBU → 0.1 nm/DBU
        layer = LayerSpec(layout.layer, layout.datatype)
        bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生逐层，不物化）
        if bounds is None:  # 目标层在顶层子树内无图形
            # 空层无法规划网格
            raise ValueError(f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")
        # nm→DBU 换算、context 契约与边段配置构造集中在 resolve_prepare_config。
        runtime = resolve_prepare_config(partition, litho, edge, dbu_nm)
        # 单遍位移（允许负值=沿法向反向）
        displacement_dbu = exact_dbu(single_pass.displacement_nm, dbu_nm, "displacement_nm")
        # 位移契约：|d| ≤ max_displacement（单遍专属；context 契约已在 resolve 内）。
        if abs(displacement_dbu) > runtime.fragmentation.max_displacement_dbu:
            raise ValueError("displacement_nm 的绝对值不得超过 max_displacement_nm")
        # 两级网格规划（像素整除/画布容量在此校验）
        macros = plan_macros(
            bounds,
            macro_grid=partition.macro_grid,
            macro_size_dbu=runtime.grid.macro_size_dbu,
            core_size_dbu=runtime.grid.core_dbu,
            context_dbu=runtime.grid.context_dbu,
            pixel_dbu=runtime.grid.pixel_dbu,
            canvas_pixels=litho.canvas_pixels,
        )
        # 规划复核：ownership 面积和恰等于父框即无正面积重叠（O(macro 数)）。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")
        prepare_seconds = time.perf_counter() - prepare_started
        # 执行阶段：逐 macro 全内存完成 prepare → 单遍位移 → 重建 → 权威裁剪。
        execute_started = time.perf_counter()
        for macro in macros:  # 按行优先顺序逐 macro
            # 完整相交物化（不裁剪 occurrence，不引入假边）
            batch = database.query([layer], macro.query_box).materialize_intersecting()
            # 全内存 problem：提边/分段/切线分裂/ownership
            problem = prepare_macro_problem(
                batch, layer, layout.polarity, runtime.fragmentation, macro, data_bounds=bounds
            )  # 单遍无处理框：数据包络即 layer bbox
            # 单遍位移：owner 段沿外法向统一移动 displacement，context 段保持零。
            # 法向约定为「材料指向空区」，因此带孔图形的孔壁法向指向孔内——
            # 统一正值位移自动实现「外环外扩、孔壁内收」的双向扩张。
            displacements = np.where(problem.owner_indices >= 0, float(displacement_dbu), 0.0)
            region = reconstruct_region(problem, displacements)
            ownership = kdb.Region(macro.ownership_box.to_native())
            clipped = region & ownership  # 权威覆盖选择：只保留自己负责的部分
            # 收集 patch（seam 由写出端全局 merge 消除）
            patches.append(GeometryPatch(macro.macro_id, layer, clipped, macro.ownership_box))
            segment_count_sum += problem.segments.segment_count
            core_count_sum += macro.core_count
            del batch, problem, region, clipped  # 释放当前 macro 再进入下一个
        execute_seconds = time.perf_counter() - execute_started
    # 写出阶段：双模式原子写出唯一产物；single_cell 全局 merge 消除表示层 seam。
    write_started = time.perf_counter()
    output.final_layout.parent.mkdir(parents=True, exist_ok=True)
    # 权威 patch 集中写出
    written = PatchWriter.write_macro_results(
        patches, output.final_layout, float(dbu_nm / 1000), cell_mode=output.final_cell_mode
    )
    write_seconds = time.perf_counter() - write_started
    total_seconds = time.perf_counter() - total_started
    print("单遍偏置扩张完成：")
    # 网格规模
    print(f"  macro 数：{len(macros)}，core 数：{core_count_sum}，段数总计：{segment_count_sum}")
    print(f"  位移：{single_pass.displacement_nm} nm（沿外法向）")
    # 分段耗时
    print(
        f"  准备 {prepare_seconds:.2f}s，执行 {execute_seconds:.2f}s，"
        f"写出 {write_seconds:.2f}s，总计 {total_seconds:.2f}s"
    )
    print(f"  最终版图：{written}（{output.final_cell_mode}）")
    return written


def main() -> int:
    """读取唯一位置参数 config，执行并打印中文摘要与耗时。"""
    if len(sys.argv) != 2:
        print("用法：python main/run_single_pass.py <config.toml>", file=sys.stderr)
        return 2  # 参数错误退出码
    # 统一加载六 Config（单遍不使用 work_dir）
    configs = load_config(
        sys.argv[1], LayoutConfig, PartitionConfig, LithographyConfig, EdgeConfig, SinglePassConfig, OutputConfig
    )
    run_single_pass(*configs)
    return 0  # 成功退出码


if __name__ == "__main__":
    raise SystemExit(main())
