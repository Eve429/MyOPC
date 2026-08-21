"""多个真实流程共用的 macro 生命周期：problem 准备、候选写出与最终合并。"""

import os  # 原子替换与文件系统操作
import sys  # 把仓库根加入模块路径，保证免安装直接运行
import tempfile  # 创建与目标同目录的临时文件
import time  # perf_counter 阶段计时
import warnings  # 处理框大于 layer bbox 的风险提示
from collections.abc import Mapping  # merge 映射参数的只读类型
from decimal import Decimal  # nm→DBU 的精确十进制换算
from pathlib import Path  # 全部路径统一使用 Path 对象
from typing import Literal  # cell_mode 的字面量类型

import klayout.db as kdb  # 写出 macro GDS 与所有权裁剪的原生版图对象
import numpy as np  # PNG 栅格与像素变换
import psutil  # 阶段 RSS 峰值测量；缺失时直接 ImportError 不降级
import torch  # 最终光刻留档的 no_grad 推理
from PIL import Image  # 最终光刻 PNG 留档

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/geometry 可导入

from common.io import atomic_write_json  # JSON 原子写出
from common.units import exact_dbu  # 处理框 nm→DBU 精确换算
from geometry import GeometryPatch, PatchWriter  # 权威 patch 与双模式最终写出
from layout import (  # 版图打开、层规格与坐标框
    DbuBox,
    LayerNotFoundError,
    LayerSpec,
    LayoutDB,
)

# 统一配置体系（含 nm→DBU 解析集中）
from main.configuration import (
    EdgeConfig,
    LayoutConfig,
    LithographyConfig,
    OutputConfig,
    PartitionConfig,
    resolve_prepare_config,
)

# 两级网格规划与留档栅格
from opc.input import (
    MaskPolarity,
    plan_macros,
    rasterize_mask_canvas,
)
from opc.input.edge import prepare_macro_problem  # problem 构造

_PLAN_FORMAT_VERSION = 1  # plan.json 结构版本


def resolve_field_bounds(layout: LayoutConfig, layer_bounds: DbuBox,
                         dbu_nm: Decimal) -> DbuBox:
    """解析处理框：双 None 用 layer bbox；box 直用；size 以 layer bbox 居中推导。

    环带（field 减 layer bbox 的无图形区）光学语义 = 极性背景外推
    （clear→不透光、opaque→透光），由 prepare 的 transmission/coverage
    变换天然给出；处理框绝不作为图形进入 Region（00_PAST field_box
    契约——不产生虚假可动边）。严格大于 layer bbox 时给出 warning。
    """
    if layout.field_box_nm is not None:
        left, bottom, right, top = (
            exact_dbu(value, dbu_nm, f"field_box_nm[{index}]")
            for index, value in enumerate(layout.field_box_nm))
        field = DbuBox(left, bottom, right, top)
    elif layout.field_size_nm is not None:
        width = exact_dbu(layout.field_size_nm[0], dbu_nm, "field_size_nm[0]")
        height = exact_dbu(layout.field_size_nm[1], dbu_nm, "field_size_nm[1]")
        # 逐轴居中：slack//2 归低侧、余量归高侧（与 _center_padding 的奇数
        # 余量约定一致）；宽高小于 layer 尺寸时 slack 为负，交由下方包含性
        # 校验统一报错，不在此重复分支。
        slack_x = width - layer_bounds.width
        slack_y = height - layer_bounds.height
        low_x, low_y = slack_x // 2, slack_y // 2
        field = DbuBox(
            layer_bounds.left - low_x, layer_bounds.bottom - low_y,
            layer_bounds.right + (slack_x - low_x),
            layer_bounds.top + (slack_y - low_y))
    else:
        return layer_bounds  # 未配置：保持 layer bbox 现行行为，零行为变化
    if (field.left > layer_bounds.left or field.bottom > layer_bounds.bottom
            or field.right < layer_bounds.right
            or field.top < layer_bounds.top):
        raise ValueError(
            f"处理框 ({field.left},{field.bottom})-({field.right},{field.top}) DBU"
            " 必须四向包含 layer bbox "
            f"({layer_bounds.left},{layer_bounds.bottom})-"
            f"({layer_bounds.right},{layer_bounds.top}) DBU"
            "——配置比版图小或偏移出界")
    if (field.left < layer_bounds.left or field.bottom < layer_bounds.bottom
            or field.right > layer_bounds.right
            or field.top > layer_bounds.top):
        scale = Decimal(dbu_nm)
        warnings.warn(
            "处理框大于 layer bbox：field "
            f"({Decimal(field.left) * scale},{Decimal(field.bottom) * scale})"
            f"-({Decimal(field.right) * scale},{Decimal(field.top) * scale}) nm"
            " ⊃ layer "
            f"({Decimal(layer_bounds.left) * scale},"
            f"{Decimal(layer_bounds.bottom) * scale})-"
            f"({Decimal(layer_bounds.right) * scale},"
            f"{Decimal(layer_bounds.top) * scale}) nm；"
            "环带恒不透光（光学开孔边界=layer 数据包络）", stacklevel=2)
    return field


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
        layer_bounds = database.layer_bbox(layer)  # 目标层整体 bbox（原生逐层，不物化）
        if layer_bounds is None:  # 目标层在顶层子树内无图形
            # 空层无法规划网格
            raise ValueError(
                f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")
        # 处理框（field_box/field_size）：未配置时即 layer bbox，零行为变化
        bounds = resolve_field_bounds(layout, layer_bounds, dbu_nm)
        # nm→DBU 换算、context 契约与边段配置构造集中在 resolve_prepare_config。
        runtime = resolve_prepare_config(partition, litho, edge, dbu_nm)
        # 两级网格规划（内部完成像素/画布校验）
        macros = plan_macros(
            bounds, macro_grid=partition.macro_grid,
            macro_size_dbu=runtime.grid.macro_size_dbu,
            core_size_dbu=runtime.grid.core_dbu,
            context_dbu=runtime.grid.context_dbu,
            pixel_dbu=runtime.grid.pixel_dbu,
            canvas_pixels=litho.canvas_pixels)
        # ownership 复核——面积和恰等于父框即无正面积重叠。
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
            # 完整相交物化（不裁剪 occurrence）
            batch = database.query(
                [layer], macro.query_box).materialize_intersecting()
            # 一次完成提边/分段/切线分裂/ownership
            problem = prepare_macro_problem(
                batch, layer, layout.polarity, runtime.fragmentation, macro,
                dark_box=layer_bounds)
            problem_path = problem.save(problems_dir / f"{macro.macro_id}.npz")  # 原子落盘
            problem_bytes = problem_path.stat().st_size  # 文件字节数即持久字节数
            segment_count_sum += problem.segments.segment_count  # 累计段数
            membership_count_sum += len(problem.member_segment_indices)  # 累计 membership
            if problem_bytes > maximum_problem_bytes:  # 更新最大 problem
                maximum_problem_bytes = problem_bytes  # 记录字节
                maximum_problem_macro_id = macro.macro_id  # 记录 macro
            # 单 macro 计划条目
            entries.append({
                "macro_id": macro.macro_id,
                "ownership_box": [macro.ownership_box.left, macro.ownership_box.bottom,
                                  macro.ownership_box.right, macro.ownership_box.top],
                "core_count": macro.core_count,
                "segment_count": problem.segments.segment_count,
                "membership_count": len(problem.member_segment_indices),
                "problem_file": str(problem_path),
                "problem_bytes": problem_bytes})
            peak_rss = max(peak_rss, process.memory_info().rss)  # 采样峰值
            del batch, problem  # 立即释放当前 macro 大对象再进入下一个
    # 全部 problem 成功且 LayoutDB 已关闭，才允许写出表示「准备完成」的 plan。
    prepare_seconds = time.perf_counter() - started  # 阶段耗时
    # 完整计划（后续阶段唯一允许消费的产物）
    plan = {
        "format_version": _PLAN_FORMAT_VERSION,
        "layout": str(layout.layout),
        "top_cell": top_cell_name,
        "dbu_um": float(dbu_nm / 1000),
        "layer": [layer.layer, layer.datatype],
        "polarity": layout.polarity.value,
        "dark_box": [layer_bounds.left, layer_bounds.bottom,
                     layer_bounds.right, layer_bounds.top],
        "core_size_dbu": runtime.grid.core_dbu,
        "context_dbu": runtime.grid.context_dbu,
        "pixel_dbu": runtime.grid.pixel_dbu,
        "canvas_pixels": litho.canvas_pixels,
        "macro_count": len(macros),
        "core_count": sum(macro.core_count for macro in macros),
        "fragmentation": {
            "corner_length_dbu": runtime.fragmentation.corner_length_dbu,
            "max_segment_length_dbu": runtime.fragmentation.max_segment_length_dbu,
            "max_displacement_dbu": runtime.fragmentation.max_displacement_dbu,
            "miter_limit": runtime.fragmentation.miter_limit},
        "work_dir": str(output.work_dir),
        "final_layout": str(output.final_layout),
        "final_cell_mode": output.final_cell_mode,
        "macros": entries,
        "segment_count_sum": segment_count_sum,
        "membership_count_sum": membership_count_sum,
        "maximum_problem_bytes": maximum_problem_bytes,
        "maximum_problem_macro_id": maximum_problem_macro_id,
        "prepare_seconds": prepare_seconds,
        "prepare_peak_rss_bytes": peak_rss}
    atomic_write_json(output.work_dir / "plan.json", plan)  # 原子写出计划
    return plan  # 返回内存计划供调用方直接消费


def write_macro_gds(layer: LayerSpec, region: kdb.Region, path: Path,
                    dbu_um: float) -> Path:
    """把单 macro 当前完整候选 Region 写入 RESULT Cell，供检查和最终合并。

    layer 显式传入：写出行为只需要目标层号，不绑定 edge MacroProblem——
    像素 ILT 等非边段方法可直接复用同一写出契约。
    """
    layout = kdb.Layout()  # 独立原生版图对象
    layout.dbu = dbu_um  # 与源版图一致的 DBU，整数坐标物理尺寸不变
    cell = layout.create_cell("RESULT")  # 固定结果 Cell 名
    # 目标层
    index = layout.layer(kdb.LayerInfo(layer.layer, layer.datatype))
    region.insert_into(layout, cell.cell_index(), index)  # 插入完整候选 Region
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    # 同目录临时文件
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
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
            # 空 macro 候选是合法形态（如像素 ILT 的无材料区域或全暗
            # 优化结果）：GDS 不保存空层，层缺失等价于零覆盖 Region，
            # 不按损坏拒绝；真正的文件损坏仍由读盘/校验路径暴露。
            try:
                layer_bounds = database.layer_bbox(layer)  # 候选层真实包络
            except LayerNotFoundError:
                layer_bounds = None
            if layer_bounds is None:  # 候选无任何图形
                region = kdb.Region()  # 空覆盖直接构造，不查询
            else:
                # 层包络内查询物化（不用魔法框：
                batch = database.query(
                    [layer], layer_bounds).materialize()
                region = batch.region(layer)  # 候选 Region
        if not region.has_valid_polygons():  # 无效 polygon
            raise RuntimeError(f"{macro_id} 候选 Region 含无效 polygon")  # 明确失败
        ownership = DbuBox(*entry["ownership_box"])  # macro ownership 框
        # 权威覆盖选择：完整候选只贡献自身 ownership 内的部分，消除相邻 macro
        # context 的正面积重复；裁剪不是最终结果，seam 由写出端全局 merge 消除。
        clipped = region & kdb.Region(ownership.to_native())  # 精确相交
        area_before += int(clipped.area())  # 统计覆盖面积
        patches.append(GeometryPatch(macro_id, layer, clipped, ownership))  # 收集
    # 按配置模式写出最终版图
    written = PatchWriter.write_macro_results(
        patches, output_path, dbu_um,
        cell_mode=cell_mode)
    # 回读验证：merge/normalize 只能改变表示方式，不得改变物理覆盖面积。
    # 逐 macro 在自身 ownership 窗口统计面积后累加——ownership 半开不重叠、
    # 最终图形不越出 bbox，分块求和与全量面积数学等价，且避免第二个全量
    # Region 常驻（每窗口 Region 用完即弃）；失败时可定位到具体 macro。
    area_after = 0  # 回读累计面积
    with LayoutDB.open(written) as database:  # 回读最终版图
        for entry in plan["macros"]:  # 逐 macro 窗口
            ownership = DbuBox(*entry["ownership_box"])  # 该 macro 计分框
            # 与候选回读同款容忍：全部 macro 均空时最终 GDS 不含目标层，
            # 该窗口面积按 0 计（与 merge 前空覆盖守恒）。
            try:
                window = (database.query([layer], ownership)
                          .materialize_intersecting())
                # 完整相交会带入跨界 polygon 伸出窗口的部分，必须显式裁回
                # ownership（与主路径 clipped 同款），否则相邻窗口重复计数。
                region = (window.region(layer)
                          & kdb.Region(ownership.to_native()))
            except LayerNotFoundError:
                region = kdb.Region()
            if not region.has_valid_polygons():  # 无效 polygon
                # 明确失败并定位 macro
                raise RuntimeError(
                    f"{entry['macro_id']} 窗口含无效 polygon")
            area_after += int(region.area())  # 累计窗口面积
    if area_after != area_before:  # 覆盖面积被 normalize 改变
        # 明确失败
        raise RuntimeError(
            f"merge 前后覆盖面积改变：{area_before} -> {area_after}")
    return written  # 返回最终版图路径


def save_final_lithography(
        plan: dict, final_layout: Path, model, batch_size: int,
        output_dir: Path,
) -> dict:
    """流式保存最终版图每 tile 的 nominal 连续/二值 PNG 和 manifest。"""
    layer = LayerSpec(plan["layer"][0], plan["layer"][1])  # 目标层
    polarity = MaskPolarity(str(plan["polarity"]))  # 极性枚举
    pixel_dbu = int(plan["pixel_dbu"])  # 栅格像素
    canvas_pixels = int(plan["canvas_pixels"])  # 画布
    core_dbu = int(plan["core_size_dbu"])  # tile 尺寸
    context_dbu = int(plan["context_dbu"])  # tile 上下文
    dark_box = DbuBox(*plan["dark_box"])  # 光学暗界与迭代期 mask 同源
    with LayoutDB.open(final_layout) as database:  # 打开一次，全程在内物化消费
        bounds = database.layer_bbox(layer)  # 目标层真实包络（不用魔法框）
        if bounds is None:  # 空层无法出图
            raise ValueError("最终版图目标层为空")
        # 独立规整 tile 网格：单 macro 全 ROI 按 core 切分。可视化网格不必复刻
        # 迭代期 macro 边界，网格参数全部写入 manifest 供对账。
        macro = plan_macros(bounds, macro_grid=(1, 1), core_size_dbu=core_dbu,
                            context_dbu=context_dbu, pixel_dbu=pixel_dbu,
                            canvas_pixels=canvas_pixels)[0]
        output_dir.mkdir(parents=True, exist_ok=True)  # 留档目录
        threshold = float(model.config.print_threshold)  # 二值阈值
        core_count = macro.core_count  # tile 总数
        tiles = []  # manifest 条目

        def _window_region(spec):
            """只物化该 tile context 窗口相交的局部几何，用完即弃。"""
            # 窗口查询
            return (database.query([layer], spec.context_box)
                    .materialize_intersecting()
                    .region(layer))

        with torch.no_grad():  # 纯推理
            for batch_start in range(0, core_count, batch_size):  # 流式分批
                # 本批 tile
                specs = [macro.core(index) for index in range(
                    batch_start, min(batch_start + batch_size, core_count))]
                # 每 tile 窗口就地栅格
                masks = np.stack([rasterize_mask_canvas(
                    _window_region(spec), spec.context_box, pixel_dbu,
                    canvas_pixels, polarity=polarity,
                    dark_box=dark_box) for spec in specs])
                mask_tensor = torch.from_numpy(masks).to(model.device)  # 送设备
                # 一次标称前向
                printed = model.forward_many(
                    mask_tensor, (model.condition("nominal"),))["nominal"]
                images = printed.cpu().numpy()  # 取回 CPU
                del printed, mask_tensor  # 每 batch 写完立即释放
                for spec, image in zip(specs, images):  # 逐 tile 写 PNG
                    tile_id = spec.core_id  # 稳定 tile 编号
                    nominal_png = output_dir / f"{tile_id}_nominal.png"  # 连续灰度
                    # 连续值 0~255
                    Image.fromarray(
                        np.rint(image * 255.0).astype(np.uint8), mode="L").save(
                        nominal_png)
                    binary_png = output_dir / f"{tile_id}_binary.png"  # 阈值二值
                    # 阈值以上 255、其余 0
                    Image.fromarray(
                        np.where(image >= threshold, 255, 0).astype(np.uint8),
                        mode="L").save(binary_png)
                    # manifest 条目
                    tiles.append({
                        "tile_id": tile_id,
                        "ownership_box": [spec.ownership_box.left,
                                          spec.ownership_box.bottom,
                                          spec.ownership_box.right,
                                          spec.ownership_box.top],
                        "context_box": [spec.context_box.left,
                                        spec.context_box.bottom,
                                        spec.context_box.right,
                                        spec.context_box.top],
                        "nominal_png": nominal_png.name,
                        "binary_png": binary_png.name})
    # 完整清单
    manifest = {
        "format_version": 1,
        "pixel_dbu": pixel_dbu, "canvas_pixels": canvas_pixels,
        "threshold": threshold,
        "grid": {"core_size_dbu": core_dbu, "context_dbu": context_dbu},
        "tile_count": len(tiles), "tiles": tiles}
    atomic_write_json(output_dir / "manifest.json", manifest)  # 落盘清单
    return manifest  # 供 summary 消费
