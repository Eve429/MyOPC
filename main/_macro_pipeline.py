"""多个真实流程共用的 macro 生命周期：problem 准备、候选写出与最终合并。"""

import os
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Literal

import klayout.db as kdb
import numpy as np
import psutil
import torch
from PIL import Image

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.io import atomic_write_json
from common.units import exact_dbu
from geometry import GeometryPatch, PatchWriter
from layout import (
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
from opc.input.edge import prepare_macro_problem

# v2 移除 dark_box 键（2026-08-22 环带改几何方案）；v1 显式拒绝。
_PLAN_FORMAT_VERSION = 2


def resolve_field_bounds(layout: LayoutConfig, layer_bounds: DbuBox, dbu_nm: Decimal) -> DbuBox:
    """解析处理框：双 None 用 layer bbox；box 直用；size 以 layer bbox 居中推导。"""
    if layout.field_box_nm is not None:
        left, bottom, right, top = (
            exact_dbu(value, dbu_nm, f"field_box_nm[{index}]") for index, value in enumerate(layout.field_box_nm)
        )
        field = DbuBox(left, bottom, right, top)
    elif layout.field_size_nm is not None:
        width = exact_dbu(layout.field_size_nm[0], dbu_nm, "field_size_nm[0]")
        height = exact_dbu(layout.field_size_nm[1], dbu_nm, "field_size_nm[1]")
        # 逐轴居中：slack//2 归低侧、余量归高侧（与 _center_padding 的奇数余量约定一致）；
        # 宽高小于 layer 尺寸时 slack 为负，交由下方包含性校验统一报错，不在此重复分支。
        slack_x = width - layer_bounds.width
        slack_y = height - layer_bounds.height
        low_x, low_y = slack_x // 2, slack_y // 2
        field = DbuBox(
            layer_bounds.left - low_x,
            layer_bounds.bottom - low_y,
            layer_bounds.right + (slack_x - low_x),
            layer_bounds.top + (slack_y - low_y),
        )
    else:
        return layer_bounds  # 未配置：保持 layer bbox 现行行为，零行为变化
    if (
        field.left > layer_bounds.left
        or field.bottom > layer_bounds.bottom
        or field.right < layer_bounds.right
        or field.top < layer_bounds.top
    ):
        raise ValueError(
            f"处理框 ({field.left},{field.bottom})-({field.right},{field.top}) DBU"
            " 必须四向包含 layer bbox "
            f"({layer_bounds.left},{layer_bounds.bottom})-"
            f"({layer_bounds.right},{layer_bounds.top}) DBU"
            "——配置比版图小或偏移出界"
        )
    if (
        field.left < layer_bounds.left
        or field.bottom < layer_bounds.bottom
        or field.right > layer_bounds.right
        or field.top > layer_bounds.top
    ):
        scale = Decimal(dbu_nm)
        warnings.warn(
            "处理框大于 layer bbox：field "
            f"({Decimal(field.left) * scale},"
            f"{Decimal(field.bottom) * scale})-"
            f"({Decimal(field.right) * scale},"
            f"{Decimal(field.top) * scale}) nm ⊃ layer "
            f"({Decimal(layer_bounds.left) * scale},"
            f"{Decimal(layer_bounds.bottom) * scale})-"
            f"({Decimal(layer_bounds.right) * scale},"
            f"{Decimal(layer_bounds.top) * scale}) nm；",
            stacklevel=2,
        )
    return field


def prepare_problems(
    layout: LayoutConfig, partition: PartitionConfig, litho: LithographyConfig, edge: EdgeConfig, output: OutputConfig
) -> dict:
    """执行阶段 0/1，逐 macro 生成 problem，并写出 plan.json。"""
    if output.work_dir is None:
        raise ValueError("此流程要求 [output].work_dir")
    layer = LayerSpec(layout.layer, layout.datatype)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with LayoutDB.open(layout.layout, layout.top_cell) as database:
        top_cell_name = database.top_cell_name
        dbu_nm = Decimal(str(database.dbu_um)) * 1000
        layer_bounds = database.layer_bbox(layer)
        if layer_bounds is None:
            raise ValueError(f"目标层 {layer.layer}/{layer.datatype} 不含任何图形")
        # 处理框（field_box/field_size）,未配置时即 layer bbox
        bounds = resolve_field_bounds(layout, layer_bounds, dbu_nm)
        # nm→DBU 换算、context 契约与边段配置构造集中在 resolve_prepare_config。
        runtime = resolve_prepare_config(partition, litho, edge, dbu_nm)
        # 两级网格规划（内部完成像素/画布校验）
        macros = plan_macros(
            bounds,
            macro_grid=partition.macro_grid,
            macro_size_dbu=runtime.grid.macro_size_dbu,
            core_size_dbu=runtime.grid.core_dbu,
            context_dbu=runtime.grid.context_dbu,
            pixel_dbu=runtime.grid.pixel_dbu,
            canvas_pixels=litho.canvas_pixels,
        )
        # ownership 复核——面积和恰等于父框即无正面积重叠。
        if sum(macro.ownership_box.area for macro in macros) != bounds.area:
            raise RuntimeError("macro ownership 面积和不等于版图 bbox 面积")
        problems_dir = output.work_dir / "problems"
        problems_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        segment_count_sum = 0
        membership_count_sum = 0
        maximum_problem_bytes = 0
        maximum_problem_macro_id = ""
        for macro in macros:
            # 完整相交物化（不裁剪 occurrence）
            batch = database.query([layer], macro.query_box).materialize_intersecting()
            # 一次完成提边/分段/切线分裂/ownership
            problem = prepare_macro_problem(
                batch, layer, layout.polarity, runtime.fragmentation, macro, data_bounds=layer_bounds
            )
            problem_path = problem.save(problems_dir / f"{macro.macro_id}.npz")
            problem_bytes = problem_path.stat().st_size
            segment_count_sum += problem.segments.segment_count
            membership_count_sum += len(problem.member_segment_indices)
            if problem_bytes > maximum_problem_bytes:
                maximum_problem_bytes = problem_bytes
                maximum_problem_macro_id = macro.macro_id
            entries.append(
                {
                    "macro_id": macro.macro_id,
                    "ownership_box": [
                        macro.ownership_box.left,
                        macro.ownership_box.bottom,
                        macro.ownership_box.right,
                        macro.ownership_box.top,
                    ],
                    "core_count": macro.core_count,
                    "segment_count": problem.segments.segment_count,
                    "membership_count": len(problem.member_segment_indices),
                    "problem_file": str(problem_path),
                    "problem_bytes": problem_bytes,
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            del batch, problem
    # 全部 problem 成功且 LayoutDB 已关闭，才允许写出表示「准备完成」的 plan。
    prepare_seconds = time.perf_counter() - started
    # 完整计划（后续阶段唯一允许消费的产物）
    plan = {
        "format_version": _PLAN_FORMAT_VERSION,
        "layout": str(layout.layout),
        "top_cell": top_cell_name,
        "dbu_um": float(dbu_nm / 1000),
        "layer": [layer.layer, layer.datatype],
        "polarity": layout.polarity.value,
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
            "miter_limit": runtime.fragmentation.miter_limit,
        },
        "work_dir": str(output.work_dir),
        "final_layout": str(output.final_layout),
        "final_cell_mode": output.final_cell_mode,
        "macros": entries,
        "segment_count_sum": segment_count_sum,
        "membership_count_sum": membership_count_sum,
        "maximum_problem_bytes": maximum_problem_bytes,
        "maximum_problem_macro_id": maximum_problem_macro_id,
        "prepare_seconds": prepare_seconds,
        "prepare_peak_rss_bytes": peak_rss,
    }
    atomic_write_json(output.work_dir / "plan.json", plan)
    return plan


def write_macro_gds(layer: LayerSpec, region: kdb.Region, path: Path, dbu_um: float) -> Path:
    """把单 macro 当前完整候选 Region 写入 RESULT Cell，供检查和最终合并。

    layer 显式传入：写出行为只需要目标层号，不绑定 edge MacroProblem——
    像素 ILT 等非边段方法可直接复用同一写出契约。
    """
    layout = kdb.Layout()  # 独立原生版图对象
    layout.dbu = dbu_um  # 与源版图一致的 DBU，整数坐标物理尺寸不变
    cell = layout.create_cell("RESULT")
    index = layout.layer(kdb.LayerInfo(layer.layer, layer.datatype))
    region.insert_into(layout, cell.cell_index(), index)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        layout.write(str(temporary))
        os.replace(temporary, path)  # 原子替换
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def merge_macro_results(
    plan: dict,
    macro_gds_paths: Mapping[str, Path],
    output_path: Path,
    *,
    cell_mode: Literal["single_cell", "macro_cells"],
) -> Path:
    """按 plan 选择各 macro ownership 权威覆盖并写出一个全局结果。"""
    if cell_mode not in ("single_cell", "macro_cells"):
        raise ValueError(f"未知 cell_mode：{cell_mode}")
    layer = LayerSpec(plan["layer"][0], plan["layer"][1])
    dbu_um = float(plan["dbu_um"])  # 源版图 DBU
    identifiers = set()
    for entry in plan["macros"]:  # 先核对计划条目自身无重复
        macro_id = entry["macro_id"]
        if macro_id in identifiers:
            raise ValueError(f"重复 macro ID：{macro_id}")
        identifiers.add(macro_id)
    # 映射完整性：调用方必须显式给出每个 macro 的 GDS，函数不猜任何路径。
    missing = sorted(identifiers - set(macro_gds_paths))
    if missing:
        raise ValueError(f"macro GDS 映射缺失：{missing}")
    extra = sorted(set(macro_gds_paths) - identifiers)
    if extra:
        raise ValueError(f"macro GDS 映射多余：{extra}")
    patches: list[GeometryPatch] = []
    area_before = 0
    for entry in plan["macros"]:
        macro_id = entry["macro_id"]
        gds_path = Path(macro_gds_paths[macro_id])
        if not gds_path.is_file():
            raise FileNotFoundError(f"缺失 macro GDS：{gds_path}")
        with LayoutDB.open(gds_path) as database:  # 回读完整候选
            # 空 macro 候选是合法形态（如像素 ILT 的无材料区域或全暗
            # 优化结果）：GDS 不保存空层，层缺失等价于零覆盖 Region，
            # 不按损坏拒绝；真正的文件损坏仍由读盘/校验路径暴露。
            try:
                layer_bounds = database.layer_bbox(layer)  # 候选层真实包络
            except LayerNotFoundError:
                layer_bounds = None
            if layer_bounds is None:
                region = kdb.Region()  # 空覆盖直接构造，不查询
            else:
                # 层包络内查询物化（不用魔法框：
                batch = database.query([layer], layer_bounds).materialize()
                region = batch.region(layer)
        if not region.has_valid_polygons():
            raise RuntimeError(f"{macro_id} 候选 Region 含无效 polygon")
        ownership = DbuBox(*entry["ownership_box"])
        # 权威覆盖选择：完整候选只贡献自身 ownership 内的部分，消除相邻 macro
        # context 的正面积重复；裁剪不是最终结果，seam 由写出端全局 merge 消除。
        clipped = region & kdb.Region(ownership.to_native())
        area_before += int(clipped.area())
        patches.append(GeometryPatch(macro_id, layer, clipped, ownership))
    # 按配置模式写出最终版图
    written = PatchWriter.write_macro_results(patches, output_path, dbu_um, cell_mode=cell_mode)
    # 回读验证：merge/normalize 只能改变表示方式，不得改变物理覆盖面积。
    # 逐 macro 在自身 ownership 窗口统计面积后累加——ownership 半开不重叠、
    # 最终图形不越出 bbox，分块求和与全量面积数学等价，且避免第二个全量
    # Region 常驻（每窗口 Region 用完即弃）；失败时可定位到具体 macro。
    area_after = 0
    with LayoutDB.open(written) as database:  # 回读最终版图
        for entry in plan["macros"]:  # 逐 macro 窗口
            ownership = DbuBox(*entry["ownership_box"])  # 该 macro 计分框
            # 与候选回读同款容忍：全部 macro 均空时最终 GDS 不含目标层，
            # 该窗口面积按 0 计（与 merge 前空覆盖守恒）。
            try:
                window = database.query([layer], ownership).materialize_intersecting()
                # 完整相交会带入跨界 polygon 伸出窗口的部分，必须显式裁回
                # ownership（与主路径 clipped 同款），否则相邻窗口重复计数。
                region = window.region(layer) & kdb.Region(ownership.to_native())
            except LayerNotFoundError:
                region = kdb.Region()
            if not region.has_valid_polygons():
                raise RuntimeError(f"{entry['macro_id']} 窗口含无效 polygon")
            area_after += int(region.area())
    if area_after != area_before:  # 覆盖面积被 normalize 改变
        raise RuntimeError(f"merge 前后覆盖面积改变：{area_before} -> {area_after}")
    return written


def save_lithography_pngs(
    gds_path: Path,
    layer: LayerSpec,
    polarity: MaskPolarity,
    core_dbu: int,
    context_dbu: int,
    pixel_dbu: int,
    canvas_pixels: int,
    model,
    batch_size: int,
    output_dir: Path,
    *,
    top_cell: str | None = None,
) -> dict:
    """流式保存指定版图每 tile 的 nominal 连续/二值 PNG 和 manifest。

    网格按给定 GDS 自身 layer bbox 规划（不依赖迭代期 plan）；top_cell
    仅源版图需要——多顶层不指名即歧义失败，最终合并 GDS 恒单顶层不传。
    """
    with LayoutDB.open(gds_path, top_cell) as database:  # 打开一次，全程在内物化消费
        bounds = database.layer_bbox(layer)  # 目标层真实包络（不用魔法框）
        if bounds is None:  # 空层无法出图
            raise ValueError("最终版图目标层为空")
        # 独立规整 tile 网格：单 macro 全 ROI 按 core 切分。可视化网格不必复刻
        # 迭代期 macro 边界，网格参数全部写入 manifest 供对账。
        macro = plan_macros(
            bounds,
            macro_grid=(1, 1),
            core_size_dbu=core_dbu,
            context_dbu=context_dbu,
            pixel_dbu=pixel_dbu,
            canvas_pixels=canvas_pixels,
        )[0]
        output_dir.mkdir(parents=True, exist_ok=True)  # 留档目录
        threshold = float(model.config.print_threshold)  # 二值阈值
        core_count = macro.core_count  # tile 总数
        tiles = []  # manifest 条目

        def _window_region(spec):
            """只物化该 tile context 窗口相交的局部几何，用完即弃。"""
            # 窗口查询
            region = database.query([layer], spec.context_box).materialize_intersecting().region(layer)
            if polarity is MaskPolarity.OPAQUE:
                # 负板：最终版图包络外到 tile 查询边界补铬（与迭代期
                # prepare 同一规则），边界 PNG 不出现虚假亮环
                region = region + (kdb.Region(spec.context_box.to_native()) - kdb.Region(bounds.to_native()))
            return region

        with torch.no_grad():  # 纯推理
            for batch_start in range(0, core_count, batch_size):  # 流式分批
                # 本批 tile
                specs = [macro.core(index) for index in range(batch_start, min(batch_start + batch_size, core_count))]
                # 每 tile 窗口就地栅格
                masks = np.stack(
                    [
                        rasterize_mask_canvas(
                            _window_region(spec), spec.context_box, pixel_dbu, canvas_pixels, polarity=polarity
                        )
                        for spec in specs
                    ]
                )
                mask_tensor = torch.from_numpy(masks).to(model.device)  # 送设备
                # 一次标称前向
                printed = model.forward_many(mask_tensor, (model.condition("nominal"),))["nominal"]
                images = printed.cpu().numpy()  # 取回 CPU
                del printed, mask_tensor  # 每 batch 写完立即释放
                for spec, image in zip(specs, images):  # 逐 tile 写 PNG
                    tile_id = spec.core_id
                    # PNG 行 0 显示在顶部而模型数组行 0 是最低 Y，只在此 I/O
                    # 边界翻转一次（项目方向不变量）；翻转与灰度/阈值变换可交换
                    top_down = np.flipud(image)
                    nominal_png = output_dir / f"{tile_id}_nominal.png"
                    Image.fromarray(np.rint(top_down * 255.0).astype(np.uint8), mode="L").save(nominal_png)
                    binary_png = output_dir / f"{tile_id}_binary.png"
                    Image.fromarray(np.where(top_down >= threshold, 255, 0).astype(np.uint8), mode="L").save(binary_png)
                    # manifest 条目
                    tiles.append(
                        {
                            "tile_id": tile_id,
                            "ownership_box": [
                                spec.ownership_box.left,
                                spec.ownership_box.bottom,
                                spec.ownership_box.right,
                                spec.ownership_box.top,
                            ],
                            "context_box": [
                                spec.context_box.left,
                                spec.context_box.bottom,
                                spec.context_box.right,
                                spec.context_box.top,
                            ],
                            "nominal_png": nominal_png.name,
                            "binary_png": binary_png.name,
                        }
                    )
    # 完整清单
    manifest = {
        "format_version": 1,
        "pixel_dbu": pixel_dbu,
        "canvas_pixels": canvas_pixels,
        "threshold": threshold,
        "grid": {"core_size_dbu": core_dbu, "context_dbu": context_dbu},
        "tile_count": len(tiles),
        "tiles": tiles,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)  # 落盘清单
    return manifest  # 供 summary 消费


def _plan_lithography_arguments(plan: dict) -> tuple:
    """从 plan 提取留档内核所需六实参（层/极性/网格），final/source 包装共用。"""
    return (
        LayerSpec(plan["layer"][0], plan["layer"][1]),
        MaskPolarity(str(plan["polarity"])),
        int(plan["core_size_dbu"]),
        int(plan["context_dbu"]),
        int(plan["pixel_dbu"]),
        int(plan["canvas_pixels"]),
    )


def save_final_lithography(
    plan: dict,
    final_layout: Path,
    model,
    batch_size: int,
    output_dir: Path,
) -> dict:
    """从 plan 提取六键，对最终合并版图留档（save_lithography_pngs 薄包装）。"""
    layer, polarity, core_dbu, context_dbu, pixel_dbu, canvas_pixels = _plan_lithography_arguments(plan)
    return save_lithography_pngs(
        final_layout, layer, polarity, core_dbu, context_dbu, pixel_dbu, canvas_pixels, model, batch_size, output_dir
    )


def save_source_lithography(
    plan: dict,
    source_layout: Path,
    model,
    batch_size: int,
    output_dir: Path,
) -> dict:
    """同一内核对源（未 OPC）版图留档：同参数，可逐 tile 对照。"""
    layer, polarity, core_dbu, context_dbu, pixel_dbu, canvas_pixels = _plan_lithography_arguments(plan)
    return save_lithography_pngs(
        source_layout,
        layer,
        polarity,
        core_dbu,
        context_dbu,
        pixel_dbu,
        canvas_pixels,
        model,
        batch_size,
        output_dir,
        top_cell=plan["top_cell"],
    )
