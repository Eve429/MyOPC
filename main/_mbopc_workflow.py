"""两个 MB-OPC 入口共享的配置加载、逐 macro 求解、最终合并与光刻输出。"""

import os  # 原子替换 result NPZ 的临时文件
import re  # device 字符串格式校验
import sys  # 把仓库根加入模块路径，保证免安装直接运行
import tempfile  # 与目标同目录的临时 NPZ
import time  # perf_counter 阶段计时
from dataclasses import asdict, dataclass  # 配置结构与记录序列化
from decimal import Decimal  # nm→DBU 精确换算
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # result NPZ 数组载体
import torch  # CUDA 可用性判定与张量搬运
from PIL import Image  # 最终光刻 PNG 留档
from tomllib import loads as toml_loads  # Python 3.12 标准库 TOML 解析

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from layout import DbuBox, LayerSpec, LayoutDB  # 最终光刻输出的版图回读
from lithography import ICCAD13Lithography  # 固定 ICCAD13 光刻模型
from main._macro_pipeline import (  # 共用 macro 生命周期
    MacroPipelineConfig,
    atomic_write_json,
    exact_dbu,
    load_macro_config,
    merge_macro_results,
    prepare_problems,
    write_macro_gds,
)
from opc.input import (  # 最终光刻输出的 tile 规划与栅格
    MaskPolarity,
    plan_macros,
    rasterize_mask_canvas,
)
from opc.input.edge import MacroProblem, reconstruct_region  # problem 与重建
from opc.iteration.mbopc import (  # 求解器结构与入口
    SimpleMBOPCConfig,
    SimpleMBOPCResult,
    TargetCanvasCache,
    optimize_macro,
)

# [mbopc] 段允许的键；未知键一律拒绝。
_MBOPC_KEYS = {"iterations", "initial_step_nm", "decay_every", "epe_distance_nm",
               "batch_size", "target_cache_mb", "device",
               "save_final_lithography", "show_progress"}
_RESULT_FORMAT_VERSION = 1  # 每 macro result NPZ 结构版本
# device 只接受 auto / cpu / cuda / cuda:N（N 为非负整数）。
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(:[0-9]+)?)$")


@dataclass(frozen=True, slots=True)
class MBOPCRunConfig:
    """保存公共 Macro 配置和 simple MB-OPC 物理单位参数。"""

    pipeline: MacroPipelineConfig  # 宏管线六段配置
    iterations: int                # 最多发布更新次数
    initial_step_nm: Decimal       # 初始步长（nm，运行期精确换算 DBU）
    decay_every: int               # 步长减半周期
    epe_distance_nm: Decimal       # EPE 探针距离（nm）
    batch_size: int                # 一次 forward 的 core 数
    target_cache_mb: int           # target uint8 LRU 上限（MiB）
    device: str                    # auto / cpu / cuda[:N]
    save_final_lithography: bool   # 是否保存最终光刻 PNG
    show_progress: bool            # 是否显示 tqdm 进度


def load_config(path: str | Path) -> MBOPCRunConfig:
    """读取宏管线六段与 [mbopc] 段并完成全部配置层校验。"""
    config_path = Path(path).expanduser().resolve()  # 配置绝对路径
    pipeline = load_macro_config(  # 六段共享配置（放行 MB-OPC 专属段）
        config_path, extra_sections=("mbopc",))  # 白名单放行
    raw = toml_loads(config_path.read_text(encoding="utf-8"))  # 再读一次取段
    section = raw.get("mbopc", {})  # MB-OPC 段
    unknown = set(section) - _MBOPC_KEYS  # 段内未知键
    if unknown:  # 拒绝拼错键
        raise ValueError(f"[mbopc] 含未知键：{sorted(unknown)}")
    required = ("iterations", "initial_step_nm", "decay_every",
                "epe_distance_nm", "batch_size", "target_cache_mb", "device")
    missing = [key for key in required if key not in section]  # 缺失键
    if missing:  # 显式报错
        raise ValueError(f"[mbopc] 缺少必填键：{missing}")
    iterations = int(section["iterations"])  # 迭代数
    decay_every = int(section["decay_every"])  # 衰减周期
    batch_size = int(section["batch_size"])  # 批大小
    target_cache_mb = int(section["target_cache_mb"])  # 缓存上限
    if (iterations < 1 or decay_every < 1 or batch_size < 1
            or target_cache_mb < 0):
        raise ValueError("iterations/decay_every/batch_size 必须为正，cache 为非负")
    initial_step_nm = Decimal(str(section["initial_step_nm"]))  # 十进制步长
    epe_distance_nm = Decimal(str(section["epe_distance_nm"]))  # 十进制探针距
    if initial_step_nm <= 0 or epe_distance_nm <= 0:
        raise ValueError("initial_step_nm 与 epe_distance_nm 必须为正")
    if initial_step_nm > pipeline.max_displacement_nm:  # 步长不得超过位移上限
        raise ValueError("initial_step_nm 不得超过 max_displacement_nm")
    if epe_distance_nm > pipeline.context_nm:  # 探针必须在 context 内
        raise ValueError("epe_distance_nm 不得超过 context_nm")
    device = str(section["device"])  # 设备字符串
    if not _DEVICE_PATTERN.match(device):  # 枚举校验
        raise ValueError(f"未知 device：{device}（只接受 auto/cpu/cuda[:N]）")
    save_final = section.get("save_final_lithography", False)  # 默认不保存
    show_progress = section.get("show_progress", False)  # 默认关闭（测试友好）
    if not isinstance(save_final, bool) or not isinstance(show_progress, bool):
        # 配置层全部错误统一 ValueError（与同函数其余校验一致），不改用 TypeError。
        raise ValueError("save_final_lithography/show_progress 必须是布尔值")  # noqa: TRY004
    return MBOPCRunConfig(  # 组装冻结配置
        pipeline=pipeline, iterations=iterations,  # 管线与迭代
        initial_step_nm=initial_step_nm, decay_every=decay_every,  # 步长
        epe_distance_nm=epe_distance_nm, batch_size=batch_size,  # 探针与批
        target_cache_mb=target_cache_mb, device=device,  # 缓存与设备
        save_final_lithography=save_final,  # 光刻留档
        show_progress=show_progress)  # 进度显示


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """把 NPZ 载荷经同目录临时文件原子写出。"""
    handle, temporary_name = tempfile.mkstemp(  # 同目录临时文件
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)  # 命名
    os.close(handle)  # 关闭句柄
    temporary = Path(temporary_name)  # Path 化
    try:  # 写出并原子替换
        with temporary.open("wb") as stream:  # 二进制写
            np.savez(stream, **arrays)  # 不压缩 NPZ
        os.replace(temporary, path)  # 原子替换
    finally:  # 清理
        if temporary.exists():  # 尚存
            temporary.unlink()  # 删除
    return path  # 返回路径


def solve_macro(
        problem: MacroProblem,
        model: ICCAD13Lithography,
        config: SimpleMBOPCConfig,
        target_cache: TargetCanvasCache,
        output_dir: Path,
        *,
        dbu_um: float,
        show_progress: bool,
        progress_position: int,
        leave_progress: bool,
) -> tuple[SimpleMBOPCResult, Path]:
    """显示 tile 进度，让一个 macro 完成全部迭代并写出 best GDS。"""
    bar = None  # 进度条（show_progress=False 时保持 None）
    if show_progress:  # 局部导入：关闭进度或未安装 tqdm 时不受影响
        from tqdm import tqdm  # 进度显示库
        bar = tqdm(  # baseline 与每个移动后状态都要评价全部 tile
            total=(config.iterations + 1) * problem.macro.core_count,
            desc=f"macro {problem.macro.macro_id}", unit="tile",  # tile 单位
            position=progress_position, leave=leave_progress)  # 多层条位置
    on_tiles = None if bar is None else bar.update  # 批完成且张量已释放后回调
    result = optimize_macro(  # 独立完成 baseline 与全部离散 EPE 轮次
        problem, model, config, target_cache, on_tiles_completed=on_tiles)
    if bar is not None:  # 提前停止按实际完成量收尾，不伪造 100%
        bar.close()
    output_dir.mkdir(parents=True, exist_ok=True)  # macro 专属目录
    best_region = reconstruct_region(  # best 位移的最终候选几何
        problem, result.best_displacements)
    best_gds = write_macro_gds(  # 完整候选 GDS（RESULT Cell）
        problem, best_region, output_dir / "best.gds", dbu_um)
    return result, best_gds  # 结果与 GDS 路径


def _run_mbopc(config_path: str | Path, *, require_multiple_macros: bool) -> dict:
    """执行两个入口共有的准备、逐 macro 求解、最终合并和结果保存。"""
    total_started = time.perf_counter()  # 全流程计时
    run_config = load_config(config_path)  # 配置层全部校验
    pipeline = run_config.pipeline  # 宏管线配置
    plan = prepare_problems(pipeline)  # 阶段 0/1（共用生命周期）
    dbu_nm = Decimal(str(plan["dbu_um"])) * 1000  # DBU 的 nm 值
    solver_config = SimpleMBOPCConfig(  # nm 参数精确换算为 DBU
        iterations=run_config.iterations,
        initial_step_dbu=float(exact_dbu(
            run_config.initial_step_nm, dbu_nm, "initial_step_nm")),
        decay_every=run_config.decay_every,
        epe_distance_dbu=float(exact_dbu(
            run_config.epe_distance_nm, dbu_nm, "epe_distance_nm")),
        batch_size=run_config.batch_size,
        target_cache_bytes=run_config.target_cache_mb * 1024 * 1024)
    device = run_config.device  # 设备解析
    if device == "auto":  # 有 CUDA 用 CUDA，否则 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ICCAD13Lithography(device=device)  # 固定 ICCAD13 模型
    macro_count = plan["macro_count"]  # macro 总数
    if require_multiple_macros:  # 多 macro 入口的数量约束
        if macro_count <= 1:
            raise ValueError(f"多 macro 入口要求 macro 数大于 1，实际 {macro_count}")
    elif macro_count != 1:  # 单 macro 入口的数量约束
        raise ValueError(f"单 macro 入口要求恰好 1 个 macro，实际 {macro_count}")
    for entry in plan["macros"]:  # 每个 macro 必须有多个 tile
        if entry["core_count"] <= 1:
            raise ValueError(
                f"{entry['macro_id']} 只有 {entry['core_count']} 个 tile，"
                "入口要求每 macro 至少 2 个 tile")
    target_cache = TargetCanvasCache(solver_config.target_cache_bytes)  # 跨 macro 共享
    macros_dir = pipeline.work_dir / "macros"  # 逐 macro 产物根目录
    macro_gds: dict[str, Path] = {}  # macro_id → best GDS（merge 显式映射）
    macro_summaries = []  # 逐 macro 摘要
    outer_bar = None  # 多 macro 外层进度条
    if require_multiple_macros and run_config.show_progress:
        from tqdm import tqdm  # 进度显示库
        outer_bar = tqdm(total=macro_count, desc="macros",  # 外层 macro 单位
                         unit="macro", position=0)  # 占第 0 行
    for entry in plan["macros"]:  # 稳定顺序逐 macro 独立求解
        macro_id = entry["macro_id"]  # macro 编号
        problem = MacroProblem.load(Path(entry["problem_file"]))  # 加载 problem
        started = time.perf_counter()  # 单 macro 计时
        result, best_gds = solve_macro(  # 全部迭代 + best GDS
            problem, model, solver_config, target_cache,
            macros_dir / macro_id,  # 专属产物目录
            dbu_um=float(plan["dbu_um"]),  # GDS 写出需要源 DBU（NPZ 不含）
            show_progress=run_config.show_progress,
            progress_position=1 if outer_bar is not None else 0,  # 外层占 0
            leave_progress=outer_bar is None)  # 多 macro 内层条不留存
        elapsed = time.perf_counter() - started  # 单 macro 耗时
        macro_dir = macros_dir / macro_id  # 产物目录
        _atomic_write_npz(  # result NPZ（位移与停止信息）
            macro_dir / "result.npz",
            format_version=np.array([_RESULT_FORMAT_VERSION], np.int32),
            macro_id=np.array([macro_id]),
            best_round=np.array([result.best_round], np.int32),
            best_displacements=np.ascontiguousarray(
                result.best_displacements, dtype=np.float64),
            stop_reason=np.array([result.stop_reason]))
        atomic_write_json(macro_dir / "metrics.json", {  # 逐轮标量与原因
            "macro_id": macro_id,
            "best_round": result.best_round,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "records": [asdict(record) for record in result.records]})
        macro_gds[macro_id] = best_gds  # 记录显式映射
        best_record = result.records[result.best_round]  # 最佳轮指标
        macro_summaries.append({  # 摘要（全量记录在 metrics.json）
            "macro_id": macro_id,
            "best_round": result.best_round,
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "round_count": len(result.records),
            "best_epe": best_record.epe, "best_l2": best_record.l2,
            "best_pvband": best_record.pvband,
            "best_gds": str(best_gds),
            "elapsed_seconds": elapsed})
        if outer_bar is not None:  # 外层条按完成 macro 计数
            outer_bar.update(1)
        del problem  # 释放当前 macro 再处理下一个
    if outer_bar is not None:  # 外层条收尾
        outer_bar.close()
    # 全部 macro 完成后只合并一次（独立 macro 策略，不做逐轮全局合并）。
    merge_started = time.perf_counter()  # 合并计时
    final_path = merge_macro_results(  # 统一 ownership 权威覆盖写出
        plan, macro_gds, pipeline.final_layout,
        cell_mode=pipeline.final_cell_mode)
    merge_seconds = time.perf_counter() - merge_started  # 合并耗时
    manifest = None  # 最终光刻留档
    if run_config.save_final_lithography:  # 只对最终合并 GDS 运行一次
        manifest = save_final_lithography(  # 逐 tile 流式 PNG
            plan, final_path, model, run_config.batch_size,
            pipeline.work_dir / "final_lithography")
    summary = {  # 完整摘要
        "macro_count": macro_count,
        "core_count": plan["core_count"],
        "segment_count_sum": plan["segment_count_sum"],
        "device": str(model.device),
        "iterations": solver_config.iterations,
        "macros": macro_summaries,
        "final_layout": str(final_path),
        "final_cell_mode": pipeline.final_cell_mode,
        "merge_seconds": merge_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "final_lithography_tiles": None if manifest is None else manifest["tile_count"]}
    atomic_write_json(pipeline.work_dir / "summary.json", summary)  # 落盘
    return summary  # 返回摘要


def run_single_macro(config_path: str | Path) -> dict:
    """准备并求解恰好一个 macro、多个 tile，使用统一 merge 写最终结果。"""
    return _run_mbopc(config_path, require_multiple_macros=False)


def run_multi_macro(config_path: str | Path) -> dict:
    """逐个独立求解多个 macro，全部完成后只执行一次 merge。"""
    return _run_mbopc(config_path, require_multiple_macros=True)


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
    with LayoutDB.open(final_layout) as database:  # 回读最终版图
        bounds = database.layer_bbox(layer)  # 目标层 bbox
        if bounds is None:  # 空层无法出图
            raise ValueError("最终版图目标层为空")
        batch = database.query(  # 全框物化最终覆盖
            [layer], DbuBox(-(2 ** 30), -(2 ** 30), 2 ** 30, 2 ** 30)).materialize()
    region = batch.region(layer)  # 最终 Region
    # 独立规整 tile 网格：单 macro 全 ROI 按 core 切分。可视化网格不必复刻
    # 迭代期 macro 边界，网格参数全部写入 manifest 供对账。
    macro = plan_macros(bounds, macro_grid=(1, 1), core_size_dbu=core_dbu,
                        context_dbu=context_dbu, pixel_dbu=pixel_dbu,
                        canvas_pixels=canvas_pixels)[0]
    output_dir.mkdir(parents=True, exist_ok=True)  # 留档目录
    threshold = float(model.config.print_threshold)  # 二值阈值
    core_count = macro.core_count  # tile 总数
    tiles = []  # manifest 条目
    with torch.no_grad():  # 纯推理
        for batch_start in range(0, core_count, batch_size):  # 流式分批
            specs = [macro.core(index) for index in range(  # 本批 tile
                batch_start, min(batch_start + batch_size, core_count))]
            masks = np.stack([rasterize_mask_canvas(  # 每 tile 居中画布
                region, spec.context_box, pixel_dbu, canvas_pixels,
                polarity=polarity) for spec in specs])
            mask_tensor = torch.from_numpy(masks).to(model.device)  # 送设备
            printed = model.forward_many(  # 一次标称前向
                mask_tensor, (model.condition("nominal"),))["nominal"]
            images = printed.cpu().numpy()  # 取回 CPU
            del printed, mask_tensor  # 每 batch 写完立即释放
            for spec, image in zip(specs, images):  # 逐 tile 写 PNG
                tile_id = spec.core_id  # 稳定 tile 编号
                nominal_png = output_dir / f"{tile_id}_nominal.png"  # 连续灰度
                Image.fromarray(  # 连续值 0~255
                    np.rint(image * 255.0).astype(np.uint8), mode="L").save(
                    nominal_png)
                binary_png = output_dir / f"{tile_id}_binary.png"  # 阈值二值
                Image.fromarray(  # 阈值以上 255、其余 0
                    np.where(image >= threshold, 255, 0).astype(np.uint8),
                    mode="L").save(binary_png)
                tiles.append({  # manifest 条目
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
    manifest = {  # 完整清单
        "format_version": 1,
        "pixel_dbu": pixel_dbu, "canvas_pixels": canvas_pixels,
        "threshold": threshold,
        "grid": {"core_size_dbu": core_dbu, "context_dbu": context_dbu},
        "tile_count": len(tiles), "tiles": tiles}
    atomic_write_json(output_dir / "manifest.json", manifest)  # 落盘清单
    return manifest  # 供 summary 消费
