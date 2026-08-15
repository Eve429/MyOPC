"""统一运行 Simple、LevelSet、CurvMulti 和 Multilevel ILT。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import estimate_rectangular_shots, evaluate_binary_l2, evaluate_pvband
from layout import DbuBox, LayerSpec
from lithography import ICCAD13Lithography
from main.artifacts import (
    atomic_json, atomic_npz, atomic_png, save_final_lithography_result,
)
from main.offline_inputs import (
    add_layout_source_arguments,
    resolve_raster_input,
)
from main.configuration import ConfiguredArgumentParser, glp_layer_map
from opc.input import process_memory_snapshot
from opc.iteration.ilt import (
    CurvMultiConfig,
    LevelSetConfig,
    MultilevelConfig,
    SimpleILTConfig,
    SimpleILTResult,
    optimize,
    optimize_curvmulti,
    optimize_levelset,
    optimize_multilevel,
)


def run_ilt(
        input_path: str | Path, output_dir: str | Path, *, method: str = "simple",
        iterations: int | None = None, step_size: float | None = None,
        weight_pvband: float | None = None,
        weight_process_l2: float | None = None,
        curvature_weight: float | None = None,
        scales: tuple[int, ...] | None = None,
        stage_iterations: tuple[int, ...] | None = None,
        stage_step_sizes: tuple[float, ...] | None = None,
        smoothing_kernel: int = 7,
        sigmoid_steepness: float = 4.0, sigmoid_offset: float = 0.5,
        mask_threshold: float = 0.5, device: str = "auto",
        save_png: bool = True, layer: LayerSpec | None = None,
        top_cell: str | None = None,
        box: DbuBox | tuple[int, int, int, int] | None = None,
        pixel_nm: float = 8.0, canvas: int = 256,
        max_file_gib: float = 4.0,
        max_shape_occurrences: int = 5_000_000,
        max_source_vertices: int = 20_000_000,
        max_estimated_gib: float = 8.0, polarity: str = "clear",
        glp_layers: dict[str, LayerSpec] | None = None,
        run_configuration: dict[str, object] | None = None,
        return_result: bool = False
        ) -> dict[str, Any] | tuple[SimpleILTResult, dict[str, Any]]:
    """运行指定 ILT；可选同时返回内存结果，并始终保存统一产物与统计。

    这是像素域 ILT（反演光刻技术，Inverse Lithography Technology）的统一入口。
    与边段 OPC 不同，ILT 不依赖几何边段，而是直接在像素域优化一张掩膜（由软
    mask soft→硬 mask binary），目标是让成像结果逼近目标图形。本入口按
    --method 分派四种方法：simple（基础梯度）、levelset（水平集）、
    curvmulti（多尺度曲率）、multilevel（多级由粗到细）。输入是像素 target
    （版图 ROI 当场栅格化，或离线 raster NPZ），输出软/硬 mask 与最终成像。
    管线在 `layout → geometry + lithography → opc.iteration.ilt`。
    输入：像素 target 来源、输出目录、方法与各方法超参。
    输出：summary dict（return_result=True 时额外返回内存结果对象，绝不重新优化）。
    """
    loaded_started = perf_counter()
    memory_checkpoints = {"start": process_memory_snapshot()}
    # 阶段①取得像素 target。GDS/OASIS 分支在 Region 物化前执行文件、层级图形、
    # 顶点、预计内存和 canvas 上限检查；NPZ 分支验证版本及解压上限。两条路径
    # 只在内存中汇合。
    target_array, metadata = resolve_raster_input(
        input_path, layer=layer, top_cell=top_cell, box=box,
        pixel_nm=pixel_nm, canvas=canvas, max_file_gib=max_file_gib,
        max_shape_occurrences=max_shape_occurrences,
        max_source_vertices=max_source_vertices,
        max_estimated_gib=max_estimated_gib, polarity=polarity,
        glp_layers=glp_layers)
    loaded = perf_counter()
    memory_checkpoints["input"] = process_memory_snapshot()
    model = ICCAD13Lithography(device=device)
    memory_checkpoints["model"] = process_memory_snapshot()
    if target_array.shape[0] > model.config.canvas or target_array.shape[1] > model.config.canvas:
        raise ValueError("ILT target 超过当前光刻模型 canvas")
    target = torch.as_tensor(target_array, device=model.device)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    optimized_started = perf_counter()
    # 阶段②按方法分派优化。各方法对像素 mask 的优化思路不同：simple 直接对软
    # mask 做带 sigmoid 的梯度下降；levelset 维护隐式水平集函数演化零等值线；
    # curvmulti 在多尺度上叠加曲率正则；multilevel 由粗尺度到细尺度逐级精化。
    # 分派只构造各求解器已有配置，不建立注册器或包装基类。前三阶段分别对
    # levelset、curvmulti、multilevel 做完整质量承诺。None 表示使用入口既有默认
    # 或新增方法自身默认，避免不同算法的损失权重被静默共用。
    if method == "simple":
        config: object = SimpleILTConfig(
            10 if iterations is None else iterations,
            0.2 if step_size is None else step_size,
            sigmoid_steepness=sigmoid_steepness,
            weight_pvband=0.0 if weight_pvband is None else weight_pvband,
            weight_process_l2=(1.0 if weight_process_l2 is None else
                               weight_process_l2),
            curvature_weight=(0.0 if curvature_weight is None else
                              curvature_weight),
            mask_threshold=mask_threshold)
        result = optimize(target, model, config)
    elif method == "levelset":
        config = LevelSetConfig(
            10 if iterations is None else iterations,
            0.2 if step_size is None else step_size,
            1.0 if weight_process_l2 is None else weight_process_l2,
            0.0 if weight_pvband is None else weight_pvband,
            0.0 if curvature_weight is None else curvature_weight)
        result = optimize_levelset(target, model, config)
    elif method == "curvmulti":
        config = CurvMultiConfig(
            (4, 2, 1) if scales is None else scales,
            10 if iterations is None else iterations,
            0.5 if step_size is None else step_size,
            smoothing_kernel,
            sigmoid_steepness, sigmoid_offset,
            0.0 if weight_process_l2 is None else weight_process_l2,
            1.0 if weight_pvband is None else weight_pvband,
            200.0 if curvature_weight is None else curvature_weight,
            mask_threshold)
        result = optimize_curvmulti(target, model, config)
    elif method == "multilevel":
        resolved_scales = (2, 1) if scales is None else scales
        if stage_iterations is not None:
            resolved_iterations = stage_iterations
        elif iterations is not None:
            resolved_iterations = (iterations,) * len(resolved_scales)
        elif resolved_scales == (2, 1):
            resolved_iterations = (20, 100)
        else:
            raise ValueError("自定义 Multilevel scales 时必须指定 iterations 或 stage_iterations")
        resolved_steps = (stage_step_sizes if stage_step_sizes is not None else
                          ((0.2 if step_size is None else step_size),) *
                          len(resolved_scales))
        config = MultilevelConfig(
            resolved_scales, resolved_iterations, resolved_steps,
            smoothing_kernel, sigmoid_steepness, sigmoid_offset,
            0.0 if weight_process_l2 is None else weight_process_l2,
            1.0 if weight_pvband is None else weight_pvband,
            0.0 if curvature_weight is None else curvature_weight,
            mask_threshold)
        result = optimize_multilevel(target, model, config)
    else:
        raise ValueError(f"未知 ILT 方法：{method}")
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    optimized = perf_counter()
    memory_checkpoints["optimization"] = process_memory_snapshot()
    conditions = tuple(model.condition(name) for name in (
        "nominal", "dose_max", "defocus_min"))
    # 阶段③统一评价最佳硬 mask。L2 衡量标称成像与目标差，PVBand 衡量工艺漂移下
    # 的边缘不确定性，shots 估算掩膜矩形碎块数。只在优化结束后追加一次，no_grad
    # 避免为纯报告保存 autograd 图；forward_many 让三个条件共享 mask FFT。
    # 最佳硬 mask 只在优化结束后追加一次统一评价；no_grad 避免为纯报告保存
    # autograd 图。forward_many 让三个条件共享 mask FFT，不重复占用输入频谱。
    with torch.no_grad():
        printed = model.forward_many(result.binary_mask.to(torch.float32), conditions)
        l2 = evaluate_binary_l2(target, printed["nominal"], model.config.print_threshold)
        pvband = evaluate_pvband(
            printed["dose_max"], printed["defocus_min"], model.config.print_threshold)
        shots = estimate_rectangular_shots(result.binary_mask)
    evaluated = perf_counter()
    memory_checkpoints["evaluation"] = process_memory_snapshot()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 阶段④归档产物。GPU Tensor 只在这里各拷回一次；后续 NPZ/PNG 全部复用 CPU
    # 数组，防止每个产物重复触发设备同步和临时主机内存分配。
    parameters = result.best_parameters.detach().cpu().numpy().astype(np.float32, copy=False)
    soft_mask = result.soft_mask.detach().cpu().numpy().astype(np.float32, copy=False)
    binary_mask = result.binary_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    result_path = atomic_npz(output / "ilt_result.npz", {
        "format_name": np.array("myopc.ilt-result"),
        "format_version": np.array(1, dtype=np.int32),
        "method": np.array(method), "best_parameters": parameters,
        "soft_mask": soft_mask, "binary_mask": binary_mask,
        "best_iteration": np.array(result.best_iteration, dtype=np.int32),
    })
    final_lithography = save_final_lithography_result(
        output, binary_mask, printed, save_png=save_png)
    images: dict[str, str] = {}
    if save_png:
        for name, values in (("target", target_array), ("soft_mask", soft_mask),
                             ("binary_mask", binary_mask)):
            images[name] = str(atomic_png(output / f"{name}.png", values))
    finished = perf_counter()
    memory_checkpoints["output"] = process_memory_snapshot()
    record_values: list[dict[str, Any]] = []
    for index, record in enumerate(result.records):
        value = asdict(record)
        if method == "curvmulti":
            # 复用公共 ILTIterationRecord，不为 scale 重复建立第二套记录类型；
            # 仅在报告边界补充可由固定每阶段轮数精确推导的阶段位置。
            stage_index = index // config.iterations_per_stage
            value.update({"stage_index": stage_index,
                          "stage_scale": config.scales[stage_index],
                          "stage_iteration": index % config.iterations_per_stage})
        elif method == "multilevel":
            # 每级迭代数可不同，按累计边界定位当前记录；阶段数通常只有 2–3，
            # 这里只运行于最终 JSON 构造，不进入 GPU 优化热路径。
            stage_start = 0
            for stage_index, stage_count in enumerate(config.stage_iterations):
                if index < stage_start + stage_count:
                    value.update({"stage_index": stage_index,
                                  "stage_scale": config.scales[stage_index],
                                  "stage_iteration": index - stage_start})
                    break
                stage_start += stage_count
        record_values.append(value)
    summary: dict[str, Any] = {
        "run_configuration": run_configuration,
        "status": "completed", "method": method,
        "input": str(Path(input_path).expanduser().resolve()),
        "source_layout": metadata.get("source"), "device": str(model.device),
        "shape": list(target_array.shape), "config": asdict(config),
        "best_iteration": result.best_iteration,
        "records": record_values,
        "evaluation": {"binary_l2": l2, "pvband": pvband,
                       "rectangular_shot_estimate": shots},
        "timing_seconds": {"input": loaded - loaded_started,
                           "optimization": optimized - optimized_started,
                           "evaluation": evaluated - optimized,
                           "output": finished - evaluated,
                           "total": finished - loaded_started},
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda" else 0),
        "memory_checkpoints": memory_checkpoints,
        "artifacts": {"result_npz": str(result_path), "images": images,
                      "final_lithography": final_lithography,
                      "summary": str(output / "summary.json")},
    }
    atomic_json(output / "summary.json", summary)
    # 兼容统一入口原有 summary 返回值；SimpleILT 兼容脚本显式请求同一次执行
    # 已生成的内存结果，绝不重新优化、重读产物或维护第二套评价流程。
    return (result, summary) if return_result else summary


def main(argv: list[str] | None = None) -> int:
    """解析统一 ILT 命令行并返回标准退出码。"""
    parser = ConfiguredArgumentParser(
        description="运行统一 ILT 方法", workflow="ilt", entry="ilt",
        valid_entries=("ilt", "simpleilt"))
    parser.add_argument("input", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("simple", "levelset", "curvmulti", "multilevel"))
    parser.add_argument("--iterations", type=int,
                        help="不指定时使用方法默认；Multilevel 指定后各级相同")
    parser.add_argument("--step-size", type=float,
                        help="不指定时使用所选方法自己的默认步长")
    parser.add_argument("--weight-pvband", type=float,
                        help="不指定时使用所选方法自己的默认权重")
    parser.add_argument("--weight-process-l2", type=float,
                        help="不指定时使用所选方法自己的默认权重")
    parser.add_argument("--curvature-weight", type=float,
                        help="不指定时使用所选方法自己的默认权重")
    parser.add_argument("--scales", type=int, nargs="+",
                        help="CurvMulti/Multilevel 严格递减尺度，必须以 1 结束")
    parser.add_argument("--stage-iterations", type=int, nargs="+",
                        help="Multilevel 各级迭代数，数量必须与 scales 相同")
    parser.add_argument("--stage-step-sizes", type=float, nargs="+",
                        help="Multilevel 各级 Adam 实际步长，数量必须与 scales 相同")
    parser.add_argument("--smoothing-kernel", type=int)
    parser.add_argument("--sigmoid-steepness", type=float)
    parser.add_argument("--sigmoid-offset", type=float)
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--device")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--json", action="store_true", help="终端输出完整 JSON 汇总")
    add_layout_source_arguments(parser)
    parser.add_argument("--pixel-nm", type=float)
    parser.add_argument("--canvas", type=int)
    args = parser.parse_args(argv)
    try:
        summary = run_ilt(
            args.input, args.output_dir, method=args.method,
            iterations=args.iterations, step_size=args.step_size,
            weight_pvband=args.weight_pvband,
            weight_process_l2=args.weight_process_l2,
            curvature_weight=args.curvature_weight,
            scales=None if args.scales is None else tuple(args.scales),
            stage_iterations=(None if args.stage_iterations is None else
                              tuple(args.stage_iterations)),
            stage_step_sizes=(None if args.stage_step_sizes is None else
                              tuple(args.stage_step_sizes)),
            smoothing_kernel=args.smoothing_kernel,
            sigmoid_steepness=args.sigmoid_steepness,
            sigmoid_offset=args.sigmoid_offset,
            mask_threshold=args.mask_threshold, device=args.device,
            save_png=not args.no_png, layer=args.layer,
            top_cell=args.top_cell,
            box=None if args.box is None else tuple(args.box),
            pixel_nm=args.pixel_nm, canvas=args.canvas,
            max_file_gib=args.max_file_gib,
            max_shape_occurrences=args.max_shapes,
            max_source_vertices=args.max_vertices,
            max_estimated_gib=args.max_estimated_gib,
            polarity=args.polarity, glp_layers=glp_layer_map(args.glp_layers),
            run_configuration=args._configuration)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"ILT 完成：{summary['method']}，输出：{summary['artifacts']['summary']}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
