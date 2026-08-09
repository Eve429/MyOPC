"""可直接运行的 MB-OPC 前端性能、内存和精确性基准。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

# 基准脚本位于二级目录，直接 Python 执行时只有 benchmarks 在
# sys.path；仅加入仓库根目录，不安装包、不修改解释器环境。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klayout
import klayout.db as kdb
import numpy as np
import psutil

from layout import CellRef, DbuBox, LayerSpec, RegionBatch
from opc.input import RectilinearCoreGrid
from opc.input.edge import FragmentationConfig, prepare_problem, reconstruct_region


def _synthetic_batch(shape_count: int) -> tuple[RegionBatch, LayerSpec]:
    """构造大量互不重叠的矩形，使数学边和控制段数可预测扩展。"""
    if shape_count <= 0:
        raise ValueError("shape_count must be positive")
    columns = int(np.ceil(np.sqrt(shape_count * 2.0)))
    rows = int(np.ceil(shape_count / columns))
    pitch, width, height = 140, 100, 80
    region = kdb.Region()
    for index in range(shape_count):
        column, row = index % columns, index // columns
        left, bottom = column * pitch + 10, row * pitch + 10
        region.insert(kdb.Box(left, bottom, left + width, bottom + height))
    layer = LayerSpec(1, 0)
    box = DbuBox(0, 0, columns * pitch, rows * pitch)
    return RegionBatch({layer: region}, box, CellRef("MBOPC_BENCHMARK", 0)), layer


def _expanded_representation_bytes(segment_count: int) -> int:
    """估算每段持久化完整几何、父字段、key 与查找索引的内存。"""
    # 对照方案为每段保存 starts/ends(32 B)、normal(16 B)、length(8 B)、
    # polygon/ring/edge 索引(12 B)、ordinal/count(8 B)、128-bit key(16 B)和
    # key 查找顺序/token(16 B)。不计 Python 对象开销，因此是保守上限。
    return segment_count * (32 + 16 + 8 + 12 + 8 + 16 + 16)


def run_benchmark(shape_count: int = 5_000) -> dict[str, object]:
    """测量准备、查 key、按需物化、零位移重建及稀疏归属。"""
    batch, layer = _synthetic_batch(shape_count)
    x = np.linspace(batch.query_box.left, batch.query_box.right, 9, dtype=np.int64)
    y = np.linspace(batch.query_box.bottom, batch.query_box.top, 9, dtype=np.int64)
    grid = RectilinearCoreGrid(x, y, 60)
    config = FragmentationConfig(10, 20, 8)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = perf_counter()
    problem = prepare_problem(batch, layer, config, grid)
    prepare_seconds = perf_counter() - started
    rss_after_prepare = process.memory_info().rss
    segments = problem.segments
    generator = np.random.default_rng(20260809)
    lookup_count = min(50_000, segments.segment_count)
    requested = generator.integers(0, segments.segment_count, size=lookup_count)
    started = perf_counter()
    located = segments.lookup_keys(segments.keys[requested])
    lookup_seconds = perf_counter() - started
    started = perf_counter()
    geometry = segments.materialize()
    materialize_seconds = perf_counter() - started
    started = perf_counter()
    reconstructed = reconstruct_region(segments, np.zeros(segments.segment_count), config)
    reconstruct_seconds = perf_counter() - started
    xor_area = int((reconstructed ^ problem.physical_mask.region).area())
    expanded_bytes = _expanded_representation_bytes(segments.segment_count)
    compact_bytes = segments.persistent_nbytes
    return {
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "klayout": klayout.__version__,
            "numpy": np.__version__, "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024.0 ** 3),
        },
        "counts": {
            "input_shapes": shape_count,
            "mathematical_edges": segments.edges.edge_count,
            "segments": segments.segment_count,
            "cores": len(problem.ownership.cores),
            "memberships": len(problem.ownership.member_segment_indices),
            "lookup_keys": lookup_count,
        },
        "timing_ms": {
            "prepare": prepare_seconds * 1000.0,
            "lookup": lookup_seconds * 1000.0,
            "materialize": materialize_seconds * 1000.0,
            "zero_reconstruct": reconstruct_seconds * 1000.0,
        },
        "memory": {
            "compact_persistent_mib": compact_bytes / (1024.0 ** 2),
            "expanded_estimate_mib": expanded_bytes / (1024.0 ** 2),
            "compact_saving_ratio": 1.0 - compact_bytes / expanded_bytes,
            "prepare_rss_delta_mib": max(0.0, rss_after_prepare - rss_before) / (1024.0 ** 2),
        },
        "verification": {
            "lookup_exact": bool(np.array_equal(located, requested.astype(np.int32))),
            "zero_displacement_xor_area": xor_area,
            "maximum_segment_length_dbu": float(geometry.lengths.max(initial=0.0)),
            "unowned_segments": int(np.count_nonzero(problem.ownership.owner_indices < 0)),
        },
    }


def strict_failures(result: dict[str, object]) -> list[str]:
    """根据宽松但可防止架构退化的门槛生成失败原因。"""
    counts, timing = result["counts"], result["timing_ms"]
    memory, verification = result["memory"], result["verification"]
    failures: list[str] = []
    if verification["zero_displacement_xor_area"]:
        failures.append("零位移重建 XOR 面积非零")
    if not verification["lookup_exact"] or verification["unowned_segments"]:
        failures.append("稳定 key 查找或 owner 完整性失败")
    if verification["maximum_segment_length_dbu"] > 20.0 + 1e-12:
        failures.append("控制段长度超过配置上限")
    # 保留排序 key 顺序和 token 可以让每轮更新直接 searchsorted，不重建
    # Python dict 或排序。因此以 40% 作为内存防退化门槛，优先保障迭代速度。
    if memory["compact_saving_ratio"] < 0.4:
        failures.append("紧凑常驻数组相对完全展开表示节省不足 40%")
    if timing["prepare"] > 5_000.0 or timing["zero_reconstruct"] > 5_000.0:
        failures.append("准备或零位移重建超过 5 秒")
    if timing["lookup"] > 500.0 or timing["materialize"] > 1_000.0:
        failures.append("批量 key 查找或坐标物化超过性能门槛")
    if counts["memberships"] > counts["segments"] * 9:
        failures.append("稀疏 halo membership 膨胀超过每段 9 个 core")
    return failures


def main(argv: list[str] | None = None) -> int:
    """解析规模与严格模式，在终端输出可归档 JSON。"""
    parser = argparse.ArgumentParser(description="MyOPC MB-OPC 前端性能基准")
    parser.add_argument("--shapes", type=int, default=5_000, help="合成独立矩形数量")
    parser.add_argument("--strict", action="store_true", help="未达验收门槛时返回非零退出码")
    args = parser.parse_args(argv)
    result = run_benchmark(args.shapes)
    failures = strict_failures(result)
    result["strict_failures"] = failures
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
