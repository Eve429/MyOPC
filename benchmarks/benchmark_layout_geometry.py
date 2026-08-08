"""可直接运行的 Layout/Geometry 性能与内存基准。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

# 该文件位于 benchmarks 子目录。直接执行时 Python 只把该子目录加入 sys.path，
# 因此这里显式加入仓库根目录；这不是安装包，也不会修改当前 Python 环境。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klayout
import klayout.db as kdb
import numpy as np
import psutil

from geometry import EdgeBatch, GeometryEngine, UniformGridIndex
from layout import DbuBox, LayerSpec, LayoutDB


def percentile_ms(values: list[float], percentile: float) -> float:
    """把秒级样本转换为指定百分位的毫秒值。"""
    return float(np.percentile(np.asarray(values), percentile) * 1000.0)


def build_million_instance_gds(path: Path) -> None:
    """写出只含一个 AREF、但逻辑上包含一百万个矩形的紧凑 GDS。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    layer = layout.layer(kdb.LayerInfo(1, 0))
    leaf = layout.create_cell("LEAF")
    leaf.shapes(layer).insert(kdb.Box(0, 0, 100, 100))
    top = layout.create_cell("TOP")
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(),
                                kdb.Vector(200, 0), kdb.Vector(0, 200), 1000, 1000))
    layout.write(str(path))


def benchmark_hierarchy(runs: int) -> dict[str, float | int]:
    """测量单次加载后对百万逻辑实例执行小 ROI 查询的成本。"""
    process = psutil.Process()
    with tempfile.TemporaryDirectory(prefix="myopc-benchmark-") as directory:
        source = Path(directory) / "million.gds"
        build_million_instance_gds(source)
        rss_before = process.memory_info().rss
        started = perf_counter()
        database = LayoutDB.open(source)
        open_seconds = perf_counter() - started
        layer = LayerSpec(1, 0)
        box = DbuBox(90_000, 90_000, 91_000, 91_000)
        query = database.query([layer], box)
        engine = GeometryEngine()
        for _ in range(5):
            engine.clip(query.materialize(), box)
        samples: list[float] = []
        polygon_count = 0
        for _ in range(runs):
            started = perf_counter()
            clipped = engine.clip(query.materialize(), box)
            samples.append(perf_counter() - started)
            polygon_count = int(clipped.region(layer).count())
        rss_after = process.memory_info().rss
        database.close()
    return {
        "logical_instance_count": 1_000_000,
        "roi_polygon_count": polygon_count,
        "open_ms": open_seconds * 1000.0,
        "query_clip_median_ms": percentile_ms(samples, 50),
        "query_clip_p95_ms": percentile_ms(samples, 95),
        "rss_delta_mb": max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0)),
    }


def timed_queries(query: Callable[[DbuBox], np.ndarray], boxes: list[DbuBox]) -> tuple[list[float], list[np.ndarray]]:
    """逐次计时查询，并保留结果用于索引与暴力扫描一致性检查。"""
    samples: list[float] = []
    results: list[np.ndarray] = []
    for box in boxes:
        started = perf_counter()
        results.append(query(box))
        samples.append(perf_counter() - started)
    return samples, results


def benchmark_spatial(edge_count: int, query_count: int) -> dict[str, float | int]:
    """比较 tile-local 网格索引与全量 NumPy bbox 扫描。"""
    columns = int(np.ceil(np.sqrt(edge_count)))
    ids = np.arange(edge_count, dtype=np.int64)
    x = (ids % columns) * 100
    y = (ids // columns) * 100
    starts = np.column_stack((x, y))
    ends = starts + np.array([20, 0], dtype=np.int64)
    edges = EdgeBatch(LayerSpec(1, 0), starts, ends, ids, ids, np.zeros(edge_count, dtype=bool))
    started = perf_counter()
    index = UniformGridIndex(edges, cell_size_dbu=100)
    build_ms = (perf_counter() - started) * 1000.0
    rng = np.random.default_rng(20260809)
    selected = rng.integers(0, edge_count, size=query_count)
    boxes = [DbuBox(int(x[i]) - 10, int(y[i]) - 10, int(x[i]) + 30, int(y[i]) + 10)
             for i in selected]
    bboxes = edges.bboxes

    def brute_force(box: DbuBox) -> np.ndarray:
        """使用完整向量化扫描生成正确性基准。"""
        keep = ((bboxes[:, 0] <= box.right) & (bboxes[:, 2] >= box.left) &
                (bboxes[:, 1] <= box.top) & (bboxes[:, 3] >= box.bottom))
        return np.flatnonzero(keep)

    # 预热 NumPy 分配器与网格查询路径，避免把首次调用成本计入中位数。
    for box in boxes[:10]:
        index.query_box(box)
        brute_force(box)
    index_samples, index_results = timed_queries(index.query_box, boxes)
    brute_samples, brute_results = timed_queries(brute_force, boxes)
    exact = all(np.array_equal(indexed, brute) for indexed, brute in zip(index_results, brute_results))
    index_median = percentile_ms(index_samples, 50)
    brute_median = percentile_ms(brute_samples, 50)
    return {
        "edge_count": edge_count,
        "query_count": query_count,
        "build_ms": build_ms,
        "index_query_median_ms": index_median,
        "index_query_p95_ms": percentile_ms(index_samples, 95),
        "brute_query_median_ms": brute_median,
        "speedup": float("inf") if index_median == 0 else brute_median / index_median,
        "results_exact": exact,
    }


def run_benchmarks(runs: int, edge_count: int, query_count: int) -> dict[str, object]:
    """运行全部基准并附带可复现实验环境。"""
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "klayout": klayout.__version__,
            "numpy": np.__version__,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024.0 ** 3),
        },
        "hierarchy": benchmark_hierarchy(runs),
        "spatial": benchmark_spatial(edge_count, query_count),
    }


def strict_failures(result: dict[str, object]) -> list[str]:
    """根据开发方案中的性能门槛返回失败原因。"""
    hierarchy = result["hierarchy"]
    spatial = result["spatial"]
    failures: list[str] = []
    if hierarchy["roi_polygon_count"] != 25:
        failures.append("百万实例 ROI 应精确得到 25 个 Polygon")
    if hierarchy["query_clip_median_ms"] > 50.0:
        failures.append("百万实例 ROI 查询与裁剪中位数超过 50 ms")
    if hierarchy["rss_delta_mb"] > 64.0:
        failures.append("百万实例 ROI 额外 RSS 超过 64 MB")
    if not spatial["results_exact"]:
        failures.append("网格索引结果与暴力扫描不一致")
    if spatial["speedup"] < 2.0:
        failures.append("网格索引查询加速低于 2 倍")
    return failures


def main(argv: list[str] | None = None) -> int:
    """解析基准参数，输出 JSON，并在严格模式下返回可用于 CI 的退出码。"""
    parser = argparse.ArgumentParser(description="MyOPC Layout/Geometry 性能基准")
    parser.add_argument("--runs", type=int, default=100, help="层级 ROI 重复次数")
    parser.add_argument("--edges", type=int, default=100_000, help="空间索引边数量")
    parser.add_argument("--queries", type=int, default=1_000, help="空间索引查询次数")
    parser.add_argument("--strict", action="store_true", help="未达到验收门槛时返回非零退出码")
    args = parser.parse_args(argv)
    result = run_benchmarks(args.runs, args.edges, args.queries)
    failures = strict_failures(result)
    result["strict_failures"] = failures
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
