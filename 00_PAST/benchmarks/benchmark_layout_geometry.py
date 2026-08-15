"""可直接运行的 Layout/Geometry 性能与内存基准。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
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

from geometry import render_region_batch
from layout import CellRef, DbuBox, LayerSpec, LayoutDB, RegionBatch


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
        for _ in range(5):
            query.materialize()
        samples: list[float] = []
        polygon_count = 0
        for _ in range(runs):
            started = perf_counter()
            clipped = query.materialize()
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


def benchmark_raster(image_size: int) -> dict[str, float | int | bool]:
    """测量原生面积覆盖栅格化，并核对与对齐几何面积完全一致。"""
    layer = LayerSpec(1, 0)
    pixel_dbu = 5
    extent = image_size * pixel_dbu
    region = kdb.Region()
    # 构造像素网格对齐的纵横线。合并后既覆盖大量不规则交点，又能用 Region 面积
    # 精确推导应为 255 的像素数，从而同时验证速度和栅格结果，而不逐像素生成答案。
    for coordinate in range(0, extent, 200):
        region.insert(kdb.Box(coordinate, 0, coordinate + 10, extent))
        region.insert(kdb.Box(0, coordinate, extent, coordinate + 10))
    region = region.merged()
    batch = RegionBatch({layer: region}, DbuBox(0, 0, extent, extent), CellRef("BENCH", 0))
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = perf_counter()
    pixels = render_region_batch(batch, layer, 0.001, pixel_size_nm=5)
    elapsed_ms = (perf_counter() - started) * 1000.0
    rss_after = process.memory_info().rss
    expected_white_pixels = region.area() // (pixel_dbu * pixel_dbu)
    actual_white_pixels = int(np.count_nonzero(pixels == 255))
    return {
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "elapsed_ms": elapsed_ms,
        "rss_delta_mb": max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0)),
        "coverage_exact": actual_white_pixels == expected_white_pixels,
    }


def run_benchmarks(runs: int, raster_size: int) -> dict[str, object]:
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
        "raster": benchmark_raster(raster_size),
    }


def strict_failures(result: dict[str, object]) -> list[str]:
    """根据开发方案中的性能门槛返回失败原因。"""
    hierarchy = result["hierarchy"]
    raster = result["raster"]
    failures: list[str] = []
    if hierarchy["roi_polygon_count"] != 25:
        failures.append("百万实例 ROI 应精确得到 25 个 Polygon")
    if hierarchy["query_clip_median_ms"] > 50.0:
        failures.append("百万实例 ROI 查询与裁剪中位数超过 50 ms")
    if hierarchy["rss_delta_mb"] > 64.0:
        failures.append("百万实例 ROI 额外 RSS 超过 64 MB")
    if not raster["coverage_exact"]:
        failures.append("原生灰度栅格结果与对齐 Region 面积不一致")
    if raster["elapsed_ms"] > 5_000.0:
        failures.append("2048x2048 灰度栅格化耗时超过 5 秒")
    if raster["rss_delta_mb"] > 128.0:
        failures.append("2048x2048 灰度栅格化额外 RSS 超过 128 MB")
    return failures


def main(argv: list[str] | None = None) -> int:
    """解析基准参数，输出 JSON，并在严格模式下返回可用于 CI 的退出码。"""
    parser = argparse.ArgumentParser(description="MyOPC Layout/Geometry 性能基准")
    parser.add_argument("--runs", type=int, default=100, help="层级 ROI 重复次数")
    parser.add_argument("--raster-size", type=int, default=2_048, help="正方形栅格基准边长")
    parser.add_argument("--strict", action="store_true", help="未达到验收门槛时返回非零退出码")
    args = parser.parse_args(argv)
    result = run_benchmarks(args.runs, args.raster_size)
    failures = strict_failures(result)
    result["strict_failures"] = failures
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
