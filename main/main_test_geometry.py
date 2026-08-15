"""geometry 包全接口的正常运行流程演示：无断言，逐函数注释并打印结果。

定位为迁移期的「首读入口」：按真实调用顺序走完 geometry 层几乎全部公共
函数（轮廓提取/重建、结构校验、Patch 所有权与写出、覆盖率栅格化），每步
注释写清函数作用、输入输出和本步演示的语义；异常路径不在本文件（归
tests/ pytest 用例）。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 直接运行 main 下脚本时 Python 只加入 main 目录；按文件位置加入仓库根，
# 不要求安装项目包，也不依赖用户从哪个工作目录启动。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import klayout.db as kdb

from geometry import (
    GeometryPatch,
    PatchSet,
    PatchWriter,
    contours_to_region,
    extract_contour,
    extract_contours,
    iter_region_coverage_tiles,
    render_layout_region,
    render_region_batch,
    validate_contours,
)
from layout import DbuBox, LayerSpec, LayoutDB, RegionBatch


def _write_sample_gds(path: Path) -> None:
    """生成单层单矩形的确定性 GDS，供版图便利入口演示使用。

    输入：path —— 目标 GDS 路径（临时目录内）。
    输出：无返回；写出 1/0 层上 (0,0)-(100,100) 的实心矩形。
    """
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    top.shapes(layout.layer(1, 0)).insert(kdb.Box(0, 0, 100, 100))
    layout.write(str(path))


def run_demo(temp: Path) -> None:
    """按真实调用顺序执行 geometry 全部公共接口并打印每步结果。

    输入：temp —— 临时目录路径，全部中间产物写在其内。
    输出：无返回；全部结果打印到标准输出。
    """
    layer = LayerSpec(7, 1)
    # 阶段①准备测试几何：donut（外框挖孔）加一个独立小矩形，构成
    # 「一个带孔 Polygon + 一个简单 Polygon」的多 Polygon Region，后续
    # 轮廓提取的两级 CSR 结构可以同时展示 hull、hole 与多 Polygon 分组。
    region = (kdb.Region(kdb.Box(0, 0, 100, 100)) - kdb.Region(kdb.Box(20, 20, 80, 80))
              + kdb.Region(kdb.Box(150, 0, 160, 10)))

    # RegionBatch(regions, box)：layout 层物化结果的轻量容器。regions 是
    # LayerSpec→原生 Region 映射，box 是查询框（DbuBox，整数 DBU 坐标）。
    # 此处不经版图文件直接构造，等价于 query().materialize() 的产物。
    batch = RegionBatch({layer: region}, DbuBox(0, 0, 200, 100))

    # extract_contours(batch)：多层入口。只物化一次 Region 边界，把每层
    # Region 转成 ContourBatch 并按 Layer 返回只读映射；输入是 RegionBatch，
    # 输出 Mapping[LayerSpec, ContourBatch]，供后续多轮数值 OPC 重复使用。
    contours = extract_contours(batch)[layer]
    # ContourBatch 是两级 CSR 数组：vertices 是 (N,2) 整数顶点；
    # ring_offsets 记录每个环的顶点区间端点；polygon_ring_offsets 记录每个
    # Polygon 的环区间端点（首环为 hull，其余为 hole）。
    print(f"轮廓：{contours.polygon_count} 个 Polygon，{contours.ring_count} 个环，"
          f"{len(contours.vertices)} 个顶点")
    print(f"环区间端点：{contours.ring_offsets.tolist()}")
    print(f"多边形区间端点：{contours.polygon_ring_offsets.tolist()}")
    # extract_contour(region)：单 Region 快速入口。跳过 Layer 映射直接转换，
    # 与多层入口产生完全相同的数组；适合只有一层的调用方省一层间接。
    direct = extract_contour(region)
    print(f"单入口顶点数：{len(direct.vertices)}（与多层入口一致）")
    # contours_to_region(contours)：CSR 数组的逆变换。在保留孔洞拓扑的前提
    # 下重建原生 Region；输出可与原 Region 做异或面积对比验证无损往返。
    rebuilt = contours_to_region(contours)
    print(f"重建面积：{rebuilt.area()}，原面积：{region.area()}（异或为 0 即无损）")

    # 阶段②结构校验。validate_contours(contours, layer)：在不修改输入的
    # 前提下检测退化环（零长边、零面积环），输出 ValidationReport；其
    # is_valid 属性为 True 表示无需进入昂贵的原生修复流程。
    report = validate_contours(contours, layer)
    print(f"校验：is_valid={report.is_valid}，问题数={len(report.issues)}")

    # 阶段③Patch 所有权。GeometryPatch(patch_id, layer, region, ownership_box)
    # 是「单 core 切片」的最小几何交付单元；PatchSet.add() 先拒绝重复 ID 与
    # 同层正面积 ownership 重叠，再把输入 Region 精确裁到 ownership_box——
    # 跨 core 图形由相邻 core 各自唯一拥有对应部分，既不丢失也不重复。
    crossing = kdb.Region(kdb.Box(25, 20, 75, 80))
    patches = PatchSet()
    left = patches.add(GeometryPatch("core-left", layer, crossing, DbuBox(0, 0, 50, 100)))
    right = patches.add(GeometryPatch("core-right", layer, crossing, DbuBox(50, 0, 100, 100)))
    print(f"Patch：共 {len(patches)} 个，层 {patches.layers}")
    print(f"左半面积：{left.region.area()}，右半面积：{right.region.area()}"
          f"（和 = {left.region.area() + right.region.area()} = 原面积 {crossing.area()}）")
    # PatchSet.region(layer)：拼接该层全部已完成 ownership 裁剪的片段，输出
    # 独立副本的合并 Region，供最终全图输出回用。
    reclaimed = patches.region(layer)
    print(f"全局回收面积：{reclaimed.area()}（异或面积 "
          f"{(reclaimed ^ crossing).area()}）")

    # 阶段④Patch 写出与回读。PatchWriter.write(patches, path, dbu_um)：把
    # PatchSet 原子写出为只含修正结果的 GDS/OASIS 流文件（同目录临时文件 +
    # os.replace，异常不残留半截文件）；输入 dbu_um 是数据库单位微米值，
    # 输出为规范化后的路径。
    output = PatchWriter.write(patches, temp / "patch.gds", 0.001)
    print(f"Patch 已写出：{output}")
    # 用 layout 层回读验证往返一致：open + query().materialize() 走一遍完整
    # 的读入、ROI 物化流程，回读面积应与 ownership 范围内的原始几何相同。
    with LayoutDB.open(output) as db:
        box = db.bbox()
        reloaded = (db.query([layer], box).materialize().region(layer)
                    if box is not None else kdb.Region())
        print(f"回读：{reloaded.count()} 个 Polygon，面积 {reloaded.area()}"
              f"（原 {reclaimed.count()} 个，面积 {reclaimed.area()}）")

    # 阶段⑤覆盖率栅格化。iter_region_coverage_tiles(region, box, pixel_dbu,
    # shape)：geometry 栅格化的底层共享原语。把 Region 裁到 box 并在原生端
    # 合并，按 ≤1M 像素分块产出 (y0, x0, areas)——areas 是 float64 面积覆盖率
    # (0~1)，左下原点（第 0 行 = 最低 Y）。显示层和光刻模型层都消费它。
    partial = kdb.Region(kdb.Box(0, 0, 15, 10))
    tiles = list(iter_region_coverage_tiles(partial, DbuBox(0, 0, 20, 20), 10, (2, 2)))
    for y0, x0, areas in tiles:
        print(f"覆盖率块 (y0={y0}, x0={x0})：{areas.tolist()}")
    # render_region_batch(batch, layer, dbu_um, pixel_size_nm)：显示层入口。
    # 复用底层覆盖率并量化为 uint8 灰度（0-255），可选原子保存 PNG（保存时
    # 才翻为图片方向）；输出数组保持左下原点的模型方向。
    pixels = render_region_batch(RegionBatch({layer: partial}, DbuBox(0, 0, 20, 20)),
                                 layer, 0.001, pixel_size_nm=10,
                                 output_path=temp / "roi.png")
    print(f"灰度像素（左下原点）：{pixels.tolist()}，PNG：{temp / 'roi.png'}")

    # 阶段⑥版图便利入口。render_layout_region(database, box, layer,
    # pixel_size_nm)：从已打开的 LayoutDB 提取单层 ROI 并直接栅格化，等价于
    # query().materialize() + render_region_batch() 的组合；输出 uint8 数组。
    gds = temp / "sample.gds"
    _write_sample_gds(gds)
    mask_layer = LayerSpec(1, 0)
    with LayoutDB.open(gds) as database:
        layout_pixels = render_layout_region(database, DbuBox(0, 0, 100, 100),
                                             mask_layer, pixel_size_nm=10)
        print(f"版图栅格：形状 {layout_pixels.shape}，"
              f"取值 {sorted(set(layout_pixels.ravel().tolist()))}")


def main() -> int:
    """生成临时数据并执行全部演示阶段；流程完成返回 0。"""
    with tempfile.TemporaryDirectory() as temp:
        run_demo(Path(temp))
    print("演示流程完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
