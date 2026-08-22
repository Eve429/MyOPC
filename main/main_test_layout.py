"""layout 包全接口的正常运行流程演示：无断言，逐函数注释并打印结果。

定位为迁移期的「首读入口」：按真实调用顺序走完 layout 层几乎全部公共
函数，每一步的注释写清函数作用、输入输出和本步演示的语义；异常路径的
验证不在本文件（归 tests/ pytest 用例）。
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

from layout import DbuBox, LayerSpec, LayoutDB

# 跨界矩形固定使用这组 DBU 坐标：查询框 (400,400,600,600) 只覆盖其左下角
# 四分之一，使「精确裁剪」与「完整图形」两种物化语义可以靠 bbox 直接对比。
_CROSS_BOX = kdb.Box(500, 500, 900, 900)
_QUERY = DbuBox(400, 400, 600, 600)


def _write_sample_gds(path: Path) -> None:
    """生成两级层级 GDS：TOP 含跨界矩形并实例 CHILD 与空 EMPTY Cell。

    输入：path —— 目标 GDS 路径（临时目录内）。
    输出：无返回；副作用是写出一个确定性版图文件，供后续全部演示使用。
    """
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    child = layout.create_cell("CHILD")
    empty = layout.create_cell("EMPTY")
    layer = layout.layer(7, 0)
    child.shapes(layer).insert(kdb.Box(0, 0, 100, 200))
    top.shapes(layer).insert(_CROSS_BOX)
    # CHILD 实例平移到查询框之外，演示层级遍历只物化与 ROI 相交的形状；
    # EMPTY 也必须被 TOP 引用，否则会成为第二个顶层，破坏唯一 top 的默认选择。
    top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans(1000, 0)))
    top.insert(kdb.CellInstArray(empty.cell_index(), kdb.Trans()))
    layout.write(str(path))


def _write_glp(path: Path) -> None:
    """生成覆盖 RECT/PGON 两种图形与两种层映射方式的最小 GLP 文本。

    输入：path —— 目标 GLP 路径（临时目录内）。
    输出：无返回；写出的文本中 M1 需要显式映射，符号名 "7" 靠末尾数字自动映射。
    """
    path.write_text(
        "BEGIN\n"
        "EQUIV 1 1000 MICRON\n"
        "CNAME X\n"
        "LEVEL M1\n"
        "LEVEL 7\n"
        "CELL TOP PRIME\n"
        "RECT N M1 0 0 100 200\n"
        "PGON N 7 300 300 400 300 400 400 300 400\n"
        "ENDMSG\n",
        encoding="utf-8",
    )


def run_demo(gds: Path, glp: Path) -> None:
    """按真实调用顺序执行 layout 全部公共接口并打印每步结果。

    输入：gds / glp —— 预先生成好的版图文件路径。
    输出：无返回；全部结果打印到标准输出，异常路径不做演示。
    """
    # 阶段①打开版图。LayoutDB.open 一次性解析 GDS/OASIS 并按确定规则选择
    # 顶层 Cell：输入是版图路径（自动 resolve）与可选的显式 top 名；输出是
    # LayoutDB 实例。with 语义保证原生 KLayout 数据库在离开作用域时释放，
    # 所有物化必须在此之前完成——这是 layout 层最核心的生命周期不变量。
    with LayoutDB.open(gds) as db:
        # source_path 属性：返回规范化后的源文件绝对路径（Path），供产物
        # 记录与日志追溯输入来源，不参与几何计算。
        print(f"源文件：{db.source_path}")
        # dbu_um 属性：返回每个整数 DBU 对应的微米值（float）。DBU 是版图
        # 最小整数单位，所有坐标换算（nm→DBU）都以它为标尺。
        print(f"DBU：{db.dbu_um} μm")
        # top_cell_name 属性：返回 open 时唯一确定的顶层 Cell 名称（str）。
        # 迁移后只携带名称字符串，不再有独立凭证类型。
        print(f"顶层 Cell：{db.top_cell_name}")
        # layers()：按确定顺序返回版图全部已有 Layer 的 LayerSpec 元组；
        # 只读扫描，不会因查询而创建空层。
        print(f"全部 Layer：{db.layers()}")
        # bbox(cell=None)：返回指定 Cell 的层级包围盒（DbuBox，整数 DBU
        # 坐标）；默认取顶层。子 Cell 传名称即可，空 Cell 返回 None。
        print(f"TOP 包围盒：{db.bbox()}")
        print(f"CHILD 包围盒：{db.bbox('CHILD')}")
        print(f"EMPTY 包围盒：{db.bbox('EMPTY')}")
        # cell_hierarchy()：返回全部 Cell 到直接子 Cell 名称的邻接表
        # （dict[str, tuple[str, ...]]）。共享子 Cell 只存一份、重复 SREF
        # 与 AREF 不按实例展开，叶子 Cell 对应空元组。
        print(f"Cell 层级：{db.cell_hierarchy()}")

        # 阶段②惰性查询与精确裁剪物化。query() 校验 Layer/Cell 存在性后
        # 返回 ShapeQuery（frozen 描述对象，不触碰几何）；materialize() 才
        # 真正发起一次原生层级遍历，把与 ROI 相交的图形精确裁到查询框。
        # 输入：Layer 序列（可混传 tuple）、DbuBox 查询框；输出 RegionBatch。
        batch = db.query([(7, 0), LayerSpec(7, 0)], _QUERY).materialize()
        # RegionBatch.layers：批次包含的 Layer 按确定顺序排序（去重后）。
        print(f"物化 Layer：{batch.layers}")
        # RegionBatch.region(layer)：取指定层的原生 kdb.Region（只读视图，
        # 几何仍在 C++ 内存）；counts() 返回各层 Polygon 数量的映射。
        print(f"裁剪后 Polygon 数：{dict(batch.counts())}")
        # 裁剪语义验证：跨界矩形只保留落入查询框的四分之一，bbox 收缩到
        # (500,500,600,600)，查询框外的 CHILD 实例完全不出现。
        clipped = batch.region(LayerSpec(7, 0))
        print(f"裁剪后 bbox：{clipped.bbox().to_s()}")

        # 阶段③相交物化。materialize_intersecting() 与 materialize() 共用
        # 同一次层级遍历语义，唯一差异是不做 ROI 裁剪：与查询框相交的完整
        # 图形原样进入 Region（并在关闭前展平）。供裁剪前提取真实物理边。
        full = db.query([(7, 0)], _QUERY).materialize_intersecting()
        region = full.region(LayerSpec(7, 0))
        print(f"相交物化 bbox：{region.bbox().to_s()}（超出查询框，图形完整）")

        # 阶段④诊断统计。materialize(diagnostics=True) 额外重复一次遍历，
        # 统计 ROI 内 polygon-like/text/edge/other 各类图形数量与耗时，
        # 供 run 入口的 --diagnostics 输出；不开启时 stats 为 None。
        diag = db.query([(7, 0)], _QUERY).materialize(diagnostics=True)
        stats = diag.stats
        layer_stats = stats.shapes[LayerSpec(7, 0)] if stats else None
        print(f"诊断：耗时 {stats.elapsed_seconds if stats else 0:.4f}s，形状统计 {layer_stats}")

        # 阶段⑤属性保持模式。preserve_properties=True 时 GDS 属性随几何一起
        # 进入 Region：裁剪分支改用 NoPropertyConstraint 求交（普通 & 会丢
        # 属性），相交分支用 merged 保持不同属性的 shape class。
        db.query([(7, 0)], _QUERY, preserve_properties=True).materialize()
        db.query([(7, 0)], _QUERY, preserve_properties=True).materialize_intersecting()
        print("属性模式：两种物化均可执行")

        # 阶段⑥容量扫描。recursive_polygon_shapes(layer, box) 返回原生
        # RecursiveShapeIterator，只接纳 Box/Path/Polygon 类图形，供物化前
        # 预估边段数与峰值内存；迭代器借用数据库，必须在关闭前遍历完。
        iterator = db.recursive_polygon_shapes(LayerSpec(7, 0), db.bbox())
        print(f"容量扫描：Polygon 类图形 {sum(1 for _ in iterator)} 个")

    # 阶段⑦GLP 输入分派。open() 按扩展名分派到 read_glp：glp_layer_map 把
    # GLP 符号层名映射到 GDS layer/datatype（可传 tuple），仅在 GLP 分支
    # 消费；输出与 GDS 完全一致的 LayoutDB，后续接口无差别可用。
    with LayoutDB.open(glp, glp_layer_map={"M1": (5, 0)}) as db:
        print(f"GLP 分支：top={db.top_cell_name}，layers={db.layers()}，dbu={db.dbu_um} μm")


def main() -> int:
    """生成临时版图并执行全部演示阶段；流程完成返回 0。"""
    with tempfile.TemporaryDirectory() as temp:
        gds, glp = Path(temp) / "sample.gds", Path(temp) / "sample.glp"
        _write_sample_gds(gds)
        _write_glp(glp)
        run_demo(gds, glp)
    print("演示流程完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
