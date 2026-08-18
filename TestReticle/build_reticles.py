"""TestReticle 测试版图集参数化生成器（规格见同目录 reticle_build_plan.md）。

每个场景构建一次相对坐标 Region，成对写出正负两份掩膜：`_clear.gds`
为原图形（配 config polarity="clear"，图形即透光区）；`_opaque.gds` 为
图形包围盒内的补区（配 polarity="opaque"，图形=不透光材料，原图形处
被挖空即透光）。两份在各自极性下表达同一透光目标。仅依赖 klayout.db，
任何工作目录可直跑：

    python TestReticle/build_reticles.py            # 生成全部 10 场景 ×2
    python TestReticle/build_reticles.py --list     # 只列清单
    python TestReticle/build_reticles.py --only lines_dense_unit
"""

import argparse
from pathlib import Path

import klayout.db as kdb

DBU_UM = 0.001  # 1 nm/DBU，全部配置 nm 值即 DBU 值
LAYER = (11, 0)  # 唯一目标层，与 gcd 系约定一致
MARGIN = 100  # 全部图形从 (100, 100) nm 起，保持正象限
POLARITIES = ("clear", "opaque")  # 每场景成对产出
_OUTPUT_DIR = Path(__file__).resolve().parent  # 产物与脚本同目录


def _box(region: kdb.Region, x0: int, y0: int, x1: int, y1: int) -> None:
    """向 Region 插入一个整数坐标矩形（nm=DBU）。"""
    region.insert(kdb.Box(x0, y0, x1, y1))


def _hline(region: kdb.Region, x0: int, x1: int, y: int, width: int) -> None:
    """插入一条水平线段矩形（y 为线底边，width 为线宽）。"""
    _box(region, x0, y, x1, y + width)


def _vline(region: kdb.Region, x: int, y0: int, y1: int, width: int) -> None:
    """插入一条垂直线段矩形（x 为线左边，width 为线宽）。"""
    _box(region, x, y0, x + width, y1)


def _tile(source: kdb.Region, cols: int, rows: int,
          pitch_x: int, pitch_y: int) -> kdb.Region:
    """把相对坐标母题按行列平铺成新 Region（间距由 pitch 减母题包围盒给出）。"""
    bbox = source.bbox()
    result = kdb.Region()
    for row in range(rows):
        for col in range(cols):
            result.insert(source.transformed(
                kdb.Trans(col * pitch_x, row * pitch_y)))  # 平移副本
    return result.merged()  # 相邻副本若共边则合并（平铺间距 >0，通常无合并）


# ---------------------------------------------------------------- 场景构建
# 每族函数返回相对坐标 Region（原点 (MARGIN, MARGIN) 附近）；坐标单位 nm。

def _shapes_lines_dense() -> kdb.Region:
    """密排线阵：三档周期 88/132/176nm（L/S=1:1）各 8 条竖线，块间空 300。"""
    region = kdb.Region()
    x = MARGIN
    for pitch in (88, 132, 176):  # 半周期即线宽
        for index in range(8):  # 每块 8 条
            _vline(region, x + index * pitch, MARGIN, MARGIN + 600,
                   pitch // 2)  # 线长 600、线宽 = 半周期
        x += 7 * pitch + pitch // 2 + 300  # 块宽 + 块间空
    return region


def _shapes_dense_iso() -> kdb.Region:
    """密集-孤立对：同一 CD=88 的横线（长 1200）在密排/半密集/孤立三环境。"""
    region = kdb.Region()
    x0, x1, cd = MARGIN, MARGIN + 1200, 88
    y = MARGIN
    for _ in range(7):  # A 密排：7 条 pitch 176
        _hline(region, x0, x1, y, cd)
        y += 176
    y += 600 - 176 + cd  # 块间空 600（从上一条线顶起算）
    _hline(region, x0, x1, y, cd)  # B 半密集：下伴随
    _hline(region, x0, x1, y + cd + 280, cd)  # 主线（边到边 280）
    _hline(region, x0, x1, y + 2 * (cd + 280), cd)  # 上伴随
    y += 2 * (cd + 280) + cd + 600  # 块间空 600
    _hline(region, x0, x1, y, cd)  # C 孤立：单线
    return region


def _shapes_line_end() -> kdb.Region:
    """线端：悬空单线端 ×2（其一侧邻大块）+ 对接线端 gap 120/200 两组。"""
    region = kdb.Region()
    cd = 88
    y = MARGIN
    _hline(region, MARGIN, MARGIN + 600, y, cd)  # 组1：悬空单线端
    x = MARGIN + 1300  # 组间空 400（600 线长 + 400）
    _hline(region, x, x + 600, y, cd)  # 组2：线端
    _box(region, x + 800, y, x + 1200, y + 400)  # 侧邻大块（末端 gap 200）
    x = MARGIN + 3100  # 组3：对接 gap 120
    _hline(region, x, x + 500, y, cd)
    _hline(region, x + 500 + 120, x + 1120, y, cd)
    x = MARGIN + 4900  # 组4：对接 gap 200
    _hline(region, x, x + 500, y, cd)
    _hline(region, x + 500 + 200, x + 1200, y, cd)
    return region


def _shapes_via_array() -> kdb.Region:
    """接触孔阵：88nm 方块 4×4 阵 × 三档边到边间距 88/132/176，纵向叠。"""
    region = kdb.Region()
    size = 88
    y = MARGIN
    for gap in (88, 132, 176):  # 中心距 = 方块 + 间距
        pitch = size + gap
        for row in range(4):
            for col in range(4):
                _box(region, MARGIN + col * pitch, y + row * pitch,
                     MARGIN + col * pitch + size, y + row * pitch + size)
        y += 3 * pitch + size + 300  # 块高 + 块间空 300
    return region


def _shapes_corners() -> kdb.Region:
    """拐角家族：L/T/十字/U，线宽 88、臂长 500，2×2 摆位（列距/行距 ~1400）。"""
    region = kdb.Region()
    cd = 88
    arm = 500
    x0, y0 = MARGIN, MARGIN
    # 左上 L：竖臂贴横臂左端
    _vline(region, x0, y0, y0 + arm, cd)
    _hline(region, x0, x0 + arm, y0, cd)
    # 右上 T：竖臂在横臂中点
    x1 = x0 + 1400
    _vline(region, x1 + arm // 2 - cd // 2, y0, y0 + arm, cd)
    _hline(region, x1, x1 + arm, y0 + arm // 2 - cd // 2, cd)
    # 左下 十字：竖臂与横臂互相居中
    y1 = y0 + 1400
    _vline(region, x0 + arm // 2 - cd // 2, y1, y1 + arm, cd)
    _hline(region, x0, x0 + arm, y1 + arm // 2 - cd // 2, cd)
    # 右下 U：两竖臂 + 底横臂
    _vline(region, x1, y1, y1 + arm, cd)
    _vline(region, x1 + arm - cd, y1, y1 + arm, cd)
    _hline(region, x1, x1 + arm, y1, cd)
    return region


def _shapes_diagonal() -> kdb.Region:
    """斜边混合：45° 三角 + 2:1 坡五边形 + 45° 平行四边形与邻侧竖线。"""
    region = kdb.Region()
    x, y = MARGIN, MARGIN
    region.insert(kdb.Polygon([kdb.Point(x, y), kdb.Point(x + 600, y),
                               kdb.Point(x, y + 600)]))  # 组1：45° 直角三角
    x += 900  # 组间空 300
    region.insert(kdb.Polygon([  # 组2：2:1 坡（dx600:dy300）五边形
        kdb.Point(x, y), kdb.Point(x + 800, y), kdb.Point(x + 800, y + 300),
        kdb.Point(x + 200, y + 600), kdb.Point(x, y + 600)]))
    x += 1200  # 组间空 400
    region.insert(kdb.Polygon([  # 组3：45° 平行四边形（水平 400、高 500、错位 300）
        kdb.Point(x, y), kdb.Point(x + 400, y), kdb.Point(x + 700, y + 500),
        kdb.Point(x + 300, y + 500)]))
    _vline(region, x + 900, y, y + 500, 88)  # 邻侧竖线（边距 200）
    return region


def _shapes_sparse() -> kdb.Region:
    """稀疏版图：主体图形集中左下，包围盒由右下/左上角标记撑开到 ~5.8µm。

    管线按 layer bbox 规划网格，纯空白不贡献 bbox——必须有标记图形才能
    把"空白区"纳入网格域。标记刻意避开右上象限（macro [2,2] 切线在 bbox
    中心 ~2950），使右上 macro 保持 S=0（P1-2 复现素材）。
    """
    region = kdb.Region()
    for index in range(3):  # mini 线阵：3 条 pitch 88、长 400
        _vline(region, MARGIN + index * 88, MARGIN, MARGIN + 400, 44)
    y = MARGIN + 500  # 与线阵纵向空 100
    _vline(region, MARGIN, y, y + 500, 88)  # L 角竖臂
    _hline(region, MARGIN, MARGIN + 500, y, 88)  # L 角横臂
    _box(region, 5600, MARGIN, 5800, MARGIN + 200)  # 右下角标记（右下宏）
    _box(region, MARGIN, 5600, MARGIN + 200, 5800)  # 左上角标记（左上宏）
    return region


def _shapes_boundary() -> kdb.Region:
    """跨界图形：横线垂直贯穿 x=3000 切线、竖线恰压 y=3000 切线（对照），
    平行对照线两条、45° 三角跨 (3000,3000) 角点。"""
    region = kdb.Region()
    cd = 88
    half = cd // 2
    _hline(region, 200, 5800, 3000 - half, cd)  # 横穿主线（中心 y=3000）
    _vline(region, 3000 - half, 200, 5800, cd)  # 竖压切线（已知台阶对照）
    _hline(region, 200, 5800, 2500 - half, cd)  # 平行对照（不跨界）
    _hline(region, 200, 5800, 3500 - half, cd)  # 平行对照（不跨界）
    region.insert(kdb.Polygon([  # 45° 三角跨切线角点
        kdb.Point(2700, 2700), kdb.Point(3300, 2700),
        kdb.Point(2700, 3300)]))
    return region


def _shapes_motif() -> kdb.Region:
    """综合母题：六族两列自然尺寸摆位（约 9.5×11.5µm），供两档 bench 平铺。"""
    region = kdb.Region()
    # 左列（高条族）自上而下连排，族间空 1000；偏移由上一族包围盒程序化计算。
    cursor_y = MARGIN
    left_width = 0
    for builder in (_shapes_dense_iso, _shapes_via_array, _shapes_corners):
        shapes = builder().transformed(kdb.Trans(0, cursor_y - MARGIN))
        region.insert(shapes)
        left_width = max(left_width, shapes.bbox().width())
        cursor_y = shapes.bbox().top + 1000
    # 右列（横条族）：x = 左列最宽 + 2000，同样自上而下连排。
    column_x = MARGIN + left_width + 2000
    cursor_y = MARGIN
    for builder in (_shapes_lines_dense, _shapes_line_end,
                    _shapes_diagonal):
        shapes = builder().transformed(kdb.Trans(column_x - MARGIN,
                                                 cursor_y - MARGIN))
        region.insert(shapes)
        cursor_y = shapes.bbox().top + 1000
    return region.merged()


def _shapes_bench_30um() -> kdb.Region:
    """综合基准：母题 3×2 平铺（约 31×24µm，规模与 gcd_30um 同量级）。"""
    motif = _shapes_motif()
    width = motif.bbox().width() + 1000  # 母题间空 1µm
    height = motif.bbox().height() + 1000
    return _tile(motif, cols=3, rows=2, pitch_x=width, pitch_y=height)


def _shapes_bench_100um() -> kdb.Region:
    """压力级基准：母题 10×7 平铺（约 105×98µm，~9900 core）。"""
    motif = _shapes_motif()
    width = motif.bbox().width() + 1000
    height = motif.bbox().height() + 1000
    return _tile(motif, cols=10, rows=7, pitch_x=width, pitch_y=height)


# 场景注册表：名字 → (构建函数, 一句话说明)。所有函数零参数、返回相对 Region。
BUILDERS: dict[str, tuple] = {
    "lines_dense_unit": (_shapes_lines_dense, "密排线阵三档周期"),
    "dense_iso_unit": (_shapes_dense_iso, "同 CD 密集/半密集/孤立三环境"),
    "line_end_unit": (_shapes_line_end, "线端悬空/侧邻大块/对接 gap 120/200"),
    "via_array_unit": (_shapes_via_array, "88nm 方孔 4×4 阵三档间距"),
    "corners_unit": (_shapes_corners, "L/T/十字/U 拐角家族"),
    "diagonal_unit": (_shapes_diagonal, "45° 与 2:1 斜边 + 曼哈顿混排"),
    "sparse_6um": (_shapes_sparse, "稀疏版图：空 macro（P1-2 素材）"),
    "boundary_6um": (_shapes_boundary, "图形贯穿/恰压 macro 切线"),
    "bench_30um": (_shapes_bench_30um, "母题 3×2：日常综合 smoke"),
    "bench_100um": (_shapes_bench_100um, "母题 10×7：压力级（~9900 core）"),
}


def _write_pair(name: str, region: kdb.Region) -> None:
    """把一个场景的 Region 写出正负两份掩膜并打印统计。

    正板 = 原图形（clear 极性下图形即透光区）；负板 = 包围盒补区
    （frame − 原图形，opaque 极性下图形为不透光材料、挖空处透光）。
    图形贴住包围盒边的方向上补区够不到框边，负板 bbox 相应收缩
    （如密排线阵两端）；对照实验时两份的网格划分可能因此不同。
    """
    frame = kdb.Region(region.bbox())  # 负板框架 = 正板图形包围盒
    variants = (("clear", region), ("opaque", frame - region))  # 负板补区
    for polarity, shapes in variants:
        layout = kdb.Layout()
        layout.dbu = DBU_UM
        top = layout.create_cell("TOP")
        # 插入本极性图形（Region 值语义，同一 Region 可重复消费）
        shapes.insert_into(layout, top.cell_index(), layout.layer(*LAYER))
        path = _OUTPUT_DIR / f"{name}_{polarity}.gds"
        layout.write(str(path))
        box = shapes.bbox()
        # 逐份打印统计（bbox/尺寸/图形数）
        print(f"{path.name:32s} bbox=({box.left},{box.bottom})-"
              f"({box.right},{box.top}) size={box.width() / 1000:.1f}x"
              f"{box.height() / 1000:.1f}um shapes={shapes.count()}")


def main() -> int:
    """按命令行参数生成版图集并逐份打印统计。"""
    parser = argparse.ArgumentParser(
        description="TestReticle 测试版图集生成器（规格见 reticle_build_plan.md）")
    parser.add_argument("--list", action="store_true",
                        help="只列出场景清单，不生成")
    parser.add_argument("--only", choices=sorted(BUILDERS),
                        help="只生成一个场景（同样成对产出）")
    arguments = parser.parse_args()
    if arguments.list:
        for name, (_, doc) in BUILDERS.items():
            print(f"{name:22s} {doc}")
        return 0
    selected = {arguments.only: BUILDERS[arguments.only]} if arguments.only \
        else BUILDERS
    for name, (builder, _) in selected.items():
        _write_pair(name, builder())
    print(f"完成：{len(selected)} 场景 × {len(POLARITIES)} 份 → {_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
