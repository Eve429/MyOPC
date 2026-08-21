"""TestReticle 测试版图集参数化生成器（规格见同目录 reticle_build_plan.md）。

每个场景构建一次相对坐标 Region，成对写出正负两份掩膜：`_clear.gds`
为原图形（配 config polarity="clear"，图形即透光区）；`_opaque.gds` 为
图形包围盒内的补区（配 polarity="opaque"，图形=不透光材料，原图形处
被挖空即透光）。两份在各自极性下表达同一透光目标。

50nm 定尺寸组（p50_1024/p50_2048 × dense/mid/loose 档）另以固定设计区框
（边长=掩膜尺寸）为负板补区基准，成对写在子目录 p50_<边长>/ 下，写后
读回自检（详见 build plan p50 章）。仅依赖 klayout.db，任何工作目录可直跑：

    python TestReticle/build_reticles.py            # 全部 10 场景 + 6 档 p50
    python TestReticle/build_reticles.py --list     # 只列清单
    python TestReticle/build_reticles.py --only lines_dense_unit
    python TestReticle/build_reticles.py --p50      # 只生成 50nm 定尺寸组
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


# ---------------------------------------------------------------- 50nm 定尺寸组
# 两组定尺寸掩膜（p50_1024 / p50_2048）：50nm 制程、设计区边长精确
# 1.024/2.048µm（= 4nm pixel 下 256/512px，对应单宏与 [2,2] 四宏网格档）、
# 图形内缩 64nm 边框空区。与上面"场景"系的本质差别：负板补区基准是固定
# 设计区框而非图形包围盒——负板恒带 64nm 铬环框、包络恒为整版，正板包络
# = 图形区；两板配 field_size（=掩膜尺寸）运行即得同一网格、环带恒不透光。
# 间距档位是唯一自由度：结构族固定，三档变体各成对写出。

P50_CD = 50      # 50nm 制程特征尺寸，全部结构共用
P50_MARGIN = 64  # 边框空区：64 是常用 pixel（4/8/16nm）的公倍数，包络边恒整像素对齐
P50_SIZES = (1024, 2048)  # 设计区边长 nm；正板图形区 = (64,64)-(size-64,size-64)
P50_VARIANTS: dict[str, dict[str, int]] = {
    "dense": {"pitch": 100, "via_pitch": 100, "t2t": 50},   # 密集档 L/S=1:1
    "mid": {"pitch": 150, "via_pitch": 150, "t2t": 100},    # 中间档 1:2
    "loose": {"pitch": 200, "via_pitch": 250, "t2t": 150},  # 稀疏档 孤立倾向
}


def _p50_vlines(region: kdb.Region, x0: int, y0: int, y1: int,
                count: int, pitch: int) -> None:
    """竖线阵：count 条 CD 宽竖线，首条左边 x0，全部贯穿 [y0,y1]。"""
    for index in range(count):
        _vline(region, x0 + index * pitch, y0, y1, P50_CD)


def _p50_hlines(region: kdb.Region, x0: int, x1: int, y_top: int,
                count: int, pitch: int) -> None:
    """横线阵：count 条 CD 高横线贯穿 [x0,x1]，首条顶边 y_top 向下排。"""
    for index in range(count):
        _hline(region, x0, x1, y_top - P50_CD - index * pitch, P50_CD)


def _p50_via_array(region: kdb.Region, x0: int, y0: int,
                   cols: int, rows: int, pitch: int) -> None:
    """方孔阵：cols×rows 个 CD 方孔，中心距 pitch，左下角孔在 (x0,y0)。"""
    for row in range(rows):
        for col in range(cols):
            x, y = x0 + col * pitch, y0 + row * pitch
            _box(region, x, y, x + P50_CD, y + P50_CD)


def _shapes_p50_sampler_896(variant: dict[str, int]) -> kdb.Region:
    """组 1 图形区 (64,64)-(960,960) 综合采样（组 2 左上象限原样复用）。

    功能带布局：竖线阵贯穿全高（撑住包络左/下/上）；孔阵贴右下、横线阵
    贴右上（撑住右）；45° 三角卡在两阵之间的空带；x≈440..576 走廊自下而
    上放孤立孔、L 拐角、竖直对接线端。条数由带宽按档位 pitch 整除自适应
    （密集档多、稀疏档少），各族互不接触（写出前断言 merge 计数不变）。
    """
    pitch, via_pitch, t2t = (variant["pitch"], variant["via_pitch"],
                             variant["t2t"])
    left = bottom = P50_MARGIN
    right = top = 1024 - P50_MARGIN
    region = kdb.Region()
    # 竖线阵：左带全高，条数按左带宽 448 铺满
    _p50_vlines(region, left, bottom, top, (448 - P50_CD) // pitch + 1, pitch)
    # 孔阵：右下锚定（右缘 960、底 64）；行数以 45° 三角带底 y=440 为界
    via_cols = (384 - P50_CD) // via_pitch + 1
    via_rows = (440 - P50_CD - bottom) // via_pitch + 1
    _p50_via_array(region, right - (via_cols - 1) * via_pitch - P50_CD,
                   bottom, via_cols, via_rows, via_pitch)
    # 横线阵：右上锚定（右缘 960、顶边 960）；最低线底边以三角顶点 590 为界
    _p50_hlines(region, 576, right, top, (910 - 610) // pitch + 1, pitch)
    # 走廊结构（x 440..576）：孤立孔；L 拐角整块单多边形（竖臂 480..530
    # y300..450 + 顶接横臂至 580，共边拼装会破坏"各族互不接触"自检）
    _box(region, 480, 150, 480 + P50_CD, 200)
    region.insert(kdb.Polygon([kdb.Point(480, 300), kdb.Point(480, 450),
                               kdb.Point(580, 450), kdb.Point(580, 400),
                               kdb.Point(530, 400), kdb.Point(530, 300)]))
    # 竖直对接线端：gap = t2t 档位参数，两端线长随档位自然变化
    _vline(region, 480, 500, 680, P50_CD)
    _vline(region, 480, 680 + t2t, 900, P50_CD)
    # 45° 直角三角：卡在孔阵顶与横线阵底之间的空带（三档位均成立）
    region.insert(kdb.Polygon([kdb.Point(600, 440), kdb.Point(750, 440),
                               kdb.Point(600, 590)]))
    return region


def _shapes_p50_2048(variant: dict[str, int]) -> kdb.Region:
    """组 2 图形区 (64,64)-(1984,1984)：左上象限复刻组 1，其余三象限扩展。

    右上：孤立竖线 + 顶部锚定横线阵 + 底部水平对接线端；左下：大孔阵
    （左下锚定，列数上限 8 给角上孤立孔留位）+ 触顶孤立孔；右下：L/T/
    十字/U 拐角家族（错臂/分段拼装互不相叠）+ 45° 与 2:1 斜边三角，U 竖臂
    延至 y=1984 撑住包络顶。跨尺寸对照：同变体左上象限与组 1 逐位相同，
    差异只在宏划分（单宏 vs [2,2] 四宏）。
    """
    pitch, via_pitch, t2t = (variant["pitch"], variant["via_pitch"],
                             variant["t2t"])
    region = kdb.Region()
    region.insert(_shapes_p50_sampler_896(variant))
    # 右上象限（x 1024..1984, y 64..960）：孤立竖线、顶锚横线阵、底部对接
    _vline(region, 1088, 64, 960, P50_CD)
    _p50_hlines(region, 1180, 1984, 960, (910 - 190) // pitch + 1, pitch)
    _hline(region, 1180, 1560, 104, P50_CD)
    _hline(region, 1560 + t2t, 1984, 104, P50_CD)
    # 左下象限（x 64..960, y 1024..1984）：大孔阵 + 右上角触顶孤立孔
    cols = min((896 - P50_CD) // via_pitch + 1, 8)
    _p50_via_array(region, 64, 1088, cols, cols, via_pitch)
    _box(region, 910, 1934, 960, 1984)
    # 右下象限（x 1024..1984, y 1024..1984）：拐角家族，臂长 300，整块
    # 单多边形构造（分臂矩形共边拼接会破坏"各族互不接触"自检）
    region.insert(kdb.Polygon([  # L：竖臂 1088..1138 + 底接横臂至 1388
        kdb.Point(1088, 1088), kdb.Point(1388, 1088), kdb.Point(1388, 1138),
        kdb.Point(1138, 1138), kdb.Point(1138, 1388), kdb.Point(1088, 1388)]))
    region.insert(kdb.Polygon([  # T：横臂 1600..1900 + 中点下接竖臂至 1388
        kdb.Point(1600, 1088), kdb.Point(1900, 1088), kdb.Point(1900, 1138),
        kdb.Point(1775, 1138), kdb.Point(1775, 1388), kdb.Point(1725, 1388),
        kdb.Point(1725, 1138), kdb.Point(1600, 1138)]))
    region.insert(kdb.Polygon([  # 十字：竖臂 1213..1263 贯穿，横臂 1088..1388 居中
        kdb.Point(1213, 1600), kdb.Point(1263, 1600), kdb.Point(1263, 1725),
        kdb.Point(1388, 1725), kdb.Point(1388, 1775), kdb.Point(1263, 1775),
        kdb.Point(1263, 1900), kdb.Point(1213, 1900), kdb.Point(1213, 1775),
        kdb.Point(1088, 1775), kdb.Point(1088, 1725), kdb.Point(1213, 1725)]))
    region.insert(kdb.Polygon([  # U：两竖臂 1600/1850 延至 1984 撑包络顶 + 底横
        kdb.Point(1600, 1600), kdb.Point(1900, 1600), kdb.Point(1900, 1984),
        kdb.Point(1850, 1984), kdb.Point(1850, 1650), kdb.Point(1650, 1650),
        kdb.Point(1650, 1984), kdb.Point(1600, 1984)]))
    # 斜边：45° 三角（十字与 U 之间空档）+ 2:1 坡三角（T 与 U 之间空档）
    region.insert(kdb.Polygon([kdb.Point(1440, 1700), kdb.Point(1590, 1700),
                               kdb.Point(1440, 1850)]))
    region.insert(kdb.Polygon([kdb.Point(1650, 1420), kdb.Point(1930, 1420),
                               kdb.Point(1650, 1560)]))
    return region


# 50nm 组注册表：名字 → (设计区边长, 档位名)。名字即产物文件名前缀与 --only 键。
P50_GROUPS: dict[str, tuple[int, str]] = {
    f"p50_{size}_{variant}": (size, variant)
    for size in P50_SIZES for variant in P50_VARIANTS
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


def _verify_p50_gds(path: Path, size: int, polarity: str) -> kdb.Region:
    """读回单份 p50 GDS 校验身份与包络，返回 merge 后图形供互补终检。"""
    layout = kdb.Layout()
    layout.read(str(path))
    if layout.dbu != DBU_UM:
        raise SystemExit(f"{path.name}: 读回 dbu={layout.dbu} != {DBU_UM}")
    tops = layout.top_cells()
    if len(tops) != 1 or tops[0].name != "TOP":
        raise SystemExit(f"{path.name}: 顶层不是唯一 TOP")
    indices = list(layout.layer_indexes())
    if len(indices) != 1 or layout.get_info(indices[0]) != kdb.LayerInfo(*LAYER):
        raise SystemExit(f"{path.name}: 图形层不是唯一 {LAYER}")
    region = kdb.Region(tops[0].begin_shapes_rec(indices[0]))
    region.merge()
    # 正板包络 = 图形区（内缩 MARGIN）；负板自带铬环框、包络 = 整版
    expected = (kdb.Box(P50_MARGIN, P50_MARGIN, size - P50_MARGIN,
                        size - P50_MARGIN) if polarity == "clear"
                else kdb.Box(0, 0, size, size))
    if region.bbox() != expected:
        raise SystemExit(f"{path.name}: 包络 {region.bbox()} != 期望 {expected}")
    box = region.bbox()
    print(f"{path.name:34s} bbox=({box.left},{box.bottom})-({box.right},{box.top}) "
          f"size={box.width() / 1000:.3f}x{box.height() / 1000:.3f}um "
          f"shapes={region.count()}")
    return region


def _write_p50_pair(size: int, variant: str) -> None:
    """生成一档 p50 掩膜：构造图形、成对写出、读回自检（失败即非零退出）。

    与场景系不同，负板补区相对固定设计区框（而非图形包围盒）——负板恒带
    64nm 铬环框，正负板在各自极性下透光区逐位互补（终检断言并集恰为设计
    区框、交集为空），这是"真互补"的机器证明，不依赖人工核对。
    """
    region = (_shapes_p50_sampler_896(P50_VARIANTS[variant]) if size == 1024
              else _shapes_p50_2048(P50_VARIANTS[variant]))
    # 写前自检：包络恰为图形区（布局必须撑满四边），结构族互不接触
    pattern = kdb.Box(P50_MARGIN, P50_MARGIN, size - P50_MARGIN, size - P50_MARGIN)
    if region.bbox() != pattern:
        raise SystemExit(f"p50_{size}_{variant}: 图形包络 {region.bbox()} "
                         f"!= 图形区 {pattern}")
    if region.merged().count() != region.count():
        raise SystemExit(f"p50_{size}_{variant}: 结构族存在重叠/共边")
    frame = kdb.Region(kdb.Box(0, 0, size, size))
    variants = (("clear", region), ("opaque", frame - region))
    out_dir = _OUTPUT_DIR / f"p50_{size}"
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for polarity, shapes in variants:
        layout = kdb.Layout()
        layout.dbu = DBU_UM
        top = layout.create_cell("TOP")
        shapes.insert_into(layout, top.cell_index(), layout.layer(*LAYER))
        path = out_dir / f"p50_{size}_{variant}_{polarity}.gds"
        layout.write(str(path))
        paths[polarity] = path
    clear = _verify_p50_gds(paths["clear"], size, "clear")
    opaque = _verify_p50_gds(paths["opaque"], size, "opaque")
    if (clear & opaque).area() != 0:
        raise SystemExit(f"p50_{size}_{variant}: 正负板图形相交")
    if ((clear + opaque) ^ frame).area() != 0:
        raise SystemExit(f"p50_{size}_{variant}: 正负板并集 != 设计区框")


def main() -> int:
    """按命令行参数生成版图集并逐份打印统计。"""
    parser = argparse.ArgumentParser(
        description="TestReticle 测试版图集生成器（规格见 reticle_build_plan.md）")
    parser.add_argument("--list", action="store_true",
                        help="只列出场景清单，不生成")
    parser.add_argument("--p50", action="store_true",
                        help="只生成 50nm 定尺寸组（p50_1024/p50_2048）")
    parser.add_argument("--only", choices=sorted({*BUILDERS, *P50_GROUPS}),
                        help="只生成一个场景或一档 p50 掩膜（同样成对产出）")
    arguments = parser.parse_args()
    if arguments.list:
        for name, (_, doc) in BUILDERS.items():
            print(f"{name:22s} {doc}")
        for name, (size, variant) in P50_GROUPS.items():
            print(f"{name:22s} 50nm 定尺寸组 {size}nm {variant} 档")
        return 0
    if arguments.only:
        if arguments.only in BUILDERS:
            _write_pair(arguments.only, BUILDERS[arguments.only][0]())
        else:
            size, variant = P50_GROUPS[arguments.only]
            _write_p50_pair(size, variant)
        return 0
    legacy = {} if arguments.p50 else BUILDERS
    for name, (builder, _) in legacy.items():
        _write_pair(name, builder())
    for size, variant in P50_GROUPS.values():
        _write_p50_pair(size, variant)
    print(f"完成：{len(legacy)} 场景 + {len(P50_GROUPS)} 档 p50 × "
          f"{len(POLARITIES)} 份 → {_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
