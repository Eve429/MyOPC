"""生成包含确定层级、变换、图形类型和 Layer 的测试版图。"""

from pathlib import Path

import klayout.db as kdb


def write_advanced_layout(path: Path) -> Path:
    """写出覆盖 SREF、AREF、旋转、镜像、孔洞以及 Path/Text 的紧凑版图。"""
    layout = kdb.Layout()
    layout.dbu = 0.001
    mask = layout.layer(kdb.LayerInfo(1, 0))
    auxiliary = layout.layer(kdb.LayerInfo(2, 5))
    leaf = layout.create_cell("LEAF")
    leaf.shapes(mask).insert(kdb.Box(0, 0, 100, 50))
    leaf.shapes(mask).insert(kdb.Path([kdb.Point(0, 100), kdb.Point(100, 100)], 20))
    donut = kdb.Polygon([kdb.Point(0, 150), kdb.Point(100, 150), kdb.Point(100, 250), kdb.Point(0, 250)])
    donut.insert_hole([kdb.Point(25, 175), kdb.Point(25, 225), kdb.Point(75, 225), kdb.Point(75, 175)])
    leaf.shapes(mask).insert(donut)
    leaf.shapes(mask).insert(kdb.Text("IGNORED", kdb.Trans(0, 300)))
    top = layout.create_cell("TOP")
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(kdb.Trans.R90, 1000, 0)))
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(kdb.Trans.M0, 0, 1000)))
    top.insert(kdb.CellInstArray(leaf.cell_index(), kdb.Trans(0, 2000), kdb.Vector(300, 0), kdb.Vector(0, 400), 3, 2))
    top.shapes(mask).insert(
        kdb.Polygon([kdb.Point(-100, -50), kdb.Point(100, -50), kdb.Point(100, 50), kdb.Point(-100, 50)])
    )
    top.shapes(auxiliary).insert(kdb.Box(-200, -200, -100, -100))
    options = kdb.SaveLayoutOptions()
    options.set_format_from_filename(str(path))
    layout.write(str(path), options)
    return path
