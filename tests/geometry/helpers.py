"""Geometry 单元测试共享的小型原生 Region 构造工具。"""

from collections.abc import Mapping

import klayout.db as kdb

from layout import CellRef, DbuBox, LayerSpec, RegionBatch


def region_batch(regions: Mapping[LayerSpec, kdb.Region],
                 box: DbuBox = DbuBox(-1000, -1000, 1000, 1000),
                 cell: CellRef = CellRef("TOP", 0)) -> RegionBatch:
    """不读取流文件，直接构造测试用 RegionBatch。"""
    return RegionBatch(regions, box, cell)
