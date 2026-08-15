"""几何单元测试共享的小型原生区域构造工具。"""

from collections.abc import Mapping

import klayout.db as kdb

from layout import DbuBox, LayerSpec, RegionBatch


def region_batch(regions: Mapping[LayerSpec, kdb.Region], box: DbuBox | None = None) -> RegionBatch:
    """不读取流文件，直接构造测试用 RegionBatch。"""
    return RegionBatch(regions, box or DbuBox(-1000, -1000, 1000, 1000))
