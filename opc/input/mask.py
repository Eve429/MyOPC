"""把版图局部数据规范化为 OPC 共享的合并物理覆盖 Region。"""

from __future__ import annotations

from enum import Enum

import klayout.db as kdb

from layout import LayerSpec, RegionBatch
from opc.errors import PhysicalMaskError


class MaskPolarity(str, Enum):
    """定义源多边形在光学上的明确含义，禁止根据版图内容猜测。"""

    CLEAR = "clear"
    OPAQUE = "opaque"


def normalize_mask(batch: RegionBatch, layer: LayerSpec) -> kdb.Region:
    """消除内部切割线并恢复孔洞，返回合并后的物理覆盖 Region。"""
    region = batch.region(layer).dup()
    # OPC 处理的是当前 layer 的物理覆盖集合，Shape 属性不能阻止相接区域合并。
    # 先在副本上删除属性，既不修改查询结果，也避免不同属性把内部 cut-line 留下来。
    region.remove_properties()
    # 最小连通规则把仅角点接触的区域保留为不同 Polygon；随后显式合并会把 GDS
    # 为表达孔洞而引入的零宽桥接线恢复成 hull/hole，两条重合反向桥边不会进入提边。
    region.min_coherence = True
    region = region.merged()
    if not region.has_valid_polygons():
        raise PhysicalMaskError("physical mask contains invalid polygons after merge")
    return region
