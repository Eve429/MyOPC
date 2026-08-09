"""把版图局部数据规范化为各 OPC 方法共享的物理 mask 边界。"""

from __future__ import annotations

from dataclasses import dataclass

import klayout.db as kdb

from geometry import ContourBatch, EdgeBatch, extract_contours, extract_edges
from layout import DbuBox, LayerSpec, RegionBatch
from opc.errors import PhysicalMaskError


@dataclass(frozen=True, slots=True)
class PhysicalMask:
    """单层合并物理区域及其紧凑轮廓、数学边表示。"""

    layer: LayerSpec
    region: kdb.Region
    contours: ContourBatch
    edges: EdgeBatch
    query_box: DbuBox


def normalize_physical_mask(batch: RegionBatch, layer: LayerSpec) -> PhysicalMask:
    """消除内部切割线、恢复孔洞，并一次性提取可重复使用的边界。"""
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
    normalized = RegionBatch({layer: region}, batch.query_box, batch.cell)
    # 给出具体的边缘，多边形数量、环数量、是不是hole
    contours = extract_contours(normalized)[layer]
    edges = extract_edges(contours)
    return PhysicalMask(layer, region, contours, edges, batch.query_box)
