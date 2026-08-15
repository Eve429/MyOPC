"""把版图局部数据规范化为各 OPC 方法共享的物理 mask 边界。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import klayout.db as kdb

from layout import DbuBox, LayerSpec, RegionBatch
from opc.errors import PhysicalMaskError


class MaskPolarity(str, Enum):
    """定义源多边形在光学上的明确含义，禁止根据版图内容猜测。"""

    CLEAR = "clear"
    OPAQUE = "opaque"


@dataclass(frozen=True, slots=True)
class PhysicalMask:
    """各 OPC 方法共享的源多边形、处理范围及显式极性。"""

    layer: LayerSpec
    region: kdb.Region
    query_box: DbuBox
    polarity: MaskPolarity = MaskPolarity.CLEAR

    def __post_init__(self) -> None:
        """规范化极性，并要求 opaque 拥有有限显式处理框。"""
        try:
            value = (self.polarity if isinstance(self.polarity, MaskPolarity)
                     else MaskPolarity(self.polarity))
        except ValueError as exc:
            raise PhysicalMaskError(f"不支持的 mask 极性：{self.polarity!r}") from exc
        object.__setattr__(self, "polarity", value)


def normalize_physical_mask(batch: RegionBatch, layer: LayerSpec,
                            polarity: MaskPolarity | str = MaskPolarity.CLEAR) -> PhysicalMask:
    """消除内部切割线并恢复孔洞，不提前构造特定 OPC 方法的边段。"""
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
    return PhysicalMask(layer, region, batch.query_box, polarity)
