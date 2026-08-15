"""版图查询与几何批次共享的小型不可变数据约束。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import klayout.db as kdb


@dataclass(frozen=True, slots=True, order=True)
class DbuBox:
    """使用整数数据库单位（DBU）表示的轴对齐矩形区域。"""

    left: int
    bottom: int
    right: int
    top: int

    def __post_init__(self) -> None:
        """统一整数类型，并在进入底层查询前拒绝空框或反向框。"""
        for name in ("left", "bottom", "right", "top"):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer DBU coordinate")
            object.__setattr__(self, name, int(value))
        if self.left >= self.right or self.bottom >= self.top:
            raise ValueError("DbuBox must have positive width and height")

    @property
    def width(self) -> int:
        """返回矩形宽度，单位为 DBU。"""
        return self.right - self.left

    @property
    def height(self) -> int:
        """返回矩形高度，单位为 DBU。"""
        return self.top - self.bottom

    @property
    def area(self) -> int:
        """返回矩形面积，单位为 DBU²。"""
        return self.width * self.height

    def expanded(self, margin: int) -> DbuBox:
        """返回四边同时向外扩展指定距离的新矩形。"""
        if not isinstance(margin, Integral) or margin < 0:
            raise ValueError("margin must be a non-negative integer DBU value")
        margin = int(margin)
        return DbuBox(self.left - margin, self.bottom - margin,
                      self.right + margin, self.top + margin)

    def intersection(self, other: DbuBox) -> DbuBox | None:
        """返回正面积交集；分离或仅接触时返回 None。"""
        left, bottom = max(self.left, other.left), max(self.bottom, other.bottom)
        right, top = min(self.right, other.right), min(self.top, other.top)
        return None if left >= right or bottom >= top else DbuBox(left, bottom, right, top)

    def to_native(self) -> kdb.Box:
        """仅在进入 KLayout 批处理边界时转换为原生 Box。"""
        return kdb.Box(self.left, self.bottom, self.right, self.top)

    @classmethod
    def from_native(cls, box: kdb.Box) -> DbuBox:
        """根据 KLayout 原生 Box 构造公共 DBU 矩形。"""
        if box.empty():
            raise ValueError("an empty native box cannot become DbuBox")
        return cls(box.left, box.bottom, box.right, box.top)


@dataclass(frozen=True, slots=True, order=True)
class LayerSpec:
    """对外稳定的 GDS/OASIS layer 与 datatype 标识。"""

    layer: int
    datatype: int = 0

    def __post_init__(self) -> None:
        """在参数进入 KLayout 前拒绝非法层号。"""
        if not isinstance(self.layer, Integral) or not isinstance(self.datatype, Integral):
            raise TypeError("layer and datatype must be integers")
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "datatype", int(self.datatype))
        if self.layer < 0 or self.datatype < 0:
            raise ValueError("layer and datatype must be non-negative")

@dataclass(frozen=True, slots=True)
class LayerShapeStats:
    """局部区域迭代器返回图形的可选诊断计数。"""

    polygon_like: int = 0
    text: int = 0
    edge: int = 0
    other: int = 0


@dataclass(frozen=True, slots=True)
class MaterializationStats:
    """各层可选诊断信息以及原生物化耗时。"""

    elapsed_seconds: float
    shapes: Mapping[LayerSpec, LayerShapeStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结映射，避免调用方修改计数而造成诊断信息失真。"""
        object.__setattr__(self, "shapes", MappingProxyType(dict(self.shapes)))


@dataclass(frozen=True, slots=True)
class RegionBatch:
    """统一顶层坐标系下、按 Layer 索引的局部原生 Region 批次。"""

    regions: Mapping[LayerSpec, kdb.Region]
    query_box: DbuBox
    stats: MaterializationStats | None = None

    def __post_init__(self) -> None:
        """只复制很小的 Layer 映射，实际 Region 数据继续留在 C++ 内存中。"""
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        """按确定顺序返回批次包含的 Layer。"""
        return tuple(sorted(self.regions))

    def region(self, layer: LayerSpec) -> kdb.Region:
        """返回指定层的原生 Region，不在 Python 中展开 Polygon。"""
        try:
            return self.regions[layer]
        except KeyError as exc:
            raise KeyError(f"batch does not contain layer {layer.layer}/{layer.datatype}") from exc

    def counts(self) -> Mapping[LayerSpec, int]:
        """利用原生 Region 元数据返回各层 Polygon 数量。"""
        return MappingProxyType({layer: region.count() for layer, region in self.regions.items()})
