"""严格读取 GDS/OASIS/GLP，并统一构造内存 KLayout 数据库。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import klayout.db as kdb

from .errors import LayoutOpenError
from .types import LayerSpec

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_NUMERIC_LAYER = re.compile(r".*?(\d+)$")


def _glp_layer(name: str, explicit: Mapping[str, LayerSpec]) -> LayerSpec:
    """按显式映射或名称末尾数字解析 GLP 符号层。"""
    if name in explicit:
        return explicit[name]
    match = _NUMERIC_LAYER.fullmatch(name)
    if match is None:
        raise LayoutOpenError(f"GLP 层 {name!r} 无法自动映射，请提供显式 layer/datatype")
    return LayerSpec(int(match.group(1)), 0)


def _integer_tokens(values: list[str], line: int) -> list[int]:
    """严格解析 GLP 整数坐标，拒绝静默舍入浮点数。"""
    try:
        return [int(value, 10) for value in values]
    except ValueError as exc:
        raise LayoutOpenError(f"GLP 第 {line} 行包含非整数坐标") from exc


def read_glp(path: Path, layer_map: Mapping[str, LayerSpec] | None = None) -> kdb.Layout:
    """读取受控 GLP 子集并直接生成内存 Layout，不创建临时 GDS。"""
    explicit = dict(layer_map or {})
    try:
        source = _COMMENT.sub("", path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise LayoutOpenError(f"无法读取 GLP {path}: {exc}") from exc
    layout = kdb.Layout()
    declared: set[str] = set()
    resolved: dict[str, LayerSpec] = {}
    current: kdb.Cell | None = None
    saw_equiv = saw_end = False
    for line_number, raw in enumerate(source.splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        keyword = fields[0].upper()
        if saw_end:
            raise LayoutOpenError(f"GLP 第 {line_number} 行位于 ENDMSG 之后")
        if keyword == "BEGIN":
            if len(fields) != 1:
                raise LayoutOpenError(f"GLP 第 {line_number} 行 BEGIN 格式无效")
            continue
        if keyword == "EQUIV":
            if (saw_equiv or len(fields) not in (4, 5) or
                    fields[3].upper() != "MICRON" or
                    (len(fields) == 5 and fields[4].upper() != "+X,+Y")):
                raise LayoutOpenError(f"GLP 第 {line_number} 行 EQUIV 格式无效")
            numerator, denominator = _integer_tokens(fields[1:3], line_number)
            if numerator <= 0 or denominator <= 0:
                raise LayoutOpenError(f"GLP 第 {line_number} 行 EQUIV 必须为正")
            layout.dbu = numerator / denominator
            saw_equiv = True
        elif keyword == "CNAME":
            if len(fields) != 2:
                raise LayoutOpenError(f"GLP 第 {line_number} 行 CNAME 格式无效")
        elif keyword == "LEVEL":
            if len(fields) != 2:
                raise LayoutOpenError(f"GLP 第 {line_number} 行 LEVEL 格式无效")
            name = fields[1]
            if name in declared:
                raise LayoutOpenError(f"GLP 第 {line_number} 行重复声明 LEVEL {name}")
            declared.add(name)
        elif keyword == "CELL":
            if len(fields) not in (2, 3) or (len(fields) == 3 and fields[2].upper() != "PRIME"):
                raise LayoutOpenError(f"GLP 第 {line_number} 行 CELL 格式无效")
            if layout.cell(fields[1]) is not None:
                raise LayoutOpenError(f"GLP 第 {line_number} 行重复定义 CELL {fields[1]}")
            current = layout.create_cell(fields[1])
        elif keyword in {"RECT", "PGON"}:
            if not saw_equiv or current is None:
                raise LayoutOpenError(f"GLP 第 {line_number} 行图形出现在 EQUIV 或 CELL 之前")
            minimum = 7 if keyword == "RECT" else 9
            if len(fields) < minimum or fields[1].upper() != "N":
                raise LayoutOpenError(f"GLP 第 {line_number} 行 {keyword} 格式或极性标记无效")
            layer_name = fields[2]
            if layer_name not in declared:
                raise LayoutOpenError(f"GLP 第 {line_number} 行使用了未声明 LEVEL {layer_name}")
            if layer_name not in resolved:
                spec = _glp_layer(layer_name, explicit)
                if spec in resolved.values():
                    raise LayoutOpenError(
                        f"GLP 层 {layer_name!r} 与其他已使用符号层映射到同一 layer/datatype")
                resolved[layer_name] = spec
            values = _integer_tokens(fields[3:], line_number)
            layer_index = layout.layer(resolved[layer_name].layer, resolved[layer_name].datatype)
            if keyword == "RECT":
                if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
                    raise LayoutOpenError(f"GLP 第 {line_number} 行 RECT 宽高必须为正")
                x, y, width, height = values
                current.shapes(layer_index).insert(kdb.Box(x, y, x + width, y + height))
            else:
                if len(values) < 6 or len(values) % 2:
                    raise LayoutOpenError(f"GLP 第 {line_number} 行 PGON 坐标数量无效")
                polygon = kdb.Polygon([kdb.Point(values[index], values[index + 1])
                                       for index in range(0, len(values), 2)])
                region = kdb.Region(polygon)
                if region.area() <= 0 or not region.has_valid_polygons():
                    raise LayoutOpenError(f"GLP 第 {line_number} 行 PGON 不是有效正面积多边形")
                current.shapes(layer_index).insert(polygon)
        elif keyword == "ENDMSG":
            if len(fields) != 1:
                raise LayoutOpenError(f"GLP 第 {line_number} 行 ENDMSG 格式无效")
            saw_end = True
        else:
            raise LayoutOpenError(f"GLP 第 {line_number} 行不支持语句 {fields[0]!r}")
    if not saw_equiv or not saw_end or current is None:
        raise LayoutOpenError("GLP 必须包含 EQUIV、CELL 和 ENDMSG")
    return layout


def read_layout(path: Path, glp_layer_map: Mapping[str, LayerSpec] | None = None) -> kdb.Layout:
    """按扩展名读取版图；GLP 使用严格子集解析，其余交给 KLayout。"""
    if path.suffix.lower() == ".glp":
        return read_glp(path, glp_layer_map)
    if glp_layer_map:
        raise LayoutOpenError("glp_layer_map 只能用于 .glp 输入")
    layout = kdb.Layout()
    try:
        layout.read(str(path))
    except Exception as exc:
        raise LayoutOpenError(f"failed to read layout {path}: {exc}") from exc
    return layout
