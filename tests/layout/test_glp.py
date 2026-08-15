"""验证严格 GLP 输入、层映射和错误边界。"""

from pathlib import Path

import pytest

from layout import DbuBox, LayerSpec, LayoutDB, LayoutOpenError


def _write_glp(path: Path, body: str) -> Path:
    """写出 UTF-8 GLP 测试文本。"""
    path.write_text(body, encoding="utf-8")
    return path


def test_glp_rect_polygon_and_explicit_layer_map(tmp_path: Path) -> None:
    """GLP 应直接进入 LayoutDB，并保持 DBU、矩形和多边形面积。"""
    source = _write_glp(tmp_path / "sample.glp", """
BEGIN /* 可被忽略的头注释 */
EQUIV 1 1000 MICRON +X,+Y
CNAME TOP
LEVEL METAL
CELL TOP PRIME
RECT N METAL 10 20 30 40
PGON N METAL 100 100 140 100 120 130
ENDMSG
""")
    with LayoutDB.open(source, glp_layer_map={"METAL": (7, 2)}) as database:
        assert database.dbu_um == pytest.approx(0.001)
        assert database.layers() == (LayerSpec(7, 2),)
        batch = database.query([LayerSpec(7, 2)], DbuBox(0, 0, 200, 200)).materialize()
        assert batch.region(LayerSpec(7, 2)).area() == 1800


def test_glp_numeric_layer_mapping_and_top_selection(tmp_path: Path) -> None:
    """名称末尾数字应映射 layer，多个 CELL 仍遵守显式顶层选择。"""
    source = _write_glp(tmp_path / "tops.glp", """
EQUIV 1 1000 MICRON
LEVEL M12
CELL A PRIME
RECT N M12 0 0 10 10
CELL B PRIME
RECT N M12 20 0 10 10
ENDMSG
""")
    with LayoutDB.open(source, top_cell="B") as database:
        assert database.layers() == (LayerSpec(12, 0),)
        assert database.bbox() == DbuBox(20, 0, 30, 10)


def test_glp_ignores_unused_unmapped_level(tmp_path: Path) -> None:
    """声明但未承载图形的非数字辅助层不应阻止读取目标层。"""
    source = _write_glp(tmp_path / "unused.glp", """
EQUIV 1 1000 MICRON +X,+Y
LEVEL M1
LEVEL E1TARGET
CELL TOP PRIME
RECT N M1 0 0 10 10
ENDMSG
""")
    with LayoutDB.open(source) as database:
        assert database.layers() == (LayerSpec(1, 0),)


@pytest.mark.parametrize("body,match", [
    ("EQUIV 1 1000 MICRON\nLEVEL METAL\nCELL TOP PRIME\nRECT N METAL 0 0 1 1\nENDMSG\n", "显式"),
    ("EQUIV 1 1000 MICRON\nLEVEL M1\nCELL TOP PRIME\nPATH N M1 0 0 1 1\nENDMSG\n", "不支持"),
    ("EQUIV 1 1000 MICRON\nLEVEL M1\nCELL TOP PRIME\nRECT P M1 0 0 1 1\nENDMSG\n", "极性"),
])
def test_glp_rejects_ambiguous_or_unsupported_input(
        tmp_path: Path, body: str, match: str) -> None:
    """无法确定层、未实现语句和非 N 图形必须明确失败。"""
    with pytest.raises(LayoutOpenError, match=match):
        LayoutDB.open(_write_glp(tmp_path / "invalid.glp", body))
