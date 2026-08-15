"""规范化 ownership Patch，并原子写出只包含修正结果的版图文件。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import klayout.db as kdb

from layout.types import DbuBox, LayerSpec

from .errors import PatchConflictError


@dataclass(frozen=True, slots=True)
class GeometryPatch:
    """由一个 core 矩形负责、位于全局坐标系的单层结果。"""

    patch_id: str
    layer: LayerSpec
    region: kdb.Region
    ownership_box: DbuBox

    def __post_init__(self) -> None:
        """尽早拒绝空标识或非 KLayout Region 数据。"""
        if not self.patch_id.strip():
            raise ValueError("patch_id must be non-empty")
        if not isinstance(self.region, kdb.Region):
            raise TypeError("region must be a KLayout Region")


class PatchSet:
    """按确定顺序收集 core ownership 不重叠的 Patch。"""

    def __init__(self) -> None:
        """创建空 Patch 集合。"""
        self._patches: list[GeometryPatch] = []
        self._ids: set[str] = set()
        self._ownership: dict[LayerSpec, kdb.Region] = {}
        self._regions: dict[LayerSpec, kdb.Region] = {}

    def add(self, patch: GeometryPatch) -> GeometryPatch:
        """按 ownership 裁剪，并拒绝重复 ID 或同层 core 重叠。"""
        if patch.patch_id in self._ids:
            raise PatchConflictError(f"duplicate patch_id: {patch.patch_id}")
        ownership = kdb.Region(patch.ownership_box.to_native())
        layer_ownership = self._ownership.setdefault(patch.layer, kdb.Region())
        if (layer_ownership & ownership).area() > 0:
            raise PatchConflictError(
                f"ownership overlap on {patch.layer.layer}/{patch.layer.datatype}: {patch.patch_id}")
        # 关键正确性边界：输入 Region 可以包含跨越多个 core 的完整图形，但当前 Patch
        # 只能拥有 ownership_box 内的部分。此处先做精确相交，保证相邻 core 对同一图形
        # 的两部分既不丢失，也不会产生正面积重复；仅共享边界不构成 ownership 冲突。
        clipped = patch.region & ownership
        normalized = GeometryPatch(patch.patch_id, patch.layer, clipped, patch.ownership_box)
        self._patches.append(normalized)
        self._ids.add(normalized.patch_id)
        layer_ownership += ownership
        self._regions.setdefault(patch.layer, kdb.Region())
        self._regions[patch.layer] += clipped
        return normalized

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        """返回至少含有一个 Patch 的所有 Layer。"""
        return tuple(sorted(self._regions))

    def region(self, layer: LayerSpec) -> kdb.Region:
        """拼接指定 Layer 中已经完成 ownership 裁剪的原始片段。"""
        region = self._regions.get(layer)
        return kdb.Region() if region is None else region.dup()

    def __len__(self) -> int:
        """返回 Patch 数量，其中包括显式记录的空 ownership 结果。"""
        return len(self._patches)

    def __iter__(self) -> Iterator[GeometryPatch]:
        """按照稳定的 Layer、矩形和 ID 顺序迭代。"""
        return iter(sorted(self._patches,
                           key=lambda item: (item.layer, item.ownership_box, item.patch_id)))


class PatchWriter:
    """序列化位于全局坐标系、已经完成 core ownership 裁剪的 Patch。"""

    _FORMATS: ClassVar[dict[str, str]] = {
        ".gds": "GDS2", ".gds2": "GDS2", ".oas": "OASIS", ".oasis": "OASIS",
    }

    @classmethod
    def write(cls, patches: PatchSet, output_path: str | Path, dbu_um: float,
              top_name: str = "OPC_PATCHES") -> Path:
        """原子写出只包含 Patch 的流文件，并返回规范化路径。"""
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() not in cls._FORMATS:
            raise ValueError("output extension must be .gds/.gds2/.oas/.oasis")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {output.parent}")
        if dbu_um <= 0:
            raise ValueError("dbu_um must be positive")
        if not top_name.strip():
            raise ValueError("top_name must be non-empty")
        layout = kdb.Layout()
        layout.dbu = float(dbu_um)
        top = layout.create_cell(top_name)
        for layer in patches.layers:
            index = layout.layer(kdb.LayerInfo(layer.layer, layer.datatype))
            patches.region(layer).insert_into(layout, top.cell_index(), index)
        return cls._save_layout(layout, output)

    @classmethod
    def write_macro_results(cls, patches: Iterable[GeometryPatch],
                            output_path: str | Path, dbu_um: float, *,
                            cell_mode: Literal["single_cell", "macro_cells"],
                            top_name: str = "OPC_RESULT") -> Path:
        """按单 Cell 全局 merge 或每 macro 子 Cell 两种方式原子写出权威 patch。"""
        collected = list(patches)  # 物化输入，保证重复 ID 检查与两次遍历一致
        if cell_mode not in ("single_cell", "macro_cells"):
            raise ValueError(f"未知 cell_mode：{cell_mode}")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() not in cls._FORMATS:
            raise ValueError("output extension must be .gds/.gds2/.oas/.oasis")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {output.parent}")
        if dbu_um <= 0:
            raise ValueError("dbu_um must be positive")
        if not top_name.strip():
            raise ValueError("top_name must be non-empty")
        layout = kdb.Layout()
        layout.dbu = float(dbu_um)
        top = layout.create_cell(top_name)
        # 两种模式都信任 grid 产生的不重叠 ownership：patch 输入已完成权威覆盖
        # 选择，这里不再建立第二套全局 ownership 冲突检测结构。
        if cell_mode == "single_cell":
            # 汇总同层覆盖后做一次全局 merge/normalize：被 ownership 选择在表示
            # 层切开的连续 polygon 重新连接，重复边与零面积碎片一并清理。这是
            # 消除 macro seam 的唯一正确性路径，也是最终输出的一次全局内存峰值。
            for layer in sorted({patch.layer for patch in collected}):
                index = layout.layer(kdb.LayerInfo(layer.layer, layer.datatype))
                merged = kdb.Region()
                for patch in collected:
                    if patch.layer == layer:
                        merged += patch.region
                merged.min_coherence = True
                merged = merged.merged()
                merged.insert_into(layout, top.cell_index(), index)
        else:
            # macro_cells：每个 patch 裁到自身 ownership 后写入独立子 Cell，顶层
            # 以单位变换引用。物理覆盖与 single_cell 完全相同，但跨 Cell polygon
            # 无法 merge，表示层 seam 保留——该模式只用于调试与降低临时内存，
            # 不得在报告中描述为已经 normalize。
            for patch in collected:
                ownership = kdb.Region(patch.ownership_box.to_native())
                clipped = patch.region & ownership
                child = layout.create_cell(patch.patch_id)
                index = layout.layer(kdb.LayerInfo(patch.layer.layer, patch.layer.datatype))
                clipped.insert_into(layout, child.cell_index(), index)
                top.insert(kdb.CellInstArray(child.cell_index(), kdb.Trans()))
        return cls._save_layout(layout, output)

    @classmethod
    def _save_layout(cls, layout: kdb.Layout, output: Path) -> Path:
        """按目标扩展名把原生 Layout 原子写出到目标路径。"""
        options = kdb.SaveLayoutOptions()
        options.format = cls._FORMATS[output.suffix.lower()]
        # 临时文件必须与目标位于同一目录和 Windows 卷。等待 KLayout 完整关闭输出后
        # 再原子替换，异常时旧结果仍然可用，也不会留下看似有效的半截 GDS/OASIS。
        handle, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=output.suffix,
                                                  dir=output.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            layout.write(str(temporary), options)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output
