"""仅补丁式 GDS/OASIS 原子写出；源版图数据库始终保持只读。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import ClassVar

import klayout.db as kdb

from geometry.patch import PatchSet


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
        options = kdb.SaveLayoutOptions()
        options.format = cls._FORMATS[output.suffix.lower()]
        # 关键可靠性步骤：临时文件与目标文件位于同一目录、同一 Windows 卷，
        # 等待版图内核完整写出并关闭文件后，再用 os.replace 原子替换，避免异常中断时
        # 留下看似存在但内容不完整的 GDS/OASIS 结果。
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
