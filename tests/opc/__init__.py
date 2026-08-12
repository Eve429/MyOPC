"""OPC 公共层和方法层测试包，并提供跨 ILT 专项复用的轻量测试对象。"""

from pathlib import Path
from types import SimpleNamespace

import klayout.db as kdb
import torch

from lithography import ProcessCondition


class RecordingLithography:
    """以廉价可微映射记录 ILT 送入模型的 shape 和条件。"""

    def __init__(self, canvas: int = 64) -> None:
        """建立 CPU 设备、固定 canvas 和调用记录。"""
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(canvas=canvas)
        self.shapes: list[tuple[int, ...]] = []
        self.names: list[tuple[str, ...]] = []

    def condition(self, name: str) -> ProcessCondition:
        """按默认名称返回互相独立的伪工艺条件。"""
        return ProcessCondition(
            name, "defocus" if name == "defocus_min" else "focus", 1.0)

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """记录完整网格并返回保持梯度的条件偏移图。"""
        self.shapes.append(tuple(mask.shape))
        self.names.append(tuple(condition.name for condition in conditions))
        offsets = {"dose_max": 0.04, "defocus_min": -0.04}
        return {condition.name: torch.clamp(
            mask + offsets.get(condition.name, 0.0), 0.0, 1.0)
                for condition in conditions}


class FlatWaferLithography(RecordingLithography):
    """返回与 mask 保持计算图关系但空间恒定的 wafer。"""

    def forward_many(self, mask: torch.Tensor,
                     conditions: tuple[ProcessCondition, ...]) -> dict[str, torch.Tensor]:
        """用零倍 mask 证明曲率作用对象是 wafer 而非输入 mask。"""
        self.shapes.append(tuple(mask.shape))
        self.names.append(tuple(condition.name for condition in conditions))
        return {condition.name: mask * 0.0 for condition in conditions}


def write_ilt_gds(path: Path) -> None:
    """写入供 ILT 函数入口和仓库外 CLI 共用的确定性矩形版图。"""
    layout = kdb.Layout(); layout.dbu = 0.001
    layer_index = layout.layer(1, 0); top = layout.create_cell("TOP")
    top.shapes(layer_index).insert(kdb.Box(8, 8, 40, 40))
    layout.write(str(path))
