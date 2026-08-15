"""定义当前 OPC/ILT 求解器实际消费的最小光刻模型能力。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from .iccad13 import ProcessCondition


@runtime_checkable
class LithographyConfigView(Protocol):
    """暴露求解器进行画布和二值评价校验所需的只读配置。"""

    canvas: int
    print_threshold: float


@runtime_checkable
class LithographyModel(Protocol):
    """描述可供像素 ILT 和边段 OPC 共用的可微批量光刻接口。"""

    device: torch.device
    config: LithographyConfigView

    def condition(self, name: str) -> ProcessCondition:
        """按稳定名称返回一个独立工艺条件。"""
        ...

    def forward_many(
            self, mask: torch.Tensor,
            conditions: Sequence[ProcessCondition]) -> dict[str, torch.Tensor]:
        """一次计算多个独立条件，并保留 mask 的 autograd 计算图。"""
        ...
