"""求解器消费的最小批量光刻模型契约（首个调用方为 simple MB-OPC）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from .iccad13 import ProcessCondition


@runtime_checkable
class LithographyConfigView(Protocol):
    """暴露求解器所需 canvas 和二值阈值。"""

    canvas: int
    print_threshold: float


@runtime_checkable
class LithographyModel(Protocol):
    """描述边段 OPC 与未来 ILT 消费的最小批量可微光刻接口。"""

    @property
    def device(self) -> torch.device: ...

    @property
    def config(self) -> LithographyConfigView: ...

    def condition(self, name: str) -> ProcessCondition: ...

    def forward_many(
        self, mask: torch.Tensor,
        conditions: Sequence[ProcessCondition],
    ) -> dict[str, torch.Tensor]: ...
