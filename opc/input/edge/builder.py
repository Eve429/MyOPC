"""组合公共物理 mask、MB-OPC 分段、归属和采样的准备入口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from geometry import extract_contour
from layout import LayerSpec, RegionBatch
from opc.input import MaskPolarity, PhysicalMask, RectilinearCoreGrid, normalize_physical_mask
from opc.input._arrays import as_vector

from .fragmentation import FragmentationConfig, SegmentBatch, fragment_edges
from .ownership import _build_ownership


@dataclass(frozen=True, slots=True)
class MBOPCProblem:
    """多轮 MB-OPC 可重复使用的参考边界、重建配置和归属索引。"""

    physical_mask: PhysicalMask
    config: FragmentationConfig
    segments: SegmentBatch
    grid: RectilinearCoreGrid
    owner_indices: NDArray[np.int32]
    core_offsets: NDArray[np.int64]
    member_segment_indices: NDArray[np.int32]

    def __post_init__(self) -> None:
        """校验问题级 owner 与 context membership CSR 的交叉不变量。"""
        owners = as_vector(self.owner_indices, np.dtype(np.int32), "owner_indices")
        offsets = as_vector(self.core_offsets, np.dtype(np.int64), "core_offsets")
        members = as_vector(self.member_segment_indices, np.dtype(np.int32),
                            "member_segment_indices")
        segment_count, core_count = self.segments.segment_count, self.grid.core_count
        if len(owners) != segment_count:
            raise ValueError("owner_indices must match segment count")
        if len(owners) and (np.any(owners < 0) or np.any(owners >= core_count)):
            raise ValueError("每个 segment 必须具有一个有效 owner")
        if (len(offsets) != core_count + 1 or offsets[0] != 0 or
                offsets[-1] != len(members) or np.any(np.diff(offsets) < 0)):
            raise ValueError("core membership offsets are invalid")
        if len(members) and (np.any(members < 0) or np.any(members >= segment_count)):
            raise ValueError("member_segment_indices 超出 segment 范围")
        object.__setattr__(self, "owner_indices", owners)
        object.__setattr__(self, "core_offsets", offsets)
        object.__setattr__(self, "member_segment_indices", members)

    @property
    def core_count(self) -> int:
        """返回规则网格中的 core 总数。"""
        return self.grid.core_count

    @property
    def persistent_nbytes(self) -> int:
        """返回问题中不重复计数的常驻 NumPy 数组字节数。"""
        contours = self.segments.contours
        arrays = (contours.vertices, contours.ring_offsets, contours.polygon_ring_offsets,
                  self.grid.x_cuts, self.grid.y_cuts, self.owner_indices,
                  self.core_offsets, self.member_segment_indices)
        return self.segments.persistent_nbytes + sum(array.nbytes for array in arrays)

    def segments_for_core(self, core_index: int) -> NDArray[np.int32]:
        """返回指定 core 的 owner 与只读 halo context segment 索引视图。"""
        if core_index < 0 or core_index >= self.core_count:
            raise IndexError("core index is out of range")
        start, end = self.core_offsets[core_index:core_index + 2]
        return self.member_segment_indices[start:end]

    def owner_segments_for_core(self, core_index: int) -> NDArray[np.int32]:
        """返回指定 core 唯一可写的 segment 索引。"""
        members = self.segments_for_core(core_index)
        # owner 必然属于自身 context，故只过滤该 core 的 CSR 局部视图；这避免每个
        # 求解器再次扫描全局 segment，并把唯一写入者语义固定在问题公共接口中。
        return members[self.owner_indices[members] == core_index]


def prepare_problem(batch: RegionBatch, layer: LayerSpec, config: FragmentationConfig,
                    grid: RectilinearCoreGrid | None = None,
                    polarity: MaskPolarity | str = MaskPolarity.CLEAR, *,
                    max_memberships: int | None = None) -> MBOPCProblem:
    """在显式 membership 上限内准备可供多轮边段 OPC 复用的内存问题。"""
    physical = normalize_physical_mask(batch, layer, polarity)
    # PhysicalMask 仅保留所有 OPC 方法共享的原生 Region。边段型输入在这里执行
    # 唯一一次数值轮廓提取，之后由 SegmentBatch 成为该拓扑的唯一持有者。
    segments = fragment_edges(extract_contour(physical.region), config, physical.polarity)
    if grid is None:
        box = batch.query_box
        # 单 core 仍走与整张 reticle 完全相同的规则网格代码，避免第二套显式 core
        # 校验和边界语义。两条切线只分配常数级数组，不影响真实多 core 路径。
        grid = RectilinearCoreGrid(
            np.array([box.left, box.right], dtype=np.int64),
            np.array([box.bottom, box.top], dtype=np.int64))
    owners, offsets, members = _build_ownership(segments, grid, max_memberships)
    return MBOPCProblem(physical, config, segments, grid, owners, offsets, members)
