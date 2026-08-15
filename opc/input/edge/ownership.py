"""在规则 core 网格上分配唯一 owner，并构造稀疏 halo membership。

输出三组数组，构成「owner 唯一写 + halo 只读」契约的数值基础：

- ``owners[S]``：每个边段唯一拥有的 core 索引（int32）。迭代阶段只有该
  core 允许修改这条边段的位移；用段中点落在哪个 core 判定归属。
- ``core_offsets[C+1]`` 与 ``members[M]``：按 core 聚合的 CSR 邻接表。
  core c 的 halo 上下文需要物化的边段集合是
  ``members[core_offsets[c]:core_offsets[c+1]]``——即所有「段 bbox 扩张
  halo 后与该 core 的 ownership box 接触」的段，无论段本身归属谁。

两者的差集正是并行 OPC 的分工依据：每个 core 评价时把 membership 内
全部段一起栅格化（邻居图形是真实光学上下文，不是零填充），但只写
owners 指向自己的那部分；全部 core 评价完经屏障后才发布下一状态。
"""

from __future__ import annotations

import numpy as np

from opc.input import RectilinearCoreGrid

from .fragmentation import SegmentBatch


def _build_ownership(
        segments: SegmentBatch, grid: RectilinearCoreGrid,
        max_memberships: int | None = None) -> tuple[np.ndarray, ...]:
    """返回 (owners, core_offsets, members)，即全部边段的唯一归属与 halo CSR。

    输入：segments —— 分段结果（只读参考，本函数不修改任何字段）；
    grid —— 规则 core 网格（切线严格递增，halo_dbu 为上下文扩张半径）；
    max_memberships —— 可选的 CSR 总条目硬上限，None 时退化为 int32 容量。
    输出：owners 为 int32[S]，core_offsets 为 int64[C+1]，members 为 int32[M]，
    三者按全局 segment 顺序与 core 行主序对齐，可直接被迭代层复用。
    """
    # 阶段一：由数学边和参数区间重建参考端点。ownership 只需要参考端点，
    # 不使用迭代阶段才需要的外法向；直接从 contours 顶点批量插值（t0/t1 是
    # 边参数），避免走 materialize 额外复制 Sx2 法向量和构造临时对象。全部
    # 数组仍按全局 segment 顺序对齐，因此 owner 与 CSR membership 语义不变。
    edge_ids = segments.edge_ids
    vertices = segments.contours.vertices
    edge_starts = vertices[edge_ids].astype(np.float64)
    vectors = (vertices[segments.edge_next_ids[edge_ids]] - vertices[edge_ids]).astype(np.float64)
    starts = edge_starts + vectors * segments.t0[:, None]
    ends = edge_starts + vectors * segments.t1[:, None]
    # 端点生成后立即释放两张 Sx2 中间表；否则 Python 局部引用会一直保留到 CSR
    # 构造结束，使峰值内存反而高于旧 materialize 路径。
    del edge_starts, vectors
    # 阶段二：段中点定唯一 owner。边段足够短（长度受 FragmentationConfig 上限
    # 约束），中点必落在某个 core 的半开区间内；locate_points 的边界归属约定
    # （内部共享边归右/上，最外沿归最后一行/列）保证恰有一个 owner，不存在
    # 平局或 -1（除非中点落在网格整体之外，由 Problem 层校验拒绝）。
    midpoints = (starts + ends) * 0.5
    owners = grid.locate_points(midpoints)
    # 阶段三：段 bbox 四向扩张 halo 得到 context 窗口。owner 决定「谁能写我」，
    # halo 窗口决定「谁评价时需要看到我」：只要扩张后的 bbox 与某个 core 的
    # ownership box 接触，该 core 就把这条段计入自己的 membership。
    halo = grid.halo_dbu
    left = np.minimum(starts[:, 0], ends[:, 0]) - halo
    right = np.maximum(starts[:, 0], ends[:, 0]) + halo
    bottom = np.minimum(starts[:, 1], ends[:, 1]) - halo
    top = np.maximum(starts[:, 1], ends[:, 1]) + halo
    # 后续只依赖 owner 和四条 bbox 边界，尽早释放三张 Sx2 数组，把准备阶段峰值
    # 限制在 CSR 展开本身，而不是让几何临时量与 membership 临时量重叠常驻。
    del starts, ends, midpoints
    # 阶段四：对 x/y 分别用 searchsorted 定位首尾候选 core，再展开每段通常很小的
    # 二维 core 范围。复杂度取决于实际 halo 邻居数量，不随全局 core 总数乘法
    # 增长，也不会建立 S×C 稠密矩阵；clip 把整体范围外的段收敛到边界 core，
    # 与 locate_points 的最外沿归属约定保持一致。
    ix0 = np.searchsorted(grid.x_cuts[1:], left, side="left")
    ix1 = np.searchsorted(grid.x_cuts[:-1], right, side="right") - 1
    iy0 = np.searchsorted(grid.y_cuts[1:], bottom, side="left")
    iy1 = np.searchsorted(grid.y_cuts[:-1], top, side="right") - 1
    ix0, ix1 = np.clip(ix0, 0, grid.column_count - 1), np.clip(ix1, 0, grid.column_count - 1)
    iy0, iy1 = np.clip(iy0, 0, grid.row_count - 1), np.clip(iy1, 0, grid.row_count - 1)
    # 阶段五：逐段 membership 计数 = x 跨度 × y 跨度，并在任何大数组分配之前
    # 执行容量保护。np.repeat/arange/argsort 产生的数组会同时达到 membership
    # 总量级；若先分配再由 Problem 校验，内存保护已经失去意义。上限取调用方
    # 显式值与 int32 容量的较小者（members 用 int32 存全局段号）。
    x_spans = np.maximum(ix1 - ix0 + 1, 0)
    y_spans = np.maximum(iy1 - iy0 + 1, 0)
    membership_counts = x_spans * y_spans
    membership_total = int(np.sum(membership_counts, dtype=np.int64))
    int32_limit = int(np.iinfo(np.int32).max)
    if grid.core_count > int32_limit:
        raise OverflowError("core 数量超过紧凑 int32 owner 容量")
    if max_memberships is not None and (
            not isinstance(max_memberships, int) or isinstance(max_memberships, bool) or
            max_memberships < 0):
        raise ValueError("max_memberships 必须是非负整数或 None")
    limit = int32_limit if max_memberships is None else min(max_memberships, int32_limit)
    # 必须在 np.repeat/arange/argsort 之前拒绝；这些数组会同时达到 membership
    # 数量级，若先分配再由 Problem 校验，容量保护已经失去意义。
    if membership_total > limit:
        raise MemoryError(
            f"membership 数量 {membership_total:,} 超过构造上限 {limit:,}")
    # 阶段六：CSR 展开。先把每段的计数累积成 membership_offsets（段视角的
    # CSR 前缀和），再用 repeat 把段号铺满 M 个条目；local 是每个条目在所属
    # 段的二维窗口内的序号，经模/除展开成 (row, column)，合并成行主序
    # core 索引。一次稳定 argsort 后即得到「按 core 聚合」的最终顺序。
    membership_offsets = np.empty(segments.segment_count + 1, dtype=np.int64)
    membership_offsets[0] = 0
    np.cumsum(membership_counts, out=membership_offsets[1:])
    members = np.repeat(np.arange(segments.segment_count, dtype=np.int32), membership_counts)
    local = np.arange(membership_total, dtype=np.int64) - np.repeat(
        membership_offsets[:-1], membership_counts)
    columns = ix0[members] + local % x_spans[members]
    rows = iy0[members] + local // x_spans[members]
    core_indices = rows * grid.column_count + columns
    order = np.argsort(core_indices, kind="stable")
    sorted_cores = core_indices[order]
    # 阶段七：bincount 直接给出每个 core 的 membership 数量，累积成 core 视角
    # 的 CSR 前缀和；members 按 order 重排后，切片
    # members[core_offsets[c]:core_offsets[c+1]] 就是 core c 的 halo 段集合。
    core_offsets = np.empty(grid.core_count + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(np.bincount(sorted_cores, minlength=grid.core_count), out=core_offsets[1:])
    return owners, core_offsets, members[order]

# 边段s0、1、2、3，分别由core0、1、2、3改写
# owners = [
#     0,1,2,3
# ]
# 每个核可见的边段分别是0123、013、23、3
# member_segment_indices = [
#     0,1,2,3,
#     0,1,3,
#     2,3,
#     3
# ]
# 表示上面的偏置
# core_offsets = [
#     0,4,7,9,10
# ]
