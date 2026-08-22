"""定义单个 macro 的持久化参考问题、ownership 构造与 NPZ 存取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import klayout.db as kdb
import numpy as np
from numpy.typing import NDArray

from common.arrays import as_vector
from common.io import atomic_write_npz
from geometry import ContourBatch, extract_contour
from layout import DbuBox, LayerSpec, RegionBatch
from opc.input import MacroSpec, MaskPolarity
from opc.input.mask import normalize_mask

from .fragmentation import FragmentationConfig, SegmentBatch, fragment_edges

Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]

# NPZ 格式版本号；不兼容的结构变更必须递增并在 load 中显式拒绝旧版本。
# v2 曾含 dark_box（透光率置零方案）；v3 移除该键（2026-08-22 改负板
# prepare 前补铬几何方案），v1/v2 一并显式拒绝。
_FORMAT_VERSION = 3


@dataclass(frozen=True, slots=True)
class MacroProblem:
    """一个 macro 可独立保存、加载和重复迭代的全部参考输入。"""

    macro: MacroSpec
    # 当前任务的 macro/core 网格、context、pixel 和 canvas 契约。

    layer: LayerSpec
    # 当前 problem 处理和最终输出的唯一 GDS layer/datatype。

    polarity: MaskPolarity
    # 源 polygon 的 mask 极性；栅格输出仍统一使用 1=透光、0=不透光。

    fragmentation: FragmentationConfig
    # 参考边段长度、最大允许位移和 miter 限制；阶段二不得重新计算。

    segments: SegmentBatch
    # 完整候选 polygon 的轮廓拓扑、数学边和控制边段，是参考几何唯一数组真源。

    owner_indices: Int32Array
    # 长度 S。owner_indices[s] 是 segment s 唯一可写的 macro 局部 core 编号；
    # -1 表示该 segment 只因 context 被当前 macro 看见，当前 macro 不得修改它。

    core_offsets: Int64Array
    # 长度 C+1 的 CSR 偏移。core c 的可见 segment 位于
    # member_segment_indices[core_offsets[c]:core_offsets[c+1]]。
    # 使用 int64 是因为 membership 总量 M 可能超过 int32 累计范围。

    member_segment_indices: Int32Array
    # 长度 M。按 core 连续存储 context 内所有 segment 的局部 segment 编号；
    # 同一 segment 可以因 context 同时出现在多个 core 的 membership 中，
    # 但 owner_indices 仍只允许一个 core 写入。segment 局部编号使用 int32 节省内存。

    def __post_init__(self) -> None:
        """规范化索引数组并校验 owner/CSR 交叉不变量。"""
        try:
            polarity = self.polarity if isinstance(self.polarity, MaskPolarity) else MaskPolarity(self.polarity)
        except ValueError as exc:
            raise ValueError(f"不支持的 mask 极性：{self.polarity!r}") from exc
        owners = as_vector(self.owner_indices, np.dtype(np.int32), "owner_indices")
        offsets = as_vector(self.core_offsets, np.dtype(np.int64), "core_offsets")
        members = as_vector(self.member_segment_indices, np.dtype(np.int32), "member_segment_indices")
        segment_count, core_count = self.segments.segment_count, self.macro.core_count
        if len(owners) != segment_count:
            raise ValueError("owner_indices must match segment count")
        if len(owners) and (np.any(owners < -1) or np.any(owners >= core_count)):
            raise ValueError("owner core indices are out of range")
        if (
            len(offsets) != core_count + 1
            or offsets[0] != 0
            or offsets[-1] != len(members)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("core membership offsets are invalid")
        if len(members) and (np.any(members < 0) or np.any(members >= segment_count)):
            raise ValueError("member_segment_indices 超出 segment 范围")
        # own ⊆ membership：每个 owner>=0 的 segment 必然出现在其 owner 的 CSR
        # 区间内；逐条目标记命中后与「owner>=0」全集比较，一次向量化完成。
        # 空 membership 时 seen 全 False＝全 -1 的合法纯 context 状态，与任何
        # owner>=0 状态不等——截断 NPZ/手工构造的损坏对象仍被拒绝。
        entry_cores = np.repeat(np.arange(core_count, dtype=np.int64), np.diff(offsets))
        matched = owners[members] == entry_cores
        seen = np.zeros(segment_count, dtype=np.bool_)
        seen[members[matched]] = True
        if not np.array_equal(seen, owners >= 0):
            raise ValueError("owned segment missing from its owner membership")
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "owner_indices", owners)
        object.__setattr__(self, "core_offsets", offsets)
        object.__setattr__(self, "member_segment_indices", members)

    def segments_for_core(self, core_index: int) -> Int32Array:
        """返回一个 core 可读取的 owned + context segment 索引视图。"""
        if core_index < 0 or core_index >= self.macro.core_count:
            raise IndexError("core index is out of range")
        start, end = self.core_offsets[core_index : core_index + 2]
        return self.member_segment_indices[start:end]

    def owner_segments_for_core(self, core_index: int) -> Int32Array:
        """从该 core 的 membership 中筛出唯一允许当前 core 更新的 segment。"""
        members = self.segments_for_core(core_index)
        # owner>=0 的 segment 必然属于自身 owner 的 membership（构造时已验证），
        # 因此在 CSR 局部视图内过滤即可，不需要扫描全局 segment 列表。
        return members[self.owner_indices[members] == core_index]

    def _arrays(self) -> dict[str, np.ndarray]:
        """按当前格式版本的键名打包全部待持久化数组。"""
        macro, segments = self.macro, self.segments
        contours = segments.contours
        return {
            "format_version": np.array([_FORMAT_VERSION], dtype=np.int32),
            "macro_id": np.array([macro.macro_id]),
            "macro_ownership_box": np.array(
                [
                    macro.ownership_box.left,
                    macro.ownership_box.bottom,
                    macro.ownership_box.right,
                    macro.ownership_box.top,
                ],
                dtype=np.int64,
            ),
            "macro_x_cuts": macro.x_cuts.astype(np.int64, copy=False),
            "macro_y_cuts": macro.y_cuts.astype(np.int64, copy=False),
            "context_dbu": np.array([macro.context_dbu], dtype=np.int64),
            "pixel_dbu": np.array([macro.pixel_dbu], dtype=np.int64),
            "canvas_pixels": np.array([macro.canvas_pixels], dtype=np.int64),
            "layer": np.array([self.layer.layer], dtype=np.int32),
            "datatype": np.array([self.layer.datatype], dtype=np.int32),
            "polarity": np.array([self.polarity.value]),
            "corner_length_dbu": np.array([self.fragmentation.corner_length_dbu], dtype=np.float64),
            "max_segment_length_dbu": np.array([self.fragmentation.max_segment_length_dbu], dtype=np.float64),
            "max_displacement_dbu": np.array([self.fragmentation.max_displacement_dbu], dtype=np.float64),
            "miter_limit": np.array([self.fragmentation.miter_limit], dtype=np.float64),
            "contour_vertices": contours.vertices.astype(np.int64, copy=False),
            "contour_ring_offsets": contours.ring_offsets.astype(np.int64, copy=False),
            "contour_polygon_ring_offsets": contours.polygon_ring_offsets.astype(np.int64, copy=False),
            "edge_next_ids": segments.edge_next_ids.astype(np.int32, copy=False),
            "edge_polygon_ids": segments.edge_polygon_ids.astype(np.int32, copy=False),
            "edge_normals": segments.edge_normals.astype(np.float64, copy=False),
            "ring_segment_offsets": segments.ring_segment_offsets.astype(np.int64, copy=False),
            "segment_edge_ids": segments.edge_ids.astype(np.int32, copy=False),
            "segment_t0": segments.t0.astype(np.float64, copy=False),
            "segment_t1": segments.t1.astype(np.float64, copy=False),
            "owner_indices": self.owner_indices.astype(np.int32, copy=False),
            "core_offsets": self.core_offsets.astype(np.int64, copy=False),
            "member_segment_indices": self.member_segment_indices.astype(np.int32, copy=False),
        }

    def save(self, path: str | Path) -> Path:
        """把 problem 以不压缩 NPZ 原子保存，不写重复几何数组。"""
        output = Path(path).expanduser().resolve()
        if not output.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {output.parent}")
        # 原子写经 common.io（npz 载荷跨模块唯一实现；np.savez 写入文件对象
        # 避免按文件名追加 .npz 后缀，失败时旧完整文件保持可用）。
        atomic_write_npz(output, **self._arrays())
        return output

    @classmethod
    def load(cls, path: str | Path) -> MacroProblem:
        """使用 allow_pickle=False 读取并通过现有结构不变量校验 problem。"""
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"][0]) != _FORMAT_VERSION:
                raise ValueError("unsupported problem format version")
            macro = MacroSpec(
                str(data["macro_id"][0]),
                DbuBox(*[int(v) for v in data["macro_ownership_box"]]),
                data["macro_x_cuts"],
                data["macro_y_cuts"],
                int(data["context_dbu"][0]),
                int(data["pixel_dbu"][0]),
                int(data["canvas_pixels"][0]),
            )
            layer = LayerSpec(int(data["layer"][0]), int(data["datatype"][0]))
            fragmentation = FragmentationConfig(
                corner_length_dbu=float(data["corner_length_dbu"][0]),
                max_segment_length_dbu=float(data["max_segment_length_dbu"][0]),
                max_displacement_dbu=float(data["max_displacement_dbu"][0]),
                miter_limit=float(data["miter_limit"][0]),
            )
            contours = ContourBatch(
                data["contour_vertices"], data["contour_ring_offsets"], data["contour_polygon_ring_offsets"]
            )
            segments = SegmentBatch(
                contours=contours,
                edge_ids=data["segment_edge_ids"],
                edge_next_ids=data["edge_next_ids"],
                edge_polygon_ids=data["edge_polygon_ids"],
                edge_normals=data["edge_normals"],
                ring_segment_offsets=data["ring_segment_offsets"],
                t0=data["segment_t0"],
                t1=data["segment_t1"],
            )
            # 构造即校验：MacroProblem.__post_init__ 会复查 owner 范围、CSR
            # 边界与 own⊆membership；损坏或被篡改的 NPZ 在这里直接失败。
            return cls(
                macro,
                layer,
                MaskPolarity(str(data["polarity"][0])),
                fragmentation,
                segments,
                owner_indices=data["owner_indices"],
                core_offsets=data["core_offsets"],
                member_segment_indices=data["member_segment_indices"],
            )


def _split_segments_at_ownership_cuts(
    segments: SegmentBatch,
    x_cuts: NDArray[np.int64],
    y_cuts: NDArray[np.int64],
) -> SegmentBatch:
    """在 macro/core ownership 切线交点处分裂控制段，保证一段不跨两个 owner。"""
    count = segments.segment_count
    # 空 macro（查询框不接触任何图形）没有可分裂的段，原样返回；后续的
    # repeat/bincount 展开都假设至少存在一个段。
    if not count:
        return segments
    edge_ids = segments.edge_ids
    vertices = segments.contours.vertices
    starts_v = vertices[edge_ids].astype(np.int64)
    ends_v = vertices[segments.edge_next_ids[edge_ids]].astype(np.int64)
    x0, y0 = starts_v[:, 0], starts_v[:, 1]
    x1, y1 = ends_v[:, 0], ends_v[:, 1]

    def _crossings(
        cuts: NDArray[np.int64], origin: NDArray[np.int64], target: NDArray[np.int64]
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (段号, 穿越参数 t) 平铺数组；仅统计严格穿过切线的边。"""
        lower = np.minimum(origin, target)
        upper = np.maximum(origin, target)
        # searchsorted 双侧开区间：跳过 ≤min 与 ≥max 的切线，落在边端点上的
        # 切线不产生分裂（端点本身不是内部穿越）。
        lo = np.searchsorted(cuts, lower, side="right")
        hi = np.searchsorted(cuts, upper, side="left")
        per_segment = np.maximum(hi - lo, 0)
        total = int(per_segment.sum())
        seg = np.repeat(np.arange(count, dtype=np.int64), per_segment)
        local = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(per_segment)[:-1])), per_segment
        )
        cut_values = cuts[lo[seg] + local].astype(np.float64)
        delta = (target - origin).astype(np.float64)
        # t 由原始整数端点与全局整数切线计算：共享 macro 边界两侧的 problem
        # 使用同一公式得到同一浮点 t，禁止把斜边裁成整数短边后重新均分
        # （那会在边界两侧产生 33/34 DBU 分歧）。
        return seg, (cut_values - origin[seg].astype(np.float64)) / delta[seg]

    x_seg, x_t = _crossings(x_cuts, x0, x1)
    y_seg, y_t = _crossings(y_cuts, y0, y1)
    seg_all = np.concatenate((x_seg, y_seg))
    t_all = np.concatenate((x_t, y_t))
    # 与现有片段边界重合的穿越点不产生新段；微小容差吸收端点插值与切线
    # 换算之间的浮点噪声，避免生成 1e-16 量度的退化碎段。
    inside = (t_all > segments.t0[seg_all] + 1e-12) & (t_all < segments.t1[seg_all] - 1e-12)
    seg_all, t_all = seg_all[inside], t_all[inside]
    # 段内按参数排序，保证分裂点单调递增且新段保持全局稳定顺序。
    order = np.lexsort((t_all, seg_all))
    seg_all, t_all = seg_all[order], t_all[order]
    # 斜边精确穿过 x/y 切线交点时，两条切线会在同一参数处各产生一个穿越点；
    # 重复分裂点必须去重，否则相邻碎段零长度并被 SegmentBatch 拒绝。
    if len(t_all) > 1:
        duplicate = (seg_all[1:] == seg_all[:-1]) & np.isclose(t_all[1:], t_all[:-1], atol=1e-12, rtol=0.0)
        keep = np.concatenate(([True], ~duplicate))
        seg_all, t_all = seg_all[keep], t_all[keep]
    counts = np.bincount(seg_all, minlength=count)
    cross_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # 每段输出 counts+2 个边界点、counts+1 个新段；全零穿越时新批次与
    # 原批次逐值相等，调用方无需为「无切线几何」维护第二套路径。
    boundary_counts = counts + 2
    boundary_offsets = np.concatenate(([0], np.cumsum(boundary_counts)[:-1]))
    boundary_seg = np.repeat(np.arange(count, dtype=np.int64), boundary_counts)
    boundary_local = np.arange(int(boundary_counts.sum()), dtype=np.int64) - np.repeat(
        boundary_offsets, boundary_counts
    )
    last = boundary_local == (counts + 1)[boundary_seg]
    first = boundary_local == 0
    if len(t_all):
        # np.where 会对全部分支求值：last/first 行的穿越索引可能越界，先夹回
        # 有效范围；被夹位置的值不会被选中，只保证花式索引本身不抛错。
        mid_index = np.minimum(cross_starts[boundary_seg] + np.maximum(boundary_local - 1, 0), len(t_all) - 1)
        mid_values = t_all[mid_index]
    else:
        mid_values = np.zeros(len(boundary_seg), dtype=np.float64)
    values = np.where(first, segments.t0[boundary_seg], np.where(last, segments.t1[boundary_seg], mid_values))
    piece_offsets = np.concatenate(([0], np.cumsum(counts + 1)))
    # 新段沿用原段的数学边号：分裂点只是控制段边界，同一条真实数学边上的
    # 全部碎片共享同一个 edge_id，重建时才能按同边同位移规则合并 junction。
    return SegmentBatch(
        contours=segments.contours,
        edge_ids=segments.edge_ids[boundary_seg[~last]],
        edge_next_ids=segments.edge_next_ids,
        edge_polygon_ids=segments.edge_polygon_ids,
        edge_normals=segments.edge_normals,
        ring_segment_offsets=piece_offsets[segments.ring_segment_offsets],
        t0=values[~last],
        t1=values[~first],
    )


def _build_macro_ownership(
    segments: SegmentBatch,
    macro: MacroSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成每段唯一 owner 和每个 core 的 context membership CSR。"""
    edge_ids = segments.edge_ids
    vertices = segments.contours.vertices
    # ownership 只需要参考端点：从数学边与参数区间直接插值，不构造完整
    # SegmentGeometry（省掉 S×2 法向复制与临时对象）。
    edge_starts = vertices[edge_ids].astype(np.float64)
    vectors = (vertices[segments.edge_next_ids[edge_ids]] - vertices[edge_ids]).astype(np.float64)
    starts = edge_starts + vectors * segments.t0[:, None]
    ends = edge_starts + vectors * segments.t1[:, None]
    del edge_starts, vectors
    # 段中点定唯一 owner：段长受 FragmentationConfig 上限约束，中点必落在
    # 恰一个 core 半开区间内；macro ownership 外的点保持 -1 = 只读 context。
    midpoints = (starts + ends) * 0.5
    owners = macro.locate_owned_points(midpoints)
    # context 接触窗口：段 bbox 四向扩 context_dbu 后与 core ownership 框
    # 相交的 core 集合。context 是均匀扩张，因此可由切线 searchsorted 精确
    # 求出候选列/行范围，无需逐 core 布尔过滤，也不建立 S×C 稠密矩阵。
    halo = macro.context_dbu
    left = np.minimum(starts[:, 0], ends[:, 0]) - halo
    right = np.maximum(starts[:, 0], ends[:, 0]) + halo
    bottom = np.minimum(starts[:, 1], ends[:, 1]) - halo
    top = np.maximum(starts[:, 1], ends[:, 1]) + halo
    del starts, ends, midpoints
    x_cuts, y_cuts = macro.x_cuts, macro.y_cuts
    columns_total, rows_total = macro.column_count, macro.row_count
    # 候选列范围 [ix0, ix1]：最小列 = 首条右切线不小于 left 的列，最大列 =
    # 末条左切线不大于 right 的列。越出 macro 的段（邻居图形整体进入查询）
    # 得到空范围而不是被裁剪到边界 core——否则远端不接触 context 的段会被
    # 误计入 membership；其 owner 已是 -1，不需要任何 membership。
    ix0 = np.searchsorted(x_cuts[1:], left, side="left")
    ix1 = np.minimum(np.searchsorted(x_cuts[:-1], right, side="right") - 1, columns_total - 1)
    iy0 = np.searchsorted(y_cuts[1:], bottom, side="left")
    iy1 = np.minimum(np.searchsorted(y_cuts[:-1], top, side="right") - 1, rows_total - 1)
    x_spans = np.maximum(ix1 - ix0 + 1, 0)
    y_spans = np.maximum(iy1 - iy0 + 1, 0)
    membership_counts = x_spans * y_spans
    membership_total = int(np.sum(membership_counts, dtype=np.int64))
    # members 用 int32 保存全局段号，总条目必须在分配展开数组前卡住容量。
    if membership_total > int(np.iinfo(np.int32).max):
        raise OverflowError("membership 数量超过紧凑 int32 容量")
    # CSR 展开：membership_offsets 是段视角前缀和；local 模/除展开成
    # (行, 列)，合并成行主序 core 索引后一次稳定排序即得按 core 聚合顺序。
    membership_offsets = np.empty(segments.segment_count + 1, dtype=np.int64)
    membership_offsets[0] = 0
    np.cumsum(membership_counts, out=membership_offsets[1:])
    members = np.repeat(np.arange(segments.segment_count, dtype=np.int32), membership_counts)
    local = np.arange(membership_total, dtype=np.int64) - np.repeat(membership_offsets[:-1], membership_counts)
    columns = ix0[members] + local % x_spans[members]
    rows = iy0[members] + local // x_spans[members]
    core_indices = rows * columns_total + columns
    order = np.argsort(core_indices, kind="stable")
    sorted_cores = core_indices[order]
    # bincount 直接给出每个 core 的 membership 数量；累积成 core 视角 CSR，
    # members[core_offsets[c]:core_offsets[c+1]] 即 core c 的 context 段集合。
    core_offsets = np.empty(macro.core_count + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(np.bincount(sorted_cores, minlength=macro.core_count), out=core_offsets[1:])
    return owners, core_offsets, members[order]


def _opaque_surround(query: DbuBox, data_bounds: DbuBox) -> kdb.Region:
    """返回负板数据包络外到查询边界的补铬区（2026-08-22 几何方案）。

    补到查询边界而非处理框边界：铬连续越过 field 边界，该处不产生轮廓（外
    框边不成为可优化段）；field 外 context 扩张带同样被覆盖，与透光率置零
    方案光学逐位同值（T=0）。clear 无对应操作——包络外无图形天然恒暗。
    """
    return kdb.Region(query.to_native()) - kdb.Region(data_bounds.to_native())


def prepare_macro_problem(
    batch: RegionBatch,
    layer: LayerSpec,
    polarity: MaskPolarity | str,
    fragmentation: FragmentationConfig,
    macro: MacroSpec,
    *,
    data_bounds: DbuBox,
) -> MacroProblem:
    """从完整相交图形一次生成可供多轮迭代复用的 macro 参考问题。

    data_bounds 是全局数据包络（layer bbox，须由调用方显式提供——macro
    局部 region 的 bbox 只是包络的局部投影，不可代推）。负板在提边之前
    补画包络外到查询边界的不透光图形：共线相接处在布尔并中融合（不产生
    虚假可动边）；补区外缘恒为 context-only 段（owner=-1，不可动、不进
    输出）；包络边透光缺口处形成真实铬|石英边，正常参与优化。
    """
    if batch.query_box != macro.query_box:
        raise ValueError("batch.query_box 必须等于 macro.query_box")
    try:
        normalized = polarity if isinstance(polarity, MaskPolarity) else MaskPolarity(polarity)
    except ValueError as exc:
        raise ValueError(f"不支持的 mask 极性：{polarity!r}") from exc
    # 完整相交物化（不裁剪 occurrence）→ 合并物理覆盖 → 提取一次真实轮廓：
    # 查询框从不参与布尔相交，其四条边不会进入 SegmentBatch 成为虚假 OPC 边
    # （负板补铬外缘虽落在查询边界，它是真实几何的边，属 context-only 段）。
    region = normalize_mask(batch, layer)
    if normalized is MaskPolarity.OPAQUE:
        # 布尔并的输出表示可能保留与既有铬共线相接的内部边（物理覆盖已
        # 融合、ring 表示未融合，实测会多出 48 个虚假段），必须显式
        # merged() 消除——等价于 normalize_mask 对 GDS cut-line 的处理。
        region = (region + _opaque_surround(macro.query_box, data_bounds)).merged()
    segments = fragment_edges(extract_contour(region), fragmentation, normalized)
    # 先按长度分段、再按 ownership 切线分裂：保证可写段的内部不跨两个 owner，
    # 否则跨界段会被某一侧独占更新而另一侧副本停在旧位置，边界处不一致。
    segments = _split_segments_at_ownership_cuts(segments, macro.x_cuts, macro.y_cuts)
    owners, core_offsets, members = _build_macro_ownership(segments, macro)
    return MacroProblem(
        macro=macro,
        layer=layer,
        polarity=normalized,
        fragmentation=fragmentation,
        segments=segments,
        owner_indices=owners,
        core_offsets=core_offsets,
        member_segment_indices=members,
    )
