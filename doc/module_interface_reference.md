# MyOPC 模块输入输出接口参考

本文以当前源码为准，逐模块说明 MyOPC 的输入、输出、数据形状、坐标与单位、对象生命周期、异常和性能边界。它回答“一个模块接收什么、返回什么、谁负责保存、哪些数据可以复用”，不把尚未实现的大 reticle macro/shard 方案描述为当前能力。

调用顺序图见[函数调用关系与数据流](function_call_architecture.md)，开发约束见[开发手册](development_manual.md)。

## 1. 接口分级与通用约定

### 1.1 接口分级

| 级别 | 识别方式 | 稳定性 |
|---|---|---|
| 包级公共接口 | 在包的 `__init__.py.__all__` 中导出 | 上层代码优先从这里导入 |
| 模块级工作接口 | 未进入包级 `__all__`，但被当前生产入口调用 | 可复用，但调用方需了解所属模块 |
| 内部接口 | 名称以 `_` 开头 | 只服务同层实现，不应成为新算法依赖 |
| 命令行入口 | `main/*.py` 的 `main()` | 可直接 `python main/<file>.py`，无需安装项目 |

`opc.iteration.ilt.optimize` 和 `opc.iteration.mbopc.optimize` 名称相同但语义不同，顶层 `opc.iteration` 有意不聚合导出。调用方应从明确的算法子包导入，必要时使用别名。

### 1.2 坐标、单位和数组记号

| 记号 | 含义 |
|---|---|
| DBU | 版图整数数据库单位；`dbu_um` 表示一个 DBU 对应多少微米 |
| `N` | 顶点或探针数量，依具体接口而定 |
| `E` | 数学边数量；在闭合轮廓中与顶点数量相同 |
| `S` | 切分后的控制边段数量 |
| `C` | core 数量 |
| `M` | 全部 core context membership 数量 |
| `(x, y)` | 所有几何坐标均按 x、y 顺序；x 向右，y 向上 |

- `DbuBox(left, bottom, right, top)` 使用整数 DBU，必须有正宽高。
- KLayout `Region`、轮廓和边段均位于版图全局坐标系。
- `geometry.render_*` 与 `opc.input.raster` 的公共数组都以最低 Y 为第 0 行，可在有效 ROI 内直接按行对齐；只有 PNG、Pillow 查看器和标注图底图会在输出边界上下翻转。
- 边段法向始终从 mask 材料指向空区。正位移沿外法向外移，负位移向材料内部移动。
- 除非接口明确写入文件，否则生产数据都只在内存中返回；诊断 PNG/GDS/NPZ 只由 `main/` 或 `opc.diagnostics` 显式生成。

### 1.3 主数据流

```text
GDS/OASIS
  -> LayoutDB
  -> ShapeQuery
  -> RegionBatch
  -> PhysicalMask
  -> ContourBatch
  -> SegmentBatch + Grid + owner/membership
  -> MBOPCProblem
  -> MB-OPC optimize
  -> best_displacements
  -> reconstructed Region/GDS

GDS/OASIS 或 raster NPZ
  -> float32 raster target
  -> ICCAD13Lithography / SimpleILT
  -> wafer Tensor 或 ILT mask
```

## 2. `layout`：版图数据库与 ROI 查询

### 2.1 [`layout/__init__.py`](../layout/__init__.py)

包级导出 `LayoutDB`、`ShapeQuery`、`DbuBox`、`LayerSpec`、`CellRef`、`RegionBatch`、层级/诊断数据类和全部 Layout 异常。上层模块应从 `layout` 导入这些对象，不依赖 KLayout cell/layer index 的内部表示。

### 2.2 [`layout/types.py`](../layout/types.py)

#### `DbuBox`

| 项目 | 契约 |
|---|---|
| 输入字段 | `left, bottom, right, top: int`，单位 DBU |
| 输出属性 | `width`、`height` 为 DBU；`area` 为 DBU² |
| 几何操作 | `expanded(margin)` 返回新框；`intersection(other)` 返回正面积交框或 `None` |
| 原生转换 | `to_native() -> kdb.Box`；`from_native(box) -> DbuBox` |
| 不变量 | 坐标必须为整数，`left < right` 且 `bottom < top` |
| 异常 | 非整数为 `TypeError`；空框、反向框或负 margin 为 `ValueError` |

内部共享边界的“归谁”不由 `DbuBox` 决定，而由 `RectilinearCoreGrid.locate_points` 的半开区间规则决定。

#### `LayerSpec`

输入 `layer: int`、`datatype: int = 0`，两者必须是非负整数。对象可排序、可作为字典键，作为所有公共接口的 GDS/OASIS Layer 标识。它不是 KLayout 内部 layer index。

#### `CellRef`

字段为 `name: str`、`index: int`。它避免把可变的 `kdb.Cell` 暴露给上层；只在创建它的 `LayoutDB` 生命周期内可用于重新解析原生 cell。

#### `LayerShapeStats` / `MaterializationStats`

- `LayerShapeStats`：`polygon_like`、`text`、`edge`、`other` 四类计数。
- `MaterializationStats`：`elapsed_seconds` 与只读 `shapes: Mapping[LayerSpec, LayerShapeStats]`。
- 仅 `ShapeQuery.materialize(diagnostics=True)` 返回这些信息。诊断会增加一次逐 shape 遍历，不应默认进入性能路径。

#### `RegionBatch`

| 字段/方法 | 输入 | 输出与语义 |
|---|---|---|
| `regions` | `Mapping[LayerSpec, kdb.Region]` | 构造时只复制小型映射，Region 仍在 C++ 内存 |
| `query_box` | `DbuBox` | 本批次的精确 planner ROI |
| `cell` | `CellRef` | 查询所基于的 cell |
| `stats` | `MaterializationStats | None` | 可选诊断 |
| `layers` | 无 | 排序后的 `tuple[LayerSpec, ...]` |
| `region(layer)` | `LayerSpec` | 返回对应原生 Region；缺层抛 `KeyError` |
| `counts()` | 无 | 返回只读 Layer→Polygon 数量映射 |

`RegionBatch` 是 ROI 物化后的跨层批次。其 Region 可在 `LayoutDB` 关闭后继续用于当前进程计算；但尚未物化的 `ShapeQuery` 不能在数据库关闭后执行。

### 2.3 [`layout/database.py`](../layout/database.py)

#### `LayoutDB.open(path, top_cell=None) -> LayoutDB`

| 项目 | 契约 |
|---|---|
| 输入 | GDS/OASIS 路径；多 top 时可显式给 `top_cell` 名称 |
| 输出 | 持有一次解析结果的只读数据库对象 |
| top 选择 | 未指定时必须恰好一个 top；指定时按精确名称选择 |
| 异常 | 文件/解析失败 `LayoutOpenError`；top 不唯一 `AmbiguousTopCellError`；cell 不存在 `CellNotFoundError` |
| 生命周期 | 推荐 `with LayoutDB.open(...) as db:`；离开上下文后所有数据库查询均失败 |

#### 查询与元数据接口

| 接口 | 输入 | 输出 | 是否物化图形 |
|---|---|---|---|
| `source_path` | 无 | 规范化 `Path` | 否 |
| `dbu_um` | 无 | `float`，微米/DBU | 否 |
| `top_cell` | 无 | `CellRef` | 否 |
| `layers()` | 无 | 排序后的现有 `LayerSpec` | 否 |
| `cell(name)` | cell 名 | `CellRef` | 否 |
| `bbox(cell=None)` | 可选 `CellRef` | 层级 bbox `DbuBox`；空 cell 为 `None` | 否 |
| `hierarchy_summary()` | 无 | `HierarchySummary` | 否 |
| `query(layers, box, cell=None, preserve_properties=False)` | Layer 列表、ROI、可选 cell | 惰性 `ShapeQuery` | 否 |
| `close()` | 无 | `None` | 释放原生数据库 |

`query` 接受 `LayerSpec` 或 `(layer, datatype)`，会去重并排序；空 Layer 集合在接触 KLayout 前拒绝。`preserve_properties=True` 的含义是导入并在精确裁剪后继承原图属性，**不是**“只选择有属性的图形”。

### 2.4 [`layout/query.py`](../layout/query.py)

#### `ShapeQuery.materialize(diagnostics=False) -> RegionBatch`

输入是 `ShapeQuery` 已保存的数据库、cell、Layer 和 ROI。执行时：

1. 每层建立一次 KLayout `RecursiveShapeIterator`；
2. 原生侧只允许 Box、Path、Polygon；Text/Edge 不进入物理 Region；
3. ROI 迭代先筛相交候选，再用一次原生 Region 相交精确裁剪到 `query_box`；
4. 属性模式使用 `NoPropertyConstraint` 保留左侧图形属性；
5. 可选诊断再统计接触 ROI 的图形类型。

输出 `RegionBatch` 中的坐标已经变换到所选 cell 的统一坐标系并精确裁到 ROI。精确裁剪可能在 ROI 框上形成新的轮廓边，因此该接口适合局部显示/计算，但未来 macro 提边不能把这些裁剪边误当物理边。

#### `ShapeQuery.materialize_intersecting(diagnostics=False) -> RegionBatch`

只用查询框筛选相交 occurrence，不与框做布尔相交；完整图形已变换到所选 cell 坐标，不相交图形不加载。deep Region 在原生端一次展平，所以结果可在 `LayoutDB` 关闭后消费；属性模式使用保持 shape class 的原生 `merged`，避免 `flatten()` 静默丢属性。该接口只供边段 macro 提取真实边；显示、ILT 像素 ROI 和普通查询继续使用精确 `materialize()`。

### 2.5 [`layout/hierarchy.py`](../layout/hierarchy.py)

- `CellInfo`：`ref`、可选 `bbox`、直接 `child_cells`、实例记录数 `instance_records`、展开阵列后的逻辑实例数 `logical_instances`。
- `HierarchySummary`：`top_cells` 和全部 `cells` 的不可变元组。
- `build_hierarchy_summary(db) -> HierarchySummary`：只遍历 cell/instance 元数据，不复制或扁平化图形。它是模块级实现函数，公共调用优先使用 `db.hierarchy_summary()`。

### 2.6 [`layout/errors.py`](../layout/errors.py)

`LayoutError(RuntimeError)` 是统一基类；细分为打开失败、top 歧义、cell 不存在、Layer 不存在和数据库已关闭。命令行入口通常捕获 `LayoutError` 并返回退出码 2。

## 3. `geometry`：轮廓、Patch、栅格与验证

### 3.1 [`geometry/__init__.py`](../geometry/__init__.py)

包级导出轮廓转换、Patch、可视化栅格、结构验证及 Geometry 异常。它不包含 OPC 的 segment、owner 或迭代语义。

### 3.2 [`geometry/contour.py`](../geometry/contour.py)

#### `ContourBatch`

使用两级 CSR 表达 `Polygon -> ring -> vertex`：

| 字段 | shape / dtype | 语义 |
|---|---|---|
| `vertices` | `[N,2] int64` | 全局 DBU 顶点；每个顶点对应一条从自身到下一点的数学边 |
| `ring_offsets` | `[R+1] int64` | ring 在 `vertices` 中的起止；首项 0，末项 N |
| `polygon_ring_offsets` | `[P+1] int64` | Polygon 在 ring 表中的起止；每个 Polygon 首 ring 是 hull，后续是 hole |

构造时数组被转为连续 `int64` 并验证：每 ring 至少三个点、每 Polygon 至少一个 ring、两级 offsets 单调且覆盖完整数据。`ring_count` 返回 R，`polygon_count` 返回 P。

#### 转换函数

| 接口 | 输入 | 输出 | 性能/生命周期 |
|---|---|---|---|
| `extract_contour(region)` | 一个 `kdb.Region` | 一个 `ContourBatch` | 逐 KLayout ring 读取一次，写入连续 64 位缓冲 |
| `extract_contours(batch)` | 多 Layer `RegionBatch` | 只读 Layer→`ContourBatch` 映射 | 每层调用一次转换 |
| `contours_to_region(contours)` | `ContourBatch` | 新 `kdb.Region` | 在明确重建边界逐 Polygon/ring 创建原生点 |

`extract_contour` 是物理边界数值化点；在 MB-OPC 中只应准备一次，后续多轮复用。

### 3.3 [`geometry/patch.py`](../geometry/patch.py)

#### `GeometryPatch`

字段：非空 `patch_id`、`LayerSpec`、全局坐标 `kdb.Region`、唯一责任框 `ownership_box`。构造只验证 ID 和 Region 类型，不自动裁剪。

#### `PatchSet`

- `add(patch) -> GeometryPatch`：先拒绝重复 ID 和同层 ownership 正面积重叠，再把输入 Region 精确裁到 ownership box，返回并保存规范化 Patch。
- `layers`：已保存 Layer 的排序元组。
- `region(layer) -> kdb.Region`：返回拼接结果副本；不存在的层返回空 Region。
- `__iter__`：按 Layer、ownership box、ID 的稳定顺序迭代。

相邻 ownership 可共享边界，但不能有正面积重叠。跨 core 图形可以完整传入不同 Patch，`add` 会各自只保留责任框内部分。

#### `PatchWriter.write(patches, output_path, dbu_um, top_name='OPC_PATCHES') -> Path`

输出扩展名支持 `.gds/.gds2/.oas/.oasis`。目标父目录必须已经存在；文件通过同目录临时文件原子替换。输出只包含 Patch 结果，不回写源版图。

### 3.4 [`geometry/raster.py`](../geometry/raster.py)

#### `iter_region_coverage_tiles(...)`

| 输入 | 约束 |
|---|---|
| `region` | 任意 KLayout Region；内部先裁到 box 并 merged |
| `box` | 全局 DBU ROI |
| `pixel_dbu` | 正整数 DBU/像素 |
| `shape=(height,width)` | 正整数输出有效区域尺寸 |
| `dtype` | 默认 `float64` |
| `max_tile_pixels` | 单个临时块像素上限，默认 1,000,000 |

按块 yield `(y0, x0, areas)`；`areas` 是 `[rows,columns]`、范围 `[0,1]` 的面积覆盖率，原点在左下。空 Region 不 yield 任何块。该函数是 Geometry 显示层和 OPC 模型栅格的共享底层。

#### `render_region_batch(...) -> uint8[H,W]`

把一个 `RegionBatch` 的单 Layer 变成左下原点灰度数组；第 0 行对应最低 Y，0 表示空，255 表示完全覆盖。`pixel_size_nm` 必须能被当前 `dbu_um` 精确表示成整数 DBU。`max_pixels` 在分配前限制总像素数；可选 PNG/查看器只在输出边界上下翻转，不改变返回数组。

#### `render_layout_region(...) -> uint8[H,W]`

输入已打开 `LayoutDB`、ROI、Layer 和显示参数。内部执行 `database.query(...).materialize()` 后调用 `render_region_batch`。它是便利接口，不缓存查询结果；同一区域重复显示时应自行复用 `RegionBatch`。

### 3.5 [`geometry/validate.py`](../geometry/validate.py)

- `ValidationIssue`：`code`、`message`、`layer`、相关 `indices`。
- `ValidationReport.issues`：稳定顺序的问题元组；`is_valid` 在无问题时为 `True`。
- `validate_contours(contours, layer)`：检测零长度边和有向面积为零的 ring，不修改输入，也不自动修复。

### 3.6 [`geometry/errors.py`](../geometry/errors.py)

`GeometryError(RuntimeError)` 是基类；`PatchConflictError` 表示 Patch ID/ownership 冲突，`RasterizationError` 表示物理像素、画布或 PNG 请求不合法。

## 4. `opc.input`：共享物理输入、网格和容量保护

### 4.1 [`opc/input/__init__.py`](../opc/input/__init__.py)

包级导出 `PhysicalMask`、`CoreSpec`、`RectilinearCoreGrid`、物理 mask 规范化及 preflight/内存统计。栅格函数当前从 `opc.input.raster` 显式导入，不在包级导出。

### 4.2 [`opc/input/_arrays.py`](../opc/input/_arrays.py)

内部数组校验接口：

- `as_vector(value, dtype, name)`：返回指定 dtype 的连续一维数组；否则 `ValueError`。
- `as_matrix(value, dtype, columns, name)`：返回连续 `[N,columns]` 数组。
- `as_points(value, name)`：返回有限 `float64[N,2]` 坐标。

它只统一输入层内部的不变量，不是面向业务的通用 utils。

### 4.3 [`opc/input/grid.py`](../opc/input/grid.py)

#### `axis_cuts_by_size(start, end, tile_dbu) -> int64[K+1]`

以 `start` 为锚点按固定 DBU 步长生成严格递增切线，最后一个 tile 自动裁短并令末切线等于 `end`。范围或 tile 非正时抛 `ValueError`。

#### `CoreSpec`

字段为非空 `core_id`、唯一写入 `ownership_box` 和只读计算 `context_box`。context 必须四向完整包含 ownership。

#### `RectilinearCoreGrid`

| 字段/属性 | 契约 |
|---|---|
| `x_cuts`, `y_cuts` | 严格递增连续 `int64` 一维数组，每轴至少两项 |
| `halo_dbu` | 非负整数 |
| `column_count`, `row_count`, `core_count` | 从 cuts 推导，不常驻 CoreSpec 列表 |
| `bounds` | 全部 ownership 的整体 `DbuBox` |
| `core(index)` | O(1) 按全局行优先索引即时构造一个 `CoreSpec`，不展开整张网格 |
| `cores()` | 按先行后列生成 `tuple[CoreSpec,...]`，context 为 core 四向扩 halo |
| `locate_points(points)` | 输入有限 `[N,2]`，输出 `int32[N]` owner；范围外为 -1 |

内部共享边界按 `[left,right)`、`[bottom,top)` 归右/上 core；网格整体最大 x/y 边界显式归最后列/行，保证外沿中点仍有 owner。

### 4.4 [`opc/input/mask.py`](../opc/input/mask.py)

#### `PhysicalMask`

字段为 `layer`、合并后的原生 `region`、`query_box`。它只表达单层物理覆盖，不包含轮廓、边段或 owner。

#### `normalize_physical_mask(batch, layer) -> PhysicalMask`

复制目标 Region、删除属性、启用最小连通、执行 merged 并验证原生 Polygon。输出不会修改输入 `RegionBatch`。不同属性不会阻止物理覆盖合并；GDS 为表达孔洞引入的零宽桥边不会成为可移动边。

### 4.5 [`opc/input/raster.py`](../opc/input/raster.py)

#### `rasterize_region_canvas(region, box, pixel_dbu, canvas) -> float32[canvas,canvas]`

有效 ROI 左下对齐到固定方形画布，数组第 0 行对应最低 y；右侧和上方不足部分为 0。覆盖率范围 `[0,1]`。当 ROI 所需宽/高超过 canvas 时拒绝，不隐式切 tile。

#### `ownership_canvas(core, context, pixel_dbu, canvas) -> bool[canvas,canvas]`

按像素中心生成唯一计分 mask：core 内为 True，halo 为 False。它只影响指标累计，不裁剪送入光刻模型的 context。

### 4.6 [`opc/input/preflight.py`](../opc/input/preflight.py)

#### 内存接口

- `default_memory_budget_bytes()`：读取启动时系统可用内存的 70%。
- `resolve_memory_budget_bytes(value_gib)`：显式 GiB→字节；`None` 使用默认值。
- `process_memory_snapshot()`：返回 `rss_bytes`、`uss_bytes`、`private_bytes`、`peak_working_set_bytes`、`system_available_bytes`。

#### `preflight_layout(...) -> dict`

必需输入为版图路径、明确 top cell、Layer、DBU ROI 和正内存预算。可选传分段长度与 grid 估算 segment/membership；两个分段长度必须同时提供。可选文件、occurrence、源顶点上限会在扫描过程中立即拒绝。

返回字段：

| 字段 | 含义 |
|---|---|
| `source_file_bytes` | 源文件大小 |
| `shape_occurrences` / `source_vertices` | 已扫描的层级图形和原始顶点 |
| `estimated_segments` / `estimated_memberships` | 按生产公式估算的规模 |
| `estimated_prepare_peak_bytes` / `estimated_solver_peak_bytes` | 保守 CPU 峰值估算 |
| `memory_budget_bytes` | 本次预算 |
| `int32_capacity_ok` / `memory_budget_ok` | 两类独立判断 |
| `scan_complete` | 是否扫描完整 ROI |
| `counts_are_lower_bounds` | 提前停止时为 True |
| `accepted` / `reason` | 当前全局内存实现能否继续 |
| `recommended_mode` | `in_memory` 或 `sharded_required` |

当预算或 int32 下界已经超限时会提前停止；此时数字只是足以拒绝的下界。`sharded_required` 是建议状态，当前代码不会自动进入尚未实现的 macro/shard 求解。

## 5. `opc.input.edge`：边段问题构造与矢量重建

### 5.1 [`opc/input/edge/__init__.py`](../opc/input/edge/__init__.py)

包级导出配置、边段数据、完整问题、准备入口、探针和重建接口。owner 构造保持内部实现，不提供可替换策略注册器。

### 5.2 [`opc/input/edge/fragmentation.py`](../opc/input/edge/fragmentation.py)

#### `FragmentationConfig`

字段均为 DBU：`corner_length_dbu`、`max_segment_length_dbu`、`max_displacement_dbu`，以及无量纲 `miter_limit=4.0`。长度必须有限，角段/最大段为正，最大段至少是角段两倍，最大位移非负，miter limit 不小于 1。

#### `SegmentBatch`

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `contours` | `ContourBatch` | 固定参考拓扑的唯一数值所有者 |
| `edge_next_ids` | `[E] int32` | 每条数学边终点顶点索引；ring 末边回到首点 |
| `edge_polygon_ids` | `[E] int32` | 数学边所属 Polygon，用于 tile 选完整 Polygon |
| `edge_normals` | `[E,2] float64` | 数学边外法向单位向量 |
| `ring_segment_offsets` | `[R+1] int64` | 每个 ring 对应的 segment 范围 |
| `edge_ids` | `[S] int32` | 每段引用的数学边 |
| `t0`, `t1` | `[S] float64` | 段在参考数学边上的参数区间，`0 <= t0 < t1 <= 1` |

`segment_count` 返回 S；`persistent_nbytes` 只统计本对象新增数组，不重复统计 `ContourBatch`。构造会交叉验证 edge cache 与轮廓拓扑。

#### `SegmentBatch.materialize(displacements=None) -> SegmentGeometry`

无位移时由参考边和 `t0/t1` 输出 `starts, ends, normals`，均为连续 `float64[S,2]`。传入位移时要求有限 `float64[S]`，每段端点统一加 `normal * displacement`。本方法不检查配置的最大位移；范围检查由重建入口负责。

`SegmentGeometry` 是按需临时对象，不应常驻 `MBOPCProblem`。

#### `fragment_edges(contours, config) -> SegmentBatch`

一次性向量化切分全部数学边。长边保留两端角段，中间均衡分段；短边按最大段长均分。结果严格保持 Polygon/ring/edge 的全局拓扑顺序，S 超过 int32 容量时抛 `OverflowError`。

### 5.3 [`opc/input/edge/ownership.py`](../opc/input/edge/ownership.py)

内部 `_build_ownership(segments, grid)` 返回：

1. `owner_indices: int32[S]`：按参考 segment 中点定位的唯一 owner；
2. `core_offsets: int64[C+1]`；
3. `member_segment_indices: int32[M]`：每 core 可读取的 owner+halo segment CSR。

membership 由参考 segment bbox 四向扩 halo 后与规则 core 范围相交得到。一个 segment 可出现在多个 core 的 context 中，但其 owner 只有一个。算法只展开实际二维邻居范围，不构造 `S×C` 密集矩阵。

### 5.4 [`opc/input/edge/builder.py`](../opc/input/edge/builder.py)

#### `MBOPCProblem`

字段：`physical_mask`、`config`、`segments`、`grid`、`owner_indices`、`core_offsets`、`member_segment_indices`。构造验证每段都有合法 owner、CSR 覆盖完整 membership、所有 member 索引在范围内。

| 接口 | 输出 |
|---|---|
| `core_count` | 从 grid 推导 C |
| `persistent_nbytes` | 去重后的全部常驻 NumPy 数组字节数；不包含 KLayout Region 和 Python 对象开销 |
| `segments_for_core(core_index)` | `member_segment_indices` 的 `int32` 视图；含 owner 与只读 halo segment |

`segments_for_core` 返回的是 context membership，不是“本 core 有权更新的边”。要取 owner 段需再过滤 `problem.owner_indices[members] == core_index`。

#### `prepare_problem(batch, layer, config, grid=None) -> MBOPCProblem`

组合顺序为 `normalize_physical_mask -> extract_contour -> fragment_edges -> _build_ownership`。省略 grid 时创建覆盖查询框的 1×1 规则网格。该函数不生成探针、PNG、NPZ 或 GDS；返回问题可在 LayoutDB 关闭后用于多轮计算。

当前实现会完整物化所选 ROI 的 Region、轮廓、segment 和 membership 到 CPU 内存。preflight 能提前拒绝超限输入，但不会分批展开另一部分。

#### Macro 前端接口

`macro_boxes(tile_grid, maximum_span_dbu)` 只选择已有 tile 切线，返回行优先 macro ownership 框；不会切开 tile。

`prepare_macro(...) -> MacroPreparation` 接收未裁剪完整候选，返回当前 macro 生命周期内的真实 `segments`、活跃 segment/全局 tile owner、当前 macro 的 tile ID 和局部 membership CSR。处理 ROI 外但进入 tile halo 的真实边保留为 `owner=-1` 的固定只读 context；`owned_segments()` 只按需推导当前 macro 的唯一可发布集合，不常驻重复数组。它不定义磁盘 shard、跨进程稳定 ID 或多轮状态，不能替代完整 `MBOPCProblem`。

`preflight_layout(..., include_layout_load_bytes=True)` 默认把一次源版图加载的保守成本计入预算；同一个已打开 `LayoutDB` 的逐 macro 局部预检传 `False`，避免把相同文件解析成本重复计入每个 macro。该开关只改变内存估算，不跳过文件大小上限或层级扫描。

### 5.5 [`opc/input/edge/sampling.py`](../opc/input/edge/sampling.py)

`edge_probe_points(starts, ends, normals, distance_dbu)` 要求三组同形 `[S,2]` 数组和正有限距离，返回两个连续 `float64[S,2]`：

```text
midpoint = (start + end) / 2
inner = midpoint - outward_normal * distance
outer = midpoint + outward_normal * distance
```

探针围绕固定参考边定义。若窄图形使 inner/outer 落到错误目标区域，后续评价会把该探针判为无效，而不是改变其几何定义。

### 5.6 [`opc/input/edge/reconstruction.py`](../opc/input/edge/reconstruction.py)

- `reconstruct_contours(problem, displacements) -> ContourBatch`：位移必须是有限 `[S]` 且绝对值不超过配置上限；同数学边等位移时删除内部切分点，不等位移时形成 jog；原始角使用解析 miter，平行或超 miter limit 时 bevel；最后取整到整数 DBU 并验证退化环。
- `reconstruct_region(problem, displacements) -> kdb.Region`：在上述结果上恢复 Polygon/hole 并验证原生合法性。

位移始终相对固定参考边计算，不把上一轮取整结果重新提边或切分。core 不参与最终矢量裁剪，因此跨 core 斜边只经历一次全局重建。

## 6. `lithography`：可微 ICCAD13 Hopkins 模型

### 6.1 [`lithography/__init__.py`](../lithography/__init__.py)

包级导出 `ICCAD13Config`、`ProcessCondition` 和 `ICCAD13Lithography`。

### 6.2 [`lithography/iccad13.py`](../lithography/iccad13.py)

#### `ICCAD13Config.from_file(path) -> ICCAD13Config`

读取“名称 值”文本，要求 `KernelNum`、`TargetDensity`、`PrintThresh`、`PrintSteepness`、三种 Dose、`Canvas`、`Resolution`。核数/canvas/resolution 必须为正，阈值在 `(0,1)`，剂量满足 min ≤ nominal ≤ max。

#### `ProcessCondition`

字段：非空唯一 `name`、`kernel`（仅 `focus` 或 `defocus`）、正有限 `dose`。条件彼此独立，不再绑定为固定三元组。

#### `ICCAD13Lithography`

构造输入为可选配置路径、资产目录和 device。默认从模块内加载 focus/defocus kernel 与 scale；`device='auto'` 或 `None` 时 CUDA 可用则选 CUDA，否则 CPU。kernel/scale 注册为 buffer，可随 `.to()` 移动但不会成为优化参数。

| 接口 | 输入 | 输出 |
|---|---|---|
| `device` | 无 | buffer 所在 `torch.device` |
| `condition(name)` | `nominal` / `dose_max` / `defocus_min` | 默认 `ProcessCondition` |
| `forward(mask, condition)` | `[H,W]` 或 `[B,H,W]` Tensor | 同 shape 连续 wafer Tensor |
| `forward_many(mask, conditions)` | 同上；非空、名称不重复的条件序列 | `dict[name, Tensor]`，每项 shape 与输入一致 |

输入会转为设备上的 float32，居中补零到 canvas，并在 resolution 不同时最近邻缩放。模型只检查尺寸，不自动限制 mask 值域。一次 `forward_many` 只做一次 mask FFT，相同 kernel bank 只传播一次单位剂量强度，再按 dose² 缩放。输出是 sigmoid 后的连续值，通常在 `(0,1)`。

所有计算由普通 PyTorch 算子组成：MB-OPC 可在 `torch.no_grad()` 下调用，ILT/梯度 OPC 可直接 `loss.backward()`。模型没有单独 `backward` 方法。

## 7. `evaluation`：公共评价指标

### 7.1 [`evaluation/__init__.py`](../evaluation/__init__.py)

包级导出二值 L2、PVBand、EPE 探针评价、矩形 shot 估计和 `EPEEvaluation`。

### 7.2 [`evaluation/metrics.py`](../evaluation/metrics.py)

#### 像素指标

| 接口 | 输入 | 输出 |
|---|---|---|
| `evaluate_binary_l2(target, nominal, threshold=0.5, ownership_mask=None)` | 同 shape/device 的 `[H,W]` 或 `[B,H,W]` | 二值不一致像素数 `int` |
| `evaluate_pvband(maximum, minimum, threshold=0.5, ownership_mask=None)` | 同 shape/device 连续 wafer | 两条件二值不一致像素数 `int` |
| `estimate_rectangular_shots(mask, threshold=0.5, shape=(512,512))` | `[H,W]` 或 `[B,H,W]` | 确定性水平 run 合并后的矩形数 `int` |

ownership mask 必须与提升后的 batch shape/device 一致；它只选择计分像素。Shot 会最近邻缩放、detach 并转 CPU，适合最终诊断，不可微，也不保证全局最小 shot 数。

#### `evaluate_edge_probes(...) -> EPEEvaluation`

输入：target/nominal 图、`batch_indices: [S]`、`inner_xy/outer_xy: [S,2]` 和阈值。xy 坐标先四舍五入为像素索引；越界、inner/outer 同像素、目标 inner 非材料或目标 outer 非空区时 `valid=False`。

输出字段均与探针对齐：

| 字段 | dtype/语义 |
|---|---|
| `valid` | bool Tensor |
| `inner_violations` | inner 应打印但未打印 |
| `outer_violations` | outer 应为空但打印 |
| `ambiguous` | inner/outer 同时违规 |
| `directions` | int8；+1 外移，-1 内移，0 不移 |
| `violation_count` | 至少一个探针违规的 segment 数，Python int |

同一 segment 两侧同时要求相反移动时记录歧义并保持 0，不用任意优先级覆盖。

## 8. `opc.iteration`：具体优化方法

### 8.1 [`opc/iteration/__init__.py`](../opc/iteration/__init__.py)

只声明算法容器，不聚合具体方法。这保证更换迭代方法时不会让同名 `optimize` 混淆。

### 8.2 [`opc/iteration/ilt/simple.py`](../opc/iteration/ilt/simple.py)

包导出定义见 [`opc/iteration/ilt/__init__.py`](../opc/iteration/ilt/__init__.py)。

#### `SimpleILTConfig`

字段：正 `iterations`、正 `step_size`、正 `sigmoid_steepness`、非负 `weight_pvband`、`weight_process_l2`、`curvature_weight`，以及 `(0,1)` 的 `mask_threshold`。全部浮点值必须有限。

#### `optimize(...) -> SimpleILTResult`

| 输入 | 契约 |
|---|---|
| `target` | `[H,W]` 或 `[B,H,W]`，转到模型设备 float32 后 detach |
| `model` | `ICCAD13Lithography` |
| `config` | `SimpleILTConfig` |
| `initial_parameters` | 可选，与 target shape 和 batch 维语义完全一致 |
| `optimization_mask` | 可选同形 `[0,1]`；0 固定、1 可优化、中间值为混合权重 |
| `nominal_condition` | 可选独立条件；默认 nominal |
| `process_conditions` | 可选任意条件序列；默认 dose_max/defocus_min，可传空元组 |

每轮以参数的 sigmoid 作为软 mask，损失为 nominal 连续 L2、所有 process 条件 L2、条件范围平方、可选 3×3 曲率项。使用 SGD 和原生 autograd。条件名称必须全局唯一。

输出：

- `best_parameters`：总损失最优状态的参数 Tensor；
- `soft_mask`：对应软 mask；
- `binary_mask`：按 `mask_threshold` 得到的 bool Tensor；
- `best_iteration`：从 0 开始；
- `records`：每轮 `ILTIterationRecord`，含五类损失和耗时。

记录和最佳状态对应该轮执行参数更新**之前**评价出的状态；二维输入会在结果中去掉 batch 维。

### 8.3 [`opc/iteration/ilt/levelset.py`](../opc/iteration/ilt/levelset.py)

#### `LevelSetConfig`

字段为正 `iterations`、正有限 `step_size` 以及非负有限的 `weight_process_l2`、`weight_pvband`、`curvature_weight`。水平集硬边界固定为零等值线，不提供会改变算法语义的第二个 mask 阈值。

#### `signed_distance_initialization(target, threshold=0.5) -> Tensor`

输入 `[H,W]` 或 `[B,H,W]` 有限 Tensor；阈值必须在 `(0,1)`。输出与输入 shape/device 一致的 float32 精确像素中心欧氏 SDF，前景为负、背景为正。CPU 使用两遍一维下包络距离变换，时间和临时内存均为 `O(BHW)`；只用于一次性初始化，不进入光刻迭代热路径。

#### `optimize_levelset(...) -> SimpleILTResult`

输入契约与 SimpleILT 的 target、优化窗口和独立工艺条件一致，但可选初值是同形 `initial_levelset`。target 必须为 `[0,1]` 内有限数；初值和窗口也必须有限。硬前向固定为 `phi < 0`，自定义 backward 以 `-|∇phi|` 调制光刻模型给出的上游梯度；Adam 更新 phi。曲率复用 SimpleILT 的 3×3 零和核。

输出复用 `SimpleILTResult`：`best_parameters` 是最优 phi，`soft_mask=sigmoid(-phi)` 只作连续诊断，`binary_mask` 严格按 `phi < 0` 生成。显式空 `process_conditions` 只计算 nominal，不回退默认工艺窗；二维输入会去掉 batch 维。

### 8.4 [`opc/iteration/ilt/curvmulti.py`](../opc/iteration/ilt/curvmulti.py)

#### `CurvMultiConfig`

字段：严格递减且以 1 结束的正整数 `scales`；正 `iterations_per_stage/step_size`；正奇数 `smoothing_kernel`；正有限 `sigmoid_steepness`；`[0,1]` 有限 `sigmoid_offset`；非负 process/PVBand/curvature 权重；`(0,1)` 的 `mask_threshold`。target 高宽必须能被全部尺度整除，最粗控制网格边长不得小于平滑核。

#### `optimize_curvmulti(...) -> SimpleILTResult`

输入 target、可选完整尺度 initial parameters/optimization mask、独立 nominal/process conditions。target 与窗口均为 `[H,W]` 或 `[B,H,W]`；target/窗口范围 `[0,1]`，所有值有限。显式空 process conditions 只计算 nominal。

每个尺度把完整参考用 area 缩为控制参数；跨阶段最优参数用 nearest warm-start。控制参数经均值池化和 offset sigmoid 生成 soft mask，再 nearest 恢复到完整 target shape，最后才进入固定像素网格 Hopkins 模型。损失为 nominal L2、process L2、process range 平方和，以及施加于 nominal wafer 的曲率。每阶段独立使用 SGD 并丢弃上一阶段优化器/计算图。

结果字段复用 `SimpleILTResult`；`records` 的 iteration 是跨尺度全局递增下标，长度为 `len(scales)*iterations_per_stage`。`best_parameters` 是最终尺度的最优控制参数，最终尺度固定为 1，故 shape 与 target 一致；`soft_mask/binary_mask` 是完整网格结果。统一 runner 的 JSON 记录额外附加 `stage_index/stage_scale/stage_iteration`，这些字段不是求解器内的第二套记录结构。

### 8.5 [`opc/iteration/ilt/multilevel.py`](../opc/iteration/ilt/multilevel.py)

#### `MultilevelConfig`

`scales` 为严格递减、以 1 结束的正整数元组；`stage_iterations`、`stage_step_sizes` 必须与尺度等长，分别为每级正迭代数和 Adam 实际正步长。其余平滑、sigmoid、损失和阈值字段与 CurvMulti 范围一致。默认 `(2,1)/(20,100)/(0.2,0.2)` 对应 OpenILT Low/Mid 调度，但不保留其物理尺度错误。

#### `optimize_multilevel(...) -> SimpleILTResult`

target、可选 initial/optimization mask 和独立工艺条件契约与 CurvMulti 一致。每级用 area 生成 target/reference 和 nearest 生成窗口/warm-start；级别 soft mask 必须先恢复完整 target shape，再调用 Hopkins，完整 wafer 随后 area 汇聚到级别 shape 计算 nominal/process/PVBand/wafer-curvature。每级新建 Adam，仅传递该级历史最优有效参数，不传 optimizer 状态或计算图。

返回继续复用 `SimpleILTResult/ILTIterationRecord`；记录数为 `sum(stage_iterations)`，iteration 全局递增。最终 scale 固定为 1，所以参数、soft/binary mask 与 target 同形。runner JSON 按累计迭代边界附加 `stage_index/stage_scale/stage_iteration`。

### 8.6 [`opc/iteration/mbopc/contracts.py`](../opc/iteration/mbopc/contracts.py)

包导出定义见 [`opc/iteration/mbopc/__init__.py`](../opc/iteration/mbopc/__init__.py)。

#### `SimpleMBOPCConfig`

字段：正迭代数、正初始步长 DBU、正衰减周期、正 EPE 距离 DBU、正整数像素 DBU、canvas、batch size，以及非负 target cache 字节上限。

#### 输出记录

- `IterationRecord`：`iteration` 是已评价状态下标，另含该状态的 EPE/L2/PVBand、有效/歧义探针数，以及由该状态提出的 `step_dbu`、成功移动/拒绝段数和耗时；最终只评价状态的三项更新字段为 0。
- `SimpleMBOPCResult`：`best_displacements: float64[S]`、状态记录元组、最佳已评价状态下标 `best_iteration` 和 `stop_reason`。

`stop_reason` 当前可能为 `iteration_limit`、`zero_epe` 或 `no_legal_update`。

### 8.7 [`opc/iteration/mbopc/solver.py`](../opc/iteration/mbopc/solver.py)

#### `optimize(problem, model, config) -> SimpleMBOPCResult`

前置约束：每个 core 的 context 按 `pixel_dbu` 计算后必须装入固定 canvas；solver canvas 不得超过模型 canvas。

状态与更新接口语义：

1. CPU 的 `current: float64[S]` 是本轮只读全局绝对位移；
2. 每个 batch 构造 target `uint8[B,H,W]`、current mask `float32[B,H,W]`、ownership `bool[B,H,W]`；
3. GPU/CPU 模型计算三个默认条件；L2/PVBand 只在 ownership 像素累计；
4. 只为 owner segment 评价参考探针，并把方向写入 `next_values`；halo 只读；
5. batch 输出释放后继续下一批，但任何批都看不到 `next_values`；
6. 全部 core 完成后全局重建候选，检查 ring 绕向和 hole 包含关系；合法才跨轮屏障发布，非法则整次更新回滚；最后一次允许更新也执行相同检查；
7. `iterations=N` 最多发布 N 次更新，初态和每次发布后状态都执行评价；完整执行产生 N+1 条状态记录，最后一条不再提出更新；
8. 最佳状态只按 EPE 严格改善选择，EPE 相同保留更早状态，L2/PVBand 不影响几何选择。

固定 target tile 使用字节上限 LRU 缓存；上限 0 可关闭。GPU 只常驻当前 batch，但 CPU 当前仍常驻完整 problem、全局位移和少量索引。输出的 `best_displacements` 不一定是最后一轮位移，调用方必须用它重建最终结果。

### 8.8 [`opc/iteration/diffopc/`](../opc/iteration/diffopc)

包级原型导出为 `DiffOPCConfig`、`DiffOPCIterationRecord`、`DiffOPCResult` 和 `optimize`。`contracts.py` 保存配置/记录/结果；`rasterizer.py` 的 `rasterize_soft_edges` 以参考 mask、`[S,2]` 起终点/法向和 `[S]` 位移生成可微软 mask；`solver.py` 在现有 `MBOPCProblem` 上优化全局位移。

这些符号已有测试和 runner 调用方，因此保留在当前目录，但尚未通过后续阶段的连续 EPE、MRC/SRAF、大图内存和完整产物验收。当前接口参考只记录原型现实，不把它声明为生产完成能力。

## 9. `opc` 异常与显式诊断

### 9.1 [`opc/errors.py`](../opc/errors.py) 与 [`opc/__init__.py`](../opc/__init__.py)

`OPCError(Exception)` 是基类；`PhysicalMaskError`、`OwnershipError`、`ReconstructionError` 分别表示物理 mask、归属/提交和重建失败。当前 `OwnershipError` 是公共领域异常，但多数数组不变量由数据类直接抛 `ValueError`。

### 9.2 [`opc/diagnostics.py`](../opc/diagnostics.py)

这些是模块级显式副作用接口，不被输入构造或 solver 热路径导入。

| 接口 | 输入 | 输出/副作用 |
|---|---|---|
| `save_problem_npz` | problem、`[S]` 位移、路径 | 原子保存诊断快照 v3，返回 Path |
| `write_debug_gds` | reference/reconstructed Region、DBU、Layer | GDS，含独立 `REFERENCE` 与 `RECONSTRUCTED` top cell |
| `render_boundary_overlay` | Region、box、`[S,2]` 端点/法向、可选 owner/probe/core | 顶部原点标注 PNG |
| `build_geometry_cases` | 无 | 五个确定性 `dict[str,kdb.Region]` 用例 |
| `run_geometry_suite` | 输出目录、是否写图、探针距离 | JSON 兼容汇总，并可写五张 PNG 与 JSON |

`save_problem_npz` 没有 `format_name/metadata_json`，也没有公开 loader；它只用于当前前端人工检查，**不能**传给 `load_segment_input`。可恢复问题归档由 `main.offline_inputs.prepare_segment_input` 生成。

## 10. `main/offline_inputs.py`：文件级输入契约

源码：[`main/offline_inputs.py`](../main/offline_inputs.py)。该模块负责输入物化、归档版本与损坏校验；原子写入实现统一位于 [`main/artifacts.py`](../main/artifacts.py)。raster 归档压缩，完整 segment 归档不压缩以降低准备时 CPU/峰值内存。

### 10.1 内存 raster 接口

#### `materialize_raster_input(...) -> (mask, metadata)`

输入 GDS/OASIS、可选 Layer/top/DBU box，以及 pixel/canvas 和安全上限。若未指定 Layer，版图必须只有一个 Layer；未指定 box 使用 top bbox。像素 nm 必须能精确换算为整数 DBU。

执行顺序为文件限制→打开 LayoutDB→画布尺寸检查→层级 preflight→精确 ROI 物化→左下原点栅格化。输出：

- `mask`: 连续 `float32[canvas,canvas]`，范围 `[0,1]`；
- `metadata`: source/top/layer/box/DBU/pixel/canvas/有效宽高/方向/Polygon 数/preflight。

该接口不写 NPZ。

#### `prepare_raster_input(...) -> Path`

调用同一内存接口后保存 raster NPZ v2。不会产生第二套栅格逻辑。

#### `load_raster_input(path, max_archive_gib=8.0) -> (mask, metadata)`

先限制 NPZ 文件与声明解压总量，禁止加密与 pickle，再验证格式、版本、Layer、ROI、DBU、方向、shape、有限值和 `[0,1]` 范围。

#### `resolve_raster_input(path, ...) -> (mask, metadata)`

仅按扩展名分派：`.npz` 调 `load_raster_input`，其他扩展名按版图调用 `materialize_raster_input`。NPZ 分支的 Layer/ROI/pixel 已由 metadata 固定，调用参数不会覆盖归档内容。

### 10.2 raster NPZ v2 字段

| 字段 | 类型 |
|---|---|
| `format_name` | 标量字符串 `myopc.raster-input` |
| `format_version` | int32 标量 2；loader 兼容历史 v1 clear 输入 |
| `metadata_json` | JSON 字符串标量 |
| `mask` | `float32[canvas,canvas]` |

### 10.3 segment 问题接口

#### `materialize_segment_input(...) -> (problem, metadata)`

执行版图范围选择、preflight、ROI 物化、分段和 owner/membership 构造，直接在内存中返回 `MBOPCProblem` 与 JSON 兼容 metadata，不写中间文件。正式 DiffOPC 的 GDS/OASIS 分支使用此接口，避免大问题先写再读 NPZ。

#### `prepare_segment_input(...) -> Path`

输入版图、输出 NPZ、Layer/top/ROI、tile/halo/角段/最大段/最大位移 nm 和安全上限。所有物理长度必须能由当前 DBU 精确表达。它调用相同的 `materialize_segment_input` 后只增加版本化原子归档；构造后仍用真实数组规模复核估算。

#### `load_segment_input(path, max_archive_gib=8.0) -> (problem, metadata)`

恢复顺序为 `ContourBatch -> SegmentBatch -> RectilinearCoreGrid -> Region -> PhysicalMask -> MBOPCProblem`，再校验单位法向、segment 参数连续覆盖、ring 对齐、每 core membership 严格递增无重复、每段在 owner context 恰好出现一次、metadata 计数与数组一致。v3 保存 polarity，loader 兼容历史 v2 clear 输入。

### 10.4 segment NPZ v3 字段

| 分组 | 字段 |
|---|---|
| 头 | `format_name=myopc.mbopc-input`、`format_version=3`、`metadata_json` |
| 轮廓 | `contour_vertices`、`contour_ring_offsets`、`contour_polygon_ring_offsets` |
| 数学边 | `edge_next_ids`、`edge_polygon_ids`、`edge_normals` |
| segment | `segment_ring_offsets`、`segment_edge_ids`、`segment_t0`、`segment_t1` |
| 归属 | `owner_indices`、`core_offsets`、`member_segment_indices` |
| 网格 | `grid_x_cuts`、`grid_y_cuts`、`grid_halo_dbu` |

metadata 包含源路径、top、Layer、ROI、DBU、分段/网格配置、计数、preflight 和构造后估算。数组是迭代权威数据，metadata 用于物理单位、报告和一致性校验。

### 10.5 命令行

```powershell
python main\offline_inputs.py raster INPUT.gds raster_input.npz [版图与像素参数]
python main\offline_inputs.py segments INPUT.gds mbopc_input.npz [版图、tile 与边段参数]
```

成功输出路径并返回 0；可预期错误输出到 stderr 并返回 2。

## 11. `main`：九个可直接运行入口

### 11.1 [`main/run_layout_geometry.py`](../main/run_layout_geometry.py)

输入 GDS/OASIS、可重复 Layer、可选 top/DBU ROI。`run(args)` 返回 JSON 兼容字典，包括源、top、DBU、box，以及每层 Polygon 数、面积、bbox；`--arrays` 增加顶点/ring/边数，`--diagnostics` 增加 shape 类型计数。可选写 Patch GDS/OASIS、单 Layer PNG或显示图片。

### 11.2 [`main/run_mbopc_frontend.py`](../main/run_mbopc_frontend.py)

这是不运行光刻的前端验证器。输入可省略以使用合成多图形，也可给真实版图；网格可按列×行或固定 nm tile；支持 preflight-only 和跳过产物。

`run(args)` 输出 `status`、范围、preflight、tiling、Polygon/Ring/Edge/Segment/core/membership 数、problem 数组内存、各阶段时间与进程内存、零位移/覆盖/重叠/合法性验证，以及产物路径。默认诊断产物为 summary JSON、诊断 NPZ v3、reference/reconstruction GDS、标注 PNG 和几何图集。

它只施加每 core 一段示范位移来验证前端，不代表真实 OPC 结果。

### 11.3 [`main/run_mbopc.py`](../main/run_mbopc.py)

直接完成 `GDS/OASIS -> preflight -> Region -> MBOPCProblem -> 光刻迭代 -> best Region`。输入包括 Layer/top/ROI、tile/halo/pixel、边段配置、迭代参数、GPU batch、target cache、device 和内存预算。

`run(args)` 返回 JSON 兼容字典，状态可能为 `preflight_only`、`rejected` 或 `completed`。完成时保存 summary JSON、双 cell 结果 GDS 和可选 preview PNG；额外在固定 512² 画布计算一次 shot 估计。`--preview/--no-preview` 可显式覆盖 TOML 默认值，这不是底层 solver 的强制输出。

注意：完成摘要中的 `top_cell` 当前回显命令行 `args.top_cell`；自动选择唯一 top 时该字段可能为 `null`，权威几何仍来自 `LayoutDB` 实际选择的 top。

### 11.4 [`main/run_lithography.py`](../main/run_lithography.py)

`run_lithography_test(input_path, output_dir=None, ...)` 接受 GDS/OASIS 或 raster NPZ。返回设备上的 `dict[str, Tensor]`，键为 `nominal`、`dose_max`、`defocus_min`。给 output dir 时写 `lithography_result.npz`、summary JSON；`save_png=True` 时再写 mask 和三个条件 PNG。

结果 NPZ v1 字段是 `nominal`、`maximum`、`minimum`，均为 CPU float32 数组；这里的 `maximum/minimum` 文件名对应内存结果的 `dose_max/defocus_min`。

### 11.5 [`main/run_simpleilt.py`](../main/run_simpleilt.py)

`run_simpleilt(input_path, output_dir, ...) -> (SimpleILTResult, summary)` 接受 GDS/OASIS 或 raster NPZ。它是 `run_ilt(method="simple", return_result=True)` 的参数适配层，保留历史默认值和 Python 返回值，不再重复输入、优化、评价或保存实现。统一写 `ilt_result.npz`、summary JSON、最终光刻 NPZ和可选 PNG；不会再生成 `simpleilt_result.npz`。

### 11.6 [`main/run_ilt.py`](../main/run_ilt.py)

`run_ilt(input_path, output_dir, method=..., return_result=False, ...) -> summary | (SimpleILTResult, summary)` 接受 GDS/OASIS 或 raster NPZ。默认返回契约不变；兼容入口只有在显式 `return_result=True` 时取得同次执行的内存结果。已验收 `method=simple/levelset/curvmulti/multilevel`：输入参数包括迭代/损失、device、Layer/top/DBU ROI、pixel/canvas 和容量上限；CurvMulti/Multilevel 另接受 scales、平滑核、sigmoid steepness/offset 和 mask threshold，Multilevel 再接受逐级 iterations/step sizes。保存 `ilt_result.npz`、最终三工艺角光刻结果、可选 target/soft/binary PNG 及 summary JSON。summary 包含配置、逐轮/阶段损失、二值 L2/PVBand/shot、输入/优化/评价/输出时间、进程内存检查点和 GPU 峰值。

### 11.7 [`main/run_diffopc.py`](../main/run_diffopc.py)

读取 GDS/OASIS 或 segment NPZ 并调用已验收 DiffOPC。入口保存结果 NPZ、参考/重建 GDS、summary、可选 preview 和 ownership-only 最终光刻 tile；当前仍是 CPU 常驻完整问题、GPU 流式 batch，不等同于未来 macro shard。

### 11.8 [`main/run_mbopc_iteration.py`](../main/run_mbopc_iteration.py)

`run_mbopc_iteration_test(input_path, output_dir, ...) -> SimpleMBOPCResult` **只接受** `prepare_segment_input` 生成的 segment NPZ v2/v3。它不读取源 GDS、不重新提边、不重新分 owner。输出 GDS、`mbopc_result.npz`、summary JSON和可选 preview；结果 NPZ v1 保存最佳位移、最佳轮次和停止原因。

### 11.9 [`main/artifacts.py`](../main/artifacts.py)、[`main/offline_inputs.py`](../main/offline_inputs.py) 与 [`main/__init__.py`](../main/__init__.py)

`main/__init__.py` 不聚合导出。`artifacts.py` 公开 `atomic_json/atomic_npz/atomic_png` 与两种最终光刻保存函数；`offline_inputs.py` 只保留版图到 raster/segment 的内存物化、归档读写校验和准备 CLI。纳米到 DBU 的严格转换是 `main.configuration.exact_dbu`，所有入口直接使用公共名称，不再跨模块导入下划线私有符号。

## 12. 常见组合方式

### 12.1 直接版图运行光刻

```python
from main.run_lithography import run_lithography_test

printed = run_lithography_test(
    "TestReticle/simple.gds",
    box=(-2000, -1100, -200, 948),
    pixel_nm=8.0,
    device="cuda",
)
nominal = printed["nominal"]
```

### 12.2 只准备一次像素，重复优化模型/ILT

```python
from main.offline_inputs import prepare_raster_input, load_raster_input

prepare_raster_input("input.gds", "target.npz", layer=None, pixel_nm=8.0)
target, metadata = load_raster_input("target.npz")
```

### 12.3 只准备一次边段，重复优化 MB-OPC

```python
from main.offline_inputs import prepare_segment_input, load_segment_input
from lithography import ICCAD13Lithography
from opc.iteration.mbopc import SimpleMBOPCConfig, optimize

prepare_segment_input("input.gds", "problem.npz")
problem, metadata = load_segment_input("problem.npz")
model = ICCAD13Lithography(device="cuda")
result = optimize(problem, model, SimpleMBOPCConfig(
    iterations=8, initial_step_dbu=8.0, decay_every=4,
    epe_distance_dbu=16.0, pixel_dbu=8,
))
```

实际 DBU 参数应由 metadata 的 `dbu_um` 换算，不能假设 1 nm/DBU。

## 13. 当前能力边界

- `ShapeQuery.materialize` 和 `prepare_problem` 面向一个明确 ROI；当前 edge problem 会把该 ROI 的完整轮廓/segment/membership 常驻 CPU 内存。
- MB-OPC 的像素与光刻中间量按 core batch 流式释放，但还没有 CPU macro shard 或 memmap 双代位移状态。
- 当前版图引用在 Region 物化后成为 top 全局 occurrence 几何，结果不会回写 master cell，也不会自动传播到其他引用。
- 当前 segment ID 只是一次 `MBOPCProblem` 内的数组下标；显式 remesh 后必须重建 owner、membership 和优化状态。
- 诊断 NPZ v3、raster NPZ v2、segment problem NPZ v3、各算法结果 NPZ v1 是四类不同协议，不应按扩展名相同而互换。

未来大版图接口方案见[大 Reticle 流式处理方案](large_reticle_streaming_plan.md)，其中未实施内容不属于本文公共接口。

## 14. 模块覆盖清单

本文已覆盖全部当前生产 Python 模块：

- `layout`：`__init__`、`types`、`database`、`query`、`hierarchy`、`errors`；
- `geometry`：`__init__`、`contour`、`patch`、`raster`、`validate`、`errors`；
- `opc`：`__init__`、`errors`、`diagnostics`；
- `opc.input`：`__init__`、`_arrays`、`grid`、`mask`、`raster`、`preflight`；
- `opc.input.edge`：`__init__`、`builder`、`fragmentation`、`ownership`、`sampling`、`reconstruction`；
- `opc.iteration`：顶层 `__init__`、共享有界 tile 缓存、ILT 的 `__init__/simple/levelset/curvmulti/multilevel`、MB-OPC 的 `__init__/contracts/solver`，以及已验收 DiffOPC 的四个模块；
- `lithography`：`__init__`、`iccad13`；
- `evaluation`：`__init__`、`metrics`；
- `main`：`__init__`、`offline_inputs` 和八个 `run_*.py`。

## 15. 包级公共符号速查

以下名称与各包当前 `__all__` 完全一致；未列入的下划线函数和具体 runner 辅助函数不属于包级公共接口。

| 包 | 公共符号 |
|---|---|
| `layout` | `AmbiguousTopCellError`、`CellInfo`、`CellNotFoundError`、`CellRef`、`ClosedLayoutError`、`DbuBox`、`HierarchySummary`、`LayerNotFoundError`、`LayerShapeStats`、`LayerSpec`、`LayoutDB`、`LayoutError`、`LayoutOpenError`、`MaterializationStats`、`RegionBatch`、`ShapeQuery` |
| `geometry` | `ContourBatch`、`GeometryError`、`GeometryPatch`、`PatchConflictError`、`PatchSet`、`PatchWriter`、`RasterizationError`、`ValidationIssue`、`ValidationReport`、`contours_to_region`、`extract_contour`、`extract_contours`、`iter_region_coverage_tiles`、`render_layout_region`、`render_region_batch`、`validate_contours` |
| `opc` | `OPCError`、`OwnershipError`、`PhysicalMaskError`、`ReconstructionError` |
| `opc.input` | `CoreSpec`、`PhysicalMask`、`RectilinearCoreGrid`、`default_memory_budget_bytes`、`normalize_physical_mask`、`preflight_layout`、`process_memory_snapshot`、`resolve_memory_budget_bytes` |
| `opc.input.edge` | `FragmentationConfig`、`MBOPCProblem`、`SegmentBatch`、`SegmentGeometry`、`edge_probe_points`、`fragment_edges`、`prepare_problem`、`reconstruct_contours`、`reconstruct_region` |
| `opc.iteration.ilt` | `CurvMultiConfig`、`ILTIterationRecord`、`LevelSetConfig`、`SimpleILTConfig`、`SimpleILTResult`、`optimize`、`optimize_curvmulti`、`optimize_levelset`、`signed_distance_initialization` |
| `opc.iteration.mbopc` | `IterationRecord`、`SimpleMBOPCConfig`、`SimpleMBOPCResult`、`optimize` |
| `opc.iteration.diffopc` | `DiffOPCConfig`、`DiffOPCIterationRecord`、`DiffOPCResult`、`optimize`；owner-only、逐 batch backward、拓扑屏障已验收 |
| `lithography` | `ICCAD13Config`、`ICCAD13Lithography`、`ProcessCondition` |
| `evaluation` | `EPEEvaluation`、`estimate_rectangular_shots`、`evaluate_binary_l2`、`evaluate_edge_probes`、`evaluate_pvband` |
