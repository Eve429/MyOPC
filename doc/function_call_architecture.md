# MyOPC 函数调用关系与数据流

## 1. 文档用途

本文档回答四个问题：

1. 从 `run_mbopc_frontend.py` 开始，一次完整运行会调用哪些函数？
2. 哪些函数每个任务只执行一次，哪些函数会在 OPC 优化迭代中重复执行？
3. `layout`、`geometry`、`opc.input` 和 `opc.input.edge` 之间通过什么数据对象衔接？
4. 未来增加 ILT、光学模型、MB-OPC solver 或拓扑安全检查时，应该接在哪一层？

图中实线箭头表示“直接调用”，虚线箭头表示“主要数据传递”或“可选调用”。

## 2. 分层依赖关系

```mermaid
flowchart TB
    CLI["run_mbopc_frontend.py<br/>参数解析与流程编排"]
    MB["opc.input.edge<br/>分段、归属、更新载体、重建"]
    COMMON["opc.input<br/>物理 mask、core、通用输入契约"]
    ITERATION["opc.iteration.&lt;method&gt;<br/>未来具体优化迭代"]
    LITHOGRAPHY["lithography<br/>未来光刻模型"]
    EVALUATION["evaluation<br/>未来评估指标"]
    GEOMETRY["geometry<br/>Region、轮廓、边、Patch、栅格"]
    LAYOUT["layout<br/>GDS/OASIS、层级、ROI、Layer"]
    KLAYOUT["KLayout C++ API"]
    NUMPY["NumPy 批量数组"]

    CLI --> MB
    CLI --> COMMON
    CLI --> GEOMETRY
    CLI --> LAYOUT
    MB --> COMMON
    MB --> GEOMETRY
    MB --> LAYOUT
    COMMON --> GEOMETRY
    COMMON --> LAYOUT
    GEOMETRY --> LAYOUT
    LAYOUT --> KLAYOUT
    GEOMETRY --> KLAYOUT
    MB --> NUMPY
    COMMON --> NUMPY
    ITERATION -.-> MB
    ITERATION -.-> COMMON
    ITERATION -.-> LITHOGRAPHY
    ITERATION -.-> EVALUATION
```

这是单向依赖：

- `layout` 不知道任何 OPC 方法。
- `geometry` 只依赖 `layout` 的坐标、Layer 和批次契约。
- `opc.input` 可供 MB-OPC、ILT 和后续方法共用，不导入边段输入或具体迭代方法。
- `opc.input.edge` 向下复用通用输入层，不把自己的位移/重建规则泄漏到 `geometry`。
- `opc.iteration.<method>` 可独立更换，并组合顶层 `lithography` 与 `evaluation`；三个目录当前均不含实现文件。
- CLI 是流程编排者，不是 solver 公共 API。库调用方应从 `prepare_problem` 开始。

## 3. 一次完整主程序调用

```mermaid
flowchart TD
    MAIN["main(argv)"] --> PARSER["build_parser() / parse_layer()"]
    PARSER --> RUN["run(args)"]

    RUN --> SOURCE{"layout 参数是否存在？"}
    SOURCE -->|"否"| DEMO["_demo_batch()"]
    SOURCE -->|"是"| OPEN["LayoutDB.open()"]
    OPEN --> LOAD["_load_database_batch()"]
    LOAD --> QUERY["LayoutDB.query()"]
    QUERY --> MATERIALIZE["ShapeQuery.materialize()"]

    DEMO --> PREPARE_ADAPTER["_prepare_input_problem()"]
    MATERIALIZE --> PREPARE_ADAPTER
    PREPARE_ADAPTER --> PREPARE["prepare_problem()"]

    PREPARE --> DEMO_UPDATE["_demo_updates()"]
    DEMO_UPDATE --> MERGE["merge_owner_updates()"]
    MERGE --> MATERIALIZE_SEG["SegmentBatch.materialize()"]
    MATERIALIZE_SEG --> SAMPLE["sample_lines()"]
    MERGE --> RECON_REF["reconstruct_region(零位移)"]
    MERGE --> RECON_MOVED["reconstruct_region(当前位移)"]

    RECON_MOVED --> PATCH["PatchSet.add() / region()"]
    RECON_MOVED --> NPZ["save_problem_npz()"]
    RECON_REF --> GDS["write_debug_gds()"]
    RECON_MOVED --> GDS
    RECON_MOVED --> PNG["render_boundary_overlay()"]
    SAMPLE --> PNG
    RUN -.-> SUITE["run_geometry_suite()\n可选"]

    PATCH --> SUMMARY["summary.json"]
    NPZ --> SUMMARY
    GDS --> SUMMARY
    PNG --> SUMMARY
    SUITE --> SUMMARY
    SUMMARY --> PRINT{"--json？"}
    PRINT -->|"是"| JSON_OUT["JSON 终端输出"]
    PRINT -->|"否"| TEXT_OUT["print_text()"]
```

### 3.1 真实版图的生命周期

`LayoutDB.open()` 到 `prepare_problem()` 必须处于同一个 `with` 上下文中：

```text
LayoutDB 打开
  └─ ShapeQuery.materialize()
       └─ prepare_problem()
            └─ 建立独立 PhysicalMask + NumPy 数组
LayoutDB 关闭
  └─ 后续迭代、重建和输出不再访问源版图
```

原因是 `ShapeQuery.materialize()` 返回的原生 Region 在准备阶段仍与 KLayout 数据库生命周期相关。`prepare_problem()` 完成物理合并和紧凑数组构建后，后续计算即可脱离源文件。

## 4. `prepare_problem()` 内部调用树

`prepare_problem()` 是最重要的库入口，每个 Layer/ROI/配置组合通常只执行一次。

```mermaid
flowchart TD
    PREPARE["prepare_problem(batch, layer, config, cores, ...)"]

    PREPARE --> MASK["normalize_physical_mask()"]
    MASK --> REGION["Region.dup() / remove_properties() / merged()"]
    MASK --> CONTOUR["extract_contours()"]
    CONTOUR --> CONTOUR_BATCH["ContourBatch"]
    MASK --> EDGE["extract_edges()"]
    EDGE --> EDGE_BATCH["EdgeBatch"]
    CONTOUR_BATCH -.-> PHYSICAL["PhysicalMask"]
    EDGE_BATCH -.-> PHYSICAL

    PREPARE --> FRAGMENT["fragment_edges()"]
    PHYSICAL -.-> FRAGMENT
    FRAGMENT --> EDGE_KEY["_edge_keys() -> _splitmix64()"]
    FRAGMENT --> NORMAL["_outward_normals()"]
    FRAGMENT --> SEGMENT_BATCH["SegmentBatch.__post_init__()"]
    SEGMENT_BATCH --> LOOKUP_INDEX["排序 key token 索引"]

    PREPARE --> POLICY["OwnershipPolicy.assign()"]
    SEGMENT_BATCH -.-> POLICY
    POLICY --> POLICY_TYPE{"core 类型"}
    POLICY_TYPE -->|"RectilinearCoreGrid"| GRID_MEMBER["_grid_membership()"]
    POLICY_TYPE -->|"CoreSpec 列表"| EXPLICIT_MEMBER["_explicit_membership()"]
    GRID_MEMBER --> OWNERSHIP["OwnershipBatch"]
    EXPLICIT_MEMBER --> OWNERSHIP

    PREPARE --> TEMPLATE["build_sample_template()"]
    TEMPLATE --> SAMPLE_TEMPLATE["BoundarySampleTemplate"]

    PHYSICAL -.-> PROBLEM["MBOPCProblem"]
    SEGMENT_BATCH -.-> PROBLEM
    OWNERSHIP -.-> PROBLEM
    SAMPLE_TEMPLATE -.-> PROBLEM
```

### 4.1 物理 mask 准备

| 函数 | 直接调用 | 输出 | 设计意图 |
|---|---|---|---|
| `normalize_physical_mask` | `RegionBatch.region`、KLayout `merged`、`extract_contours`、`extract_edges` | `PhysicalMask` | 消除内部 cut-line，恢复 hull/hole，不修改输入批次 |
| `extract_contours` | `_extract_layer` | `ContourBatch` | 把 Region 一次性转成 CSR 式轮廓数组 |
| `extract_edges` | NumPy 闭环索引 | `EdgeBatch` | 每个轮廓顶点对应一条有向数学边 |

### 4.2 边段准备

| 函数 | 直接调用 | 输出 | 设计意图 |
|---|---|---|---|
| `fragment_edges` | `_edge_keys`、`_outward_normals`、`SegmentBatch(...)` | `SegmentBatch` | 用 edge ID + `t0/t1` 保存分段，不常驻重复端点 |
| `_edge_keys` | `_splitmix64` | 每条数学边 128-bit key | key 只依赖 Layer 和有向端点，跨进程稳定 |
| `_outward_normals` | NumPy 环面积/方向计算 | edge 单位外法向 | hull 和 hole 都指向材料外的空区 |
| `SegmentBatch.__post_init__` | `_vector`、`_matrix`、`argsort` | 规范化数组 + key 查找索引 | 查找索引只构建一次，多轮复用 |

### 4.3 owner 与 halo context

| 函数 | 调用条件 | 输出 |
|---|---|---|
| `MidpointOwnerPolicy.assign` | `RectilinearCoreGrid` 进入快速路径 | `_grid_membership` 的 `OwnershipBatch` |
| `_grid_membership` | 规则 x/y cuts | 每段唯一 owner + 按 core 排列的 CSR membership |
| `_explicit_membership` | 少量不规则 `CoreSpec` | 显式 core 的 owner/context membership |
| `_validate_explicit_cores` | `_explicit_membership` 调用 | 拒绝空列表、重复 ID 和正面积 ownership 重叠 |

`OwnershipBatch.owner_indices` 与 `member_segment_indices` 含义不同：

- `owner_indices[segment]`：谁有权修改这个 segment，必须唯一。
- `segments_for_core(core)`：该 core 计算光学上下文时能看见哪些 segment，可在多个 core 中重复。

## 5. 优化迭代热路径

真正的 MB-OPC solver 尚未实现，但前端已将每轮需要的调用压缩到下图的数组操作。

```mermaid
flowchart LR
    SOLVER["未来 solver<br/>按 core 生成更新"]
    UPDATE_BATCH["SegmentUpdateBatch[]"]
    MERGE["merge_owner_updates()"]
    LOOKUP["SegmentBatch.lookup_keys()"]
    UPDATE_RESULT["UpdateResult<br/>displacements + dirty IDs"]
    MATERIALIZE["SegmentBatch.materialize()"]
    GEOMETRY["SegmentGeometry"]
    SAMPLE["sample_lines()"]
    SAMPLE_BATCH["BoundarySampleBatch"]
    OPTICAL["未来光学模型/损失函数"]

    SOLVER --> UPDATE_BATCH --> MERGE
    MERGE --> LOOKUP --> UPDATE_RESULT
    UPDATE_RESULT --> MATERIALIZE --> GEOMETRY
    GEOMETRY --> SAMPLE --> SAMPLE_BATCH
    SAMPLE_BATCH -.-> OPTICAL
    OPTICAL -.-> SOLVER
```

### 5.1 `merge_owner_updates()`

`merge_owner_updates()` 内部调用 `SegmentBatch.lookup_keys()`，然后依次检查：

1. key 是否存在。
2. 提交者 core 是否在范围内。
3. 提交者是否是该 segment 的唯一 owner。
4. 同一轮是否对同一 segment 重复提交。
5. 绝对位移是否超过 `max_displacement_dbu`。

返回的 `UpdateResult` 包含：

- `displacements`：与全局 segment 对齐的绝对法向位移。
- `changed_segment_indices`：当轮变化的 segment。
- `dirty_polygon_ids`：可供后续增量重建/光学缓存失效使用。

### 5.2 `SegmentBatch.materialize()`

```text
edge_starts + (edge_ends - edge_starts) * t0/t1
  └─ + edge_normal * displacement
       └─ SegmentGeometry(starts, ends, normals, lengths)
```

调用时可传 `indices`，只物化某个 core 或某些 dirty segment；不传时物化全部 segment。

### 5.3 `sample_lines()`

`build_sample_template()` 在准备阶段预先生成 `line_indices`、切向位置和法向偏移。`sample_lines()` 每轮只做 `take` 和 NumPy 广播，并可重用调用方提供的 `out` 缓冲区。

> 当前采样函数只按法向偏移坐标，不检查采样点是否穿过对面边界。例如 2 nm 线宽配置 8 nm 内偏移时，需要未来的局部间距/有效性层介入。

## 6. 轮廓重建调用链

```mermaid
flowchart TD
    REGION["reconstruct_region()"]
    CONTOURS["reconstruct_contours()"]
    VALIDATE_D["_validated_displacements()"]
    MATERIALIZE["SegmentBatch.materialize(displacements)"]
    JUNCTION["同边 jog / 拐角 miter / bevel"]
    CONTOUR_BATCH["ContourBatch"]
    VALIDATE_C["validate_contours()"]
    TO_REGION["contours_to_region()"]
    NATIVE_VALID["Region.has_valid_polygons()"]
    RESULT["kdb.Region"]

    REGION --> CONTOURS
    CONTOURS --> VALIDATE_D
    CONTOURS --> MATERIALIZE
    MATERIALIZE --> JUNCTION
    JUNCTION --> CONTOUR_BATCH
    CONTOUR_BATCH --> VALIDATE_C
    CONTOUR_BATCH --> TO_REGION
    TO_REGION --> NATIVE_VALID --> RESULT
```

`reconstruct_contours()` 的 junction 规则：

- 同一数学边、位移相同：不输出内部分割点，避免斜边 DBU 取整毛刺。
- 同一数学边、位移不同：输出前段终点和当前段起点，形成 jog。
- 不同数学边：优先使用解析交点 miter；平行或 miter 过长时使用 bevel。

### 6.1 当前拓扑验证的边界

`validate_contours()` 当前检查零长边、零面积环和每个 polygon 的唯一 hull；`has_valid_polygons()` 检查 KLayout 是否能表示输出 Polygon。它们尚不保证：

- hole 始终完整位于原 hull 内。
- 外边不会越过内边。
- 矩形左边不会越过右边。
- 新旧轮廓的拓扑关系不会改变。

因此未来几何安全层应该接在下列位置，而不是塞入 `layout` 或基础 `geometry` 查询中：

```mermaid
flowchart LR
    MERGE["merge_owner_updates()"] --> GUARD["未实现：<br/>MB 位移可行性/碰撞检查"]
    GUARD -->|"接受/限幅"| MATERIALIZE["materialize() / sample_lines()"]
    GUARD -->|"拒绝"| SOLVER["solver 缩小步长"]
    MATERIALIZE --> RECONSTRUCT["reconstruct_region()"]
    RECONSTRUCT --> POST_GUARD["未实现：<br/>环自交、绕向、hole 包含验证"]
```

## 7. 输出与诊断调用关系

```mermaid
flowchart LR
    REF["参考 Region"]
    MOVED["重建 Region"]
    PROBLEM["MBOPCProblem"]
    DISPLACEMENT["displacements"]
    GEOMETRY["SegmentGeometry"]
    SAMPLES["BoundarySampleBatch"]

    MOVED --> PATCH["PatchSet.add()"]
    PATCH --> STITCH["PatchSet.region()"]
    STITCH --> XOR["跨 core XOR 验证"]

    PROBLEM --> NPZ["save_problem_npz()"]
    DISPLACEMENT --> NPZ
    REF --> GDS["write_debug_gds()"]
    MOVED --> GDS
    MOVED --> PNG["render_boundary_overlay()"]
    GEOMETRY --> PNG
    SAMPLES --> PNG
    PNG --> RASTER["geometry.render_region_batch()"]

    NPZ --> FILE1["segments.npz"]
    GDS --> FILE2["reconstruction.gds"]
    PNG --> FILE3["overview.png"]
```

| 函数 | 是否进入 solver 热路径 | 用途 |
|---|---|---|
| `PatchSet.add` / `region` | 否 | 按 ownership box 精确裁剪并验证跨 core 拼接 |
| `save_problem_npz` | 否 | 保存可复现的纯数值调试状态 |
| `write_debug_gds` | 否 | 把参考/重建 mask 分别写入两个顶层 Cell |
| `render_boundary_overlay` | 否 | 绘制 mask、segment owner、core、法向和采样点 |
| `run_geometry_suite` | 否 | 运行确定性多图形零位移验证并生成图集 |

## 8. 核心数据对象关系

```mermaid
flowchart LR
    RB["RegionBatch<br/>某 Layer/ROI 的原生 Region"]
    PM["PhysicalMask<br/>Region + ContourBatch + EdgeBatch"]
    SB["SegmentBatch<br/>edge ID + t0/t1 + normals + keys"]
    OB["OwnershipBatch<br/>owner + CSR memberships"]
    ST["BoundarySampleTemplate<br/>不含当前坐标"]
    PROBLEM["MBOPCProblem<br/>多轮复用的参考问题"]
    UB["SegmentUpdateBatch[]<br/>各 core 提交"]
    UR["UpdateResult<br/>位移 + dirty IDs"]
    SG["SegmentGeometry<br/>当前坐标"]
    BS["BoundarySampleBatch<br/>当前采样坐标"]
    CR["ContourBatch / kdb.Region<br/>当前重建结果"]

    RB --> PM --> SB
    SB --> OB
    SB --> ST
    PM --> PROBLEM
    SB --> PROBLEM
    OB --> PROBLEM
    ST --> PROBLEM
    PROBLEM --> UB --> UR
    PROBLEM --> SG
    UR --> SG --> BS
    UR --> CR
```

### 8.1 参考态与迭代态

| 类别 | 数据 | 生命周期 |
|---|---|---|
| 固定参考态 | `PhysicalMask`、`SegmentBatch`、`OwnershipBatch`、`BoundarySampleTemplate` | 任务准备后复用到优化结束 |
| 每轮变化态 | `displacements`、`UpdateResult` | 每次 solver 更新 |
| 按需物化态 | `SegmentGeometry`、`BoundarySampleBatch` | 光学评估或输出时创建，可复用缓冲区 |
| 最终/调试态 | `ContourBatch`、`kdb.Region`、NPZ/GDS/PNG | 最终输出或按需诊断 |

`MBOPCProblem` 使用 frozen dataclass 固定结构关系，但 NumPy 数组底层仍然是可变内存。上层 solver 应把 `problem` 内的参考数组当作只读数据，把所有优化状态放在独立 `displacements` 中。

## 9. 调用频率与性能重点

| 频率 | 函数 | 性能策略 |
|---|---|---|
| 每文件一次 | `LayoutDB.open` | 只解析一次 GDS/OASIS，保留层级 |
| 每 Layer/ROI 一次 | `ShapeQuery.materialize` | KLayout C++ 递归迭代，不在 Python 中 flatten |
| 每问题一次 | `normalize_physical_mask` | 合并、轮廓和边界缓存 |
| 每问题一次 | `fragment_edges` | 批量分段、法向和 key 构建 |
| 每问题一次 | `OwnershipPolicy.assign` | 规则网格不构建 segment×core 稠密矩阵 |
| 每迭代 | `merge_owner_updates` | 复用排序 key 索引，不建 Python dict |
| 每光学评估 | `SegmentBatch.materialize` | 端点/法向按需物化，可只取 core/dirty 子集 |
| 每光学评估 | `sample_lines` | 重用模板和输出缓冲区 |
| 最终或按需 | `reconstruct_region` | 生成完整轮廓，不必放入每次光学评估 |
| 调试或交付 | NPZ/GDS/PNG 函数 | 不进入 solver 热路径 |

## 10. 如何接入真正的 MB-OPC solver

下面是调用顺序示意，不是已实现的 solver 代码：

```python
# 任务级：只准备一次。
problem = prepare_problem(batch, layer, config, core_grid)
displacements = np.zeros(problem.segments.segment_count)
sample_buffer = np.empty((len(problem.sample_template.line_indices), 2))

for iteration in range(max_iterations):
    # 只物化当前几何和采样点，不重做版图查询、合并、分段和 owner 分配。
    geometry = problem.segments.materialize(displacements)
    samples = sample_lines(geometry.starts, geometry.ends, geometry.normals,
                           problem.sample_template, sample_buffer)
    loss, core_updates = optical_model_and_optimizer(samples, problem.ownership)

    # 未来应在这里插入位移可行性/碰撞限幅。
    update_result = merge_owner_updates(problem, core_updates, displacements)
    displacements = update_result.displacements

final_region = reconstruct_region(problem.segments, displacements, problem.config)
```

实际开发时，`optical_model_and_optimizer` 不应被放入 `opc.input`；具体循环应位于 `opc.iteration.<method>`，并通过顶层 `lithography`、`evaluation` 组合模型与指标，再用 `MBOPCProblem`、`BoundarySampleBatch` 和 `SegmentUpdateBatch` 与输入前端交互。

## 11. ILT 和其他方法的复用路径

```mermaid
flowchart LR
    BATCH["RegionBatch"] --> MASK["normalize_physical_mask()"]
    MASK --> COMMON_EDGE["ContourBatch / EdgeBatch"]
    GRID["RectilinearCoreGrid"] --> METHOD["ILT / 其他 OPC 方法"]
    COMMON_EDGE --> METHOD
    TEMPLATE["build_sample_template() / sample_lines()"] --> METHOD
    METHOD --> VIS["render_boundary_overlay()\n可选诊断"]

    MASK -.-> MB_FRAGMENT["fragment_edges()\n仅 MB-OPC 需要"]
    MB_FRAGMENT -.-> MB_RECON["reconstruct_region()\n仅 MB-OPC 需要"]
```

ILT 如果使用像素 mask，可以在 `PhysicalMask.region` 上调用 `geometry.render_region_batch()`；如果使用边界评估，可直接复用 `ContourBatch`、`EdgeBatch` 和通用采样。它不应为了复用这些能力而依赖 `SegmentBatch` 或 MB 轮廓重建。

## 12. 源码导航

| 主题 | 文件 |
|---|---|
| 直接运行与完整编排 | [`run_mbopc_frontend.py`](../run_mbopc_frontend.py) |
| 可复用准备入口 | [`opc/input/edge/builder.py`](../opc/input/edge/builder.py) |
| 物理 mask | [`opc/input/mask.py`](../opc/input/mask.py) |
| core 和通用输入数据契约 | [`opc/input/types.py`](../opc/input/types.py) |
| 采样物化 | [`opc/input/edge/sampling.py`](../opc/input/edge/sampling.py) |
| 控制段切分 | [`opc/input/edge/fragmentation.py`](../opc/input/edge/fragmentation.py) |
| 控制段、归属和更新数据契约 | [`opc/input/edge/types.py`](../opc/input/edge/types.py) |
| owner 和 halo membership | [`opc/input/edge/ownership.py`](../opc/input/edge/ownership.py) |
| owner-only 更新合并 | [`opc/input/edge/updates.py`](../opc/input/edge/updates.py) |
| 轮廓重建 | [`opc/input/edge/reconstruction.py`](../opc/input/edge/reconstruction.py) |
| NPZ/GDS 产物 | [`opc/input/edge/artifacts.py`](../opc/input/edge/artifacts.py) |
| 标注可视化 | [`opc/input/edge/visualize.py`](../opc/input/edge/visualize.py) |
| 多图形验证套件 | [`opc/input/edge/verification.py`](../opc/input/edge/verification.py) |
| 层级版图生命周期 | [`layout/database.py`](../layout/database.py) |
| ROI 局部物化 | [`layout/query.py`](../layout/query.py) |
| Region/轮廓互转 | [`geometry/contour.py`](../geometry/contour.py) |
| 轮廓提边 | [`geometry/edge.py`](../geometry/edge.py) |
| 轮廓基础验证 | [`geometry/validate.py`](../geometry/validate.py) |
| core Patch 裁剪与拼接 | [`geometry/patch.py`](../geometry/patch.py) |

## 13. 建议阅读顺序

1. 先阅读第 2、3 节，了解分层和完整流程。
2. 阅读第 4 节，理解为什么 `prepare_problem()` 是架构中心。
3. 阅读第 5 节，理解多轮优化时哪些数据保持不变。
4. 阅读第 6 节，理解轮廓重建和当前拓扑安全边界。
5. 开发 solver 时使用第 10 节作为调用骨架，并在位移合并后插入几何可行性层。
