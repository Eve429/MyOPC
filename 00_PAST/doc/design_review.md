# MyOPC 架构评审：过度设计与缺失能力

## 0. 复核更新（2026-08-09）：代码已按本评审推进

用户在本评审之后实施了一轮重构，沿 design review 的方向落地了大量工作。本节记录各条目的**处置状态**与复核中新发现的问题；**原文第 1–7 节为评审时（代码变更前）的判断，保留作历史记录**，其中的“文件:行号”引用对应变更前代码。本节引用的 commit 均为本地 `master`。复核时质量门全绿：`ruff`（限定范围）、`compileall`、`pytest`（114 项）均通过。

### 0.1 处置状态总表

| 评审条目 | 现状 | 处置方式 / 证据 |
|---|---|---|
| 3.1 `GeometryEngine` | **已删除** | `geometry/region.py` 整体移除，`geometry/__init__.py` 不再导出；`layout/types.py` 的 `backend` 硬编码字段同步删除（`6a2f353`） |
| 3.2 `UniformGridIndex` | **已删除** | `geometry/spatial.py` 与 `tests/geometry/test_spatial.py` 移除（`6a2f353`） |
| 3.3 `OwnershipPolicy` Protocol + 显式 core | **已删除** | 收敛为单一 `build_ownership`；`prepare_problem` 第 4 参由 `cores`+`ownership_policy` 改为可选 `grid: RectilinearCoreGrid 或 None`，默认用 `query_box` 建单 core、走与多 core 完全相同的规则网格代码，消除第二套显式 core 校验语义（`09f898f`，`builder.py:15`） |
| 3.4 128-bit key + 每轮 argsort | **已移出热路径** | `SegmentBatch` 不再持有 `lookup_keys`/token；`splitmix64`/`merge_owner_updates`/`updates.py` 删除；solver 改为预分配 `next_values` + `written` 布尔位直接 scatter，正落地 AGENTS.md 第 20 行“不得每轮全局排序”（`09f898f` + `7485204`） |
| 4.1 分块/流式编排 | **部分处置** | 流式落在**求解器评价层**：`optimize()`（`opc/iteration/mbopc/solver.py`）按 `batch_size` 流式取 core、`_current_tile` 用 `(mask − 参考) + 当前` 局部差分不重建整张 reticle Region。但 `prepare_problem`→`normalize_physical_mask` **仍是单批单层一次性全局 `merged()`**（`mask.py:34`），“整 reticle 装不进内存”的硬约束在**前端物化层尚未流式**（`7485204`） |
| 4.2 跨 tile `d_current`/`d_next` + owner-only 写 | **核心语义已落地（部分）** | solver 内 `current`/`next_values` 为全局数组，owner 唯一写、运行时 `written` 重复写硬失败、整轮屏障后才 `current=next_values`，与 AGENTS.md 第 18 行一致。**未达**：仍是单进程单次 `optimize()` 内的“逻辑跨 tile”，无跨调用/跨进程持久位移存储，去重用布尔数组而非 epoch 号（`7485204`） |
| 4.3 几何安全/位移可行性 | **部分处置** | `_preserves_reference_topology`（环数/polygon_id/is_hole 不变 + 有符号面积不翻转 + 每个 hole 仍被原 hull 包含）在发布屏障前检查，非法则**整轮回滚**（`a822efe`）。**未覆盖**：最小宽度、环自交；矩形左右穿越仅以“绕向翻转”间接覆盖 |
| 4.4 采样有效性 | **基本处置（探针级）** | 求解器评价改用 `edge_probe_points` 的 inner/outer 探针；`evaluate_edge_probes` 以 `target_inner`/`target_outer` 判定探针是否穿过对侧边界并标记 `valid`，无效探针不计入方向（`test_two_dbu_hollow_wall_invalid_long_edge_probes...` 验证 2 DBU 壁 + 8 DBU 探针距） |
| 4.5 受控 remesh API | **未处置** | `SegmentBatch` 仍由 `fragment_edges` 一次性生成，无重切入口 |
| 4.6 层级 source 追溯与输出复用 | **未处置** | 物化仍把 occurrence 扁平化到 top 全局坐标，丢失 cell/instance 信息 |
| 4.7 求解器/模型/指标 | **已落地（首版）** | `opc/iteration/mbopc/`（流式 simple solver）+ `lithography/iccad13.py`（OpenILT ICCAD13 Hopkins 三工艺角）+ `evaluation/metrics.py`（L2/PVBand/EPE），三者经 `MBOPCProblem` 与输入前端交互，未下沉进 `opc.input`（`7485204` + `6cf885a`） |
| 4.8 checkpoint 读回端 | **未处置** | NPZ 写出迁移至 `opc/diagnostics.py:save_problem_npz`，仍无加载器/续算路径 |
| 4.9 多层假设 | **未处置** | `prepare_problem` 仍接受单个 `layer` |
| 3.5 `HierarchySummary` planner | **已处置** | 删除独立模块和统计结构体；当前 `LayoutDB.cell_hierarchy()` 只返回完整 Cell DAG 邻接字典，不展开 occurrence |

### 0.2 复核新发现（本轮新增，原文档未涉及）

**A. 迭代语义 off-by-one（待确认）【中】**
`solver.optimize` 主循环在末轮 `if iteration == config.iterations − 1: break`（`solver.py:312`）**早于**提交语句 `current = next_values`（`:319`），导致最后一轮算出的位移永远不提交、也不进入 `best_displacements` 候选。后果：`iterations=1` 提交 **0** 次位移（仅评估参考态），`iterations=N` 最多提交 **N−1** 次。`best_displacements = current.copy()`（`:287`）本身正确（score 反映被评估的 `current`），问题纯粹是末轮 `next_values` 被丢弃。建议二选一：(a) 文档明确“iterations = 评估轮数，返回最佳已评估态”；(b) 末轮也做拓扑检查并提交，使 `iterations=N` 实际应用 N 步。属语义选择，需用户拍板。

**2026-08-12 处置**：用户确认按更新次数修复。当前 `iterations=N` 最多发布 N 次合法更新，评价初态及每次发布后状态；完整执行时产生 N+1 条状态记录，最后记录不再提出候选。末次更新执行同一全局拓扑守卫，最佳位移只来自已评价状态。原段落保留为历史问题说明，不再代表当前行为。

**B. ICCAD13 FFT 归一化是移植不变量（移植脆弱点）【中】**
`lithography/iccad13.py` 的 `_aerial` 对 `fft2`/`ifft2` **均**用 `norm="forward"`（`:171`/`:172`），两者叠加使 aerial 相对标准卷积多一个 1/N 因子。当前自洽**完全依赖**随附的 OpenILT `.pt` scale 在同一约定下生成。建议加一条“对已知 OpenILT 参考 aerial image 校验绝对强度”的锚定断言，防止日后重生成资产或改 `norm=` 时静默失真。非 bug，是移植校准脆弱点。

**C. 拓扑 hole 检查是每轮 Python+KLayout 循环【低】**
`_preserves_reference_topology`（`solver.py:167-185`）对每个 hole ring 单独建 `kdb.Region` 做 `(hole − hull).is_empty()`。当前仅带孔 polygon 才进入此路径，可接受；若将来出现密集 hole 版图，该检查会成为每轮热点。代码注释已标注为第一版。

### 0.3 复核结论

- **过度设计（第 3 节）已清零**：3.1/3.2/3.3 删除，3.4 移出热路径——`AGENTS.md`“新抽象必须有当前调用方”的违规全部消除。
- **缺失（第 4 节）过半落地**：4.7（本体）、4.1/4.2（流式 + owner/barrier 核心语义）、4.4（探针有效性）已处置或部分处置；4.3 拓扑守卫部分到位。
- **仍待推进**：4.5 remesh、4.6 层级追溯/复用、4.8 checkpoint 读回为完全未决；4.1 的前端物化层流式、4.2 的跨进程持久状态、4.3 的最小宽度/自交为部分项的剩余部分。
- 新发现 **A 需用户决策**，**B/C** 为记录性提示。

## 1. 评审范围与方法

本文评审 `layout/`、`geometry/`、`opc/input/`（含 `opc/input/edge/`）三个基础包，判断在“**完整 reticle 的 OPC 处理**”这一最终目标下，哪些设计属于过度设计，哪些能力缺失、需要被增加。评审结论与 `AGENTS.md` 的“未来优化内容”路线图逐条对照（见第 5 节），以区分“我自己独立得出的判断”与“项目已声明但尚未实现的方向”。

评审基于完整源码阅读（非仅接口），并辅以全仓库调用点检索验证“无调用方”的断言。每条结论标注置信度（高/中/低）与依据（`文件:行号`）。本文仅为评审意见，不构成对受保护目录 `layout/`、`geometry/` 的修改授权——任何涉及这两个目录的处置仍须按 `AGENTS.md` 取得逐次确认。

## 2. 结论速览

| 类别 | 条目 | 置信度 | 一句话 |
|---|---|---|---|
| 过度设计 | `GeometryEngine` 后端门面 | 高 | OPC 全程不经过它，仅在 layout/geometry 演示与自测中使用 |
| 过度设计 | `UniformGridIndex` 空间索引 | 高 | 零生产调用方，仅被自身单元测试引用 |
| 过度设计 | `OwnershipPolicy` Protocol + 显式 core 路径 | 中 | 只有一个实现，显式路径基本只服务单 core 默认值 |
| 过度设计（前置投资） | 128 位稳定 key 的“跨进程/checkpoint”用途 | 中 | 机制在进程内被用，但所述跨进程/checkpoint 场景尚不存在 |
| 已处置的过度设计 | `HierarchySummary` 的 planner 用途 | 高 | 已删除专用结构体与统计字段，收敛为 `LayoutDB.cell_hierarchy()` 普通邻接字典 |
| 缺失 | 分块/流式编排层 | 高 | `prepare_problem` 是单批单层一次性管线，无法装下整张 reticle |
| 缺失 | 跨 tile 共享 `d_current`/`d_next` + owner-only 写入 | 高 | 位移是单次 `run()` 内的局部数组，无跨 tile 状态 |
| 缺失 | 几何安全 / 位移可行性层 | 高 | 自交、hole 包含、最小宽度、左右穿越均未保证 |
| 缺失 | 采样有效性层 | 中 | 法向偏移可穿越对侧边界，已知未实现 |
| 缺失 | 受控 remesh API | 高 | 分段一次性生成，无重切入口 |
| 缺失 | 层级 source 追溯与层级输出复用 | 高 | 物化即扁平化到全局坐标，丢失 cell/instance 信息 |
| 缺失（已声明占位） | `lithography` / `evaluation` / `opc.iteration` | 高 | 真正的求解器/模型/指标，是“完整 reticle”的直接阻塞项 |
| 缺失 | checkpoint 的读回端 | 中 | 有 NPZ 写出，无加载与续算 |

## 3. 过度设计

> 判据：抽象为其所述扩展目的而建，但当前无调用方，或调用方仅为演示/自测；与 `AGENTS.md`“新抽象必须有当前调用方”的规则相悖。

### 3.1 `GeometryEngine`：未被产品路径使用的“可替换后端”门面【高】

- 位置：`geometry/region.py:14`，注释自称“无状态几何门面，保持原生调用为粗粒度且便于后续替换后端”。
- 调用点检索结果：`GeometryEngine` 及其 `union/combine/intersection/difference/xor/offset/merge/clip` 方法仅出现在 `run_layout_geometry.py:77`、`benchmarks/benchmark_layout_geometry.py`、`tests/geometry/test_region.py`。**OPC 前端（`opc/input/edge/*` 与 `run_mbopc_frontend.py`）完全不经过它**。
- OPC 实际怎么做的：`mask.py:34` 直接 `region.merged()`；`patch.py:37` 直接 `region & clip`；`run_mbopc_frontend.py:258/272` 直接 `reference ^ ...`。也就是说“门面”抽象被绕过，几何操作散落在各处以原生 `kdb.Region` 调用完成。
- 进一步看“可替换后端”这个理由本身不成立：`RegionBatch.backend == "klayout"` 是硬编码字段（`layout/types.py:132`），`_check_backend`（`region.py:93`）只是拒绝其他字符串，全链路对 `kdb.Region`/`kdb.Box` 强耦合。
- **公平性补充**：`GeometryEngine` 的布尔运算方法本身是有用的工具，未来 ILT 或其他方法（架构文档 §11）确实可能复用。因此“过度设计”的指控应**收窄到两点**：（a）其自称的“可替换后端”理由不成立、且被硬编码耦合证伪；（b）它不在当前 OPC 产品路径上，却与 OPC 内联的原生调用构成第二套并列入口。并非主张“整类方法都是废代码”。
- **判断**：典型的“为假设中的多后端预先抽象”——既无第二后端，也未在产品路径上提供价值，反而制造了第二套几何入口（门面 vs 原生直调），增加阅读与维护成本。
- **处置建议**：因位于受保护的 `geometry/`，不擅自改动。建议优先去掉“可替换后端”的措辞与 `_check_backend`/`backend` 字段所暗示的多后端语义；方法本身保留为可选工具，待确有第二调用方（如 ILT）再决定是公共门面还是内联回原生。需用户确认。

### 3.2 `UniformGridIndex`：零生产调用方的空间索引【高】

- 位置：`geometry/spatial.py:16`，含超长边单独建表（`:37`）、`oversized_count`、`query_box`、`query_radius`（`:67`）等完整功能。
- 调用点检索结果：除自身定义与 `geometry/__init__` 导出外，**仅 `tests/geometry/test_spatial.py` 引用**。无任何生产管线、演示脚本或基准调用。
- 为什么 OPC 不需要它：halo context 归属已经在 `ownership.py:25` 的 `_grid_membership` 中用 `grid.locate_points` + 在 `x_cuts/y_cuts` 上 `searchsorted` 直接展开（`:38-55`），复杂度只随 halo 邻居数增长，不需要通用边空间索引。
- **判断**：这是最明确的一处过度设计——一个完整的、带内存上界处理的均匀网格索引，只有单元测试在用。它服务的“高频边邻域查询”场景在当前架构里已经被 ownership 的 CSR membership 取代。
- **处置建议**：等未来光学模型确实需要“任意边对边的邻近查询”（而非 core↔segment 的网格归属）时再启用；在此之前属于待激活资产。是否从公共导出中撤下，需用户确认（`geometry/` 受保护）。

### 3.3 `OwnershipPolicy` Protocol 与显式 core 路径【中】

- 位置：`ownership.py:16`（Protocol）、`:114`（`MidpointOwnerPolicy`，唯一实现）、`:78`（`_explicit_membership`）、`:64`（`_validate_explicit_cores`，O(C²) 重叠校验）；`builder.py:24` 暴露 `ownership_policy` 参数。
- 现状：只有一个策略 `MidpointOwnerPolicy`，且它内部按 `isinstance(cores, RectilinearCoreGrid)` 二选一。显式 `CoreSpec` 列表路径（含闭区间兜底归属 `:101-106` 与 O(C²) 校验）实际只被 `prepare_problem` 的“单 core 默认值”（`builder.py:29`）和测试驱动。
- `AGENTS.md` 路线图只描述了规则正交网格 + 流式 tile，**未提及显式不规则 core**。
- **判断**：Protocol 与显式路径是为“多种跨 core 协调策略”预留，但当前没有第二个策略，显式路径也没有真实工作流。属于“轻度过建”，并非纯冗余——参数化本身无害。
- **处置建议**：保留 `ownership_policy` 参数与 `MidpointOwnerPolicy`；在出现第二个真实策略前，可把 Protocol 收敛为函数约定，`_explicit_membership` 的闭区间兜底与 O(C²) 校验可作为“少量不规则 core”的明确语义保留，但不必扩展。

### 3.4 128 位稳定 key：机制被用，所述用途未被用【中，前置投资】

- 位置：`fragmentation.py:12`（`_splitmix64`）、`:22`（`_edge_keys`，128 位）、`opc/input/edge/types.py:197`（`lookup_keys`，searchsorted + token）、`:144`（token 碰撞硬失败）。
- 机制确实在热路径上被调用：`updates.py:30` 的 `merge_owner_updates` 用 `lookup_keys` 把提交的 key 映射回 segment 索引。
- 但“为什么用稳定 key”的理由，按 `doc/function_call_architecture.md` §9 原文是“**stable key 主要用于跨进程校验、checkpoint、恢复和诊断**”——而这些场景（跨进程、checkpoint、恢复）目前都不存在（见第 4 节“checkpoint 读回端缺失”）。当前是单进程、单 `run()`、用 `SegmentUpdateBatch` 携带 key 而非直接用 segment 索引。
- **判断**：这不是浪费——key 身份对未来层级复用、跨 tile 去重、续算确实关键，且后加成本高，属合理前置投资。但需如实标注：**目前 key 的复杂度（128 位 + token + 碰撞检测 + searchsorted）所服务的核心用途尚未落地**。在单进程求解器落地前，求解器与 `merge_owner_updates` 之间其实可以直接用 segment 索引通信，key 只在跨边界（NPZ/诊断/未来 checkpoint）出现。
- **处置建议**：保留 key 机制；但避免在尚未有跨进程/checkpoint 消费者时继续围绕它扩展（例如不要再为 key 增设注册器/索引服务）。

### 3.5 `HierarchySummary`：为不存在的 planner 而读【中，轻】

- 位置：`layout/hierarchy.py:25`（`HierarchySummary`）、`:33`（`build_hierarchy_summary`），`CellInfo` 注释“供诊断和后续 planner 使用”（`:16`）。
- 调用点：`LayoutDB.hierarchy_summary()`（`database.py:120`）被 `run_layout_geometry.py` 与 `tests/layout/test_database.py` 调用，**OPC 不读**。
- 这与第 4 节“层级未被利用”是一体两面：层级被**读取统计**了，却没有被**计算利用**。统计对象本身不算冗余，但其存在理由（planner）尚不存在。
- **处置建议**：保留对象；真正的处置是第 4.6 节——让 OPC 实际利用层级，而非删除统计。

**2026-08-13 处置**：用户确认当前只需要轻量完整 Cell 层级，不需要 bbox、实例统计或 planner 专用对象。已删除 `layout/hierarchy.py`、`CellInfo`、`HierarchySummary` 与旧方法，在 `LayoutDB` 内收敛为普通 `dict[str, tuple[str, ...]]` 邻接表。该变化只精简层级检查接口，不代表第 4.6 节的 source occurrence 追溯已经实现。

## 4. 缺失 / 需要增加

> 判据：面向“完整 reticle”目标，当前架构无法承载或存在已知正确性缺口的能力。

### 4.1 分块 / 流式编排层（最关键缺失）【高】

- `prepare_problem`（`builder.py:20`）签名是 `prepare_problem(batch, layer, config, cores, ...)`——**单批、单层、一次性**。`normalize_physical_mask`（`mask.py:34`）对整批做一次全局 `merged()`。
- 这等于把“整张 reticle 作为单个 Region 物化 + 全局合并 + 全局分段”写进了唯一入口。完整 reticle 在此路径下会：
  - 触发分段总数的 int32 上限保护（`fragmentation.py:70-72`，`> iinfo(int32).max` 直接抛错）——这其实是“强制你必须分块”的硬墙，但**目前没有分块路径**。
  - 在 `normalize_physical_mask` 的全局 `merged()` 上内存/时间爆炸。
- **需要增加**：在 `prepare_problem` 之上增加一个 tile 调度/流式编排层：按 tile 物化“tile + halo”局部 Region → 局部 `prepare_problem` → 局部求解 → 写回 → 推进。这正对应 `AGENTS.md` 的“流式 GPU batch / 大 reticle 普通轮次”。
- **注意**：这不应放进受保护的 `layout/` 或基础 `geometry/`；它是 `opc/` 层（或 `opc.iteration`）的编排职责。

### 4.2 跨 tile 共享位移状态 `d_current`/`d_next`【高】

- 当前位移是 `run()` 内的局部 numpy 数组（`run_mbopc_frontend.py` 里 `np.zeros(segment_count)`），生命周期等于一次演示运行。
- `AGENTS.md` 明确要求：“全部 tile 必须基于同一只读 `d_current`；owner 是 segment 的唯一写入者；非 owner core 只能通过 halo 读取或提交只读误差贡献；所有 tile 完成后才能发布 `d_next`”。当前没有任何对象承载这个跨 tile 状态。
- 同条还要求“预分配 `d_next` 并按 owner segment index 直接 scatter，使用 epoch/bitset 检测重复写入”。而现有 `merge_owner_updates`（`updates.py:27-38`）每轮 `np.concatenate` 全部提交 key 再 `argsort` + `searchsorted`——**这正是路线图说“不得为每轮全局排序”的那条路径**。当前实现对正确性无碍，但与目标热路径设计相悖，未来需要重写为预分配 + scatter。
- **需要增加**：一个全局位移存储（按稳定 key 索引），owner-only 写、halo 读、轮次屏障发布，配套 epoch/bitset 去重。

### 4.3 几何安全 / 位移可行性层【高，正确性风险】

- `geometry/validate.py:10` 的 `validate_contours` 只查：零长边、零面积环、每 polygon 恰一个 hull。`reconstruction.py:99` 在重建后调用它。
- `doc/function_call_architecture.md` §6.1 明确承认它**不保证**：hole 完整位于原 hull 内、外边不越过内边、矩形左边不越过右边、新旧轮廓拓扑关系不变。
- 真实求解器每轮都推位移，没有任何一层做位移可行性/碰撞检查——这是“边算边产生非法几何”的直接风险。
- **需要增加**：在 `merge_owner_updates` 之后、`materialize`/`sample_lines` 之前接入位移可行性/碰撞限幅层；在 `reconstruct_region` 之后接入环自交、绕向、hole 包含验证。架构文档已给出接线位置，但实现为空。

### 4.4 采样有效性层【中】

- `doc/function_call_architecture.md` §5.3 承认：`sample_lines`（`sampling.py:31`）只按法向偏移坐标，**不检查采样点是否穿过对面边界**（举例：2 nm 线宽配置 8 nm 内偏移）。
- 对薄特征/密集图形，这会让光学评估采样落到材料内部或对侧外部，污染损失。
- **需要增加**：局部间距/有效性层，对每个采样点判断其与对侧边界的关系，标记或剔除无效采样。

### 4.5 受控 remesh API【高】

- 当前分段在 `fragment_edges`（`fragmentation.py:59`）一次性生成，`prepare_problem` 之后**没有任何重切入口**。
- `AGENTS.md` 要求：普通轮次固定参考边段、不重切；但“只有显式 remesh 才能改变分段，并必须同步重建 key、归属和优化器状态”。
- 长时间迭代中掩膜边移动足以使原分段/halo 假设失效时，没有 remesh 就无法继续；当前架构对“需要 remesh”没有出口。
- **需要增加**：显式 remesh 操作，原子地重建 `SegmentBatch`（key/normal/topo）+ `OwnershipBatch` + 优化器状态，并保证与旧位移向量的映射。

### 4.6 层级 source 追溯与层级输出复用【高，规模化关键】

- `AGENTS.md` 原文：当前物化把 SREF/AREF occurrence 转为 top 全局坐标，物理合并后失去 master cell、源 shape、instance path 和 transform；对一个 occurrence 的修正不修改源 cell，也不传播到其他引用。
- 这意味着完整 reticle 中被引用上百万次的同一 cell 会被**完全扁平化**，层级带来的内存与计算优势全部丧失（这与 `LRN-20260809-001`“保留层级、惰性 ROI”的初衷在 OPC 输出端被抵消）。
- **需要增加**：输入端按需保留 `source_cell_id`/`source_shape_id`/`instance_path`/`instance_transform`/`occurrence_id`；输出端按上下文等价性复用修正（相同修正共享 OPC cell，不同修正克隆 variant 并重定向引用），源 GDS 只读。

### 4.7 求解器 / 模型 / 指标层（已声明占位，但仍是直接阻塞）【高】

- `lithography/` 与 `evaluation/` 为空目录，`opc/iteration/` 尚未建立（当前 `opc/` 下只有 `errors.py`、`input/`）。`doc/function_call_architecture.md` §10 给出了求解器调用骨架（伪码），但无任何实现。
- 这三项不是“锦上添花”，而是“完整 reticle 处理”的本体——没有它们，前端再完善也只是几何演示。
- **需要增加**：`opc.iteration.<method>` 的迭代循环、`lithography` 的光学模型/损失、`evaluation` 的指标；三者通过 `MBOPCProblem`/`BoundarySampleBatch`/`SegmentUpdateBatch` 与输入前端交互，不得下沉进 `opc.input`。

### 4.8 checkpoint 的读回端【中】

- `artifacts.py:14` 的 `save_problem_npz` 能写出纯数值 NPZ（含 key、归属、位移）。
- 但**没有对应的加载器**，也没有“从 NPZ 恢复 `MBOPCProblem` 并续算”的路径。这使第 3.4 节 key 的“checkpoint/恢复”用途处于“能写不能读”的半成品状态。
- **需要增加**：NPZ → `MBOPCProblem` 的反序列化与一致性校验，作为大 reticle 分段续算/崩溃恢复的支撑。

### 4.9 多层假设（备注，低）

- `prepare_problem` 只接受单个 `layer`。若未来 OPC 需要多层掩膜或上下文层（如 SADP/多重图形），当前入口无法表达。**暂记为假设**，是否立项取决于工艺需求。

## 5. 与 `AGENTS.md` 路线图对照

下表把上述发现映射到 `AGENTS.md` “未来优化内容”已声明的条目，证明本评审与项目既定方向一致，而非另起炉灶。

| 本评审条目 | 对应 `AGENTS.md` 路线图表述 | 状态 |
|---|---|---|
| 4.1 分块/流式编排 | “大 reticle 使用流式 GPU batch”“普通 OPC 轮次固定参考边段” | 未实现 |
| 4.2 跨 tile `d_current`/`d_next` | “全部 tile 基于同一只读 `d_current`…所有 tile 完成后才发布 `d_next`” | 未实现 |
| 4.2 scatter/epoch 去重 | “预分配 `d_next`…epoch/bitset 检测重复写入，不得每轮全局排序” | 当前实现恰好是“每轮全局排序”，待重写 |
| 4.3 几何安全层 | （架构文档 §6.1）“未来几何安全层应接在 merge 之后 / reconstruct 之后” | 未实现 |
| 4.5 remesh | “只有显式 remesh 才能改变分段，并同步重建 key、归属、优化器状态” | 未实现 |
| 4.6 层级追溯与复用 | “未来层级 OPC 输入应按需保留 source_cell_id…”“未来层级输出优先按上下文等价性复用” | 未实现 |
| 3.4 key 用途 | “stable key 主要用于跨进程校验、checkpoint、恢复和诊断”（架构文档 §9） | 机制在，用途未落地 |
| 3.1 / 3.2 未用抽象 | `AGENTS.md`“新抽象必须有当前调用方” | `GeometryEngine`/`UniformGridIndex` 违反此条 |

## 6. 若推进“完整 reticle”的优先级建议

1. **先封口缺失的“使用方”**：落地最小可用 `opc.iteration`（哪怕零位移/规则位移的占位求解器）+ `evaluation`，让前端有真实多轮调用方。没有它，第 3 节的“过度设计”判断与第 4 节的“缺失”判断都无法被真实工作流验证。
2. **再做 4.1 + 4.2**：tile 编排 + 跨 tile 位移状态。这是把“单批管线”升级为“整 reticle 管线”的结构性前提，也直接决定 4.3/4.4 的接线方式。
3. **同步 4.3 几何安全层**：在求解器产出真实位移后立刻需要，否则位移即非法几何。
4. **随后 4.6 层级复用**：是规模化（上百万实例）的必要条件，也是 3.5 层级摘要与 3.4 key 投资真正兑现价值的地方。
5. **最后清理第 3 节**：在真实工作流稳定后，再决定 `GeometryEngine`/`UniformGridIndex`/显式 core 路径是激活还是退役（涉及受保护目录，须单独授权）。

## 7. 评审者不确定性与边界

- 第 3.1 / 3.2 条“无生产调用方”是**检索确证**的客观事实，置信度高。
- 第 3.3 / 3.4 条“是否算过度设计”带主观成分：Protocol 与 key 都是**可辩护的前置投资**，我标注为“中”并在 3.4 明确写了“非浪费”，避免一刀切。
- 本文不评估 `run_layout_geometry.py` / `run_mbopc_frontend.py` / benchmark 的工程质量，仅评估三个被评审包的设计。
- 本文不构成对 `layout/`、`geometry/` 的修改授权；任何处置这些目录条目的建议均需按 `AGENTS.md` 取得用户逐次确认。
