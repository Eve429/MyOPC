---
id: CHG-20260818-levelset-ilt
title: LevelSet ILT 迁移
type: implementation-spec
status: draft
baseline_commit: aa03d2486e2cd93b4297f1b85847f88857b221a2
baseline_worktree: unknown
baseline_dirty_paths: []
scope:
  - opc/input/pixel
  - opc/iteration/ilt
  - main
  - config
  - tests
  - doc
depends_on:
  - doc/changes/completed/CHG-20260818-simple-ilt/implementation_spec.md
  - doc/contracts/ilt.md
  - doc/contracts/lithography.md
supersedes: []
---

# LevelSet ILT 迁移

## 0. Document Contract

本文档是本 change 的唯一实现规格。实施 AI MUST 先以当前 `migration` 分支真实源码重新核对
Simple ILT、`PixelMacroProblem`、`ILTMacroResult`、`ILTMethod` 与 `_ilt_workflow`，并在开始实施前
更新 `baseline_commit`、worktree 状态与全量 test count；若基线发生实质漂移，必须先修订本文。

实现 AI MUST：只修改 §14 文件；复用已完成的 pixel problem、宏级参数域、core-batched 光刻、
`ILTMacroResult` 与公共 artifact/merge 生命周期；以 OpenILT LevelSet 为算法参考，以当前 MyOPC
macro/core ownership 契约为工程事实；阻塞问题或算法语义冲突时停止。

实现 AI MUST NOT：修改 `layout/geometry/00_PAST/lithography/evaluation`；为 LevelSet 复制整套 ILT
workflow；建立尚无必要的 ILT 基类/注册器；把 core ownership 当成参数 ownership；为不同 core
独立生成同一物理位置的 SDF/phi；静默修正 NaN、I/O 或 CUDA 错误。

## 1. Objective

在现有 Simple ILT 已建立的像素型 macro/core 管线基础上迁移 OpenILT 的 hard LevelSet 方法，并
把公共 ILT workflow 中残留的 Simple 专属 mask/context 语义移出公共层，使用户可直接运行
`main/run_levelset_ilt.py`，且后续 CurvMulti/Multilevel 可继续复用同一方法无关生命周期。

LevelSet 的核心工程约束是：**一个 macro 内每个物理像素只有一个权威 phi；core 仅是显存受限的
光刻计算窗口，不拥有独立参数或独立 SDF。**

## 2. Baseline and Evidence

### 2.1 Baseline

- 文档审查基线：`aa03d2486e2cd93b4297f1b85847f88857b221a2`。
- 实施前 MUST 重新记录当前 `migration` HEAD、worktree 状态与全量测试数量，不得沿用旧的
  `2fa75.../446 tests` 设计事实。
- 当前 Simple ILT 已完成并建立 `PixelMacroProblem -> optimize_simple_macro -> ILTMacroResult ->
  run_ilt_workflow` 生产路径；本文以该真实源码而不是旧 draft 假设为准。

### 2.2 Confirmed Facts

| Fact ID | Confirmed fact | Evidence | Verification |
|---|---|---|---|
| FACT-001 | 当前像素 ILT 参数域定义在 macro ownership，core 通过 `trainable_index_canvas` 映射同一 macro 参数 | `opc/iteration/ilt/simple.py`、`opc/input/pixel/problem.py` | 静态阅读+现有测试 |
| FACT-002 | 同一 macro 参数可出现在多个 core context；Simple 将各 core 梯度 scatter-add 求和后仅做一次 macro 同步 step | `optimize_simple_macro`、`TestCrossCoreGradient` | 源码+测试 |
| FACT-003 | OpenILT LevelSet hard forward 为 `phi<0`，代理反向为 `-|∇phi|*grad_output`，optimizer 为 Adam | `OpenILT/pyilt/levelset.py` | 只读参考 |
| FACT-004 | 旧迁移提供精确二维 EDT、strict-zero、全空/全满与 LevelSet STE 参考实现 | `00_PAST/opc/iteration/ilt/levelset.py` | 只读源码与旧测试 |
| FACT-005 | 当前 `_ilt_workflow` 的 final binary canvas 直接读取 `config.sigmoid_steepness` 并使用 Simple 的 `σ(β(2T−1))` 固定 context，尚非真正 method-independent | `main/_ilt_workflow.py::_binary_canvas/_evaluate_best_binary` | 静态阅读 |
| FACT-006 | 当前公共结果类型是 `ILTMacroResult`，不是旧规格中的 `ILTBatchResult` | `opc/iteration/ilt/_common.py` | 静态阅读 |

### 2.3 Uncertainty Boundary

- OpenILT 的原始执行单元是一张完整 tile；MyOPC 为 full-layout 工程需要 macro/core 两级切分。因此
  “OpenILT tile”在参数/SDF 语义上对应 MyOPC macro query，而不是单个 core。
- LevelSet STE 是代理梯度，不是 hard threshold 的数学真导数；本 change 不宣称单调收敛或几何最优。
- macro 之间仍独立优化、固定外部 context；macro seam 联合优化不属于本 change。

### 2.4 External and Archive References

| Reference | Role | Adopt | Reject/Adapt | Reason |
|---|---|---|---|---|
| `OpenILT/pyilt/levelset.py` | 原算法 | hard forward、空间梯度 STE、Adam、固定区混合 | 全局 DEVICE、单 tile 假设 | 保留算法，适配 macro-global 参数/core-batched optics |
| `OpenILT/pyilt/initializer.py` | 初始化依据 | 整个计算域唯一 signed-distance field 的思想 | polygon/legacy 多初始化器 | 当前输入已是持久 raster，采用精确 raster EDT |
| `00_PAST/opc/iteration/ilt/levelset.py` | 工程参考 | O(HW) EDT、全空/全满、strict zero、结构化异常 | 旧 result/simple 依赖 | 迁入当前 `ILTMacroResult`/macro 同步契约 |
| 当前 Simple ILT | 架构基线 | macro 参数唯一、core loss ownership、跨 core gradient sum、N+1 state、macro best | Simple sigmoid/context 参数化 | LevelSet 只替换参数化/优化器/context mask |

## 3. Current Behavior

当前生产行为：

1. `PixelMacroProblem` 持久保存每个 macro query 的一张 `target_u8`，并按需生成 core target、
   ownership、trainable index 与 valid-context canvas；
2. `optimize_simple_macro` 在 macro ownership 上维护唯一参数数组，同一 state 的全部 core/batch 读取
   同一快照，梯度 scatter-add 回 macro 参数，全部 core 完成后恰一次同步 SGD step；
3. `run_ilt_workflow` 负责 prepare、逐 macro solver、best binary 终评、artifact、merge 与 summary；
4. 公共 workflow 的 final canvas 仍硬编码 Simple sigmoid context，需要本 change 做一次最小通用化；
5. 当前没有 LevelSet 生产实现。

## 4. Target Behavior

### REQ-001：直接入口

系统 MUST 提供 `python main/run_levelset_ilt.py [config.toml]`；缺省配置为仓库内
`config/levelset_ilt.toml`，公共输入、macro 生命周期、artifact 与 merge 结构与 Simple ILT 对齐。

### REQ-002：LevelSet hard/STE

LevelSet MUST 前向输出严格 `(phi < 0).float()`；`phi==0` MUST 为不透光。反向 MUST 返回
`-|∇phi| * grad_output`，空间差分使用二维中心差分，画布最外沿采用 replicate boundary。

### REQ-003：macro-query 唯一 SDF

每个 macro MUST 从完整 `problem.target_u8`（即 macro query region）生成且仅生成一次权威
`initial_query_phi[Hq,Wq]`：target `>=0.5` 为前景负、背景正。不得对每个 core target canvas
独立运行 SDF。

同一物理像素若同时出现在 core A/B context 中，两者 MUST 由同一 `initial_query_phi`/同一
`macro_phi` 取值，因此始终满足 `phi_A(x,y) == phi_B(x,y)`。

### REQ-004：精确 EDT 与生命周期

SDF MUST 使用 `O(Hq*Wq)` 时间的二维精确欧氏距离变换，不依赖 SciPy/OpenCV 新依赖；全空/全满
query 必须返回有限且符号正确的常量场。EDT 仅在每 macro 初始化执行一次，不进入 state/core 热循环。

实现 SHOULD 将 float64 EDT scratch 限制为最少完整 query 工作区并复用 O(max(Hq,Wq)) 一维
workspace；不得为每 core 重复分配完整 EDT scratch。

### REQ-005：macro-global phi 与 macro 同步 Adam

LevelSet 可训练参数 MUST 是 macro ownership 上唯一 `macro_phi[Hm,Wm]`。core 通过
`trainable_index_canvas` 把自身画布中的 macro 像素 gather 成 local leaf；同一物理像素可在多个
core 画布出现，但映射到同一个 macro index。

同一 state 内：

1. 所有 core/batch MUST 读取相同 macro phi 快照；
2. 各 core loss 只在自己的 `ownership_canvas` 统计；
3. backward 得到的 local phi 梯度 MUST scatter-add 到同一 `macro_gradient`，不得按出现次数平均；
4. 全部 core 完成后才允许对 macro phi 执行恰一次 Adam update；
5. Adam 的 `m/v` state MUST 属于 macro 参数，不得按 core/batch 分裂。

因此 batch size/core 顺序不得改变算法语义。

### REQ-006：三类画布位置语义

LevelSet core canvas MUST 区分三类位置：

1. `trainable_index_canvas >= 0`：属于当前 macro ownership，使用当前 `macro_phi`，即使该位置位于
   当前 core 的 context 中仍可通过本 core loss 获得梯度；
2. macro 外但 `context_valid_canvas == True`：只读物理 context，使用 `initial_query_phi` 对应值，
   不可更新；
3. `context_valid_canvas == False`：仅为固定 canvas 的数值 padding，mask transmission 必须严格 0，
   不可把它解释成物理 T=0 后再构造非零连续值。

`core ownership` 仅定义 loss owner，绝不等于 parameter owner。

### REQ-007：上下文宽度与空间梯度

LevelSet STE 本身需要相邻 phi 计算 `|∇phi|`，因此 LevelSet MUST 要求
`context_dbu >= pixel_dbu`，与 `curvature_weight` 是否为 0 无关。该约束在 solver 入口前置校验，
避免 core ownership 边缘的空间差分读到数值 padding 而形成切分相关梯度。

### REQ-008：损失、状态与 best

LevelSet MUST 使用当前公共 `owned_continuous_losses`、`weighted_macro_loss` 与
`ILTStateRecord/ILTMacroResult`；N 次 Adam 更新对应 N+1 个完整已评价 macro state，末状态纯评价。

best MUST 只按完整 macro total loss 严格下降选择唯一 state，不得做 per-core/per-sample patchwork。
输出：`best_parameters=best_phi[Hm,Wm]`、`soft_mask=sigmoid(-best_phi)`（仅诊断）、
`binary_mask=(best_phi<0)`。

### REQ-009：曲率

曲率项启用时 MUST 作用于当前 hard mask，并只统计 ownership 有效卷积区；权重为 0 时不得执行
卷积。LevelSet 的 context>=1 pixel 已保证 valid 3x3 邻域覆盖 core ownership。

### REQ-010：公共 workflow 通用化

允许并要求对 `_ilt_workflow` 做一次向后兼容的最小重构：公共层 MUST 不再直接读取
`sigmoid_steepness`、不得自己决定具体方法的固定 context transmission。

`ILTMethod` MUST 增加一个最小方法策略钩子（建议名 `build_fixed_context_canvas`，可等价命名），职责是：
给定 `PixelMacroProblem/core_index/config` 返回当前方法在“macro 外物理 context + 数值 padding”上的
固定 mask transmission canvas。公共 `_binary_canvas` 只负责把 `result.binary_mask` 写入
`trainable_index_canvas>=0` 位置，其他位置消费该方法钩子。

- Simple adapter 返回现有 `σ(β(2T−1))` 物理 context + padding 0，数值必须零变化；
- LevelSet adapter 返回由初始 LevelSet sign 决定的 hard context（物理 query 内 `phi<0`，padding 0）；
- 公共 workflow 不得再假定所有 ILT config 都含 `sigmoid_steepness`。

若实现发现还有其它仅 Simple 专属字段泄漏到公共 workflow，可在保持接口最小的前提下同步移除。

### REQ-011：artifact 一致性

配置、进度、资源统计、macro result NPZ、metrics、best.gds、summary、final merge/final lithography
MUST 继续由公共 workflow 生成，并遵循当前 Simple ILT 已有格式；LevelSet 不单独发明 result dataclass
或 artifact schema。方法专属 result 文件名为 `levelset_ilt_result.npz`，其中 `best_parameters` 语义为 phi。

## 5. Scope

### 5.1 In Scope

- macro-query 精确 SDF、hard-forward surrogate-backward、LevelSet config、macro-global Adam solver；
- `PixelMacroProblem` 增加一个最小通用 query-array→core-canvas 切片能力，避免 LevelSet 复制私有坐标算法；
- `_ilt_workflow/ILTMethod` 去除 Simple 专属 final-context 假设；
- config 注册、LevelSet adapter/runner、Simple adapter 兼容修改；
- 测试、配置、contracts/architecture/manual 与交付报告。

### 5.2 Out of Scope

- SDF reinitialization/fast marching、窄带 LevelSet；
- macro 间参数交换、macro seam healing/联合优化；
- 拓扑/MRC/EPE/shot 约束；
- CurvMulti/Multilevel 算法实现。

### 5.3 Protected Areas

`00_PAST/**`、`layout/**`、`geometry/**`、`lithography/**`、`evaluation/**`、用户数据不得修改。

## 6. Invariants

### INV-001：唯一 phi

一个 macro 内同一个物理 `(x,y)` 只有一个权威 `phi`；所有 core canvas 仅是该 field 的不同窗口视图。

### INV-002：SDF sign

`target>=0.5` 的初始前景为负、背景为正；`phi<0 <=> binary=True`，`phi==0` 始终 False。

### INV-003：ownership 分离

macro ownership 是参数 ownership；core ownership 是 loss ownership；macro 外 query context 只读；
数值 padding 严格 0 transmission。

### INV-004：宏同步更新

同一 state 全部 core 读取同一 macro phi 快照，梯度按物理参数 index 求和，Adam 仅在全 core 屏障后更新一次。

### INV-005：macro best

best 来自完整已评价 macro state；batch size/core reorder 不改变输出（允许正常浮点累加容差）。

### INV-006：公共层算法无关

`_ilt_workflow` 不读取 Simple/LevelSet 专属参数，不自行构造具体参数化的 context transmission。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `opc.input.pixel.PixelMacroProblem` | query raster、通用 query-array→core canvas 坐标映射 | LevelSet/Simple 数学 |
| `opc.iteration.ilt.levelset` | SDF、STE、macro phi/Adam、core batch 梯度回散、best | GDS、TOML、tqdm、artifact |
| `opc.iteration.ilt._common` | 中性 result/record/共享 loss/曲率 | 方法 context 策略 |
| `main._ilt_workflow` | 方法无关 prepare/solver/终评/artifact/merge | sigmoid/levelset 参数化 |
| `main._simple_ilt_workflow` | Simple METHOD + Simple 固定 context 策略 | 公共生命周期复制 |
| `main._levelset_ilt_workflow` | LevelSet METHOD + LevelSet 固定 context 策略 | optimizer 数学/公共 workflow |
| `main.run_levelset_ilt` | 直接入口 | 业务逻辑 |

### 7.2 Dependency Direction

```text
pixel.problem ---------------------------> generic core canvas mapping
levelset -> ilt._common + pixel problem + lithography
simple   -> ilt._common + pixel problem + lithography
simple adapter   -> _ilt_workflow + simple
levelset adapter -> _ilt_workflow + levelset
runner -> levelset adapter
```

不得让 `levelset.py` import `simple.py`，不得让 `_common` 依赖具体方法。

### 7.3 Data Flow

```text
PixelMacroProblem.target_u8 [query Hq,Wq]
 -> signed_distance_initialization(query target)          [once/macro, CPU]
 -> initial_query_phi
      ├─ ownership crop -> macro_phi [Hm,Wm]              [trainable]
      └─ outside ownership -> fixed query phi             [read-only]

for state 0..N:
    macro_gradient = 0
    for core batch:
        target/ownership/trainable_index/valid canvas
        fixed_phi_canvas = crop(initial_query_phi, core context)
        local_phi = gather(macro_phi, trainable indices)
        phi_canvas = trainable ? local_phi : fixed_phi_canvas
        hard_mask = LevelSetBinarize(phi_canvas)
        hard_mask[padding] = 0 transmission
        printed = model.forward_many(three conditions)
        loss only on core ownership
        backward
        scatter-add local phi grad -> macro_gradient
    record full macro state / update macro best
    if state < N:
        Adam macro_phi once using summed macro_gradient

best_phi -> soft/binary -> ILTMacroResult -> unchanged common artifacts/merge
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| initialization | macro query target | CPU exact EDT | initial_query_phi + initial macro_phi | 每 core/state 不得重算 |
| state batch | macro phi snapshot | core canvas gather/litho/loss/backward | partial macro gradient | 不做 optimizer step |
| state barrier | all core gradients | macro total record + Adam | next macro phi | 不按 batch/core step |
| output | macro best phi | soft/binary | ILTMacroResult | 不二次选 best |
| workflow final | ILTMacroResult | method-specific fixed context + binary final eval/artifacts | final outputs | 不知道 phi/sigmoid 内部参数 |

### 7.5 Planned Call Graph

```text
main/run_levelset_ilt.py::main
└─ main/_levelset_ilt_workflow.py::run_levelset_ilt
   └─ main/_ilt_workflow.py::run_ilt_workflow(LEVELSET_ILT_METHOD, config)
      ├─ prepare_pixel_problems
      ├─ opc/iteration/ilt/levelset.py::optimize_levelset_macro
      │   ├─ signed_distance_initialization(query_target)
      │   ├─ PixelMacroProblem.query_array_canvas(...)
      │   ├─ _LevelSetBinarize.apply                    [state/core batch]
      │   ├─ model.forward_many
      │   ├─ ilt._common losses
      │   └─ macro Adam state/update
      ├─ method.build_fixed_context_canvas(...)         [final binary eval]
      └─ shared artifacts/merge/final lithography
```

## 8. Data Contracts

### `LevelSetILTConfig`

Owner：`opc.iteration.ilt.levelset`；frozen；one run。

| Field | dtype | meaning |
|---|---|---|
| `iterations` | strict int | macro Adam 更新数，`>=1` |
| `step_size` | float | Adam lr，finite `>0` |
| `weight_process_l2` | float | finite `>=0` |
| `weight_pvband` | float | finite `>=0` |
| `curvature_weight` | float | finite `>=0` |
| `batch_size` | strict int | 一次 forward 的 core 数，`>=1` |

nominal 权重固定 1；target SDF threshold 固定 0.5；LevelSet binary threshold 固定严格 0；
**不得为适配公共 workflow 人工增加 `sigmoid_steepness` 或 `mask_threshold`。**

### LevelSet tensors

| Name | dtype/shape | Resident | Lifetime |
|---|---|---|---|
| `initial_query_phi` | float32 `[Hq,Wq]` | CPU | macro solve |
| `macro_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| Adam `m/v` | float32 `2×[Hm,Wm]` | CPU | macro solve |
| `macro_gradient` | float32 `[Hm,Wm]` | CPU | one state |
| `best_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| EDT scratch | float64 query workspaces | CPU | initialization only |
| local phi / hard mask / printed | float32 `[B,256,256]` 量级 | GPU/device | one core batch |

实现 MAY 使用 PyTorch CPU Adam 或等价显式 Adam 更新；无论具体实现，Adam state 语义必须是唯一 macro 参数
state，且不得让完整 macro autograd graph/GPU tensor 常驻显存。

### 8.1 Configuration Contract

新增 `[levelset_ilt]`，六字段全部 required、无默认；共享 `[layout]/[partition]/[lithography]/[output]`
与 Simple 同源，不要求 `[edge]`。

### 8.2 Persisted Artifact Contract

复用当前公共 pixel problem/ilt_plan/final contract；方法专属文件：

- `macros/<macro>/levelset_ilt_result.npz`：沿用当前公共 result NPZ schema，`best_parameters` 为 phi；
- `metrics.json`、`best.gds`、`summary.json`：沿用当前 `_ilt_workflow` 实际字段，不为 LevelSet 单独扩格式；
- `summary.method="levelset_ilt"`。

## 9. Interface Changes

### IF-001：LevelSet solver

```python
signed_distance_initialization(
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> torch.Tensor | np.ndarray

optimize_levelset_macro(
    problem: PixelMacroProblem,
    model: LithographyModel,
    config: LevelSetILTConfig,
    *,
    on_tiles_completed: Callable[[int], None] | None = None,
) -> ILTMacroResult
```

允许实现选择 SDF 对外统一 numpy 或 torch；生产 solver 中权威 SDF 必须在 CPU macro-query 级构造一次。

### IF-002：通用 query-array canvas 映射

`PixelMacroProblem` MUST 提供一个最小公共 helper，把与 `target_u8` 同 query shape 的二维数组按当前
core context/canvas padding 规则映射为 canvas；建议：

```python
query_array_canvas(
    self,
    array: np.ndarray,
    core_index: int,
    *,
    fill_value=0,
) -> np.ndarray
```

它必须复用 `_context_window/_center_padding` 的唯一坐标事实；`target_canvas` SHOULD 改为复用该 helper，
避免 LevelSet 复制私有切片逻辑。

### IF-003：ILTMethod 方法策略

`ILTMethod` 在当前四字段基础上增加一个最小 final-context strategy，例如：

```python
build_fixed_context_canvas: Callable[[PixelMacroProblem, int, object], np.ndarray]
```

返回完整 `[canvas,canvas]` transmission canvas，其中 macro 外物理 context 按方法定义，padding 为 0；
公共 `_binary_canvas` 再把 trainable 位置替换成 best binary。

若实现发现“返回完整 fixed canvas”比“只返回 context values”更简洁，可采用等价签名，但必须满足：
公共 workflow 不再读取 `sigmoid_steepness` 或 LevelSet 内部参数。

### IF-004：配置与入口

`CONFIG_SECTIONS[LevelSetILTConfig] = "levelset_ilt"`；新增 `LEVELSET_ILT_METHOD`、
`run_levelset_ilt(config_path)->dict`；runner 接受零或一个 config path，与 Simple CLI 风格一致。

## 10. Algorithm

### 10.1 Macro-query SDF

对 `problem.target_u8.astype(float32)/255` 的完整 query raster 二值化：

- 前景：`target >= 0.5`；
- 背景：`target < 0.5`；
- mixed query：分别计算到前景/背景的精确 Euclidean distance，前景位置取负 inside distance，背景取正 outside distance；
- 全前景：有限负常量，例如 `-max(Hq,Wq)`；
- 全背景：有限正常量，例如 `+max(Hq,Wq)`。

二维 EDT 采用两次一维抛物线下包络。SDF 距离单位是 pixel center，不是 nm/DBU；当前 STE 仅使用相邻
空间差分，因此该无量纲像素距离与 OpenILT 同类实现一致。

关键约束：SDF 的定义域是 **macro query**，不是 macro ownership，更不是 core canvas。这样 ownership
边缘像素的最近边可以来自 macro 外 context，且 A/B core overlap 中同一物理像素只存在一个 SDF 值。

### 10.2 Macro state loop

```text
query_target = problem.target_u8 / 255
initial_query_phi = signed_distance_initialization(query_target)       # once
macro_phi = crop(initial_query_phi, macro ownership).copy()
fixed_query_phi = initial_query_phi                                    # read-only
Adam state belongs to macro_phi

for state_index in 0..N:
    build_gradient = state_index < N
    macro_gradient = zeros_like(macro_phi) if build_gradient else None
    sums = zero macro losses

    for core batch:
        target = target_canvas(core)
        ownership = ownership_canvas(core)                              # loss owner
        trainable = trainable_index_canvas(core)                        # parameter map
        valid = context_valid_canvas(core)
        fixed_phi = query_array_canvas(fixed_query_phi, core, fill=+C)

        local_phi = gather(macro_phi, trainable>=0)                     # leaf on device
        phi_canvas = where(trainable>=0, local_phi, fixed_phi)
        hard = LevelSetBinarize(phi_canvas)
        hard = where(valid OR trainable>=0, hard, 0)                    # padding strictly zero

        printed = model.forward_many(hard, three_conditions)
        losses = owned_continuous_losses(..., ownership)
        optional curvature(hard, ownership)
        accumulate macro scalar sums

        if build_gradient:
            backward(batch_total)
            scatter-add local_phi.grad -> macro_gradient
        release batch tensors
        progress(batch_count)

    validate/record full macro state
    update macro best if total_loss strictly lower
    if state_index < N:
        validate macro_gradient
        Adam_step(macro_phi, macro_gradient)                            # exactly once
        validate macro_phi

soft = sigmoid(-best_phi)
binary = best_phi < 0
return ILTMacroResult(...)
```

实现可以不用显式构造 `phi_canvas` 的完整 CPU 版本；关键是 GPU batch 里的所有 trainable 值来自同一
macro snapshot，fixed context 来自同一 query SDF。

### 10.3 STE

`_LevelSetBinarize.forward(phi) = (phi < 0).to(phi.dtype)`。

backward：对 `[B,H,W]` phi 做 replicate pad 后中心差分：

```text
dx = (phi[x+1]-phi[x-1]) / 2
dy = (phi[y+1]-phi[y-1]) / 2
grad_phi = -sqrt(dx^2 + dy^2) * grad_output
```

常量 phi 的 STE gradient magnitude 为 0，这是算法语义，不做 fallback。

### 10.4 Fixed context

训练时 macro 外物理 context 使用 `initial_query_phi` 的 hard sign；数值 padding transmission 恒 0。
终评必须使用与训练相同的 LevelSet fixed context 定义。Simple 的 fixed context 仍保持现有
`σ(β(2T−1))`，由 Simple adapter 提供，不得因此修改 LevelSet config。

### 10.5 State Transition

与当前 Simple 一致：N 次更新、N+1 个已评价 macro state；best 严格按 macro total loss下降更新，
平局保留更早 state。Adam step 后的新参数必须在下一 state 才参与 best 比较。

## 11. Ownership and State

### 11.1 参数所有权

`macro_phi + Adam m/v` 由 LevelSet solver 的当前 macro 独占。`trainable_index_canvas` 是从 core canvas
到 macro 参数的唯一映射；任何 core-local tensor 都不是独立参数真源。

### 11.2 loss 所有权

每个 core 的 `ownership_canvas` 唯一计分，因此全 macro loss 是不重叠 core ownership loss 的求和。
某 macro 参数虽然可能出现在多个 core context，但这些梯度来自不同 ownership loss 对同一物理参数的
真实影响，因此 MUST 求和，不按出现次数平均。

### 11.3 fixed state

`initial_query_phi` 在全部 state 只读；仅 macro ownership 对应 `macro_phi` 可更新。macro 外 context
永远保持初始 LevelSet，和 OpenILT “中央可优化区 + 外围初值固定”思想一致。

## 12. Error Handling

### ERR-001

非法 config、target range/shape、canvas mismatch、`context_dbu < pixel_dbu` 由构造或 solver 抛
`ValueError`，不得 fallback。

### ERR-002

非有限 loss、macro gradient、macro phi/Adam state 抛 `FloatingPointError`，不得跳过 pixel/core 或
重置 optimizer state。

### ERR-003

CPU EDT、CUDA、I/O 等未知错误原样传播；entry 不捕获并伪装成功。公共 workflow 失败发布语义沿用
现有 Simple 契约。

## 13. Performance and Memory Constraints

- SDF 初始化：每 macro query 恰一次 `O(HqWq)`；禁止每 state/core/batch 重算 EDT。
- macro 参数与 Adam state SHOULD 常驻 CPU：`phi + m + v + gradient + best ≈ 5×float32`，约
  `20 bytes/macro-pixel`，另有一张 `initial_query_phi float32` 与 EDT 临时 scratch。
- GPU 只常驻一个 core batch 的 local phi/hard mask/printed/autograd graph；不得把完整 macro phi 或
 完整 macro autograd graph常驻 GPU。
- 每 state 每 core batch 一次 `forward_many(three conditions)`；curvature=0 不执行 conv。
- `query_array_canvas` 不得在热路径为每 core 构造 O(macro_pixels) 的辅助索引数组；只允许 canvas/window
  量级临时数组。
- smoke 必须记录 CPU RSS、CUDA peak、总耗时、SDF 初始化耗时；与 Simple 同输入对比只记录，不声明
  LevelSet 必然更快/更优。

## 14. File-Level Change Plan

| File / Symbol | Action | Contract change | Reason |
|---|---|---|---|
| `opc/input/pixel/problem.py` | modify | 增通用 query-array→core-canvas helper；`target_canvas` 复用 | 唯一坐标事实，避免 LevelSet 复制切片 |
| `opc/iteration/ilt/levelset.py` | add | `LevelSetILTConfig`、EDT/SDF、STE、macro-global Adam solver | REQ-002..009 |
| `opc/iteration/ilt/__init__.py` | modify | 导出 LevelSet API | REQ-001 |
| `main/_ilt_workflow.py` | modify | `ILTMethod` 增 fixed-context strategy；终评去 Simple `sigmoid_steepness` 假设 | REQ-010 |
| `main/_simple_ilt_workflow.py` | modify | 提供现有 Simple fixed context strategy，数值零变化 | REQ-010 回归 |
| `main/_levelset_ilt_workflow.py` | add | LevelSet METHOD + hard fixed context strategy + thin run | REQ-001/010 |
| `main/configuration.py::CONFIG_SECTIONS` | modify | 注册 `[levelset_ilt]` | IF-004 |
| `main/run_levelset_ilt.py` | add | `[config.toml]` 直接入口 | REQ-001 |
| `config/levelset_ilt.toml` | add | smoke 配置 | §8.1 |
| `tests/opc/input/test_pixel_problem.py` | modify | query-array canvas 映射与 target_canvas 复用回归 | TEST-002 |
| `tests/opc/iteration/test_levelset_ilt.py` | add | SDF/STE/唯一 phi/macro Adam/跨 core/真实模型 | TEST-001..008 |
| `tests/main/test_levelset_ilt_runner.py` | add | config/adapter/CLI/artifacts/final context | TEST-009..011 |
| `tests/main/test_simple_ilt_runner.py` | modify as needed | workflow 通用化后 Simple 数值零变化 | TEST-010/011 |
| `tests/main/test_configuration.py` | modify | 新 section 严格解析 | TEST-009 |
| `doc/contracts/ilt.md` | modify | LevelSet API + method-independent workflow + 限制 | 交付 |
| `doc/architecture/system.md`、`doc/architecture/dataflow.md` | modify | 当前 LevelSet/ILT workflow 数据流 | 交付 |
| `doc/development_manual.md`、`doc/test_manual.md` | modify | 使用与测试 | 交付 |
| 本规格 active→completed | move/update | baseline/status/revision/evidence | 交付 |
| `development_report.md`、`test_report.md` | add | 实施/偏差/测试/性能 | 交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` | modify | 同步 | AGENTS |

除上述最小公共化修改外，不得借本 change 重构 unrelated workflow/pixel API。

## 15. Test Specification

### TEST-001：macro-query SDF 精确性

小图 rectangle/single/hole/all-empty/all-full；与 brute-force 最近前景/背景距离逐值比对；sign/strict-zero
约束明确。spy 证明一个 macro 只调用一次 SDF，不随 core/state 数增加。

### TEST-002：query-array canvas 唯一映射

对任意 query-shape 标号数组，A/B 重叠 core 在同一物理坐标裁出的值完全一致；padding fill 正确；
`target_canvas` 与改造前逐值一致。

### TEST-003：STE forward/backward

严格 `phi<0`；手算 dx/dy、符号、replicate 最外沿；常量 phi 梯度为零。

### TEST-004：同一位置唯一 phi

构造两个有 context overlap 的 core A/B：选择同时出现在两 canvas 的 macro pixel P，捕获两次 forward
输入，断言 state0 `phi_A(P)==phi_B(P)`；一轮更新后两个窗口仍从同一 macro 参数读取。

测试必须可判别“每 core 独立 SDF”错误：在 A 局部窗口之外、B 可见范围内放置最近 feature edge，
若各 core 独立 SDF 则 P 距离会不同。

### TEST-005：跨 core gradient sum

使用具有局部空间耦合的 differentiable stub lithography；同一 macro 参数 P 同时影响 A/B ownership loss，
验证 local gradients scatter-add 求和，batch=1/2 结果一致；若按出现次数平均或仅 owner-core 采样则失败。

### TEST-006：macro Adam barrier

建立 float64/独立数值参考或可手算 Adam 小例子：所有 core 在同一 state 使用同一参数快照；恰一次 macro
Adam step；禁止 batch 内提前 step；N=2 得 3 个已评价 state。

### TEST-007：ownership/context/padding

macro ownership 参数可在邻 core context 中获得梯度；macro 外 context phi/binary 始终保持初始 SDF；
padding transmission严格0；`context_dbu < pixel_dbu` 前置失败。

### TEST-008：loss/curvature/real model

共享 nominal/process/PV loss 逐值；curvature hard-mask ownership-only 且 weight0 无 conv；真实 ICCAD13
CPU 一轮 backward/update/final evaluation finite；有 CUDA 时运行 parity/smoke 并记录资源。

### TEST-009：配置/adapter/CLI

合法、缺键、未知键、非有限、bool-as-int；LevelSetConfig 无 `sigmoid_steepness` 仍可完整跑公共 workflow；
仓库外 cwd 直接入口支持默认/显式 config。

### TEST-010：workflow method independence

用 spy/fake method 证明 `_ilt_workflow` 不访问 `sigmoid_steepness`、`mask_threshold`、phi 等算法字段；
final binary canvas 的 context 完全由 method strategy 提供。

### TEST-011：Simple 零回归与 artifact

现有 Simple 全测试通过；固定 workload 对 workflow 重构前基线检查 `best_parameters/binary_mask/binary_l2/
pvband/artifact key` 数值与格式不变。LevelSet result/metrics/best/final/summary 遵循同一公共 schema。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected |
|---|---|---|
| SDF domain | query vs core | once/macro，overlap same phi |
| SDF geometry | empty/full/single/hole/edge outside A local window | sign/distance exact |
| State | N=1/2 | N+1 state / macro best |
| Cross-core | overlap P, batch1/2 | summed gradient / same result |
| Boundary | macro context / core seam / padding | fixed context / trainable overlap / padding0 |
| Optimizer | multi-core Adam | one macro step/state |
| Device | CPU/CUDA | finite/parity where available |
| Workflow | Simple + LevelSet + fake method | no algorithm-specific leakage |
| Failure | config/shape/context/nonfinite | explicit exception |

### 15.2 Verification Commands

实施环境按仓库实际 Python 路径执行，至少包括：

```bash
python -m pytest -q tests/opc/iteration/test_levelset_ilt.py
python -m pytest -q tests/opc/input/test_pixel_problem.py
python -m pytest -q tests/main/test_levelset_ilt_runner.py tests/main/test_simple_ilt_runner.py
python -m pytest -q tests/opc/iteration/test_simple_ilt.py
python -m pytest -q tests
python -m ruff check common layout geometry opc lithography evaluation main tests
python -m compileall -q common layout geometry opc lithography evaluation main tests
python main/run_levelset_ilt.py config/levelset_ilt.toml
python main/run_simple_ilt.py config/simple_ilt.toml
git diff --check
```

## 16. Requirement Traceability

| Requirement/Invariant | Implementation | Tests | AC |
|---|---|---|---|
| REQ-001/011 | adapter/runner/shared artifacts | TEST-009/011 | AC-001 |
| REQ-002 | `_LevelSetBinarize` | TEST-003 | AC-002 |
| REQ-003/004 + INV-001/002 | macro-query SDF | TEST-001/002/004 | AC-002 |
| REQ-005/006 + INV-003/004 | macro solver/index mapping | TEST-004..007 | AC-003 |
| REQ-007 | context guard | TEST-007 | AC-003 |
| REQ-008/009 + INV-005 | state/loss/best | TEST-006/008 | AC-004 |
| REQ-010 + INV-006 | `ILTMethod/_ilt_workflow` | TEST-009..011 | AC-005 |

## 17. Acceptance Criteria

- [ ] **AC-001**：直接入口完成，`method=levelset_ilt`，artifact/merge contract 与当前 Simple 共享。
- [ ] **AC-002**：macro-query SDF/STE/strict-zero 逐值测试通过；同一 overlap 物理像素在 A/B core 中 phi 相同。
- [ ] **AC-003**：macro 参数 ownership、跨 core gradient sum、context fixed、padding0、macro Adam barrier、batch invariance 通过。
- [ ] **AC-004**：N+1/macro best/共享 loss/curvature/真实 ICCAD13 backward 通过并记录资源。
- [ ] **AC-005**：`_ilt_workflow` 不再读取 Simple 专属参数；Simple 数值零回归，全量测试绿色。
- [ ] **AC-006**：文档、development_report、test_report、baseline/revision evidence 完成。

## 18. Compatibility and Migration

- API：新增 LevelSet；`ILTMethod` 做一次向后兼容字段扩展，现有 Simple adapter 同步提供 strategy。
- Pixel input：`PixelMacroProblem` format v1/NPZ 不变，仅新增派生 canvas helper，不新增持久字段。
- Data：`ILTMacroResult` 与 result NPZ schema 不变；LevelSet `best_parameters` 语义为 phi，由 method 名区分。
- CLI：新增 LevelSet 入口；Simple CLI 行为不变。
- Numerical：Simple workflow 通用化要求数值零变化；LevelSet 保留 hard/STE/Adam，工程上采用 macro-query
  精确 raster SDF 与 macro 同步更新，不承诺与 OpenILT 单 tile 逐值一致。

## 19. Decisions

### DEC-001：OpenILT tile 映射到 MyOPC macro query，而不是 core

Reason：OpenILT 一张 tile 只有一个 LevelSet field；MyOPC core 有 overlap，若各 core 独立 SDF 会让同一
物理位置出现多个 phi，并进一步改变 `|∇phi|` 与代理梯度。

### DEC-002：SDF 在 macro query 上一次生成

Reason：query 同时覆盖可优化 macro ownership 和外围物理 context；ownership 边缘最近 feature 可位于
macro 外 context。只对 ownership 或 core 计算都会截断距离信息。

### DEC-003：core 只负责计算，不拥有参数

Reason：当前 Simple 已验证 macro-global 参数 + core loss ownership + cross-core gradient sum；LevelSet 应
复用相同物理 ownership 语义，只替换参数化与 optimizer。

### DEC-004：macro Adam 而非 core Adam

Reason：Adam 含状态且为非线性更新；按 core/batch step 会使结果依赖切分与 batch order。所有 core raw
梯度必须先合并，再对唯一 macro phi 做一次 Adam。

### DEC-005：允许重构 `_ilt_workflow`

Reason：当前公共终评硬编码 Simple `sigmoid_steepness/context`，会迫使 LevelSet 增加无意义配置并污染
算法语义。项目处于早期阶段，应优先形成真正 method-independent 的最终架构，而不是维持错误兼容层。

### DEC-006：跨 core 梯度求和，不照搬 overlap 平均

Reason：当前 core ownership loss 本身互不重叠，每个 core 对同一 macro 参数产生的是不同物理 loss 项的
真实偏导；总目标是这些 loss 之和，因此数学上应 scatter-add sum。平均会改变目标函数并使 context
出现次数影响优化尺度。

## 20. Open Questions

### 20.1 Blocking

None。

### 20.2 Non-blocking

- 大 macro 下 EDT float64 scratch 的进一步内存优化可在性能实测后单独处理；本 change 先保证 once/macro
  和无 core 重算。
- SDF 定期 reinitialization、narrow-band LevelSet、macro seam healing 另立 change。

## 21. Implementation Freedom

允许等价 EDT workspace、Adam CPU 实现方式、fixed-context strategy 的具体函数名和局部辅助函数拆分；
不得改变以下核心约束：macro-query 唯一 SDF、macro-global phi、core loss ownership、跨 core gradient sum、
macro barrier 后 Adam once、hard `phi<0`、STE 公式、padding0、公共 workflow 不含 Simple 专属参数。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Verify | Commit |
|---|---|---|---|---|
| A | 通用 pixel canvas + workflow method independence | pixel problem / `_ilt_workflow` / Simple adapter/tests | Simple 数值零回归 | `refactor(ilt): 通用化像素ILT上下文与画布接口` |
| B | SDF/STE/macro-global Adam solver | levelset.py + unit tests | TEST-001..008 | `feat(ilt): 实现宏级LevelSet优化器` |
| C | config/adapter/runner/artifacts | main/config/tests | TEST-009..011 + direct main | `feat(main): 接入LevelSetILT` |
| D | full smoke/docs/audit | docs/planning/reports | all commands | `docs(ilt): 完成LevelSet迁移报告` |

GitHub 写入/提交按当前任务授权范围执行；其它 push/PR 行为仍须单独授权。

## 23. Delivery and Final Audit

交付前必须：

1. 更新 baseline/status/revision；
2. 同步 `doc/contracts/ilt.md` 与 architecture/manual；
3. 输出 development_report/test_report，记录 SDF 初始化时间、RSS/CUDA peak、Simple 对照；
4. 证明一个 macro 的 SDF 调用次数恒为 1；
5. 证明 overlap A/B 同一物理 pixel 的 phi 一致、梯度求和且 batch invariant；
6. 证明 Adam 每 state 只有一次 macro update；
7. 证明 `_ilt_workflow` 不访问 Simple/LevelSet 专属数学配置；
8. Simple/full test、ruff、compileall、diff-check 全绿；
9. 审计无 workflow 复制、无 core-local 权威 phi、无独立 core EDT、无 gradient averaging、无 padding 伪透光。
