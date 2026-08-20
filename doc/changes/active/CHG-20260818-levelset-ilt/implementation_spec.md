---
id: CHG-20260818-levelset-ilt
title: LevelSet ILT 迁移
type: implementation-spec
status: draft
baseline_commit: 02de825b4853f416b643a6a3e0092b4efe17495d
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
  - doc/architecture/dataflow/simple_ilt.md
supersedes: []
---

# LevelSet ILT 迁移

## 0. Document Contract

本文档是本 change 的唯一实现规格，已按 `migration@02de825b4853f416b643a6a3e0092b4efe17495d` 与已完成 Simple ILT 重新审查。该提交记录全量 `545 passed`；实施 AI MUST 在开始实现时再次记录实际 HEAD 与全量测试数，若共享接口已变化则先修订本文。

实现 MUST 复用当前 `PixelMacroProblem -> macro-global parameter -> core-batched lithography -> gradient scatter-add -> macro barrier -> ILTMacroResult -> run_ilt_workflow` 生命周期；以 OpenILT LevelSet 的 hard forward / surrogate backward / Adam 为算法参考，以 MyOPC 当前 macro/core ownership 为工程事实。

实现 MUST NOT：复制整套 ILT workflow；建立当前无调用价值的 ILT 基类/注册器；把 core ownership 当参数 ownership；按 core 独立生成 SDF、phi 或空间梯度场；在 core-local canvas 上重新计算同一物理像素的 `|grad(phi)|`；静默修正 NaN、I/O、CUDA 错误；修改 `00_PAST/**`、`layout/**`、`geometry/**`、`lithography/**`、`evaluation/**`。

## 1. Objective

在已经完成的 Simple ILT 像素型 macro/core 管线上实现 hard LevelSet ILT，并完成一次最小的公共 workflow 通用化，使 `python main/run_levelset_ilt.py [config.toml]` 可直接运行，同时不破坏 Simple ILT 数值行为。

核心约束升级为：**一个 macro query 内每个物理像素只有一个权威 phi，并且在同一 state 内只有一个权威 `|grad(phi)|`；core 只是光刻/loss 的计算窗口，不拥有独立参数、独立 SDF 或独立空间梯度。**

## 2. Baseline and Evidence

### 2.1 Baseline

- 审查基线：`02de825b4853f416b643a6a3e0092b4efe17495d`。
- 该基线提交说明全量测试为 `545 passed`；本文未在远端连接器环境重复执行测试。
- 当前生产 Simple ILT 已建立 `PixelMacroProblem -> optimize_simple_macro -> ILTMacroResult -> run_ilt_workflow`。
- 当前 `_ilt_workflow.py` 终评仍直接读取 `sigmoid_steepness`，属于 Simple 专属语义泄漏。
- 当前 `doc/architecture/dataflow/` 已按工作流拆分，Simple ILT 文档为 `doc/architecture/dataflow/simple_ilt.md`。

### 2.2 Confirmed Facts

| ID | Fact | Evidence |
|---|---|---|
| FACT-001 | Simple 参数域是 macro ownership；core 通过 `trainable_index_canvas` 映射同一 macro 参数 | `opc/iteration/ilt/simple.py`、`opc/input/pixel/problem.py` |
| FACT-002 | 同一参数可出现在多个 core context；Simple 将梯度 scatter-add 求和并在全 core 屏障后单次 SGD | `optimize_simple_macro` |
| FACT-003 | OpenILT LevelSet hard forward 为 `phi<0`，backward 为 `-|grad(phi)|*grad_output`，optimizer 为 Adam | `OpenILT/pyilt/levelset.py` |
| FACT-004 | OpenILT 在完整 tile levelset 上计算空间梯度；不存在 MyOPC core-local 重算问题 | `OpenILT/pyilt/levelset.py::gradImage/_Binarize` |
| FACT-005 | 旧迁移已有精确二维 EDT、全空/全满、strict-zero 的可复用参考 | `00_PAST/opc/iteration/ilt/levelset.py` |
| FACT-006 | 当前公共结果为 `ILTMacroResult`，共享 loss/curvature 位于 `ilt/_common.py` | 当前生产源码 |

### 2.3 Adaptation Boundary

OpenILT 的一张 tile 在参数/SDF/空间梯度语义上对应 MyOPC 的 **macro query**，不是单个 core。MyOPC 的 core overlap 只用于受显存限制的光刻与 loss 分批，因此任何会影响 LevelSet surrogate gradient 的 field 都必须先在 macro-query 坐标系中唯一确定，再裁给各 core。

LevelSet STE 是代理梯度，不宣称为 hard threshold 的数学真导数；macro 之间仍独立优化并使用固定外部 context，macro seam 联合优化不属于本 change。

## 3. Current Behavior

1. `PixelMacroProblem` 持久化一张 macro query `target_u8`，按需生成 core target、ownership、trainable-index、valid-context canvas。
2. `optimize_simple_macro` 在 macro ownership 上维护唯一 CPU 参数；同一 state 全部 core/batch 读同一快照，梯度回散求和后单次同步 SGD。
3. `run_ilt_workflow` 负责 prepare、逐 macro solve、binary final evaluation、artifact、merge、summary。
4. `_ilt_workflow::_binary_canvas/_evaluate_best_binary` 仍假定 config 含 `sigmoid_steepness`。
5. 当前无生产 LevelSet solver。

## 4. Target Behavior

### REQ-001：直接入口

提供 `python main/run_levelset_ilt.py [config.toml]`；默认 `config/levelset_ilt.toml`。输入、macro 生命周期、artifact、merge 与 Simple ILT 对齐。

### REQ-002：hard forward 与 external-gradient STE

前向 MUST 严格输出 `(phi < 0).float()`，`phi==0` 为不透光。

backward MUST 使用调用方提供的、与当前 state 同源的 `grad_magnitude`：

```text
grad_phi = -grad_magnitude * grad_output
```

`_LevelSetBinarize` MUST NOT 在 core-local canvas 内重新对 phi 做中心差分。推荐接口：

```python
_LevelSetBinarize.apply(phi_canvas, grad_magnitude_canvas)
```

其中 `grad_magnitude_canvas` 无梯度，仅作为 surrogate backward 系数。

### REQ-003：macro-query 唯一 SDF

每个 macro 从完整 `problem.target_u8` 生成且仅生成一次 `initial_query_phi[Hq,Wq]`：`target>=0.5` 为负，背景为正。不得按 core target 独立运行 SDF。

同一物理像素 P 出现在 core A/B 时，必须始终满足：

```text
phi_A(P) == phi_B(P)
```

### REQ-004：每 state 唯一空间梯度场

每个需要 backward 的 state，MUST 先在 CPU macro-query 坐标系构造唯一当前 field：

```text
current_query_phi = initial_query_phi.copy()
current_query_phi[macro ownership] = macro_phi_snapshot
```

然后在 **完整 current_query_phi** 上一次性计算中心差分：

```text
dx = (right - left) / 2
dy = (up - down) / 2
grad_magnitude = sqrt(dx^2 + dy^2)
```

macro-query 最外沿采用 replicate boundary。各 core 只允许从该唯一 `current_query_phi` / `grad_magnitude` 裁出窗口，不得重算。因此同一物理像素还必须满足：

```text
grad_magnitude_A(P) == grad_magnitude_B(P)
```

空间梯度仅在 `state_index < iterations` 时构造；末状态纯评价，不得无意义重复计算。

### REQ-005：精确 EDT 与生命周期

SDF 使用 `O(Hq*Wq)` 精确二维欧氏距离变换，不新增 SciPy/OpenCV 依赖；全空/全满返回有限且符号正确的常量场。EDT 每 macro 恰一次，不进入 state/core/batch 热循环。

EDT 实现 SHOULD 复用至多两张 float64 query workspace + `O(max(Hq,Wq))` 一维 scratch；foreground/background 两次 EDT 顺序执行并复用 workspace，不得同时常驻两套完整 float64 distance map。

### REQ-006：macro-global phi 与同步 Adam

可训练参数是 macro ownership 上唯一 `macro_phi[Hm,Wm]`。同一 state：所有 core 读取同一快照；loss 仅在各自 `ownership_canvas` 统计；local phi 梯度按 `trainable_index_canvas` scatter-add 到唯一 `macro_gradient`，绝不按出现次数平均；全 core 完成后恰一次 Adam update；Adam `m/v` 属于 macro 参数，不能按 core/batch 分裂。

batch size/core 顺序不得改变算法语义，只允许正常浮点累加容差。

### REQ-007：三类画布语义

1. `trainable_index_canvas>=0`：当前 macro 参数，使用 `macro_phi_snapshot`，即使位于当前 core context 也可通过该 core ownership loss 得梯度。
2. macro 外且 `context_valid_canvas=True`：只读物理 context，phi 使用 `initial_query_phi`，不可更新。
3. `context_valid_canvas=False`：纯数值 padding，mask transmission 严格 0，不解释为物理 T=0 后再做连续参数化。

core ownership 只定义 loss owner，不定义 parameter owner。

### REQ-008：context 宽度

LevelSet MUST 要求 `context_dbu >= pixel_dbu`。原因是 macro ownership 边缘的 query-level 中心差分和 3x3 曲率都至少需要一圈真实物理 context；该约束与 `curvature_weight` 是否为 0 无关。core seam 不再依赖 core-local replicate，因为空间梯度已经在 query field 上统一计算。

### REQ-009：损失、状态与 best

复用 `owned_continuous_losses`、`weighted_macro_loss`、`ILTStateRecord`、`ILTMacroResult`。N 次 Adam 更新对应 N+1 个完整已评价 macro state，末状态纯评价。best 只按完整 macro total loss 严格下降选择，平局保留更早 state。

输出：`best_parameters=best_phi`、`soft_mask=sigmoid(-best_phi)` 仅作诊断、`binary_mask=(best_phi<0)`。

### REQ-010：曲率

曲率作用于当前 hard mask，只统计 ownership 有效卷积区；权重为 0 时不得执行 conv。复用 `_common.curvature_loss`，不得复制第二套核。

### REQ-011：公共 workflow 通用化

`_ilt_workflow` MUST 不再读取 `sigmoid_steepness`、`mask_threshold`、phi 等算法字段，也不自行决定固定 context transmission。

`ILTMethod` 增加最小 final-context strategy：

```python
build_fixed_context_canvas: Callable[[PixelMacroProblem, int, object], np.ndarray]
```

公共 `_binary_canvas` 只将 `result.binary_mask` 写入 trainable 位置，其余消费 method strategy。

策略的数学实现 MUST 位于算法模块而不是 adapter：

- `simple.py::build_simple_fixed_context_canvas(...)`：真实 context=`sigmoid(beta*(2T-1))`，padding=0；Simple solver 训练时也复用同一 helper，保证训练/终评公式只有一个事实源。
- `levelset.py::build_levelset_fixed_context_canvas(...)`：真实 context=`target>=0.5` 的 hard transmission，padding=0；不得为终评重新运行 SDF，因为该 sign 与初始 LevelSet 严格等价。
- `_simple_ilt_workflow.py` / `_levelset_ilt_workflow.py` 只挂载 callable，不复制数学公式。

### REQ-012：artifact 一致性

继续由公共 workflow 生成配置、进度、资源统计、macro result NPZ、metrics、best.gds、summary、final merge/final lithography。方法文件为 `levelset_ilt_result.npz`，沿用公共 schema，`best_parameters` 语义为 phi。

## 5. Scope

### 5.1 In Scope

- macro-query 精确 SDF；query-global current phi / spatial-gradient field；hard-forward surrogate-backward；macro-global Adam。
- `PixelMacroProblem` 增通用 query-array→core-canvas helper。
- `_ilt_workflow/ILTMethod` 去除 Simple 专属 final-context 假设；Simple context helper 单一事实源。
- config、adapter、runner、tests、contracts、architecture dataflow、manual、reports。

### 5.2 Out of Scope

SDF reinitialization/fast marching/narrow-band；macro 间参数交换/seam healing；MRC/EPE/shot；CurvMulti/Multilevel 算法实现；无关 workflow/pixel API 重构。

## 6. Invariants

- **INV-001 Unique Phi**：同一 macro query 物理坐标只有一个当前 phi。
- **INV-002 Unique Spatial Gradient**：同一 state、同一物理坐标只有一个 `|grad(phi)|`，与 core/batch 无关。
- **INV-003 Sign**：`target>=0.5 -> initial phi<0`；`phi<0 <=> binary=True`；`phi==0 -> False`。
- **INV-004 Ownership Separation**：macro ownership=parameter ownership；core ownership=loss ownership；macro 外 query context 只读；padding transmission=0。
- **INV-005 Macro Barrier**：同一 state 全 core 读同一快照，raw gradient 求和，Adam 仅屏障后一步。
- **INV-006 Macro Best**：best 只来自完整已评价 macro state。
- **INV-007 Method-independent Workflow**：公共 workflow 不含 Simple/LevelSet 数学参数化。

## 7. Architecture and Data Flow

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `PixelMacroProblem` | query raster、通用 query-array→core canvas 坐标映射 | LevelSet/Simple 数学 |
| `ilt.levelset` | SDF、query spatial gradient、STE、macro phi/Adam、fixed-context helper、solver | GDS/TOML/artifact |
| `ilt.simple` | Simple solver + Simple fixed-context helper | workflow |
| `ilt._common` | 中性 result/record/loss/curvature | method context policy |
| `_ilt_workflow` | prepare/solve/final-eval/artifact/merge | sigmoid/phi 数学 |
| method adapters | `ILTMethod` 装配 + thin run | 数学复制/optimizer |

### 7.2 Data Flow

```text
PixelMacroProblem.target_u8 [Hq,Wq]
 -> signed_distance_initialization                         # once / macro
 -> initial_query_phi
      ├─ ownership crop -> macro_phi [Hm,Wm]              # trainable CPU
      └─ outside ownership -> fixed query phi             # readonly

for state in 0..N:
    current_query_phi = initial_query_phi
    overwrite ownership with macro_phi snapshot

    if state < N:
        query_grad_magnitude = spatial_gradient(current_query_phi)      # once/state
        macro_gradient = 0

    for core batch:
        phi_canvas  = query_array_canvas(current_query_phi)
        grad_canvas = query_array_canvas(query_grad_magnitude)          # backward states only
        target / ownership / trainable-index / valid

        local leaf values come from macro_phi snapshot
        hard = LevelSetBinarize(phi_canvas, grad_canvas)
        padding hard = 0
        printed = model.forward_many(three conditions)
        loss = ownership-only shared losses + optional curvature
        backward -> scatter-add local grad -> macro_gradient

    record full state / update macro best
    if state < N:
        Adam(macro_phi, summed macro_gradient) exactly once

best -> ILTMacroResult -> method-independent final evaluation/artifacts/merge
```

关键点：`phi_canvas` 与 `grad_canvas` 是 query-global field 的窗口；core 不重新定义二者。

## 8. Data Contracts

### 8.1 `LevelSetILTConfig`

六字段全部 required、无默认：`iterations:int>=1`、`step_size:finite>0`、`weight_process_l2:finite>=0`、`weight_pvband:finite>=0`、`curvature_weight:finite>=0`、`batch_size:int>=1`。bool 不得当 int。nominal 权重固定 1；SDF threshold 固定 0.5；binary threshold 固定严格 0。不得为兼容公共 workflow 增加 `sigmoid_steepness` 或 `mask_threshold`。

### 8.2 Tensor Residency

| Name | Shape/type | Resident | Lifetime |
|---|---|---|---|
| `initial_query_phi` | float32 `[Hq,Wq]` | CPU | macro solve |
| `current_query_phi` | float32 `[Hq,Wq]` | CPU | one state |
| `query_grad_magnitude` | float32 `[Hq,Wq]` | CPU | one backward state |
| `macro_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| Adam `m/v` | float32 `2×[Hm,Wm]` | CPU | macro solve |
| `macro_gradient` | float32 `[Hm,Wm]` | CPU | one backward state |
| `best_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| EDT scratch | <=2×float64 query + 1D scratch | CPU | initialization only |
| local phi/grad/hard/printed | canvas batch | GPU | one core batch |

完整 macro autograd graph 不得常驻 GPU。

## 9. Interface Changes

### IF-001 LevelSet API

```python
signed_distance_initialization(target, threshold=0.5)
spatial_gradient_magnitude(query_phi: np.ndarray) -> np.ndarray
build_levelset_fixed_context_canvas(problem, core_index, config) -> np.ndarray
optimize_levelset_macro(problem, model, config, *, on_tiles_completed=None) -> ILTMacroResult
```

`spatial_gradient_magnitude` 的权威定义域是 macro query。

### IF-002 Generic query-array canvas

```python
PixelMacroProblem.query_array_canvas(
    array: np.ndarray,
    core_index: int,
    *,
    fill_value=0,
) -> np.ndarray
```

要求 `array.shape == target_u8.shape`，复用 `_context_window/_center_padding`；`target_canvas` SHOULD 改为调用该 helper。不得为单 core 构造 O(macro_pixels) 全宏索引表。

### IF-003 ILTMethod strategy

`ILTMethod` 从四字段扩展一个 `build_fixed_context_canvas` callable。Simple 与 LevelSet adapter 均提供；公共 final evaluation 只认该接口。

### IF-004 Config/CLI

`CONFIG_SECTIONS[LevelSetILTConfig] = "levelset_ilt"`；新增 `LEVELSET_ILT_METHOD`、`run_levelset_ilt`、`main/run_levelset_ilt.py` 与 `config/levelset_ilt.toml`。

## 10. Algorithm

### 10.1 SDF

mixed target 分别计算到前景/背景的精确 EDT；前景位置取负 inside distance，背景取正 outside distance。全前景/全背景使用有限常量 `-/+max(Hq,Wq)`。距离单位为 pixel center。

### 10.2 State Loop

```text
query_target = target_u8 / 255
initial_query_phi = SDF(query_target)                  # once
macro_phi = ownership_crop(initial_query_phi).copy()
Adam state = zeros_like(macro_phi)

for state_index in 0..N:
    build_gradient = state_index < N

    current_query_phi = initial_query_phi.copy()
    current_query_phi[ownership] = macro_phi           # authoritative snapshot

    if build_gradient:
        query_grad = spatial_gradient_magnitude(current_query_phi)     # once
        macro_gradient = zeros_like(macro_phi)

    sums = 0
    for core batch:
        phi_canvas = crop(current_query_phi)
        grad_canvas = crop(query_grad) if build_gradient else None
        trainable = trainable_index_canvas

        # 为 autograd 建 local leaf；数值必须与 phi_canvas trainable 位一致
        local = gather(macro_phi, trainable>=0)
        phi_device = fixed phi canvas with trainable positions replaced by local
        hard = LevelSetBinarize(phi_device, grad_canvas) if build_gradient \
               else (phi_device < 0)
        hard[padding] = 0

        printed = forward_many(hard, nominal/dose_max/defocus_min)
        loss = shared ownership losses + optional curvature
        accumulate scalars

        if build_gradient:
            backward(loss)
            scatter-add local.grad -> macro_gradient

    record full macro state; update best on strict decrease
    if build_gradient:
        validate gradient
        Adam_step(macro_phi, macro_gradient) exactly once
        validate phi/m/v
```

禁止在 batch 内更新参数；禁止从某个 core canvas 计算 `grad_canvas`。

### 10.3 STE

```python
forward(phi, grad_mag):
    return (phi < 0).to(phi.dtype)

backward(grad_output):
    return -grad_mag * grad_output, None
```

`grad_mag` 必须 detach/no-grad；常量 field 的梯度幅值为 0，不做 fallback。

### 10.4 Fixed Context

训练时 macro 外 query context 的 phi 固定为 `initial_query_phi`；hard transmission 因而等价于 `target>=0.5`。终评 LevelSet fixed context 直接从 target hard sign 构造，避免第二次 EDT。Simple fixed context 继续保持 `sigmoid(beta*(2T-1))`，但训练和终评必须共用 `simple.py` 中同一 helper。

## 11. Ownership and State

`macro_phi + Adam m/v` 由当前 macro 独占；`trainable_index_canvas` 是 core→macro 参数的唯一映射。每个 core 的 `ownership_canvas` 唯一计分，因此不同 core loss 对同一参数的偏导是总目标的不同项，必须求和而非平均。

`initial_query_phi` 全程只读；`current_query_phi/query_grad_magnitude` 每 state 派生；core-local tensor 均为临时窗口，不得成为权威状态。

## 12. Error Handling

非法 config、target/shape/canvas mismatch、`context_dbu<pixel_dbu` -> `ValueError`。非有限 loss、macro gradient、macro phi、Adam state、query gradient -> `FloatingPointError`。EDT/CUDA/I/O 未知异常原样传播；不得跳过 core/pixel、重置 optimizer 或发布部分成功 macro。

## 13. Performance and Memory

- SDF：`1 call / macro`，`O(HqWq)`。
- query spatial gradient：`iterations calls / macro`，即每个需要 backward 的 state 恰一次；不得随 core_count/batch_size 增长。
- 每 state/core batch 仍仅一次 `forward_many(three conditions)`；curvature=0 不构建 conv。
- CPU 允许 `initial/current/query_grad` 三张 float32 query 量级临时/常驻组合；可通过 workspace 复用进一步降低，但不得牺牲语义。
- EDT 峰值目标：至多两张 float64 query workspace，foreground/background 顺序复用。
- GPU 只保留当前 core batch 的 autograd 图；不得常驻完整 macro phi/grad graph。
- 不缓存所有 core 的完整 `trainable/ownership/valid/phi` canvas；仅可缓存轻量 window/padding 元数据。
- smoke 记录 total、SDF、query-gradient 时间，CPU RSS、CUDA peak，并与同输入 Simple 只做事实对照。

## 14. File-Level Change Plan

| File / Symbol | Action | Contract change | Reason |
|---|---|---|---|
| `opc/input/pixel/problem.py` | modify | 增 `query_array_canvas`；`target_canvas` 复用 | 坐标事实唯一 |
| `opc/iteration/ilt/levelset.py` | add | config、EDT/SDF、query spatial gradient、external-grad STE、fixed context、macro Adam solver | 核心算法 |
| `opc/iteration/ilt/simple.py` | modify | 提取并复用 `build_simple_fixed_context_canvas` | 训练/终评单一公式源 |
| `opc/iteration/ilt/__init__.py` | modify | 导出 LevelSet API | 公共入口 |
| `main/_ilt_workflow.py` | modify | `ILTMethod` 增 fixed-context strategy；终评移除 Simple 专属字段 | 方法无关 workflow |
| `main/_simple_ilt_workflow.py` | modify | 挂载 Simple fixed-context helper | Simple 零回归 |
| `main/_levelset_ilt_workflow.py` | add | METHOD + thin run，仅装配 callable | 新方法接入 |
| `main/configuration.py` | modify | 注册 `[levelset_ilt]` | 配置 |
| `main/run_levelset_ilt.py` | add | 直接入口 | CLI |
| `config/levelset_ilt.toml` | add | smoke 配置 | 可运行样例 |
| `tests/opc/input/test_pixel_problem.py` | modify | query-array 映射/target 回归 | 坐标正确性 |
| `tests/opc/iteration/test_levelset_ilt.py` | add | SDF、唯一 phi/grad、STE、Adam、cross-core、real model | 核心验证 |
| `tests/opc/iteration/test_simple_ilt.py` | modify as needed | context helper 重构零回归 | 兼容 |
| `tests/main/test_levelset_ilt_runner.py` | add | config/adapter/CLI/artifact/final context | 集成 |
| `tests/main/test_simple_ilt_runner.py` | modify as needed | workflow 通用化零回归 | 兼容 |
| `tests/main/test_configuration.py` | modify | 新 section 严格解析 | 配置 |
| `doc/contracts/ilt.md` | modify | LevelSet + 新 ILTMethod contract | 契约 |
| `doc/architecture/system.md` | modify | LevelSet 组件 | 架构 |
| `doc/architecture/dataflow/index.md` | modify | 索引 LevelSet dataflow | 当前文档结构 |
| `doc/architecture/dataflow/levelset_ilt.md` | add | 函数级流向 + 伪代码 + 边界 | 当前文档结构 |
| `doc/development_manual.md`、`doc/test_manual.md` | modify | 使用/测试 | 交付 |
| change active→completed + `development_report.md` + `test_report.md` | move/add | baseline、实施偏差、测试/性能 | 交付 |

不得修改已废弃的 `doc/architecture/dataflow.md` 路径；不得借本 change 重构 unrelated workflow。

## 15. Test Specification

### TEST-001 SDF exactness/lifecycle

rectangle/single/hole/all-empty/all-full 与 brute-force 最近距离逐值比对；sign/strict-zero 明确。spy：一个 macro 的 SDF 调用数恒为 1，与 core/state/batch 无关。

### TEST-002 Query-array mapping

标号 query array 在 A/B overlap 同一物理坐标裁值一致；padding fill 正确；`target_canvas` 重构前后逐值一致。

### TEST-003 Query-global spatial gradient

手算中心差分与 replicate query outer boundary。构造物理像素 P：P 在 core A 的 local window 边缘、在 core B 内部，断言从 query-global field 裁出的 `phi_A(P)==phi_B(P)` 且 `grad_A(P)==grad_B(P)`。

测试必须能击穿错误实现：若在各 core-local canvas 上自行 pad+差分，则该场景结果不同并失败。

spy：`spatial_gradient_magnitude` 调用数恰为 `iterations`/macro，不随 core_count/batch_size 增长；末纯评价 state 不调用。

### TEST-004 STE

forward 严格 `phi<0`；给定显式 `grad_mag` 和 `grad_output`，backward 逐值等于 `-grad_mag*grad_output`；对 grad_mag 返回 None；证明 backward 不调用任何空间差分 helper。

### TEST-005 Cross-core gradient sum

局部耦合 differentiable lithography stub；同一 macro 参数 P 同时影响 A/B ownership loss，验证两个 core 使用同一 query-global grad coefficient，local raw gradients scatter-add 求和；batch=1/2 结果一致；owner-only 或平均实现失败。

### TEST-006 Macro Adam barrier

独立数值 Adam 参考：所有 core 同一快照、每 state 恰一次 macro step、m/v 唯一；N=2 -> 3 个评价 state；禁止 batch 内提前 step。

### TEST-007 Context/padding

macro ownership 参数可从邻 core context 得梯度；macro 外 context 固定；padding hard transmission=0；`context_dbu<pixel_dbu` 前置失败。

### TEST-008 Loss/curvature/real model

共享 nominal/process/PV 逐值；curvature ownership-only 且 weight0 无 conv；真实 ICCAD13 CPU 一轮 backward/update/final eval finite；有 CUDA 时 smoke/parity 并记录资源。

### TEST-009 Workflow method independence

fake method 证明 `_ilt_workflow` 不访问 `sigmoid_steepness/mask_threshold/phi`。Simple 训练与 final evaluation 的 fixed context 调用同一 helper；LevelSet final context 不触发 SDF。

### TEST-010 CLI/artifact/Simple zero regression

合法/缺键/未知键/nonfinite/bool-int；仓库外 cwd 默认/显式 config；LevelSet result/metrics/best/final/summary schema 与公共格式一致。Simple 固定 workload 的 best/binary/binary_l2/pvband/artifact key 与重构前一致；全量测试绿色。

### 15.1 Verification Commands

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

## 16. Acceptance Criteria

- [ ] AC-001：直接入口、method、artifact/merge contract 完成。
- [ ] AC-002：SDF once/macro，精确距离/sign/strict-zero 通过。
- [ ] AC-003：同一 overlap 物理像素在 A/B core 中 `phi` 与 `|grad(phi)|` 均一致；query gradient 每 backward state 只计算一次。
- [ ] AC-004：cross-core gradient sum、macro Adam barrier、batch invariance 通过。
- [ ] AC-005：context fixed、padding0、N+1 best/loss/curvature/真实模型通过。
- [ ] AC-006：`_ilt_workflow` 无方法数学字段；Simple context helper 单一事实源且数值零回归。
- [ ] AC-007：contracts、`dataflow/levelset_ilt.md`、manual、development_report、test_report、baseline/revision evidence 完成。

## 17. Compatibility and Migration

- `PixelMacroProblem` NPZ v1 不变，只增加派生 helper。
- `ILTMacroResult` 与 result NPZ schema 不变。
- `ILTMethod` 有一次向后兼容字段扩展；Simple adapter 同步提供 strategy。
- Simple CLI/数值必须零变化。
- LevelSet 不承诺与 OpenILT 单 tile 逐值一致；保留其 hard/STE/Adam 核心算法，工程上增加 macro-query 唯一 field 与 core-batched optics。
- CurvMulti/Multilevel 的 active spec 在实施前 MUST 以完成后的真实 `ILTMethod` contract 重新核对；其中“ILTMethod 必须保持四字段完全不变”的旧假设不得继续作为阻塞条件。

## 18. Decisions

### DEC-001 OpenILT tile -> MyOPC macro query

LevelSet field 是物理连续状态；core 是计算切片。按 core 建 SDF 会直接产生多份同位置 phi。

### DEC-002 Query-global spatial gradient

仅保证唯一 phi 不够。OpenILT backward 乘 `|grad(phi)|`；若在 core-local canvas 重算中心差分，同一 P 会因 local window 边缘/padding 得到不同 surrogate coefficient，导致优化依赖 core 切分。故每 backward state 必须先在 macro query 上计算唯一梯度场，再裁给 core。

### DEC-003 Macro Adam

Adam 是带状态的非线性更新；按 core/batch step 会让结果依赖切分/order，必须 raw gradient 全部求和后一步更新。

### DEC-004 Fixed-context strategy belongs to algorithm module

adapter 只装配，不复制公式；Simple solver 与 final evaluation 共用 `simple.py` helper，避免未来训练/终评漂移。LevelSet final hard context 直接由 target sign 得到，禁止重复 EDT。

### DEC-005 EDT memory bound

精确 EDT 保留，但 foreground/background 顺序执行并复用两张 float64 query workspace；算法正确性优先于极端微优化。

## 19. Open Questions

Blocking：None。

Non-blocking：SDF reinitialization、narrow-band、macro seam healing、query workspace 进一步原地复用均另立 change；本 change 不引入这些复杂度。

## 20. Implementation Freedom

允许等价 EDT workspace、显式或 PyTorch CPU Adam、局部 helper 命名调整；不得改变：macro-query 唯一 SDF、每 backward state 唯一 query phi/grad field、macro-global phi、core loss ownership、cross-core raw-gradient sum、macro Adam once、hard `phi<0`、external-grad STE、padding0、公共 workflow 无方法数学参数。

## 21. Implementation Stages

| Stage | Objective | Main files | Verify |
|---|---|---|---|
| A | 通用 query canvas + method-independent final context + Simple helper 单一事实源 | pixel problem / simple / `_ilt_workflow` / Simple adapter | Simple 数值零回归 |
| B | SDF + query-global spatial gradient + STE + macro Adam solver | levelset.py | TEST-001..008 |
| C | config/adapter/runner/artifacts | main/config/tests | TEST-009..010 + direct main |
| D | docs/full smoke/audit | contracts/dataflow/manual/reports | all commands |

## 22. Delivery and Final Audit

交付前必须证明：

1. 实施 baseline/status/test count 已更新；
2. 一个 macro 的 SDF 调用恒为 1；
3. query spatial-gradient 调用恒为 `iterations`，不随 core/batch 增长；
4. overlap A/B 同一物理 P 的 phi 和 grad magnitude 均一致；
5. 同一参数来自多个 core ownership loss 的梯度为 sum，不平均；
6. Adam 每 backward state 只有一次 macro update；
7. LevelSet final context 不重复 SDF；Simple 训练/终评 context 公式来自同一 helper；
8. `_ilt_workflow` 不访问 Simple/LevelSet 数学配置；
9. 无 workflow 复制、core-local 权威 phi/grad、独立 core EDT、gradient averaging、padding 伪透光；
10. `doc/architecture/dataflow/levelset_ilt.md` 与 index 已更新；
11. Simple/full tests、ruff、compileall、diff-check 全绿并输出 development_report/test_report。
