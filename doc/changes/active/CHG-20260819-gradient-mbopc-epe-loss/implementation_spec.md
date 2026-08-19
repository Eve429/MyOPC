---
id: CHG-20260819-gradient-mbopc-epe-loss
title: Gradient MB-OPC 可微 EPE Loss
type: implementation-spec
status: draft
baseline_commit: 540a0121eb06904bdc44ae7fe3bd491aeff22fb5
baseline_worktree: dirty
baseline_dirty_paths:
  - doc/changes/active/CHG-20260818-simple-ilt/implementation_spec.md
  - findings.md
  - progress.md
  - task_plan.md
  - .learnings/
scope:
  - opc/iteration/mbopc/gradient.py
  - main
  - config/gradient_mbopc.toml
  - tests/opc/iteration
  - tests/main
  - doc
depends_on:
  - doc/changes/completed/CHG-20260816-gradient-mbopc/implementation_spec.md
supersedes: []
---

# Gradient MB-OPC 可微 EPE Loss

## 0. Document Contract

本文档是本 change 的独立实现规格。仓库根 `AGENTS.md`、当前源码、测试及 `depends_on` 文档
仍是上位约束；本文只增量修改 Gradient MB-OPC，不重写已完成规格。

实现 AI MUST：

- 以本文的 requirements、invariants、interfaces 与 acceptance criteria 为实现目标；
- 开始前确认 baseline，保留所有既有未提交修改；
- 仅修改 File-Level Change Plan 列出的文件；扩大范围必须先修订并重新批准规格；
- 若本文状态不是 `approved`、Blocking Open Question 非空或 baseline 实质漂移，不得实施；
- 对 requirement、test、性能及兼容性逐项提供证据。

实现 AI MUST NOT：

- 把本 change 称为完整 DiffOPC、离散 EPE 的精确可微化或 MRC-clean OPC；
- 复制 NVlabs/DiffOPC 源码，或新增 OpenCV、Hydra、自定义 CUDA 等依赖；
- 修改 `layout/`、`geometry/`、`00_PAST/`、`opc/input/`、`lithography/`、
  `evaluation/` 或用户数据；
- 改变 segment ownership、core membership、macro 同步更新或 midpoint STE 语义。

规范词：MUST/MUST NOT 为强制项；SHOULD 除非有文档化证据否则必须遵守；MAY 为不改变契约
时的实现自由。

## 1. Objective

参考 DiffOPC 的 target-boundary normal-profile EPE surrogate，为现有 Gradient MB-OPC 增加
可选的第四项训练 loss。该 loss 必须支持任意方向边段、唯一 owner 计分、跨 core 光学梯度
累加、batch 不变归一化，并在关闭时保持现有算法与产物兼容。

## 2. Baseline and Evidence

### 2.1 Baseline

- Commit：`540a0121eb06904bdc44ae7fe3bd491aeff22fb5`
- Worktree：dirty；未提交项如 front matter 所列，均须保留，不得覆盖或夹带提交。
- 生产源码与测试相对 baseline 无修改。
- Linux CPU 基线：`450 passed, 8 skipped`；8 项均因当前环境无 CUDA 跳过。

### 2.2 Confirmed Facts

| Fact ID | Confirmed fact | Evidence | Verification method |
|---|---|---|---|
| FACT-001 | 每个可训练 segment 只有一个 owner 位移参数，但会从全部 core memberships 累加 mask gradient | `gradient.py::_prepare_macro_context`、`tests/opc/iteration/test_gradient_mbopc.py::test_cross_core_contributions_sum` | 静态阅读 + 测试 |
| FACT-002 | loss 在 ownership canvas 内唯一计分；全部 batch 同参数快照，屏障后一次 Adam step | `gradient.py::_evaluate_state`、`gradient.py::optimize_gradient_macro` | 静态阅读 + batch 测试 |
| FACT-003 | 当前连续目标只有 nominal/process/PV；离散 EPE 只作 `no_grad` diagnostic | `gradient.py::GradientMBOPCIterationRecord`、`gradient.py::_evaluate_state` | 静态阅读 |
| FACT-004 | hard-mask backward 在当前重构 segment midpoint 双线性采样 `dL/dMask` | `gradient.py::_EdgeGradientMask`、`tests/opc/iteration/test_gradient_mbopc.py::test_backward_is_two_times_bilinear_midpoint` | 静态阅读 + 测试 |
| FACT-005 | `epe_distance_dbu` 已用于参考 inner/outer 离散 EPE 探针 | `gradient.py::_prepare_macro_context` | 静态阅读 |
| FACT-006 | 配置解析支持 dataclass 默认字段；旧 TOML 可省略新增的尾部可选字段 | `main/configuration.py::_parse_config`、`tests/main/test_configuration.py::test_default_applied_when_field_absent` | 静态阅读 + 测试 |
| FACT-007 | DiffOPC 论文 eq. (6)-(8) 在 target boundary 法向 profile 上聚合 `(Z_nom-T)^2` 后使用 sigmoid | ICCAD 2024 论文 | 作者 PDF 公式核对 |
| FACT-008 | 官方实现只处理 H/V、硬编码 15 pixel、零误差每段仍贡献 0.5，且官方 notes 认为该 loss 尚需实验 | NVlabs/DiffOPC `src/opc/edgeilt.py::cal_epe_loss`，commit `bdc6e72` | 官方源码只读核对 |

### 2.3 External and Archive References

| Reference | Role | Adopt | Explicitly reject | Reason |
|---|---|---|---|---|
| DiffOPC 论文，DOI `10.1145/3676536.3676764`，eq. (6)-(16) | 算法依据 | 固定 target 边界法向 profile、nominal squared error、sigmoid 聚焦及同次反传 | Manhattan-only 适用域、未经定义的跨 core 工程语义 | 本项目需覆盖斜边与 macro/core batching |
| `https://github.com/NVlabs/DiffOPC`，commit `bdc6e72` | 源码行为证据 | H/V 实现作为轴对齐退化行为证据 | 源码复制、H/V Python 循环、固定 15 pixel、未归一化 loss、外部依赖 | 许可证、数值与性能契约不允许直接迁移 |
| `00_PAST/opc/iteration/diffopc/` | 只读历史比较 | owner-only 计分和 batch backward 作为比较项 | inner/outer squared-ReLU 公式及旧 Problem/API | 它不是官方 DiffOPC EPE 公式 |

官方仓库许可证限制非商业研究/评估使用。本 change MUST 根据论文公式及本项目契约独立实现，
MUST NOT 复制官方代码；`00_PAST/` 保持只读。

### 2.4 Uncertainty Boundary

- 连续 EPE surrogate 与最终二值轮廓的离散 EPE 不是同一指标；不得承诺两者在任意权重、版图或
  工艺模型下同步单调下降。
- DiffOPC 论文与源码没有定义 MyOPC 的斜边、core ownership、batch 归一化和画布边界语义；
  这些由本文明确冻结，属于 MyOPC 工程扩展。
- 独立 macro 仍不交换迭代态，macro 边界 context 仍为参考几何。

## 3. Current Behavior

1. `gradient.py::_evaluate_state` 计算
   `L_total = w_nominal*L_nominal + w_process*L_process + w_pv*L_pv`；
2. `gradient.py::_prepare_macro_context` 的 `epe_distance_dbu` 只生成离散 inner/outer probe，
   EPE 不参与 backward 或 best；
3. `gradient.py::optimize_gradient_macro` 每 state 遍历全部 core batch，累计同一参数快照的梯度，
   随后统一 Adam step；
4. `gradient.py::GradientMBOPCIterationRecord`、metrics 与 summary 只有三项连续 loss/weight；
5. `main._gradient_mbopc_workflow::save_macro_result` 的 result NPZ 保存状态与位移，不保存每项
   连续 loss。

## 4. Target Behavior

### REQ-001 — Optional configuration

`GradientMBOPCConfig` 与入口配置 MUST 增加：

```text
weight_epe: float = 0.0
epe_steepness: float = 4.0
```

`weight_epe` MUST 有限且非负；`epe_steepness` MUST 有限且严格大于 0。四个 loss 权重 MUST
至少一个为正，因此允许 EPE-only 配置。旧 TOML 省略二者时 MUST 正常加载并保持 EPE training
关闭。仓库示例 `config/gradient_mbopc.toml` MUST 显式设置 `weight_epe = 1.0`、
`epe_steepness = 4.0`，展示新功能已启用。

### REQ-002 — Fixed reference profiles

EPE measurement MUST 固定在 reference target segment，不随 current mask 移动。每个
`owner>=0` 的物理 segment MUST 只在其 owner core 建立一条 profile；context membership
MUST NOT 重复创建或重复计 loss。

设 `pixel_dbu=p`、`epe_distance_dbu=R*p`。当 `weight_epe>0` 时 MUST 要求 R 为正整数；
不满足时 MUST 在 optimizer 入口、GPU 分配前抛 `ValueError`。profile 采样偏移固定为：

```text
q = (-R+0.5, ..., -0.5, 0.5, ..., R-0.5) * p
xy_profile = reference_midpoint + q * reference_unit_normal
Q = 2R
```

所有方向、clear/opaque、outer ring/hole MUST 使用同一公式。H/V 必须退化为沿 y/x 的
对称 pixel-center line；MUST NOT 按 H/V、corner 或斜边设置不同分支。

### REQ-003 — Differentiable EPE formula

对同一 owner core 的 nominal wafer `Z_nom`、target transmission `T`：

```text
D = (Z_nom - T)^2
d_s = mean_q(bilinear_sample(D, xy_profile[s, q]))
penalty_s = 2 * (sigmoid(epe_steepness * d_s) - 0.5)
O = 当前 macro 的 owner segment 总数
L_epe = sum_s(penalty_s) / O
L_total = w_nominal*L_nominal
        + w_process*L_process
        + w_pv*L_pv
        + weight_epe*L_epe
```

`L_epe` MUST 在零 profile error 时严格为 0，范围为 `[0,1)`。profile 内用 mean 而非 sum，
segment 间按全 macro 固定 O 归一；每个 batch MUST 贡献 `sum(batch_penalty)/O`，不得用 batch
自己的 segment 数作分母。

### REQ-004 — Ownership and gradient semantics

profile 可以采到 owner core 的 simulation context，但 loss 仍属于该 owner segment，且只计一次。
其对 nominal wafer 和 hard mask 的梯度 MUST 与三项现有 loss 一起执行同一次
`batch_loss.backward()`。hard mask 梯度 MUST 继续通过全部 core memberships 的当前 midpoint
STE 累加到唯一 owner 参数；MUST NOT 建立 EPE 专用参数、owner-only mask gradient、overlap
averaging 或单独 optimizer step。

### REQ-005 — State, best and diagnostics

每个 state MUST 记录 `epe_loss`，`total_loss` 与 best 选择 MUST 包含加权 EPE。离散
`l2`/`pvband`/`epe`、valid/ambiguous probe diagnostic MUST 保持既有定义和执行时机，且不得作为
连续 total loss 的替代项或 tie-breaker。state0、更新屏障、末 state 纯评价语义不变。

### REQ-006 — Disabled-path compatibility

当 `weight_epe==0` 时 MUST：

- 不构造、传输或采样 EPE profile；
- 不执行 sigmoid 或 EPE autograd 运算；
- `epe_loss` 记录为 `0.0`；
- 在相同 seed/config/model 下，位移、三项旧 loss、total loss、best、离散指标与旧实现逐值一致。

新增 JSON 字段除外，旧 TOML、CLI 与 Gradient result NPZ MUST 向后兼容。

### REQ-007 — Vectorized and bounded execution

reference profile 坐标 MUST 每 macro 预计算一次并常驻 CPU；每 batch 只把其 owner profile
转到当前 device，并以张量化四邻域 gather 完成双线性采样。MUST NOT 在每个 state 对 segment
逐条调用 Python、KLayout、`grid_sample` per segment 或光刻 forward。

`D` MUST 复用 nominal L2 的 squared-error tensor。GPU 增量内存 MUST 为
`O(E_batch*Q)`，不得引入全 macro image、全 macro graph 或跨 batch graph 保留。

### REQ-008 — Artifacts and presentation

- `GradientMBOPCIterationRecord` 与 metrics JSON record MUST 新增有限 `epe_loss: float`；
- summary 的 `loss_weights` MUST 新增 `epe`，顶层 MUST 新增 `epe_steepness`；
- 每个 macro summary MUST 新增最佳状态的 `best_epe_loss`；
- CLI MUST 打印四项权重及 steepness；
- Gradient result NPZ version 与数组 MUST 不变；
- 开发手册、测试手册和专项报告 MUST 同步新公式、配置、结果与限制。

### REQ-009 — Failure semantics

profile 坐标越出完整 simulation canvas 表示输入/网格契约错误，MUST 抛带 core/segment 上下文的
`ValueError`，不得 clip、padding、跳过或改变分母。非有限的 profile sample、`L_epe`、总 loss 或
gradient MUST 按现有数值错误边界抛 `FloatingPointError`。其他异常不得吞掉或降级。

## 5. Scope

### 5.1 In Scope

- nominal wafer 上的可微 target-normal EPE profile loss；
- 任意方向向量化双线性 profile 采样；
- 配置、state record、metrics、summary、CLI 与文档更新；
- CPU/CUDA、斜边/hole、ownership/membership、batch/关闭兼容和真实光刻测试。

### 5.2 Out of Scope

- 修改既有离散 EPE evaluator、probe threshold 或 best tie-break；
- current-mask moving profile、contour extraction、signed-distance loss；
- MRC、SRAF、shot/curvature loss、自动权重、学习率或 optimizer 调参；
- 跨 macro 同步、全 reticle GPU、分布式执行；
- 逐值复现官方 Manhattan ray-casting benchmark。

### 5.3 Protected Areas

`layout/**`、`geometry/**`、`00_PAST/**`、`opc/input/**`、`lithography/**`、
`evaluation/**` 和用户数据 MUST NOT 修改。

## 6. Invariants

### INV-001 — Unique parameter

每个 owner segment 只有一个位移参数和一份 optimizer state；membership 只决定该参数可接收哪些
core loss 的 mask gradient。

### INV-002 — Unique EPE accounting

每个 owner segment 每 state 只产生一个 EPE penalty，并只由 owner core 计算；profile 进入
simulation context 不构成重复计分。

### INV-003 — Fixed state snapshot

同一 state 的全部 core batch 读取同一位移快照，所有 loss/gradient 完成后才能统一 step。

### INV-004 — Coordinate and polarity

profile 使用 reference midpoint、reference public unit normal、DBU 物理偏移和居中 canvas 坐标；
clear/opaque/hole 不改变公式或梯度符号约定。

### INV-005 — Normalization

`L_epe` 的分母只由当前 macro 固定 owner segment 数 O 决定，不随 core 数、membership 数、
batch size、core 顺序或有效离散 probe 数变化。

### INV-006 — Legal publication

新增 loss 只改变 Adam gradient；候选裁剪、合法重构、失败停止、已评价 best 和最终发布契约不变。

## 7. Architecture and Data Flow

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `opc.iteration.mbopc.gradient` | profile 预计算、可微采样、四项 loss、同步优化与 state record | 配置文件 I/O、离散 metric 定义、光刻模型实现 |
| `main.configuration` | TOML 类型/default 校验和 nm→DBU 配置映射 | pixel/profile 张量或 optimizer 状态 |
| `main._gradient_mbopc_workflow` | 产物、macro summary 与顶层 summary | EPE 数学或 gradient 构图 |
| `evaluation` | 继续提供离散 EPE diagnostic | 可微训练 EPE loss |

### 7.2 Dependency Direction

允许：

```text
main -> opc.iteration.mbopc.gradient -> opc.input / lithography / evaluation
```

禁止基础层反向依赖 gradient 方法；本 change 不增加新依赖方向。

### 7.3 Data Flow

```text
MacroProblem + target cache
        |
        +-- owner reference midpoint/normal --> CPU profile coordinates [O,Q,2]
        |
state d_k snapshot
        |
        +-- core batch: reconstruct/rasterize --> hard mask STE
        |       |
        |       +-- forward_many --> nominal/process wafers
        |       +-- ownership pixel losses
        |       +-- owner profile EPE loss
        |       +-- one backward --> membership midpoint gradients --> unique parameters
        |
        +-- all batches barrier --> Adam step --> legal d_(k+1)
        +-- macro total/record/best
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| Macro prepare | `MacroProblem`、config | owner profile 与分母 O | CPU static context | 每 state 重算 reference 坐标 |
| State/batch | 固定 `d_k`、core batch | raster、一次 `forward_many`、四项 loss/backward | 累积参数 gradient、detached metrics | 为 EPE 单独 forward |
| State barrier | 全部 batch gradient | 一次 Adam、裁剪、合法重构 | 已发布 `d_(k+1)` | batch 内提前 step |
| Persist | best snapshot、records | NPZ/JSON/GDS/summary | 权威产物 | 重算优化或改变 best |

## 8. Data Contracts

### 8.1 Configuration

```python
@dataclass(frozen=True)
class GradientMBOPCConfig:
    # 既有字段顺序不变
    weight_epe: float = 0.0
    epe_steepness: float = 4.0
```

`main.configuration.GradientConfig` MUST 增加同名、同默认字段。不得新增独立 EPE config
dataclass 或 mode 枚举。

### 8.2 State record

```python
@dataclass(frozen=True)
class GradientMBOPCIterationRecord:
    # 既有字段保持
    epe_loss: float = 0.0
```

该字段 MUST 放在既有字段末尾并保留默认值，使旧 positional/keyword 构造仍成立；生产记录在启用
EPE 时 MUST 显式写入实际值。空问题或关闭路径记录 `0.0`。JSON 是 additive schema 变更；
消费方不得依赖旧三键精确集合。

### 8.3 Private precomputed data

每 core 只需在既有私有 context 中追加 owner profile 坐标及对应 owner segment identity/数量映射；
数据必须是 CPU contiguous array。不得为单一调用建立公共类型、注册器或新模块。

| Data | Owner/lifetime | dtype/shape | Unit/coordinate | Mutability/residency |
|---|---|---|---|---|
| owner profile coordinates | `_GradientMacroContext` / macro solve | CPU `float64 [E_core,Q,2]` | canvas continuous `(x,y)` | immutable CPU，batch 临时转 GPU |
| nominal error | `_evaluate_state` / batch | device `float32 [B,H,W]` | transmission squared | transient GPU autograd tensor |
| batch profile samples | `_evaluate_state` / batch | device `float32 [E_batch,Q]` | transmission squared | transient GPU autograd tensor |
| macro EPE accumulator | `_evaluate_state` / state | Python/CPU float scalar | normalized loss | detached accumulation only |
| owner parameter/gradient | `optimize_gradient_macro` / macro solve | device `float32 [O]` | DBU / loss per DBU | 现有 Adam 唯一写者 |

### 8.4 Persisted Artifacts

| Artifact | Change | Required content |
|---|---|---|
| `gradient_metrics.json` | additive | 每 state 的 `epe_loss`；其他字段不变 |
| 顶层 summary JSON | additive | `loss_weights.epe`、`epe_steepness`、每 macro `best_epe_loss` |
| `gradient_result.npz` | none | version、数组、dtype、`allow_pickle=False` 契约不变 |
| GDS/PNG | none | 仍只由 best snapshot 与既有 workflow 产生 |

## 9. Interface Changes

| Symbol | Change |
|---|---|
| `GradientMBOPCConfig` | 尾部新增两个有默认值字段 |
| `GradientMBOPCIterationRecord` | 新增 `epe_loss` |
| `main.configuration.GradientConfig` | 尾部新增同名默认字段 |
| `resolve_gradient_config` | 只映射新增字段；签名不变 |
| `optimize_gradient_macro` | public 签名不变；入口新增启用条件下的单位校验 |
| metrics/summary/CLI | additive 字段、`best_epe_loss` 与显示更新 |

不得新增 public optimizer、公共 sampler、evaluation API 或 lithography API。

## 10. Detailed Algorithm

### 10.1 Precompute

1. 校验既有 config 与启用时的 distance/pixel 关系；该校验先于空 owner 快速返回；
2. 若 `weight_epe==0`，设置关闭标记并跳过本节余项；
3. 从 `problem.macro.pixel_dbu` 与 `epe_distance_dbu` 计算整数 R 和 Q；
4. 按 owner core 收集唯一 owner segment 的 reference midpoint 与 unit normal；
5. 一次向量化生成 `[E_core,Q,2]` DBU profile，再用既有坐标契约转为 canvas 浮点坐标；
6. 校验全部坐标落在 canvas 闭区间 `[0, canvas_pixels-1]`；越界立即失败；
7. 校验所有 core 的 profile 总数严格等于 O。

### 10.2 State evaluation

```text
macro_epe = 0
for each core batch using the same d_state:
    masks = exact hard raster with midpoint STE
    nominal, dose_max, defocus_min = forward_many(masks)
    nominal_error = (nominal - target)^2
    existing ownership losses use nominal_error/process/PV
    if EPE enabled:
        samples = bilinear_gather(nominal_error, owner_profiles)
        batch_epe = sum(2*(sigmoid(gamma*mean(samples, q))-0.5)) / O
        macro_epe += detached(batch_epe)
        batch_loss += weight_epe * batch_epe
    backward batch_loss once when gradient is requested
release batch graph
return four macro loss components and total
```

state N 的纯评价 MUST 构造并记录 EPE loss，但不得 backward 或 step。

### 10.3 Bilinear sampling

sampler MUST 使用当前项目 `(x,y)` canvas 契约，按 floor/ceil 四邻域和标准双线性权重一次 gather。
坐标恰位于最外侧整数 pixel center 时，高低邻居退化为同一边界 pixel；这不是越界坐标裁剪。
sampler 必须保留 `D` 的 autograd，profile coordinates 为固定常量，无需坐标梯度。禁止把真正
越界的坐标 silent clamp 回画布。

## 11. Ownership, State and Boundary Conditions

| State/data | Owner | Writers | Readers | Publish point | Lifetime |
|---|---|---|---|---|---|
| reference profile | macro static context | `_prepare_macro_context` once | `_evaluate_state` | context construction succeeds | macro solve |
| owner parameters | optimizer | Adam at state barrier only | every core batch | legal candidate reconstruction | macro solve |
| batch graph/samples | current core batch | PyTorch autograd | same batch backward | never persisted | one batch |
| `epe_loss` record | state evaluation | `_evaluate_state` | best/artifact workflow | all core batches complete | result lifetime |
| best snapshot | macro solver | strict lower total loss only | output workflow | evaluated state completes | macro solve/result |

失败时新增 EPE 计算不得发布参数或半份 state record；候选几何失败继续保留最后合法状态与历史 best。
顺序执行、不同 batch size 或未来 batch 并行不得改变 owner、分母和 state barrier 语义。

### 11.1 Boundary Conditions

| Condition | Required behavior |
|---|---|
| `weight_epe == 0` | 完全关闭新计算，记录 0，旧数值逐值兼容 |
| `weight_epe > 0`, distance 非 pixel 整数倍或 R<1 | optimizer 入口 `ValueError` |
| 合法 config 且 O == 0 | 保持现有 `no_owned_segments` 空结果，EPE 为 0 |
| 某 core/batch 的 owner profile 数为 0 | EPE contribution 为 0，不调用 sampler，不改变全局分母 O |
| profile 穿过 core ownership 边界 | 合法；从该 core 的完整 simulation canvas 采样 |
| profile 越过 simulation canvas | `ValueError`，不得裁剪或忽略 |
| 斜边、hole、corner 相邻段 | 每个固定 segment 按自身单位法向统一采样 |
| 离散 probe 为 ambiguous/invalid | 不影响连续 profile 集合或分母；diagnostic 原样记录 |
| 最后 state | 评价 EPE 并可成为 best，不建 backward graph、不 step |

## 12. Failure Semantics

| Failure | Exception/result | Handling |
|---|---|---|
| 非法 EPE 权重/steepness/distance | `ValueError` | GPU 分配前失败 |
| profile 越界或 owner 数不一致 | `ValueError`，含 macro/core/segment | 不裁剪、不跳过 |
| EPE sample/loss/gradient 非有限 | `FloatingPointError` | 不发布候选 |
| 候选几何非法 | 既有 `invalid_geometry` result | 保留历史 best，语义不变 |
| CUDA OOM/I/O/未知错误 | 原异常或明确上下文异常 | 不转换为成功结果 |

## 13. Performance and Memory

### PERF-001

每 macro 只预计算一次 profile；每 state 不重新求 reference midpoint/normal/坐标。

### PERF-002

光刻 `forward_many` 次数 MUST 与 EPE 关闭时完全相同；EPE MUST 复用 nominal output 与
`nominal_error`。

### PERF-003

GPU 仍只常驻一个 core batch。新增 GPU tensor 上界为 `O(E_batch*Q)`，CPU 静态数据上界为
`O(O*Q)`；不得随 macro/reticle 像素面积建立 dense EPE tensor。

### PERF-004

batch_size 或 core 遍历顺序不得改变公式、分母或参数更新语义；浮点归约顺序允许现有容差内差异。

性能测量 workload 为 `config/gradient_mbopc.toml` 指向的 gcd_30um、1×1 macro、10 updates；分别
记录 EPE enabled 与相同配置 `weight_epe=0` 的 total time、peak RSS、CUDA peak、每 state
`forward_many` 次数。本 change 对墙钟和峰值增量只记录实测基线、不预设无证据阈值；
`forward_many` 次数相等与上述渐进内存上界是硬性 pass/fail contract。

## 14. File-Level Change Plan

| File | Type | Action | Required minimal change | Covers |
|---|---|---|---|---|
| `opc/iteration/mbopc/gradient.py` | 业务代码 | modify | 配置/record 字段、profile 预计算、向量化 sampler、第四项 loss 与校验 | REQ-001..007, REQ-009 |
| `main/configuration.py` | 接线代码 | modify | 两个默认配置字段、四权重校验与映射 | REQ-001, REQ-006 |
| `main/_gradient_mbopc_workflow.py` | 接线代码 | modify | summary 增 EPE weight/steepness/best loss，metrics 接受新 record 字段 | REQ-008 |
| `main/run_gradient_mbopc.py` | CLI | modify | 打印四项权重与 steepness | REQ-008 |
| `config/gradient_mbopc.toml` | 配置 | modify | 显式启用示例 EPE 配置 | REQ-001 |
| `tests/opc/iteration/test_gradient_mbopc.py` | 测试 | modify | 公式、方向、owner、batch、disabled、异常、CPU/CUDA 回归 | TEST-001..007, TEST-009..011 |
| `tests/main/test_configuration.py` | 测试 | modify | 默认字段与解析契约 | TEST-006..008 |
| `tests/main/test_gradient_mbopc_runner.py` | 测试 | modify | config、JSON、summary、CLI additive schema | TEST-007, TEST-008 |
| `doc/contracts/mbopc.md` | 契约 | modify | Gradient EPE loss 契约 | REQ-001..009 |
| `doc/development_manual.md` | 手册 | modify | 配置、公式、运行和限制 | REQ-008 |
| `doc/test_manual.md` | 手册 | modify | 测试矩阵与命令 | TEST-012 |
| `doc/changes/completed/CHG-20260819-gradient-mbopc-epe-loss/development_report.md` | 报告 | add | 完成时新增实施差异、性能与清理审计 | AC-005, AC-008, AC-010 |
| `doc/changes/completed/CHG-20260819-gradient-mbopc-epe-loss/test_report.md` | 报告 | add | 完成时新增全量与专项结果 | AC-007, AC-008 |
| `task_plan.md`、`findings.md`、`progress.md` | 规划记录 | modify | 阶段、事实和验证同步 | AC-010 |
| 本规格 | 规格 | move | 实施裁决与 revision 后，完成时随目录从 active 移到 completed | Approval Gate |

若现有测试被拆分到同目录新文件，必须先在本规格 file plan 中列出，不得自行扩展生产结构。

## 15. Test Plan

### TEST-001 — Formula and zero baseline

用手算小 tensor 验证双线性采样、profile mean、zero-based sigmoid、macro owner mean；全零误差
必须得到精确 `epe_loss==0`。

### TEST-002 — Direction and topology

H/V profile 必须落在对应 y/x 法向 pixel-center；45°、clear/opaque、outer/hole 使用统一向量
公式，坐标和法向符号正确。

### TEST-003 — Autograd chain

stub differentiable lithography 下，EPE-only loss 必须对 hard mask 和 owner displacement 产生有限、
方向符合手算的非零梯度；必须经过既有 midpoint STE，而非直接对参数构造 loss。

### TEST-004 — Owner versus membership

构造一个 segment 同时属于多个 core membership 的问题：其 EPE penalty 只由 owner core 计一次，
但其他 core loss 的 mask gradient 仍累加到同一参数。反向 core 顺序时结果在规定容差内一致。

### TEST-005 — Batch invariance

相同 state 在不同 `gradient_batch_size` 下，四项 loss、total 与 displacement update 在规定容差内
一致；分母使用全 macro O。

### TEST-006 — Disabled compatibility

旧配置不含新字段、显式 `weight_epe=0` 与 baseline fixture 三路对照：profile helper 不被调用，
旧三项 loss、total、位移、best 和离散指标逐值一致，`epe_loss==0`。

### TEST-007 — Validation and boundaries

覆盖负/非有限 weight、非正/非有限 steepness、distance 非 pixel 整数倍、R=0、profile 越界和
非有限 sample；验证明确异常且没有 clip/skip。

### TEST-008 — State and artifacts

验证 state0..N、末 state 无 backward、EPE 参与 total/best、metrics record 新字段、summary 四权重
及 steepness、NPZ version/arrays 不变、旧 TOML 默认兼容。

### TEST-009 — EPE-only optimization

在确定性生成问题和可微 stub model 上设置旧三权重为 0、`weight_epe>0`；至少一次合法更新必须
使已评价 `epe_loss` 严格下降，best 快照与该 record 一致。

### TEST-010 — Real lithography and CUDA

ICCAD13 CPU 小问题验证 EPE loss/gradient 有限且可反传；有 CUDA 时验证 CPU/CUDA parity。无 CUDA
只允许按现有 marker skip，不得静默改用 CPU。

### TEST-011 — Performance guards

spy 验证启用 EPE 不增加 `forward_many` 次数、reference profile 每 macro 只建一次、batch 后 graph
可释放；源码审计确认 state 热循环无逐 segment Python 调用。

### TEST-012 — Full regression

执行全量 pytest、ruff、compileall、覆盖率与未调用/重复实现/异常入口审计；记录新增未覆盖分支及
处理结果。真实 gcd workload 记录 EPE-enabled 前后四项连续 loss、离散 EPE、耗时、RSS 与 CUDA
峰值，只作报告，不为多目标离散 EPE 设置无证据的单调门槛。

### 15.1 Planned test functions

新增测试函数分别使用中文 docstring，并在现有三个测试文件内按职责命名：

| Test IDs | Planned function(s) |
|---|---|
| TEST-001..003 | `test_epe_profile_formula_and_zero_baseline`、`test_epe_profile_coordinates_all_directions`、`test_epe_loss_backpropagates_through_midpoint_ste` |
| TEST-004..006 | `test_epe_owner_scores_once_membership_gradients_sum`、`test_epe_batch_size_invariant`、`test_epe_disabled_is_exactly_compatible` |
| TEST-007 | `test_epe_training_validation_fails_before_device_allocation`、参数化 config validation tests |
| TEST-008 | `test_epe_record_summary_and_npz_contract`、runner/config additive schema tests |
| TEST-009..011 | `test_epe_only_update_improves_evaluated_loss`、real CPU/CUDA tests、forward/profile construction spies |

### 15.2 Required Test Matrix

| Dimension | Cases | Expected distinction |
|---|---|---|
| Geometry | H、V、45°、outer、hole、跨 core | 坐标统一、owner 唯一、membership gradient sum |
| Objective | EPE-only、四项混合、EPE disabled | EPE 参与或完全退出 total/gradient |
| State/batch | state0、可更新 state、末 state；batch 1/多 core；正/逆序 | snapshot、barrier、best、分母不变 |
| Device | CPU、CUDA（可用时） | loss/gradient/update 在既有容差内一致 |
| Failure | 权重、steepness、pixel alignment、越界、nonfinite | 明确异常、无发布、无 silent clamp |
| Artifact | old/new TOML、metrics、summary、NPZ、CLI | 默认兼容、additive JSON、NPZ 不变 |

### 15.3 Verification Commands

```bash
/home/wzh/miniconda3/envs/myopc312/bin/python -m pytest -q tests/opc/iteration/test_gradient_mbopc.py tests/main/test_configuration.py tests/main/test_gradient_mbopc_runner.py
/home/wzh/miniconda3/envs/myopc312/bin/python -m pytest -q
/home/wzh/miniconda3/envs/myopc312/bin/python -m ruff check layout geometry opc lithography main tests
/home/wzh/miniconda3/envs/myopc312/bin/python -m compileall -q layout geometry opc lithography main tests
/home/wzh/miniconda3/envs/myopc312/bin/python main/run_gradient_mbopc.py config/gradient_mbopc.toml
```

CUDA parity 只在 `torch.cuda.is_available()` 为真时执行；否则仅对应 CUDA tests skip。用户 GDS
缺失时真实 gcd CLI smoke 可跳过并明确报告，但生成式 CPU integration 与其他核心 contract 不得跳过。

## 16. Requirement Traceability

| Requirement / invariant | Implementation symbol | Tests | Acceptance |
|---|---|---|---|
| REQ-001 | config dataclasses、`resolve_gradient_config` | TEST-006..008 | AC-004, AC-006 |
| REQ-002, INV-002, INV-004 | `_prepare_macro_context` | TEST-002, TEST-004, TEST-007 | AC-003 |
| REQ-003, INV-005 | private sampler、`_evaluate_state` | TEST-001, TEST-005, TEST-009 | AC-002, AC-003 |
| REQ-004, INV-001 | `_evaluate_state`、`_EdgeGradientMask.backward` | TEST-003, TEST-004 | AC-003 |
| REQ-005, INV-003, INV-006 | `_evaluate_state`、`optimize_gradient_macro` | TEST-008, TEST-009 | AC-001, AC-002 |
| REQ-006 | config default、`_prepare_macro_context`、`_evaluate_state` | TEST-006 | AC-004 |
| REQ-007 | `_prepare_macro_context`、private sampler | TEST-005, TEST-011, TEST-012 | AC-005, AC-008 |
| REQ-008 | record、workflow、CLI、docs | TEST-008, TEST-012 | AC-006, AC-010 |
| REQ-009 | config validation、optimizer entry、state finite checks | TEST-007, TEST-012 | AC-001, AC-009 |

## 17. Acceptance Criteria

- [ ] AC-001：全部 REQ 有通过的 traceability tests，无 blocking open question；
- [ ] AC-002：EPE-only 确定性优化至少一个已评价状态严格优于 state0，best 快照一致；
- [ ] AC-003：H/V、45°、hole 与跨 core owner/membership 语义测试通过；
- [ ] AC-004：旧 TOML 与 `weight_epe=0` 关闭路径通过逐值兼容测试；
- [ ] AC-005：启用 EPE 不增加光刻 forward 次数，显存仍只随 core batch 与 Q 增长；
- [ ] AC-006：metrics/summary/CLI 增量字段正确，Gradient result NPZ 不改版；
- [ ] AC-007：CPU 全量、ruff、compileall、覆盖率审计通过；CUDA 可用时 parity 通过；
- [ ] AC-008：真实 workload 的连续/离散指标及资源结果已记录，不做超出证据的质量声明；
- [ ] AC-009：未修改 Protected Areas、`00_PAST/`、用户数据或无关工作树修改；
- [ ] AC-010：完成差异、未调用函数、重复实现、异常入口和覆盖未命中分支审计，并更新两份报告及
  三份规划记录。

## 18. Compatibility

- 旧 Gradient TOML：兼容；新增字段使用默认值关闭 EPE；
- `optimize_gradient_macro`：签名兼容；
- Gradient result NPZ：version 与数组兼容；
- metrics JSON/summary：additive schema，精确键集合的消费者和测试必须同步；
- simple MB-OPC、input、lithography、evaluation：行为不变；
- 当 EPE 关闭时，数值优化路径除新增 `epe_loss=0.0` 记录外逐值兼容。

## 19. Decisions

### DEC-001 — Generalize the DiffOPC profile

采用 reference segment unit normal + bilinear sampling，统一支持 H/V 与斜边；不复制官方 H/V
分支。理由：当前 MyOPC 的 segment 契约不限于 Manhattan。

### DEC-002 — Use a normalized zero-based sigmoid

采用 `2*(sigmoid(gamma*mean(D))-0.5)`，不原样使用 `sigmoid(gamma*sum(D))`。理由：完美匹配
应为 0，且 loss 权重不应随 profile 像素数和 segment 数漂移。

### DEC-003 — Fixed target profile, unique owner loss

profile 固定在 reference segment，只由 owner core 计一次；mask gradient 仍经全部 memberships
累加。理由：保持 EPE 测量稳定、物理参数唯一和现有跨 core 光学耦合。

### DEC-004 — Reuse epe_distance_dbu

不增加第二个 EPE 距离配置；训练 profile 与离散 diagnostic 共用物理距离，并在启用训练时要求其
为 pixel 整数倍。理由：避免两个同义距离产生漂移。

### DEC-005 — Default off, example on

API 默认 `weight_epe=0` 保护旧配置与旧数值；仓库示例显式设为 1 展示功能。该权重仅是默认
示例，不宣称适用于所有 workload。

### DEC-006 — Keep implementation private and local

sampler/profile 逻辑留在 `gradient.py`，不扩展 evaluation 或新建模块。理由：当前只有一个生产
调用方，且它属于可微 optimizer 内部语义。

## 20. Open Questions

### Blocking

None. 本文处于 draft 是因为尚待用户批准，不表示存在未定义的实现语义。

### Non-blocking validation questions

- 示例 `weight_epe=1.0` 对真实 gcd workload 是否合适，只能通过实施 smoke 记录，不能预先声称最优；
- 连续 EPE 与离散 EPE 的相关性需在专项报告中比较，若不改善只记录事实，不在本 change 内自动
  调权或更换 loss。

## 21. Implementation Freedom

实现 AI MAY 决定局部变量名、张量 reshape/index 的等价写法，以及私有 sampler 是静态方法还是
模块级私有函数；前提是满足本文数值、ownership、内存和测试契约。

实现 AI MUST NOT 自行改变 profile 坐标、公式、归一分母、配置默认值、public API、持久化格式、
同步点、best 选择、依赖方向或 File-Level Change Plan。只有一个调用点且没有独立领域含义的
逻辑 SHOULD 留在调用函数内。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Required verification | Suggested local commit |
|---|---|---|---|---|
| A | 配置、profile sampler、第四项 loss 与单元测试 | `gradient.py`、configuration、iteration/config tests | TEST-001..007、ruff、compileall | `feat(mbopc): add differentiable epe loss` |
| B | workflow artifacts、CLI 与集成测试 | workflow、runner、TOML、main tests | TEST-008..011 | `feat(main): expose gradient epe loss` |
| C | 全量验证、手册、报告与审计 | docs、reports、planning records | TEST-012、全部 AC | `docs(mbopc): report gradient epe loss` |

每阶段只在验证通过后做本地 commit；不得包含用户数据、Simple ILT 未提交修改或其他无关差异。
未经用户明确授权不得推送远端。

## 23. Delivery and Final Audit

实现完成后 MUST：

- 更新开发/测试手册，在 change 目录完成时新增专项开发/测试报告，并同步三份规划记录；
- 记录实际文件与本规格偏差，contract 偏差必须事前获批；
- 记录环境、命令、pass/fail/skip、耗时、RSS/CUDA peak 与 smoke 产物；
- 搜索全部调用点，审计未调用函数、重复实现、异常吞噬、一次性抽象和覆盖未命中分支；
- 检查新增/修改 Python 的中文模块、函数、方法和测试 docstring 及关键中文注释；
- 提供 `git diff --check`、目标/全量测试、ruff、compileall 和直接 CLI smoke 证据；
- 明确 `layout/`、`geometry/`、`00_PAST/`、用户数据未修改，列出本地 commit 并注明未推送。

## 24. Known Limitations

- loss 是 target-normal 局部误差聚焦 surrogate，不直接计算 printed contour 到 target contour 的
  几何距离；
- 固定 reference profile 不追踪大位移后的 current contour；
- sigmoid 在大误差处会饱和，steepness 需要按 workload 验证；
- 独立 macro context 不随邻 macro 更新；
- 不处理 MRC、SRAF、shot count、曲率或拓扑变化；
- 任意角度扩展是 MyOPC 设计，不是官方 DiffOPC benchmark 的逐值复现。

## 25. Approval Gate

当前状态为 `draft`。只有用户明确批准本文后，才可：

1. 将 `status` 改为 `approved`；
2. 记录批准日期与必要裁决；
3. 按 File-Level Change Plan 实施、测试和本地提交。

未经明确批准不得进入实现。

## 26. Revision History

| Revision | Date | Status | Summary |
|---|---|---|---|
| 0.1 | 2026-08-19 | draft | 新增任意方向、唯一 owner、归一化且默认关闭的 Gradient EPE loss 设计 |
