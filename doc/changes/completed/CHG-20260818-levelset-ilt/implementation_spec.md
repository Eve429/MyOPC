---
id: CHG-20260818-levelset-ilt
title: LevelSet ILT 迁移
type: implementation-spec
status: approved
baseline_commit: 3bd72025aa100fa5c8d8606a3dc30314457d401f
baseline_worktree: unknown
baseline_dirty_paths: []
scope:
  - opc/iteration/ilt
  - main
  - config
  - tests
  - doc
  - requirements.txt
depends_on:
  - doc/changes/completed/CHG-20260818-simple-ilt/implementation_spec.md
  - doc/contracts/ilt.md
  - doc/contracts/lithography.md
  - doc/architecture/dataflow/simple_ilt.md
supersedes: []
---

# LevelSet ILT 迁移

## 0. Document Contract

本文档是本 change 的唯一实现规格。当前审查基线为 `migration@3bd72025aa100fa5c8d8606a3dc30314457d401f`；该提交仅更新 LevelSet 规格，生产代码与其父提交 `02de825b4853f416b643a6a3e0092b4efe17495d` 一致。父提交已完成 Simple ILT，记录全量 `545 passed`。实施 AI MUST 在开始实现时再次记录实际 HEAD、worktree 状态和全量测试数；若共享接口已变化，必须先修订本文。

实现 MUST 复用当前 `PixelMacroProblem -> macro-global parameter -> core-batched lithography -> gradient scatter-add -> macro barrier -> ILTMacroResult -> run_ilt_workflow` 生命周期。LevelSet 只替换参数化、代理梯度、优化器和固定 context 语义，不复制整套 workflow。

实现 MUST NOT：建立当前无调用价值的 ILT 基类/注册器；把 core ownership 当参数 ownership；按 core 独立生成 SDF/phi/空间梯度系数；在 core-local canvas 上重新计算同一物理参数的 `|grad(phi)|`；在 batch 内更新参数；平均跨 core 梯度；静默修正 NaN、I/O、CUDA 错误；修改 `00_PAST/**`、`layout/**`、`geometry/**`、`lithography/**`、`evaluation/**`。

## 1. Objective

在已完成的 Simple ILT 像素型 macro/core 管线上实现 hard LevelSet ILT，使 `python main/run_levelset_ilt.py [config.toml]` 可直接运行，同时保持 Simple ILT 的现有热路径和数值行为不变。

核心工程约束：

1. 一个 macro ownership 像素只有一个权威 `phi` 参数；
2. 每个需要 backward 的 state，每个可训练 macro 参数只有一个权威 `|grad(phi)|` 代理梯度系数；
3. core 只是光刻/loss 计算窗口，不拥有独立参数、独立 SDF 或独立 LevelSet 空间梯度；
4. 同一 macro 参数即使同时出现在多个 core context，也必须使用相同 `phi`、相同 `|grad(phi)|`，各 core loss 对它的梯度只做 raw sum；
5. Simple ILT solver 不因本 change 被迫改写 CPU/GPU 热路径。

## 2. Baseline and Evidence

### 2.1 Confirmed Current Facts

| ID | Fact | Evidence |
|---|---|---|
| FACT-001 | Simple 参数域定义在 macro ownership | `opc/iteration/ilt/simple.py` |
| FACT-002 | core 通过 `trainable_index_canvas` 映射同一 macro 参数 | `opc/input/pixel/problem.py` |
| FACT-003 | 同一参数可出现在多个 core context；Simple 将 local gradient scatter-add 后在 macro barrier 处单次 SGD | `optimize_simple_macro` |
| FACT-004 | 公共结果已统一为 `ILTMacroResult`；连续 loss/curvature 位于 `ilt/_common.py` | 当前生产源码 |
| FACT-005 | `_ilt_workflow` final evaluation 仍直接读取 `sigmoid_steepness`，存在 Simple 专属语义泄漏 | `main/_ilt_workflow.py` |
| FACT-006 | 旧 LevelSet 参考已包含 hard `phi<0`、空间梯度代理反向、Adam 和 raster EDT | `00_PAST/opc/iteration/ilt/levelset.py` |

### 2.2 Algorithm Adaptation Boundary

OpenILT LevelSet 的 hard forward、`-|grad(phi)|*grad_output` surrogate backward 与 Adam 是算法参考；MyOPC 的 macro/core ownership、fixed context、N+1 state、artifact/merge 生命周期是工程事实。

OpenILT 原始执行域是一张完整 tile；MyOPC 为控制显存将光刻/loss 拆成多个 core。不能因此把一个物理 LevelSet 参数复制成多份 core-local 参数或多份空间梯度系数。

MyOPC 本 change 使用 **raster signed-distance field**：距离定义在像素中心上，不宣称与 OpenILT polygon-edge initializer 逐值一致。该差异必须在测试中固定，禁止后续以“对齐 OpenILT”为由无规格修改 SDF 数值定义。

## 3. Current Behavior

1. `PixelMacroProblem` 持久化 macro query 的单张 `target_u8`，按需生成 core target、ownership、trainable-index、valid-context canvas。
2. `optimize_simple_macro` 在 macro ownership 上维护唯一 CPU 参数，同一 state 全部 core/batch 读取同一参数快照；local gradient 回散求和后仅做一次同步 SGD。
3. `run_ilt_workflow` 负责 prepare、逐 macro solve、binary final evaluation、artifact、merge、summary。
4. `_ilt_workflow::_binary_canvas/_evaluate_best_binary` 仍假定算法 config 存在 `sigmoid_steepness`。
5. 当前没有生产 LevelSet macro solver。

## 4. Target Behavior

### REQ-001：直接入口

提供 `python main/run_levelset_ilt.py [config.toml]`；默认配置 `config/levelset_ilt.toml`。输入、macro 生命周期、artifact、merge、summary 与 Simple ILT 对齐。

### REQ-002：hard forward 与 external-gradient STE

前向严格：

```text
hard = (phi < 0).float()
```

`phi==0` 必须是不透光。

代理反向严格：

```text
grad_phi = -grad_magnitude * grad_output
```

`grad_magnitude` 由调用方提供，属于当前 macro state 的只读系数，不参与 autograd。推荐：

```python
_LevelSetBinarize.apply(local_phi, local_grad_magnitude)
```

其中 `local_phi` 与 `local_grad_magnitude` 对同一 `trainable_index_canvas` gather；`_LevelSetBinarize` 内部 MUST NOT 再做空间差分。

### REQ-003：SDF once/macro，使用生产级 EDT

每个 macro 从完整 `problem.target_u8` 仅生成一次 `initial_query_phi[Hq,Wq]`：

```text
binary = target_u8 / 255 >= 0.5
background pixel: phi = +distance(pixel_center, nearest foreground pixel_center)
foreground pixel: phi = -distance(pixel_center, nearest background pixel_center)
```

mixed target 使用 `scipy.ndimage.distance_transform_edt`。本 change 允许并要求在 `requirements.txt` 增加 `scipy`；禁止把旧版逐 row/column 的 Python EDT 循环直接搬入生产热路径。旧纯 Python/暴力距离实现仅可作为小尺寸测试 oracle，不得作为 production solver。

全前景/全背景显式返回有限常量场：

```text
all foreground -> -max(Hq, Wq)
all background -> +max(Hq, Wq)
```

SDF 只在初始化执行一次，不进入 state/core/batch 循环。实现 SHOULD 顺序执行 inside/outside EDT 并及时释放中间数组，避免无意义同时常驻多套 float64 distance map。

### REQ-004：macro-global phi + 1 pixel halo

可训练参数仅为 macro ownership：

```text
macro_phi[Hm,Wm]
```

初始化值是 `initial_query_phi` 的 macro ownership crop。

每个需要 backward 的 state，不再构造完整 `current_query_phi[Hq,Wq]` 与完整 `query_grad_magnitude[Hq,Wq]`。改为构造 macro ownership 周围 1 pixel 的 LevelSet halo：

```text
phi_halo[Hm+2,Wm+2]
```

语义：

```text
phi_halo[1:-1, 1:-1] = 当前 macro_phi snapshot
phi_halo 外围 1 pixel = initial_query_phi 中对应的固定物理 context
```

然后只计算当前可训练参数的唯一梯度系数：

```text
dx = (phi_halo[1:-1,2:] - phi_halo[1:-1,:-2]) / 2
dy = (phi_halo[2:,1:-1] - phi_halo[:-2,1:-1]) / 2
macro_grad_magnitude = sqrt(dx^2 + dy^2)   # [Hm,Wm]
```

因此同一 macro 参数 P 无论出现在 core A/B 哪个位置，都只有：

```text
macro_phi[P]
macro_grad_magnitude[P]
```

两份权威值。core-local canvas 位置、padding、batch size 不得改变这两个值。

### REQ-005：context 宽度

LevelSet MUST 要求：

```text
context_dbu >= pixel_dbu
```

原因是 macro ownership 边缘参数的中心差分至少需要一圈真实物理 context。该约束与 `curvature_weight` 是否为 0 无关。

这里的 1-pixel halo 是 **macro 参数梯度的物理邻域**，不是 core padding。不得用 core-local replicate padding 替代。

### REQ-006：core batch 只 gather，不定义 LevelSet field

每个 core batch 继续复用已有：

- `target_canvas`
- `ownership_canvas`
- `trainable_index_canvas`
- `context_valid_canvas`

对 `trainable_index_canvas>=0` 的位置：

```text
local_phi      = gather(macro_phi_snapshot, trainable_index)
local_grad_mag = gather(macro_grad_magnitude, trainable_index)
local_hard     = LevelSetBinarize(local_phi, local_grad_mag)
```

固定位置：

1. macro 外且 `context_valid_canvas=True`：使用 `target>=0.5` 的 hard transmission；
2. `context_valid_canvas=False`：纯数值 padding，transmission 严格为 0。

然后：

```text
mask = where(trainable_index>=0, local_hard, fixed_context_hard)
```

不得为了 LevelSet 为每个 core 建独立 SDF、独立 phi canvas 或独立 spatial-gradient field。

### REQ-007：cross-core raw gradient sum + macro barrier

loss 仍只在每个 core 自己的 `ownership_canvas` 统计；parameter ownership 与 loss ownership 严格分离。

同一 macro 参数出现在多个 core context 时，各 core ownership loss 对该参数的偏导必须 raw sum：

```text
macro_gradient[P] += local_grad_from_core_A[P]
macro_gradient[P] += local_grad_from_core_B[P]
```

绝不按出现次数平均。所有 core/batch 必须读取同一 state 的 `macro_phi` 与 `macro_grad_magnitude` 快照；全部 core 完成后才能更新参数一次。

batch size 与 core 遍历顺序不得改变算法语义，只允许正常浮点累加误差。

### REQ-008：Adam 明确契约

使用 macro-global Adam，推荐直接使用 CPU `torch.optim.Adam`，避免重新手写 optimizer 状态机。

参数固定：

```text
lr = config.step_size
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0
amsgrad = False
```

`macro_phi`、Adam `m/v` 均属于 macro 参数域，不得按 core/batch 拆分。每个 backward state：

```text
1. 全 core gradient sum 完成
2. 将唯一 macro_gradient 赋给 macro parameter.grad
3. optimizer.step() 恰一次
4. optimizer.zero_grad(set_to_none=True)
```

若使用等价手写 Adam，必须逐值通过 PyTorch Adam reference test。

### REQ-009：N 次 update + N+1 evaluated states

MyOPC 明确保留 Simple ILT 的状态生命周期：

```text
iterations = N
评价 state 0
update 1
评价 state 1
...
update N
评价 state N
```

即 N 次 Adam 更新，N+1 个完整已评价 macro state；最后一个 state 纯评价，不计算 `macro_grad_magnitude`、不 backward、不 update。

这是对 OpenILT 原始 loop 的有意工程适配，不要求与 OpenILT “最后一次 step 后不再评价”的状态计数逐值一致。

best 只能从完整 macro state 中严格下降选择，平局保留更早 state。

### REQ-010：损失、结果、曲率

复用：

- `owned_continuous_losses`
- `weighted_macro_loss`
- `curvature_loss`
- `ILTStateRecord`
- `ILTMacroResult`

输出：

```text
best_parameters = best_phi
soft_mask = sigmoid(-best_phi)      # 仅诊断
binary_mask = (best_phi < 0)
```

曲率作用于当前 hard mask，仅统计 ownership 有效卷积区；`curvature_weight==0` 时不得执行 conv。

### REQ-011：公共 workflow 仅抽象 final-context strategy

`_ilt_workflow` MUST 不再读取 `sigmoid_steepness`、`mask_threshold`、phi 等算法数学字段。

`ILTMethod` 仅新增一个真实需要的策略：

```python
build_fixed_context_canvas: Callable[[PixelMacroProblem, int, object], np.ndarray]
```

公共 `_binary_canvas`：

1. 调 method strategy 得到 fixed context canvas；
2. 将 `result.binary_mask` 写入 `trainable_index_canvas>=0` 位置；
3. 不知道 Simple sigmoid 或 LevelSet phi 数学。

策略实现放在算法模块：

- `simple.py::build_simple_final_context_canvas(...)`：真实 context=`sigmoid(beta*(2T-1))`，padding=0；
- `levelset.py::build_levelset_final_context_canvas(...)`：真实 context=`target>=0.5`，padding=0。

### REQ-012：Simple ILT 热路径零重构

本 change 不要求 `optimize_simple_macro` 调用 `build_simple_final_context_canvas`，也不要求把其 GPU `torch.sigmoid` context 路径改成 NumPy helper。

Simple solver 当前训练热路径保持原样；新增 final-context helper 仅服务公共 final evaluation。必须通过测试证明：

```text
Simple 训练 fixed-context 公式 == Simple final-context helper 公式
```

允许正常 float32 容差，但现有 Simple 固定 workload 的 best/binary/final metrics/artifact 必须零回归。

公式的轻微代码重复优先于为消除重复而引入 CPU/GPU round-trip 或新的通用抽象层。

### REQ-013：artifact 一致性

继续由公共 workflow 生成配置、进度、资源统计、macro result NPZ、metrics、best.gds、summary、final merge/final lithography。LevelSet 方法文件为 `levelset_ilt_result.npz`；公共 schema 不变，`best_parameters` 在 LevelSet 中语义为 phi。

## 5. Scope

### 5.1 In Scope

- production raster SDF once/macro；
- macro-global `phi`；
- 1-pixel physical halo；
- macro-global `|grad(phi)|` coefficient once/backward-state；
- external-gradient STE；
- cross-core raw gradient sum；
- macro-global Adam；
- `_ilt_workflow` final-context 最小通用化；
- LevelSet config/adapter/runner/tests/docs；
- `requirements.txt` 增 `scipy`。

### 5.2 Out of Scope

- SDF reinitialization / fast marching / narrow-band；
- macro 间参数交换 / seam healing；
- MRC / EPE / shot；
- CurvMulti / Multilevel 算法实现；
- `trainable_index_canvas` 的矩形 slice 优化；
- Simple ILT optimizer/参数化重构；
- unrelated pixel/workflow API 重构。

## 6. Invariants

- **INV-001 Unique Phi**：每个 macro ownership 参数只有一个 `macro_phi[P]`。
- **INV-002 Unique Gradient Coefficient**：每个 backward state，每个 macro 参数只有一个 `macro_grad_magnitude[P]`。
- **INV-003 Cross-Core Identity**：同一 P 在 core A/B gather 到相同 phi 与 grad coefficient。
- **INV-004 Sign**：`target>=0.5 -> initial phi<0`；`phi<0 <=> binary=True`；`phi==0 -> False`。
- **INV-005 Ownership Separation**：macro ownership=parameter ownership；core ownership=loss ownership。
- **INV-006 Fixed Context**：macro 外真实 context 固定，padding transmission=0。
- **INV-007 Macro Barrier**：全部 core raw gradient sum 后 Adam 恰一次。
- **INV-008 Macro Best**：best 只来自完整已评价 macro state。
- **INV-009 Method-independent Workflow**：公共 workflow 不含 Simple/LevelSet 数学字段。
- **INV-010 Simple Zero Regression**：本 change 不改变 Simple solver 热路径的数值语义。

## 7. Architecture and Data Flow

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `PixelMacroProblem` | query raster、core target/ownership/trainable/context 映射 | LevelSet/Simple 数学 |
| `ilt.levelset` | SDF、macro halo gradient、STE、Adam、LevelSet final context、solver | workflow/GDS/TOML |
| `ilt.simple` | 现有 Simple solver + Simple final-context helper | workflow |
| `ilt._common` | 中性 result/record/loss/curvature | method context policy |
| `_ilt_workflow` | prepare/solve/final-eval/artifact/merge | sigmoid/phi 数学 |
| method adapter | `ILTMethod` 装配 + thin run | optimizer/算法公式复制 |

### 7.2 Data Flow

```text
PixelMacroProblem.target_u8 [Hq,Wq]
 -> scipy EDT once/macro
 -> initial_query_phi [Hq,Wq]
 -> ownership crop
 -> macro_phi [Hm,Wm]                         # unique trainable parameter

for state in 0..N:
    if state < N:
        phi_halo = fixed 1px context + macro_phi snapshot
        macro_grad_magnitude = central_difference(phi_halo)   # [Hm,Wm], once/state
        macro_gradient = 0

    for core batch:
        target / ownership / trainable-index / valid
        gather local_phi from macro_phi
        gather local_grad_mag from macro_grad_magnitude
        local_hard = LevelSetBinarize(local_phi, local_grad_mag)
        fixed context = hard target; padding = 0
        mask = trainable ? local_hard : fixed context

        printed = model.forward_many(three conditions)
        loss = ownership-only shared losses + optional curvature
        backward -> scatter-add local.grad -> macro_gradient

    record full macro state; update best

    if state < N:
        macro Adam step exactly once

best -> ILTMacroResult -> common final evaluation/artifacts/merge
```

## 8. Data Contracts

### 8.1 `LevelSetILTConfig`

全部字段 required、无默认：

```text
iterations: int >= 1
step_size: finite float > 0
weight_process_l2: finite float >= 0
weight_pvband: finite float >= 0
curvature_weight: finite float >= 0
batch_size: int >= 1
```

bool 不得当 int。nominal 权重固定 1；SDF threshold 固定 0.5；binary threshold 固定严格 0。不得为了兼容公共 workflow 增加 `sigmoid_steepness` 或 `mask_threshold`。

### 8.2 Tensor/Array Residency

| Name | Shape/type | Resident | Lifetime |
|---|---|---|---|
| `initial_query_phi` | float32 `[Hq,Wq]` | CPU | macro solve |
| `macro_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| `phi_halo` | float32 `[Hm+2,Wm+2]` | CPU | one backward state |
| `macro_grad_magnitude` | float32 `[Hm,Wm]` | CPU | one backward state |
| `macro_gradient` | float32 `[Hm,Wm]` | CPU | one backward state |
| Adam `m/v` | float32 `2×[Hm,Wm]` | CPU | macro solve |
| `best_phi` | float32 `[Hm,Wm]` | CPU | macro solve |
| EDT temporaries | SciPy float64 query arrays | CPU | initialization only |
| local phi/grad/hard/printed | core canvas batch | GPU | one batch |

完整 macro autograd graph 不得常驻 GPU。

## 9. Interfaces

### IF-001 LevelSet API

```python
@dataclass(frozen=True, slots=True)
class LevelSetILTConfig: ...

signed_distance_initialization(target_u8: np.ndarray) -> np.ndarray
macro_gradient_magnitude(problem, initial_query_phi, macro_phi) -> np.ndarray
build_levelset_final_context_canvas(problem, core_index, config) -> np.ndarray
optimize_levelset_macro(problem, model, config, *, on_tiles_completed=None) -> ILTMacroResult
```

内部 helper 名称允许调整，但外部语义不得变化。

### IF-002 `ILTMethod`

```python
@dataclass(frozen=True, slots=True)
class ILTMethod:
    method_name: str
    config_type: type
    optimize_macro: Callable
    evaluated_states: Callable
    build_fixed_context_canvas: Callable
```

不增加 optimizer factory、parameterization base class、registry 等当前无第二调用价值的接口。

### IF-003 Config/CLI

- `CONFIG_SECTIONS[LevelSetILTConfig] = "levelset_ilt"`
- 新增 `LEVELSET_ILT_METHOD`
- 新增 `main/_levelset_ilt_workflow.py`
- 新增 `main/run_levelset_ilt.py`
- 新增 `config/levelset_ilt.toml`
- `requirements.txt` 墘 `scipy`

## 10. Algorithm

### 10.1 SDF

```text
binary = target_u8 >= 128     # 等价于 target>=0.5 的 uint8 语义需在实现中精确确认

if all(binary):
    phi = -max(Hq,Wq)
elif none(binary):
    phi = +max(Hq,Wq)
else:
    outside = EDT(~binary)    # background -> nearest foreground
    inside  = EDT(binary)     # foreground -> nearest background
    phi = outside
    phi[binary] = -inside[binary]
```

注意：若实现直接以 `target_u8.astype(float32)/255 >= 0.5` 判断，则阈值事实源以该表达式为准；不得在代码不同位置混用 `>=127`、`>=128` 等不等价整数阈值。推荐直接使用归一化浮点比较，测试覆盖 `127/128`。

### 10.2 Macro Halo Gradient

```text
macro_phi = ownership crop(initial_query_phi)

for backward state:
    phi_halo = ownership + 1 physical pixel context
    overwrite halo center with macro_phi snapshot

    dx = (right - left) / 2
    dy = (up - down) / 2
    macro_grad = sqrt(dx^2 + dy^2)
```

`macro_grad` 只定义在 trainable macro ownership；fixed context 不需要代理参数梯度。

### 10.3 Core Batch

```text
trainable = problem.trainable_index_canvas(core)
owned = trainable >= 0
safe = where(owned, trainable, 0)

local_phi = macro_phi_flat[safe]
local_grad = macro_grad_flat[safe]
local_hard = LevelSetBinarize(local_phi, local_grad)

fixed_hard = target>=0.5 on valid physical context
fixed_hard = 0 on numerical padding
mask = where(owned, local_hard, fixed_hard)
```

local tensor 是 GPU leaf；backward 后只把 `local.grad[owned]` scatter-add 到唯一 `macro_gradient`。

### 10.4 Adam

```text
for state 0..N:
    evaluate all cores
    update best

    if state < N:
        verify macro_gradient finite
        macro_parameter.grad = macro_gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        verify macro_phi and Adam state finite
```

## 11. Performance and Memory

1. SDF：`1 call/macro`；生产实现使用 SciPy compiled EDT，禁止 Python 像素级双重热循环。
2. `macro_grad_magnitude`：`iterations calls/macro`，每次只处理 `[Hm+2,Wm+2]`，不随 core_count/batch_size 增长。
3. 不构造每 state 的完整 `current_query_phi[Hq,Wq] + query_grad_magnitude[Hq,Wq]`。
4. 每 state/core batch 仍仅一次 `forward_many(three conditions)`。
5. `curvature_weight==0` 时不构建 conv。
6. GPU 仅保留当前 batch autograd graph；macro phi/Adam state 常驻 CPU。
7. 不缓存所有 core 的 `target/ownership/trainable/valid` canvas；现有按需构造策略保持。
8. `trainable_index_canvas` 的矩形 slice 优化属于后续性能 change，本 change 不扩大范围。
9. smoke 必须记录 total/SDF/macro-gradient 时间、CPU RSS、CUDA peak，并与同输入 Simple 做事实对照。

## 12. Error Handling

以下前置失败：

- config 非法；
- target/canvas/shape mismatch；
- `context_dbu < pixel_dbu`；
- SDF/halo 索引无法取得完整 1px context。

异常类型：配置/形状/契约错误 -> `ValueError`；非有限 loss、macro gradient、macro phi、Adam state、macro grad magnitude -> `FloatingPointError`。

EDT/CUDA/I/O 未知异常原样传播；不得跳 core/pixel、重置 optimizer、发布部分成功 macro 或用零值静默替代失败结果。

## 13. File-Level Change Plan

| File / Symbol | Action | Change | Reason |
|---|---|---|---|
| `requirements.txt` | modify | 增 `scipy` | compiled exact EDT |
| `opc/iteration/ilt/levelset.py` | add | config、SciPy SDF、macro halo gradient、STE、final context、macro Adam solver | 核心算法 |
| `opc/iteration/ilt/simple.py` | modify minimal | 仅增 `build_simple_final_context_canvas`；不改 solver 热路径 | final workflow 解耦 |
| `opc/iteration/ilt/__init__.py` | modify | 导出 LevelSet API/final helper | 方法入口 |
| `main/_ilt_workflow.py` | modify | `ILTMethod` 增 final-context strategy；终评去除 Simple 专属字段 | method-independent workflow |
| `main/_simple_ilt_workflow.py` | modify minimal | 挂载 Simple final-context helper | Simple 适配 |
| `main/_levelset_ilt_workflow.py` | add | METHOD + thin run | LevelSet 适配 |
| `main/configuration.py` | modify | 注册 `[levelset_ilt]` | 配置 |
| `main/run_levelset_ilt.py` | add | CLI | 直接运行 |
| `config/levelset_ilt.toml` | add | smoke config | 可运行样例 |
| `tests/opc/iteration/test_levelset_ilt.py` | add | SDF/halo/STE/Adam/cross-core/real model | 核心验证 |
| `tests/opc/iteration/test_simple_ilt.py` | modify as needed | final helper 等价性 + zero regression | Simple 保护 |
| `tests/main/test_levelset_ilt_runner.py` | add | config/adapter/CLI/artifact/final context | 集成 |
| `tests/main/test_simple_ilt_runner.py` | modify as needed | workflow 通用化零回归 | 兼容 |
| `tests/main/test_configuration.py` | modify | 新 section 严格解析 | 配置 |
| `doc/contracts/ilt.md` | modify | LevelSet + ILTMethod contract | 契约 |
| `doc/architecture/system.md` | modify | LevelSet 组件 | 架构 |
| `doc/architecture/dataflow/index.md` | modify | 索引 LevelSet flow | 文档入口 |
| `doc/architecture/dataflow/levelset_ilt.md` | add | 函数级流向/边界/伪代码 | 当前事实 |
| `doc/development_manual.md`、`doc/test_manual.md` | modify | 使用/测试 | 交付 |
| change active→completed + reports | move/add | development/test report | 完成闭环 |

不得借本 change 修改 `PixelMacroProblem` 映射 API；现有 `trainable_index_canvas` 已足够完成正确实现。

## 14. Test Specification

### TEST-001 SDF definition / performance path

- rectangle/single/hole/mixed target 与 brute-force 最近 opposite-class pixel-center 距离逐值对比；
- `target_u8=127/128` 固定 threshold 语义；
- all-empty/all-full 有限常量与 sign；
- spy：一个 macro 的 production SDF 调用恰一次；
- production implementation 必须调用 SciPy EDT，禁止落回 Python 逐像素 EDT。

### TEST-002 Macro halo gradient exactness

构造手算 `macro_phi` + 1px fixed context，逐值验证中心差分。重点覆盖 macro ownership 边缘参数，证明其梯度使用真实 query context，而非 core-local replicate。

spy：`macro_gradient_magnitude` 调用数恰等于 `iterations`/macro；末纯评价 state 不调用。

### TEST-003 Cross-core identity

构造同一参数 P 同时出现在 core A/B context：

```text
local_phi_A(P) == local_phi_B(P)
local_grad_mag_A(P) == local_grad_mag_B(P)
```

测试必须能击穿按 core 自己做空间差分的错误实现。

### TEST-004 STE

给定显式 local phi / grad magnitude / upstream gradient：

```text
forward == (phi<0)
backward == -grad_mag * grad_output
```

`grad_mag` 返回 None；backward 不调用任何 spatial-gradient helper。

### TEST-005 Cross-core gradient sum

使用局部耦合 differentiable lithography stub；同一参数 P 同时影响 A/B ownership loss，验证两份 local raw gradient 被求和而非平均。`batch_size=1/2` 结果一致。

### TEST-006 Adam reference / macro barrier

使用 CPU `torch.optim.Adam` reference 固定参数：`betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False`。

验证：

- 所有 core 同一参数快照；
- 每 backward state 恰一次 macro step；
- m/v 唯一；
- N=2 -> 3 个 evaluated states；
- batch 内提前 step 的错误实现失败。

### TEST-007 Context/padding

- macro 外真实 context hard=`target>=0.5`；
- padding=0；
- macro ownership 参数可从邻 core ownership loss 得梯度；
- `context_dbu<pixel_dbu` 前置失败。

### TEST-008 Loss/curvature/real model

共享 nominal/process/PV 逐值；curvature ownership-only；weight0 无 conv。真实 ICCAD13 CPU 至少 1 次 update + final evaluation 全 finite；CUDA 可用时执行 smoke/parity 并记录资源。

### TEST-009 Workflow method independence

fake method 证明 `_ilt_workflow` 不访问 `sigmoid_steepness/mask_threshold/phi`。LevelSet final context 不运行 SDF；Simple final context helper 与 Simple solver 现有 context 公式数值等价。

### TEST-010 Simple zero regression / CLI / artifact

- Simple solver 源码热路径不被 LevelSet change 改写；
- 固定 Simple workload 的 best/binary/binary_l2/pvband/artifact key 与变更前一致；
- LevelSet 合法/缺键/未知键/nonfinite/bool-int 配置；
- 仓库外 cwd 默认/显式 config；
- LevelSet result/metrics/best/final/summary schema 与公共格式一致；
- 全量测试绿色。

## 15. Verification Commands

```bash
python -m pytest -q tests/opc/iteration/test_levelset_ilt.py
python -m pytest -q tests/opc/iteration/test_simple_ilt.py
python -m pytest -q tests/main/test_levelset_ilt_runner.py tests/main/test_simple_ilt_runner.py
python -m pytest -q tests/main/test_configuration.py
python -m pytest -q tests
python -m ruff check common layout geometry opc lithography evaluation main tests
python -m compileall -q common layout geometry opc lithography evaluation main tests
python main/run_levelset_ilt.py config/levelset_ilt.toml
python main/run_simple_ilt.py config/simple_ilt.toml
git diff --check
```

## 16. Acceptance Criteria

- [ ] AC-001：LevelSet CLI/method/config/artifact/merge 完成。
- [ ] AC-002：production SDF 使用 SciPy EDT，once/macro；距离/sign/127-128/all-empty/all-full 通过。
- [ ] AC-003：每 backward state 只构造一次 macro 1px halo 和一次 `[Hm,Wm]` gradient coefficient；不构造完整 query current/gradient field。
- [ ] AC-004：同一参数跨 core 的 phi/grad coefficient 完全一致。
- [ ] AC-005：cross-core raw gradient sum、batch invariance、macro Adam barrier 通过。
- [ ] AC-006：Adam 参数与 PyTorch reference 一致；N update + N+1 evaluated states 通过。
- [ ] AC-007：LevelSet fixed context hard、padding0、loss/curvature/best/真实模型通过。
- [ ] AC-008：`_ilt_workflow` 无方法数学字段，只新增 final-context strategy。
- [ ] AC-009：Simple solver 热路径保持原样，final helper 数学等价，固定 workload 零回归。
- [ ] AC-010：contracts/dataflow/manual/development_report/test_report 与最终实现同步；full tests/ruff/compileall/diff-check 全绿。

## 17. Compatibility and Migration

- `PixelMacroProblem` NPZ v1 不变；不新增 query-array canvas API。
- `ILTMacroResult` 与 result NPZ schema 不变。
- `ILTMethod` 增加一个 final-context callable；Simple adapter 同步提供。
- Simple CLI/config/solver 数值语义不得变化。
- `requirements.txt` 新增 SciPy 是本 change 唯一新增 runtime dependency。
- LevelSet 不承诺与 OpenILT initializer/单 tile 逐值一致；保留 hard/STE/Adam 核心算法，SDF 使用本文固定的 raster pixel-center 定义。
- CurvMulti/Multilevel active spec 在实施前必须以完成后的真实 `ILTMethod` contract 重新核对。

## 18. Decisions

### DEC-001 OpenILT tile -> MyOPC macro parameter domain

LevelSet 参数是物理连续状态；core 只是计算切片。按 core 建参数/SDF 会直接产生多份同位置状态。

### DEC-002 Unique coefficient defined only for trainable parameters

代理 backward 只需要可训练 `phi` 的 `|grad(phi)|`。因此无需为整个 macro query 每轮生成完整空间梯度场；`macro_phi + 1px physical halo -> macro_grad_magnitude[Hm,Wm]` 已充分且更省内存。

### DEC-003 No core-local spatial difference

同一参数在不同 core 中的位置不同。若每个 core 对自己的画布做 replicate/padding 差分，同一 P 可能得到不同 surrogate coefficient，使算法依赖 core partition。故 spatial difference 必须在 macro 参数域统一计算一次。

### DEC-004 Macro Adam

Adam 是带状态非线性更新；按 core/batch step 会使结果依赖切分/order。必须先 sum 全部 raw gradient，再一步更新。

### DEC-005 SciPy EDT over Python EDT

旧 O(HW) EDT 虽渐近复杂度正确，但主体为 Python row/column loop，不适合作为大像素图生产路径。生产实现使用 SciPy compiled EDT；旧实现只作为 correctness oracle。

### DEC-006 Minimal workflow abstraction

公共 workflow 只抽象实际发生差异的 final fixed-context policy。不为两个算法建立参数化基类、optimizer factory 或通用 solver engine。

### DEC-007 Preserve Simple hot path

Simple solver 已完成并通过现有测试。为消灭几行 sigmoid 公式重复而让训练改走 NumPy helper/CPU round-trip，收益小于回归风险，因此本 change 不做。

### DEC-008 N+1 state is intentional MyOPC behavior

MyOPC best 必须来自完整已评价状态；因此最后一次 update 后再评价一次。与 OpenILT 原始 loop 的 state count 差异是明确适配，不是 bug。

## 19. Open Questions

Blocking：None。

Non-blocking：SDF reinitialization、narrow-band、macro seam healing、`trainable_index_canvas -> rectangular slice` 性能优化均另立 change。

## 20. Implementation Freedom

允许：helper 命名、SciPy EDT workspace 组织、CPU Torch/NumPy 间零拷贝细节、等价 PyTorch Adam 管理方式调整。

不得改变：

- raster SDF 定义与 once/macro；
- macro-global phi；
- 1px physical halo；
- 每参数每 state 唯一 `macro_grad_magnitude`；
- external-gradient STE；
- cross-core raw-gradient sum；
- macro Adam once；
- Adam 参数；
- hard `phi<0`；
- fixed context hard / padding0；
- N update + N+1 evaluated states；
- 公共 workflow 只含 final-context strategy；
- Simple solver 热路径零重构。

## 21. Implementation Stages

| Stage | Objective | Main files | Verify |
|---|---|---|---|
| A | workflow final-context 最小通用化 + Simple helper | simple / `_ilt_workflow` / Simple adapter | Simple zero regression |
| B | SciPy SDF + macro halo gradient + STE + macro Adam solver | levelset.py / requirements | TEST-001..008 |
| C | config/adapter/runner/artifacts | main/config/tests | TEST-009..010 |
| D | docs/full smoke/audit | contracts/dataflow/manual/reports | all commands |

## 22. Delivery and Final Audit

交付前必须证明：

1. 实施 baseline/status/test count 已更新；
2. production SDF 使用 SciPy EDT 且一个 macro 恰一次；
3. `macro_grad_magnitude` 调用恒为 `iterations`，不随 core/batch 增长；
4. 不存在每 state 完整 `current_query_phi/query_grad_magnitude` query field；
5. overlap A/B 同一参数的 phi 和 grad coefficient 完全一致；
6. 同一参数来自多个 core ownership loss 的梯度为 sum，不平均；
7. Adam 每 backward state 只有一次 macro update，参数与 PyTorch reference 一致；
8. LevelSet final context 不重复 SDF；
9. `_ilt_workflow` 不访问 Simple/LevelSet 数学配置；
10. Simple solver 热路径未被 LevelSet change 改写，固定 workload 零回归；
11. 无 workflow 复制、core-local SDF/gradient、gradient averaging、padding 伪透光；
12. `doc/architecture/dataflow/levelset_ilt.md` 与 index 已更新；
13. full tests、ruff、compileall、direct runner、diff-check 全绿并输出 development_report/test_report。
