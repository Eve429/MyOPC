---
id: CHG-20260818-levelset-ilt
title: LevelSet ILT 迁移
type: implementation-spec
status: draft
baseline_commit: 2fa75ea89ea6cd64122214f1e2e0ed14cae518c3
baseline_worktree: dirty
baseline_dirty_paths:
  - opc/input/grid.py
scope:
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

本文档是本 change 的唯一实现规格。实现 AI MUST 先确认
`CHG-20260818-simple-ilt` 已按批准规格完成，并把本文件的 baseline 更新到该完成 commit；在此
之前本文件只能审查，不能实施。

实现 AI MUST：只修改 §14 文件；复用已完成的 pixel problem、`ILTBatchResult`、`ILTMethod` 和
workflow；以 OpenILT 与 `00_PAST` 为算法参考但以本规格为目标；阻塞问题或基线漂移时停止。

实现 AI MUST NOT：修改 `layout/geometry/00_PAST/lithography/evaluation`；复制 Simple workflow；
建立 ILT 基类/注册器；静默修正配置、NaN、I/O 或 CUDA 错误。

## 1. Objective

在不改变首个 ILT 公共输入、执行和产物接口的前提下，迁移 OpenILT 的硬二值水平集方法，使
用户可直接运行 `main/run_levelset_ilt.py`，并证明新增方法只通过具体 optimizer 与薄适配器接入。

## 2. Baseline and Evidence

### 2.1 Baseline

- 设计基线：`2fa75ea89ea6cd64122214f1e2e0ed14cae518c3`，446 tests passed。
- 实施基线：MUST 更新为 `CHG-20260818-simple-ilt` 完成 commit 并重新记录 test count。
- 设计开始工作树 clean；交付前 `opc/input/grid.py` 出现一处来源未确认的注释删改，行为不变，
  实施时 MUST 保留并排除在本 change 提交之外。

### 2.2 Confirmed Facts

| Fact ID | Confirmed fact | Evidence | Verification |
|---|---|---|---|
| FACT-001 | 当前设计基线没有 ILT 实现 | `doc/contracts/ilt.md` | 静态阅读 |
| FACT-002 | LevelSet 上游前向为 `phi<0`，反向为 `-|∇phi|*grad_output`，optimizer=Adam | `OpenILT/pyilt/levelset.py::_Binarize/LevelSetILT.solve` | 只读源码 |
| FACT-003 | 旧迁移用精确二维欧氏距离生成前景负/背景正 SDF | `00_PAST/opc/iteration/ilt/levelset.py::signed_distance_initialization` | 只读源码与旧测试 |
| FACT-004 | 旧迁移已覆盖 zero boundary、代理梯度、固定区、真实 Hopkins backward | `00_PAST/tests/opc/test_levelset_ilt.py` | 只读测试 |
| FACT-005 | 目标公共接口由依赖 change 明确定义 | `CHG-20260818-simple-ilt` §8/§9 | 规格依赖 |

### 2.3 Uncertainty Boundary

- 后续 Simple 实施若改变本规格依赖的 public symbol、字段或 workflow，必须先修订本文，不能猜测适配。
- LevelSet STE 是代理梯度，不是硬阈值的数学真导数；本 change 不宣称单调收敛或几何最优。

### 2.4 External and Archive References

| Reference | Role | Adopt | Reject | Reason |
|---|---|---|---|---|
| `OpenILT/pyilt/levelset.py` | 原算法 | hard forward、空间梯度 STE、Adam、三条件 loss | 全局 DEVICE、Module 包装层、DataParallel、旧 filter/config | 当前共享 workflow 已负责 batch/filter |
| `OpenILT/pyilt/initializer.py` | 初始化依据 | signed distance 意图 | polygon/legacy 多初始化器和全局 helper | target 已是 raster，保留一个精确 SDF 实现 |
| `00_PAST/opc/iteration/ilt/levelset.py` | 工程参考 | O(HW) EDT、全空/全满、严格 zero boundary、结构化异常 | 复用 `SimpleILTResult` 的方法间反向依赖 | 改用首 change 中性 `ILTBatchResult` |

## 3. Current Behavior

设计基线无 ILT。依赖 change 完成后的预期当前行为是：

1. `opc.input.pixel` 提供持久 pixel problem、core target/ownership 与 GDS 重建；
2. `main._ilt_workflow::run_ilt_workflow` 接受 `ILTMethod` 并完成 macro/core/产物生命周期；
3. `opc.iteration.ilt.simple::optimize_simple_batch` 是唯一方法；没有水平集参数化。

实施前必须从真实完成源码重新核对以上三项。

## 4. Target Behavior

### REQ-001

系统 MUST 提供 `python main/run_levelset_ilt.py <config.toml>`，同一 GDS/pixel problem/output
contract 下运行 LevelSet ILT。

### REQ-002

LevelSet MUST 前向输出严格 `(phi<0).float()`；`phi==0` MUST 为不透光；反向 MUST 返回
`-|∇phi|*grad_output`，空间差分边界使用 replicate padding。

### REQ-003

缺省初值 MUST 从 target `>=0.5` 构造前景负、背景正的精确像素中心 SDF；初始化只在 CPU
执行一次/批，不进入 state 热循环。

### REQ-004

SDF 必须使用 `O(BHW)` 时间、`O(HW)` 单图 scratch 的二维精确距离变换；全空/全满图必须返回
有限且符号正确的常量场，不得依赖 SciPy/OpenCV 新依赖。

### REQ-005

LevelSet MUST 使用 Adam；loss/conditions/ownership selection、N 次更新+N+1 状态、逐样本 best、
最终 binary 评价和 GDS 输出 MUST 复用 Simple change 契约。

### REQ-006

context 的初始 phi MUST 固定；只有 ownership phi 可更新。输出 `best_parameters=best_phi`、
`soft_mask=sigmoid(-best_phi)`、`binary_mask=best_phi<0`。

### REQ-007

曲率项启用时 MUST 作用于 hard binary mask，并只统计 ownership 有效卷积区；权重 0 时不执行卷积。

### REQ-008

新增方法 MUST 只新增具体算法、配置注册、薄 adapter/runner 和测试文档；MUST NOT 修改
`PixelMacroProblem`、`ILTBatchResult` 字段、`ILTMethod` 字段或 `_ilt_workflow` 行为。

### REQ-009

配置、进度、资源统计、macro artifacts、final merge/final lithography MUST 与 Simple workflow
保持同一格式，仅 method/config/算法记录数不同。

## 5. Scope

### 5.1 In Scope

- SDF、hard-forward surrogate-backward、LevelSet config 和 batch optimizer。
- config 注册、LevelSet adapter、直接 main、测试、配置与文档/报告。

### 5.2 Out of Scope

- SDF reinitialization/fast marching、窄带 level set、拓扑/MRC 约束。
- 公共 workflow/problem/result 重构；其他 ILT 方法；EPE/shot。

### 5.3 Protected Areas

`00_PAST/**`、`layout/**`、`geometry/**`、`lithography/**`、`evaluation/**`、用户数据不得修改。

## 6. Invariants

### INV-001

`phi<0 <=> binary_mask=True`；`phi==0` 始终 False，soft mask 仅诊断，不能反向决定 binary。

### INV-002

SDF target 前景为负、背景为正，距离单位是 pixel center。

### INV-003

context phi 在全部 state 固定；ownership 是唯一 optimizer writer。

### INV-004

每样本 best 独立，state 记录属于已评价 phi；batch size/reorder 不改变单样本输出。

### INV-005

共享 pixel/problem/workflow/artifact contract 不变。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `opc.iteration.ilt.levelset` | SDF、STE、Adam、LevelSet state | GDS、配置文件读取、tqdm |
| `main._levelset_ilt_workflow` | METHOD 描述和薄调用 | optimizer 数学、公共 workflow |
| `main.run_levelset_ilt` | 直接入口 | 业务逻辑 |

### 7.2 Dependency Direction

```text
levelset -> ilt._common + lithography
_levelset_ilt_workflow -> _ilt_workflow + levelset
run_levelset_ilt -> _levelset_ilt_workflow
```

不得反向依赖或横向 import `simple.py`。

### 7.3 Data Flow

```text
PixelMacroProblem core batch
 -> target/ownership
 -> signed_distance_initialization(target)
 -> phi state 0..N
    -> hard mask via custom autograd
    -> shared continuous losses
    -> per-sample best
    -> Adam step except final state
 -> ILTBatchResult
 -> unchanged shared workflow artifacts/GDS
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| initialization | target batch | CPU EDT | initial phi | 每 state 不得重算 |
| state | phi | hard forward/litho/loss/backward | record/next phi | 不处理 GDS |
| output | per-sample best phi | soft/binary | shared result | 不二次选 best |

### 7.5 Planned Call Graph

```text
main/run_levelset_ilt.py::main
└─ main/_levelset_ilt_workflow.py::run_levelset_ilt
   └─ main/_ilt_workflow.py::run_ilt_workflow(METHOD, config)
      └─ opc/iteration/ilt/levelset.py::optimize_levelset_batch
         ├─ signed_distance_initialization
         ├─ _LevelSetBinarize.apply                 [state loop N+1]
         ├─ model.forward_many
         └─ ilt._common continuous losses/result
```

## 8. Data Contracts

### `LevelSetILTConfig`

Owner：`opc.iteration.ilt.levelset`；frozen；one run。

| Field | dtype | meaning |
|---|---|---|
| `iterations` | strict int | Adam 更新数，`>=1` |
| `step_size` | float | Adam lr，finite `>0` |
| `weight_process_l2` | float | finite `>=0` |
| `weight_pvband` | float | finite `>=0` |
| `curvature_weight` | float | finite `>=0` |
| `batch_size` | strict int | core batch，`>=1` |

nominal 权重固定 1；target SDF threshold 固定 0.5；binary threshold 固定严格 0。

### LevelSet tensors

| Name | dtype/shape | Resident | Lifetime |
|---|---|---|---|
| initial/best phi | float32 `[B,256,256]` | CPU→device/device | batch |
| EDT scratch | NumPy float64 `[256,256]` ×≤2 | CPU | one image initialization |
| Adam params/m/v | float32 `3×[B,256,256]` | device | batch |
| hard mask/printed | float32 `[B,256,256]` | device | one state |

复用 `ILTStateRecord`：stage=0、scale=1；复用 `ILTBatchResult`，不得新增重复 result dataclass。

### 8.1 Configuration Contract

新增 `[levelset_ilt]`，六字段全部 required、无默认。共享段与 Simple 相同，不要求 `[edge]`。

### 8.2 Persisted Artifact Contract

复用 pixel problem/plan/final contract；方法专属文件名：

- `macros/<macro>/levelset_ilt_result.npz`：与公共 result 格式相同，`best_parameters` 语义为 phi；
- `metrics.json`/`best.gds`/`summary.json`：method=`levelset_ilt`，config 为本方法字段。

## 9. Interface Changes

### IF-001：新增 solver

```python
signed_distance_initialization(target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor

optimize_levelset_batch(
    target: torch.Tensor,
    ownership: torch.Tensor,
    model: LithographyModel,
    config: LevelSetILTConfig,
    *,
    on_states_completed: Callable[[int], None] | None = None,
) -> ILTBatchResult
```

输入输出形状同 Simple；target/initial 非有限、范围错误或 shape 错误抛 `ValueError`。

### IF-002：配置与入口

`CONFIG_SECTIONS[LevelSetILTConfig]="levelset_ilt"`；新增 `METHOD` 和
`run_levelset_ilt(config_path)->dict`；直接 main 的 CLI 只接 config path。

### IF-003：共享接口

No change required。实现若发现必须修改 `ILTMethod`、`ILTBatchResult` 或 `_ilt_workflow`，必须停止并
修订 Simple/本文规格。

## 10. Algorithm

### 10.1 SDF

对每张 binary target：若全前景返回 `-max(H,W)`；全背景返回 `+max(H,W)`；否则分别计算到前景
与背景的精确欧氏距离，把前景位置赋负的 inside distance，背景保留正 outside distance。
二维距离通过两次一维抛物线下包络完成；scratch 每图复用，结果 float32。

### 10.2 State loop

```text
fixed_phi = initial_phi.detach
phi = clone(initial_phi, requires_grad)
Adam(phi, lr)
for state 0..N:
    effective_phi = phi*ownership + fixed_phi*(1-ownership)
    hard_mask = LevelSetBinarize(effective_phi)
    printed = model.forward_many(three conditions)
    per-sample selected losses; optional hard-mask curvature
    update per-sample best effective_phi
    record/notify
    if state<N: backward(sum loss); Adam.step
soft = sigmoid(-best_phi)
binary = best_phi < 0
```

### 10.3 Boundary Conditions

| Condition | Behavior |
|---|---|
| target all 0/all 1 | finite constant SDF，正常优化 |
| target contains gray coverage | `>=0.5` 决定 SDF sign，loss 仍对原 float target |
| phi spatially constant | STE gradient magnitude=0，这是算法语义，不 fallback |
| phi==0 | binary false |
| nonfinite loss/phi | `FloatingPointError`，无 completed artifacts |
| batch tail | 真实 B，per-sample best |

### 10.4 State Transition

与 Simple 相同 N+1 已评价状态；best 严格按 total loss 下降更新。Adam step 后状态必须下轮评价。

## 11. Ownership and State

phi/Adam 由 solver 拥有；fixed_phi/context 只读；每 state 的 hard mask/printed 临时；best_phi 每样本
写入；workflow 只读结果并负责 disk/GDS。失败不发布当前 macro，其他 macro/最终语义同共享 workflow。

## 12. Error Handling

### ERR-001

非法 config/target/shape/range 由构造或 solver 抛 `ValueError`，不得 fallback。

### ERR-002

非有限 loss/phi 抛 `FloatingPointError`，不得跳过 pixel/sample 或重置 Adam。

### ERR-003

CPU EDT、CUDA、I/O 等未知错误原样传播；entry 不捕获并返回成功。

## 13. Performance and Memory Constraints

- SDF 初始化 `O(BHW)`，每 batch 恰一次；禁止每 iteration CPU/GPU 往返或重算 EDT。
- GPU 比 Simple 多两份 Adam state；仍只常驻一个 core batch，不保留全 macro/reticle tensor。
- 每 state 一次三条件 `forward_many`；curvature=0 不执行 conv。
- smoke 记录 CPU/CUDA 时间与 peak，不设阈值；与 Simple 同输入/批量对比只记录，不声称优劣。

## 14. File-Level Change Plan

| File / Symbol | File type | Action | Contract change | Reason |
|---|---|---|---|---|
| `opc/iteration/ilt/levelset.py` | 业务代码 | add | config/SDF/STE/optimizer | REQ-002..007 |
| `opc/iteration/ilt/__init__.py` | 业务代码 | modify | 导出 LevelSet API | REQ-001 |
| `main/configuration.py::CONFIG_SECTIONS` | 业务代码 | modify | 注册 `[levelset_ilt]` | IF-002 |
| `main/_levelset_ilt_workflow.py` | 方法适配器 | add | METHOD + thin run | REQ-008/009 |
| `main/run_levelset_ilt.py` | 运行入口 | add | 直接 Python 入口 | REQ-001 |
| `config/levelset_ilt.toml` | 配置 | add | smoke 配置 | §8.1 |
| `tests/opc/iteration/test_levelset_ilt.py` | 测试 | add | SDF/STE/state/batch/real model | TEST-001..007 |
| `tests/main/test_levelset_ilt_runner.py` | 测试 | add | config/adapter/CLI/artifacts | TEST-008..010 |
| `tests/main/test_configuration.py` | 测试 | modify | 新 section 严格解析 | TEST-008 |
| `doc/contracts/ilt.md` | 接口文档 | modify | 增加当前 LevelSet API/限制 | 交付 |
| `doc/architecture/system.md`、`doc/architecture/dataflow.md` | 架构文档 | modify | 增加当前 LevelSet 组件/数据流 | 交付 |
| `doc/development_manual.md`、`doc/test_manual.md` | 手册 | modify | 使用与测试 | 交付 |
| `doc/changes/active/CHG-20260818-levelset-ilt/implementation_spec.md` → `doc/changes/completed/CHG-20260818-levelset-ilt/implementation_spec.md` | 规格 | move | baseline/status/revision/evidence | 交付 |
| `doc/changes/completed/CHG-20260818-levelset-ilt/development_report.md` | 开发报告 | add | 实施、偏差、审计 | 交付 |
| `doc/changes/completed/CHG-20260818-levelset-ilt/test_report.md` | 测试报告 | add | 环境、命令、结果 | 交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` | 项目记录 | modify | 同步 | AGENTS |

`opc/input/pixel/**`、`main/_ilt_workflow.py` 不在修改清单，兼容性由“不修改”验证。

## 15. Test Specification

### TEST-001：SDF 符号与精确距离

矩形/单点/hole/全空/全满/batch；与小图 brute-force 距离逐值对照。

### TEST-002：STE forward/backward

严格 `phi<0`；手算 dx/dy、方向、replicate 边界、常量 phi 零梯度。

### TEST-003：ownership/context

固定区 phi 与 binary 不变；ownership 有梯度，context 无梯度。

### TEST-004：N+1 state 与逐样本 best

identity model 下手算 Adam/不同 best；batch=1/2/reorder 一致。

### TEST-005：loss/curvature

共享 loss 逐值；curvature hard mask ownership-only；权重 0 无 conv。

### TEST-006：几何矩阵

rectangle/hole/concave/diagonal/multi-island/窄线，结果 finite、shape/binary contract 正确。

### TEST-007：真实 ICCAD13 CPU/CUDA

一轮 backward/step/final evaluation；finite 与规定容差。

### TEST-008：配置/adapter

合法、缺键、未知键、非有限、bool-as-int；METHOD 复用公共 workflow。

### TEST-009：直接入口/产物

仓库外 cwd；generated multi-core GDS；method/config/result phi 语义和 GDS 完整。

### TEST-010：共享接口零差异

Simple/pixel/workflow 测试全绿；静态 diff 证明 `_ilt_workflow.py` 和 pixel problem 未改。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected |
|---|---|---|
| SDF | empty/full/single/hole/batch | sign/distance exact |
| Geometry | rect/hole/concave/diagonal/narrow | finite hard mask |
| State | N=1/2, distinct best | N+1/per-sample |
| Boundary | ownership/context/core seam | context immutable |
| Device | CPU/CUDA | backward finite/parity |
| Failure | config/shape/range/nonfinite | explicit exception |

### 15.2 Verification Commands

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_levelset_ilt.py tests/main/test_levelset_ilt_runner.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_simple_ilt.py tests/opc/input/test_pixel_problem.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc main tests
D:\app\miniforge\envs\myopc\python.exe -m compileall -q opc main tests
D:\app\miniforge\envs\myopc\python.exe main/run_levelset_ilt.py config/levelset_ilt.toml
git diff --check
```

## 16. Requirement Traceability

| Requirement/Invariant | Implementation | Tests | AC |
|---|---|---|---|
| REQ-001/008/009 | adapter/runner/config | TEST-008..010 | AC-001/005 |
| REQ-002/006 | `_LevelSetBinarize`/optimizer | TEST-002..004 | AC-002/003 |
| REQ-003/004 | SDF functions | TEST-001 | AC-002 |
| REQ-005/007 | optimizer+common | TEST-004..007 | AC-003/004 |
| INV-001/002 | SDF/output | TEST-001/002 | AC-002 |
| INV-003/004 | optimizer | TEST-003/004 | AC-003 |
| INV-005 | no-change audit | TEST-010 | AC-005 |

## 17. Acceptance Criteria

- [ ] **AC-001**：直接入口完成且 method=`levelset_ilt`，产物 contract 与 Simple 一致。
- [ ] **AC-002**：SDF/STE/strict-zero 逐值测试通过。
- [ ] **AC-003**：ownership、N+1、Adam、per-sample best、batch invariance 通过。
- [ ] **AC-004**：真实 ICCAD13 CPU 和可用 CUDA 完成 backward，资源基线记录。
- [ ] **AC-005**：共享 workflow/problem/result 未修改，Simple 与全量回归全绿。
- [ ] **AC-006**：文档/报告/规划和最终简化审计完成。

## 18. Compatibility and Migration

- API：纯新增 LevelSet 方法；共享 ILT API 不变。
- Data：复用 pixel problem v1；新增方法 result 的 `best_parameters` 是 phi，format/version 不变但 method
  字段区分；不读取旧归档 NPZ。
- CLI：新增直接入口，无旧 CLI 兼容。
- Numerical：保留 OpenILT hard/STE/Adam；采用旧迁移精确 SDF、N+1 状态、per-sample best 与当前
  ICCAD13，故不承诺上游逐值。

## 19. Decisions

### DEC-001：精确 SDF，不复制上游多初始化器

Reason：当前输入已是 raster；精确 SDF 为 STE 提供全场空间梯度，单一实现可测试。

### DEC-002：hard mask 参与光刻，soft mask 只诊断

Reason：这是 LevelSet 与 Simple 的核心算法差异；用 sigmoid forward 会变成另一方法。

### DEC-003：不修改共享 workflow

Reason：首 change 的兼容性验收要求；本方法所需差异完全由 optimizer/config 表达。

## 20. Open Questions

### 20.1 Blocking

None.

### 20.2 Non-blocking

- SDF 定期重初始化、窄带更新和 level-set curvature flow 另立 change。

## 21. Implementation Freedom

允许等价 EDT scratch 复用和局部命名；不得改变 hard/STE 公式、strict zero、Adam、SDF sign、共享
接口或文件清单。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Verify | Commit |
|---|---|---|---|---|
| A | SDF/STE/optimizer | levelset.py + unit tests | TEST-001..007 | `feat(ilt): 实现水平集优化器` |
| B | config/adapter/runner | main/config/tests | TEST-008..010 + direct main | `feat(main): 接入LevelSetILT` |
| C | full/smoke/docs/audit | doc/planning | all commands | `docs(ilt): 完成LevelSet迁移报告` |

只做本地 commit，未经授权不得 push。

## 23. Delivery and Final Audit

同步手册、contracts/architecture、两报告和规划；记录测试/性能/skip；审计未调用 helper、复制 loss、
吞异常、一次性 wrapper、中文 docstring；确认未修改 protected paths 和共享 workflow/problem/result。

## 24. Known Limitations and Future Work

- 继承 Simple change 的 tile-independent seam 限制。
- STE 可能在常量 phi 区域零梯度；不做 reinitialization。
- 无 MRC/EPE/shot/checkpoint。

## 25. Specification Approval Gate

- [ ] Simple change 已完成且本文 baseline 已更新；
- [ ] 共享接口与实际完成源码一致；
- [ ] hard/STE/SDF/Adam/config 已确认；
- [ ] Blocking 为 None，需求均有 test/AC；
- [ ] 文件清单不修改 shared workflow/problem 和 protected paths。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-18 | draft | 首版；依赖 Simple 公共管线，冻结 SDF、hard STE、Adam 与零共享接口改动 | 待用户审核 |
