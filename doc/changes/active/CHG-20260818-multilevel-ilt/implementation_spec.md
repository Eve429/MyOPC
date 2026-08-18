---
id: CHG-20260818-multilevel-ilt
title: Multilevel ILT 迁移
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
  - doc/changes/completed/CHG-20260818-curvmulti-ilt/implementation_spec.md
  - doc/contracts/ilt.md
  - doc/contracts/lithography.md
supersedes: []
---

# Multilevel ILT 迁移

## 0. Document Contract

本文是本 change 唯一实现规格。实施前 MUST 确认 Simple 和 CurvMulti change 均完成，把 baseline
更新到最新依赖 commit，并以真实公共接口替换本文的依赖预期；否则只能审查。

实现 AI MUST 只改 §14；复用 pixel problem、ILT workflow/result、variadic tuple parser 和
resize/smooth helper。MUST NOT 修改 protected paths、复制 CurvMulti solver 改参数冒充新方法、
建立统一万能多尺度类、吞异常或自行选择缺省 stage 配置。

## 1. Objective

迁移 OpenILT 的 Multilevel 思路：以逐级不同 Adam 迭代数/学习率优化控制网格，同时始终在完整
物理网格执行光刻，再把 printed/target area-downsample 到本级监督网格计算 loss；通过现有公共
workflow 输出完整大版图结果。

## 2. Baseline and Evidence

### 2.1 Baseline

- 设计基线：`2fa75ea89ea6cd64122214f1e2e0ed14cae518c3`，446 passed。
- 实施基线：MUST 更新为 CurvMulti 完成 commit，并记录依赖后全量基线。
- 设计开始工作树 clean；交付前 `opc/input/grid.py` 出现一处来源未确认的注释删改，行为不变，
  实施时 MUST 保留并排除在本 change 提交之外。

### 2.2 Confirmed Facts

| Fact | Confirmed fact | Evidence | Verification |
|---|---|---|---|
| FACT-001 | 上游 Multilevel 用两个/多个不同尺度 solver，跨级 nearest best params | `OpenILT/pyilt/multilevel.py` main | 只读源码 |
| FACT-002 | 上游最低级实际调用 `solve(target,target)`；参数化为 avg-pool+sigmoid | `OpenILT/pyilt/multilevel.py:252-259` | 只读源码 |
| FACT-003 | 旧迁移把多 solver 改成一个 stage loop，每级独立 Adam、不同 iterations/step | `00_PAST/opc/iteration/ilt/multilevel.py::optimize_multilevel` | 只读源码 |
| FACT-004 | 旧迁移每级 full-grid 光刻后把 printed area-downsample 到 stage target 监督 | 同上 | 只读源码 |
| FACT-005 | 旧测试区分 stage supervision grid、full physical optics 和 independent Adam | `00_PAST/tests/opc/test_multilevel_ilt.py` | 只读测试 |
| FACT-006 | 目标 variadic tuple/helper 由 CurvMulti change 提供 | 依赖规格 §8/§9 | 规格依赖 |

### 2.3 Uncertainty Boundary

- 依赖完成实现若与规格不同，本文必须先修订。
- 不同 stage 的 loss 位于不同监督网格，绝对值不可跨 stage 比较；本 change 不做跨 stage global best。
- 本 change 不声称 Multilevel 优于 CurvMulti，只提供独立可验证实现和基线。

### 2.4 External and Archive References

| Reference | Role | Adopt | Reject | Reason |
|---|---|---|---|---|
| `OpenILT/pyilt/multilevel.py` | 原算法 | stage-specific solver、nearest warm-start、平滑 sigmoid | 手工多 solver/多 config、全局 DEVICE、printedMax nominal bug、无 N+1 | 合并为明确 stage loop并修正具名条件 |
| `00_PAST/opc/iteration/ilt/multilevel.py` | 工程参考 | full optics→stage supervision、独立 Adam、显式 tuples | method-to-Simple result import、末步未评价 | 复用中性公共 result |
| `00_PAST/tests/opc/test_multilevel_ilt.py` | 测试参考 | stage loss、divisibility、batch、runner | 旧接线 | 迁到当前 workflow |

## 3. Current Behavior

设计基线无 ILT。依赖完成后应已有 Simple 公共管线、CurvMulti 的 `resize_image`/
`smooth_sigmoid_mask` 和 tuple parser，但没有 stage-specific iterations/step、stage supervision 或
Multilevel adapter/entry。实施前须以真实代码核对。

## 4. Target Behavior

### REQ-001

系统 MUST 提供 `python main/run_multilevel_ilt.py <config.toml>`，复用当前 pixel/macro/core/GDS
workflow，不增加 method string 分派。

### REQ-002

`scales`、`stage_iterations`、`stage_step_sizes` MUST 非空且长度相同；scales 严格递减、末项 1、
全部整除 256；迭代数正整数、step finite 正数。

### REQ-003

每 stage MUST 建立独立 Adam，使用对应 iterations/step；optimizer/momentum MUST NOT 跨 stage。

### REQ-004

每 state MUST 把控制 soft mask nearest 恢复为 `[B,256,256]` 后调用完整物理模型，再把每个具名
printed image 以 area resize 到 stage shape；不得在 coarse grid 调光刻。

### REQ-005

stage target MUST 是 full target 的 area resize；stage ownership MUST 是 full ownership 的 area
resize权重 `[0,1]`。nominal/process/PV 和 stage nominal-wafer curvature 的逐像素平方项 MUST 乘该
权重再求和，避免边界像素重复或丢失。

### REQ-006

每 stage 执行 N 次 Adam 更新和 N+1 次评价，逐样本选本级 best；只有本级 best parameters 进入
下一 stage，最终结果只来自 scale=1 stage。

### REQ-007

参数初值、平滑、context 固定和 full-grid 二次混合 MUST 与 CurvMulti 保持同一公式；两方法差异
仅限 optimizer、每级配置和监督/曲率网格。

### REQ-008

`_common` 连续 loss 选择参数 MUST 支持 bool mask 和 float `[0,1]` 权重，Simple/CurvMulti 的 bool
输入数值保持逐值不变；不得复制第二套 loss。

### REQ-009

共享 `ILTMethod/ILTBatchResult/PixelMacroProblem/_ilt_workflow` MUST 不变；records 使用公共 stage/
scale 坐标，进度总量为 `core_count*sum(stage_iterations[i]+1)`。

### REQ-010

最终二值评价、macro NPZ/JSON/GDS、final merge/lithography 和资源统计 MUST 复用共享 workflow。

## 5. Scope

### 5.1 In Scope

- Multilevel config/stage optimizer、weighted stage loss、小幅 widening 共用 loss selection。
- adapter/runner/config/tests/docs/reports。

### 5.2 Out of Scope

- 自动 scale/iterations/lr、scheduler、跨 stage optimizer state、macro joint field。
- CurvMulti 行为重构、其他 ILT、EPE/shot/MRC/checkpoint。

### 5.3 Protected Areas

`00_PAST/layout/geometry/lithography/evaluation/TestReticle GDS` 不得修改；共享 workflow/problem/result
不得修改。

## 6. Invariants

### INV-001

光刻始终完整 256 物理网格；监督/曲率只在 stage grid。

### INV-002

stage ownership weight 来源唯一为 area(full ownership)，范围 `[0,1]`；loss 中只乘一次。

### INV-003

每级 Adam 独立；下一 stage 只接收每样本 best parameters。

### INV-004

同一 stage 的 record 才可比较 best；final 权威输出只来自最后 stage。

### INV-005

context fixed、per-sample best、batch invariance、transmission/GDS contract 与共享管线一致。

### INV-006

Simple/CurvMulti bool ownership loss 逐值不变。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `ilt._common` | selection 从 bool 扩为 nonnegative weights | stage policy |
| `ilt.multilevel` | stage config、Adam、full optics→stage supervision | workflow/GDS |
| `_multilevel_ilt_workflow` | METHOD 描述 | solver 数学 |
| `run_multilevel_ilt` | direct entry | 业务逻辑 |

### 7.2 Dependency Direction

```text
multilevel -> ilt._common + lithography
adapter -> _ilt_workflow + multilevel
runner -> adapter
```

不得 import CurvMulti 具体 solver；只复用 `_common` helper。

### 7.3 Data Flow

```text
full target/ownership
 -> initial=target; fixed_full=smooth(initial)
 -> for (scale, count, lr):
      stage_target = area(target)
      stage_weight = area(ownership.float)
      control reference/init + independent Adam
      state 0..count:
         control smooth -> nearest full mask -> full lithography
         area printed -> stage weighted loss/wafer curvature
         per-sample stage best
      carry best to next stage
 -> final scale1 ILTBatchResult
 -> unchanged shared workflow/output
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| run init | full target | fixed full mask | immutable | 不按 stage 重算 |
| level init | scale/config/prev best | resize/new Adam | stage refs | 不继承 optimizer |
| level state | control | full optics + stage supervision | record/best | 不在 full+stage 双计 loss |
| level end | stage best | handoff/release | next initial | 不带 graph |

### 7.5 Planned Call Graph

```text
run_multilevel_ilt.main
└─ _multilevel_ilt_workflow.run_multilevel_ilt
   └─ _ilt_workflow.run_ilt_workflow(METHOD,...)
      └─ multilevel.optimize_multilevel_batch
         ├─ _common.resize_image/smooth_sigmoid_mask
         ├─ per stage: Adam
         │  └─ per state: full model.forward_many
         │     └─ area printed/target/ownership -> weighted common losses
         └─ ILTBatchResult
```

## 8. Data Contracts

### `MultilevelILTConfig`

| Field | dtype | meaning |
|---|---|---|
| `scales` | tuple[int,...] | strict descending/final1/divides256 |
| `stage_iterations` | tuple[int,...] | same length；每项 `>=1` |
| `stage_step_sizes` | tuple[float,...] | same length；finite `>0`，Adam lr |
| `smoothing_kernel` | strict int | positive odd，<=最粗 grid |
| `sigmoid_steepness` | float | finite `>0` |
| `sigmoid_offset` | float | finite `[0,1]` |
| `weight_process_l2` | float | finite `>=0` |
| `weight_pvband` | float | finite `>=0` |
| `curvature_weight` | float | finite `>=0` |
| `mask_threshold` | float | finite `(0,1)` |
| `batch_size` | strict int | `>=1` |

### Stage arrays

令 `Hs=Ws=256/scale`：target/printed float32 `[B,Hs,Ws]`；ownership weights float32 同形；control
params + Adam m/v 同形；full mask/printed `[B,256,256]`。final result 仍 `[B,256,256]`。

### Weighted selection contract

`continuous_losses(..., selection)` 接受 bool 或 float32、同 shape/device；float selection 必须 finite
且 `[0,1]`。每个逐像素平方项先乘 selection 再按 H/W 求和，返回 `[B]`；不对权重归一化。

### 8.1 Configuration Contract

新增 `[multilevel_ilt]`，上述字段全部 required。TOML：

```toml
scales = [2, 1]
stage_iterations = [20, 100]
stage_step_sizes = [0.2, 0.2]
```

不允许只写一个总 iterations/step 后静默广播。

### 8.2 Persisted Artifact Contract

复用 pixel problem/result/final v1；方法文件 `multilevel_ilt_result.npz`；metrics records 含 stage/scale；
summary method=`multilevel_ilt`，三 tuple JSON 化为同序 list。

## 9. Interface Changes

### IF-001：weighted common loss

Current（依赖 Simple/Curv 预期）：selection 是 bool ownership。

Target：同一函数允许 bool 或 finite `[0,1]` float selection；bool 路径转换为 0/1 后公式逐值相同。
非法 shape/device/range 抛 `ValueError`。全部现有调用点无需改参数。

### IF-002：solver

```python
optimize_multilevel_batch(target, ownership, model, config, *,
                          on_states_completed=None) -> ILTBatchResult
```

### IF-003：config/entry

注册 `MultilevelILTConfig -> "multilevel_ilt"`；新增 adapter METHOD、thin run 和 direct main。

### IF-004：共享 workflow/problem/result

No change required；需要修改时停止修订规格。

## 10. Algorithm

### 10.1 Run initialization

`initial=target`；fixed full mask 用共用 smooth sigmoid。previous best 初始 None。global state=0。

### 10.2 Per-stage

```text
shape=(256/scale,256/scale)
stage_target=area(full target)
stage_reference=area(initial)
stage_init=stage_reference if first else nearest(previous_best)
stage_movable=nearest(full ownership.float) for parameter fixing
stage_weight=area(full ownership.float) for loss
params=clone(stage_init); Adam(params, lr_i)
for stage_state 0..iterations_i:
    effective=params*stage_movable + reference*(1-stage_movable)
    stage_soft=smooth_sigmoid(effective)
    full_soft=nearest(stage_soft,256)
    full_mask=full_soft*full ownership + fixed_full*(1-full ownership)
    printed_full=model.forward_many(...)
    printed_stage=area(each printed_full)
    weighted stage target/process/PV/nominal-wafer curvature
    per-sample stage best
    record/notify
    if not final state: backward(sum); Adam.step
previous_best=stage_best_parameters.detach
release stage optimizer/graph
```

### 10.3 Boundary Conditions

| Condition | Behavior |
|---|---|
| tuple lengths unequal/empty | config ValueError |
| scale invalid/kernel too large | config ValueError |
| partial ownership at coarse boundary | area weight fraction，不 round/drop |
| weight all zero（理论上空 owner） | record zero selected losses；共享 workflow 不应产生空 core |
| stage loss different scale | 只选 stage-local best |
| final scale !=1 | config reject |
| nonfinite | FloatingPointError，无 macro completed artifact |

### 10.4 State Transition

每级 `S(i,0)..S(i,Ni)` 可比较；`best(i)` nearest 成 `S(i+1,0)` 初值但由新监督重新评价。最终只发布
`best(last)`。global state 只为记录/进度标识，不用于跨级 best。

## 11. Ownership and State

full target/fixed context 由 batch solver 只读；stage params/Adam 为当前 stage 唯一 writer；stage weight
只读；best per sample；stage barrier 后释放 optimizer并发布 handoff；workflow 仅消费 final result。
异常时不发布 macro/final，磁盘语义沿用共享 workflow。

## 12. Error Handling

- Config/tuple/selection/shape/range：`ValueError`，包含字段或 stage index。
- 非有限 loss/parameter：`FloatingPointError`；不得 skip stage/sample或自动降 lr。
- 模型/CUDA/I/O：原样传播；进度条 finally 收尾，summary 不写 completed。

## 13. Performance and Memory Constraints

- 每 stage 只常驻一套 control params+Adam m/v；上一 stage graph/optimizer释放后再下一级。
- full printed 仅到 area resize 建图所需生命周期；不得保存全部 stage/state wafer。
- 每 state 一次三条件 full-grid forward；area resize 批量 torch；禁止逐 pixel/sample Python loop。
- CPU/GPU 上界与一个 core batch相关，不随 reticle macro 数累积。
- smoke 记录每 stage 时间、总时间/RSS/CUDA peak；不设阈值。

## 14. File-Level Change Plan

| File/Symbol | Type | Action | Contract | Reason |
|---|---|---|---|---|
| `opc/iteration/ilt/_common.py::continuous_losses` | business | modify | weighted selection | REQ-005/008 |
| `opc/iteration/ilt/multilevel.py` | business | add | config/staged Adam solver | REQ-002..007 |
| `opc/iteration/ilt/__init__.py` | business | modify | exports | REQ-001 |
| `main/configuration.py::CONFIG_SECTIONS` | business | modify | register section | IF-003 |
| `main/_multilevel_ilt_workflow.py` | adapter | add | METHOD/thin run | REQ-009/010 |
| `main/run_multilevel_ilt.py` | entry | add | direct Python | REQ-001 |
| `config/multilevel_ilt.toml` | config | add | explicit stage smoke | §8.1 |
| `tests/opc/iteration/test_multilevel_ilt.py` | tests | add | weighted/stage/full optics/real model | TEST-001..007 |
| `tests/opc/iteration/test_simple_ilt.py`、`test_curvmulti_ilt.py` | tests | modify only if needed | bool selection zero-regression assertion | REQ-008 |
| `tests/main/test_multilevel_ilt_runner.py` | tests | add | config/runner/artifacts/progress | TEST-008/009 |
| `tests/main/test_configuration.py` | tests | modify | tuple lengths/section | TEST-010 |
| `doc/contracts/ilt.md` | 接口文档 | modify | 增加 Multilevel API/限制 | 交付 |
| `doc/architecture/system.md`、`doc/architecture/dataflow.md` | 架构文档 | modify | 增加 stage supervision 数据流 | 交付 |
| `doc/development_manual.md`、`doc/test_manual.md` | 手册 | modify | 使用与测试 | 交付 |
| `doc/changes/active/CHG-20260818-multilevel-ilt/implementation_spec.md` → `doc/changes/completed/CHG-20260818-multilevel-ilt/implementation_spec.md` | 规格 | move | baseline/status/revision/evidence | 交付 |
| `doc/changes/completed/CHG-20260818-multilevel-ilt/development_report.md` | 开发报告 | add | 实施、偏差、审计 | 交付 |
| `doc/changes/completed/CHG-20260818-multilevel-ilt/test_report.md` | 测试报告 | add | 环境、命令、结果 | 交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` | 项目记录 | modify | 同步 | AGENTS |

不得修改 `_ilt_workflow.py`、pixel problem、CurvMulti solver 或 protected paths。

## 15. Test Specification

### TEST-001：weighted loss

小图手算 bool/0.25/0.5/1 权重 nominal/process/PV；bool 与修改前逐值一致；非法 range/shape失败。

### TEST-002：config tuples

合法不同长度 stages；拒绝长度不等、空、bool、非递减、非1结尾、不整除、非正 count/lr。

### TEST-003：stage supervision formula

spy identity model，小图 area target/printed/ownership 手算；record nominal 精确。

### TEST-004：full physical optics

全部 stage/state model 输入均 `[B,256,256]`，coarse only control；forward count 精确。

### TEST-005：independent Adam/stage handoff

spy optimizer/state或可判别输入，证明新 Adam、per-sample stage best nearest handoff、final-stage-only output。

### TEST-006：context/batch invariance

full/stage 双固定；batch1/B/reorder 同一 core final一致。

### TEST-007：真实 ICCAD13 CPU/CUDA

`scales=[2,1]`、各一更新；finite backward/输出，容差与资源记录。

### TEST-008：runner/artifacts/progress

generated multi-core GDS；records stage counts、config tuples、result/GDS/summary 完整；进度 total 精确。

### TEST-009：direct main/异常收尾

仓库外 cwd 成功；中途 model 异常时 bars close，无 final summary。

### TEST-010：shared regression

Simple/Curv/pixel/workflow/config/full suite 绿；static diff 证明禁止文件未改。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected |
|---|---|---|
| Stage | [1], [2,1], unequal counts/lr | config/stage records |
| Weight | bool/fractional/zero/invalid | exact weighted sum |
| Geometry | rect/hole/diagonal/dense | full optics finite |
| Boundary | coarse fractional ownership/context | no duplicate/drop |
| Batch | 1/B/reorder/different best | invariant |
| Device | CPU/CUDA | Adam backward |
| Failure | config/nonfinite/model/I/O | explicit/no completed |

### 15.2 Verification Commands

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_multilevel_ilt.py tests/main/test_multilevel_ilt_runner.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_simple_ilt.py tests/opc/iteration/test_curvmulti_ilt.py tests/opc/input/test_pixel_problem.py tests/main/test_configuration.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc main tests
D:\app\miniforge\envs\myopc\python.exe -m compileall -q opc main tests
D:\app\miniforge\envs\myopc\python.exe main/run_multilevel_ilt.py config/multilevel_ilt.toml
git diff --check
```

## 16. Requirement Traceability

| Requirement/Invariant | Implementation | Tests | AC |
|---|---|---|---|
| REQ-001/009/010 | adapter/runner | TEST-008..010 | AC-001/006 |
| REQ-002/003 | config/solver | TEST-002/005 | AC-002/004 |
| REQ-004/005 | solver/common | TEST-001/003/004/007 | AC-003 |
| REQ-006/007 | solver | TEST-005/006 | AC-004/005 |
| REQ-008 | common loss | TEST-001/010 | AC-005 |
| INV-001..004 | solver | TEST-003..005 | AC-003/004 |
| INV-005/006 | solver/regression | TEST-006/010 | AC-005/006 |

## 17. Acceptance Criteria

- [ ] **AC-001**：direct main 完成 method=`multilevel_ilt`，共享产物和 merge 完整。
- [ ] **AC-002**：三 tuple 配置/validation 全矩阵通过，无隐式广播。
- [ ] **AC-003**：full optics→stage weighted supervision/curvature 手算和 spy 通过。
- [ ] **AC-004**：独立 Adam、N+1、stage-local per-sample best、final-stage output 通过。
- [ ] **AC-005**：context/batch invariant 和 Simple/Curv bool loss 零变化。
- [ ] **AC-006**：共享 workflow/problem/result/Curv solver 未改，全量/静态检查绿，资源记录。
- [ ] **AC-007**：文档/报告/规划/简化审计完成。

## 18. Compatibility and Migration

- API：新增 Multilevel；common loss selection 向后兼容扩展，bool 数值相同；其他共享接口不变。
- Config：依赖 Curv variadic parser，新增三个等长 tuple section；不支持旧 argparse 广播。
- Data：复用 pixel/result v1，method/config/records 区分。
- Numerical：保留上游 multilevel 意图，采用具名 nominal、full optics、weighted ownership、N+1、
  per-sample best 和当前模型；不承诺上游逐值。

## 19. Decisions

### DEC-001：Multilevel 独立实现，不把 CurvMulti 参数化成万能 solver

Reason：optimizer、每级超参、监督网格和曲率网格均有真实差异；强行合并会增加分支并降低可读性。

### DEC-002：stage ownership 使用 area 权重

Reason：coarse pixel 可能只覆盖部分 ownership；nearest bool 会丢失或重复边界贡献。

### DEC-003：不跨 stage 选 global best

Reason：监督分辨率不同，loss 数值不可直接比较；最终可制造输出必须来自 scale=1。

### DEC-004：显式 tuple，不广播总 iterations/step

Reason：避免隐藏算法配置；用户必须明确每级预算。

## 20. Open Questions

### 20.1 Blocking

None.

### 20.2 Non-blocking

- 是否按像素面积归一化不同 stage loss、自动 stage schedule，另立算法 change。

## 21. Implementation Freedom

允许等价 area resize/临时释放顺序；不得改变 weighted sum、Adam、stage-local best、full optics、final
stage、explicit tuple 或共享接口。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Verify | Commit |
|---|---|---|---|---|
| A | weighted common loss + config | common/config/tests | TEST-001/002/010 | `feat(ilt): 支持多级加权监督` |
| B | Multilevel optimizer | multilevel/tests | TEST-003..007 | `feat(ilt): 实现Multilevel优化器` |
| C | adapter/runner | main/config/tests | TEST-008/009/direct | `feat(main): 接入MultilevelILT` |
| D | full/smoke/docs/audit | doc/planning | all | `docs(ilt): 完成Multilevel迁移报告` |

只本地 commit，未经授权不得 push。

## 23. Delivery and Final Audit

同步 current docs/manual/reports/planning；记录 stage 性能/资源/数值差异；审计 full-grid model 调用、
weighted selection 只乘一次、无复制 solver/loss/resize、无吞异常/一次性 wrapper、中文 docstring；
确认 protected/shared 文件未越界修改。

## 24. Known Limitations and Future Work

- 继承 tile-independent seam 与 pixel GDS 限制。
- scales 必须整除 256；不自动选择 schedule。
- stage loss 未做像素数归一化，数值只在 stage 内用于 best。
- 无 EPE/MRC/shot/checkpoint/distributed worker。

## 25. Specification Approval Gate

- [ ] Simple/Curv changes 已完成，baseline/接口已更新；
- [ ] 用户确认 stage weighted supervision、独立 Adam 与显式 tuple；
- [ ] shared workflow/problem/result/Curv solver 不修改；
- [ ] requirements/tests/AC/file list 完整；Blocking None。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-18 | draft | 首版；冻结 full optics→weighted stage supervision、独立 Adam、显式 per-stage 配置 | 待用户审核 |
