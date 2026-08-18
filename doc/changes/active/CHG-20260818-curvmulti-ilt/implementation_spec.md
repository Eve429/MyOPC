---
id: CHG-20260818-curvmulti-ilt
title: CurvMulti ILT 迁移
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

# CurvMulti ILT 迁移

## 0. Document Contract

本文是本 change 唯一实现规格。实施前 MUST 确认 Simple ILT change 完成、把 baseline 更新到其
完成 commit，并核对共享 `ILTMethod/ILTBatchResult/_ilt_workflow/PixelMacroProblem`。

实现 AI MUST 只修改 §14 文件，按 OpenILT 原算法与 `00_PAST` 工程适配作参考；MUST NOT 修改
`layout/geometry/00_PAST/lithography/evaluation`、复制 workflow、建立 registry/base class、吞异常。

## 1. Objective

迁移采用平滑 sigmoid 控制场、粗到细 warm-start、完整物理网格光刻和 nominal-wafer 曲率约束的
CurvMulti ILT，并验证首个 ILT 公共 workflow 可在不改接口的情况下接入第一个多尺度方法。

## 2. Baseline and Evidence

### 2.1 Baseline

- 设计基线：`2fa75ea89ea6cd64122214f1e2e0ed14cae518c3`；446 passed。
- 实施基线：MUST 更新为 Simple ILT 完成 commit 并记录新全量基线。
- 设计开始工作树 clean；交付前 `opc/input/grid.py` 出现一处来源未确认的注释删改，行为不变，
  实施时 MUST 保留并排除在本 change 提交之外。

### 2.2 Confirmed Facts

| Fact | Confirmed fact | Evidence | Verification |
|---|---|---|---|
| FACT-001 | OpenILT CurvMulti 使用 7×7 avg-pool、sigmoid offset、SGD、粗到细 nearest warm-start | `OpenILT/pyilt/curvmulti.py::CurvILT.solve`/main | 只读源码 |
| FACT-002 | 上游入口最低级实际调用 `solve(target,target)`，不是 PixelInit 的 `2*target-1` | `OpenILT/pyilt/curvmulti.py:208-212` | 只读源码 |
| FACT-003 | 上游 nominal L2 误用 `printedMax`，同时曲率使用 `printedNom` 且权重硬编码 200 | `OpenILT/pyilt/curvmulti.py::CurvILT.solve` | 只读源码 |
| FACT-004 | 旧迁移修正为具名 nominal、显式 curvature weight，并保持每级完整光学网格 | `00_PAST/opc/iteration/ilt/curvmulti.py::optimize_curvmulti` | 只读源码 |
| FACT-005 | 当前 tuple parser 只正确支持定长 tuple，不能正确解析 `tuple[int,...]` | `main/configuration.py::_parse_scalar` | 静态阅读 |
| FACT-006 | Simple 规格的通用 record 已含 stage/scale 坐标 | `CHG-20260818-simple-ilt` §8 | 规格依赖 |

### 2.3 Uncertainty Boundary

- 实施前必须以 Simple 完成源码替换设计基线事实；共享接口漂移时先修订。
- CurvMulti 的“曲率”是 nominal resist 图离散二阶平滑代理，不是严格几何曲率/MRC。
- 本 change 不声明比 Simple/LevelSet 更优，只记录相同 smoke 指标与资源。

### 2.4 External and Archive References

| Reference | Role | Adopt | Reject | Reason |
|---|---|---|---|---|
| `OpenILT/pyilt/curvmulti.py` | 原算法 | target 初值、smooth sigmoid、SGD、nearest warm-start、wafer curvature | printedMax 冒充 nominal、硬编码 200、跨不同物理 grid 直接换模型、全局 DEVICE | 明确修正 bug，保持当前具名模型 |
| `00_PAST/opc/iteration/ilt/curvmulti.py` | 工程参考 | full-grid optics、显式 config、fixed area、stage best | `SimpleILTResult` 反向引用、最后 step 未评价 | 用公共中性 result/N+1 state |
| `00_PAST/tests/opc/test_curvmulti_ilt.py` | 测试参考 | 公式、全物理 grid、batch、GDS/CLI | 旧 runner/artifacts | 场景迁入当前 workflow |

## 3. Current Behavior

设计基线无 ILT。依赖 Simple change 完成后应已有 pixel problem、公共 ILT result/workflow 和 Simple
方法，但没有多尺度 helper、可变 tuple config 或 CurvMulti optimizer。实施前须重新核对。

## 4. Target Behavior

### REQ-001

系统 MUST 提供 `python main/run_curvmulti_ilt.py <config.toml>`，复用既有 pixel problem、macro/core
batch、指标、GDS 和 final merge。

### REQ-002

CurvMulti 参数初值 MUST 为 target coverage `[0,1]`；每级参数经 odd-kernel avg-pool 后使用
`sigmoid(beta*(pooled-offset))` 形成软 mask。

### REQ-003

scales MUST 是非空、严格递减、以 1 结束且全部整除 256 的正整数；每一级的控制网格为
`[256/scale,256/scale]`，跨级 best parameters 用 nearest resize warm-start。

### REQ-004

Hopkins forward MUST 在完整 `[B,256,256]` 物理网格运行；粗尺度只减少控制参数，不得把粗图
直接 padding 后送模型，也不得改变 `pixel_nm`。

### REQ-005

每级 MUST 使用独立 SGD；旧 optimizer state 不跨级。每级执行 N 次更新和 N+1 次评价，按样本
选择本级 best；仅本级 best 进入下一尺度，最终输出只取 scale=1 的 best。

### REQ-006

nominal L2 MUST 使用具名 nominal output，不能复制上游 `printedMax` bug；process/PV loss 使用共享
定义。曲率 MUST 作用于 full-grid nominal wafer 并只统计 ownership 有效卷积区。

### REQ-007

context 在平滑前用本级 reference 固定，完整网格恢复后再与 full-resolution fixed mask 混合；
context 不得因卷积邻域获得可训练自由度或被写回。

### REQ-008

`main.configuration::_parse_scalar` MUST 正确支持 `tuple[T,...]` 的非空任意长度 TOML list，同时
保持 `tuple[int,int]` macro_grid 的定长行为与错误信息可定位。

### REQ-009

共享 `ILTMethod`、`ILTBatchResult`、PixelMacroProblem 和 `_ilt_workflow` public/behavior contract
MUST 不变；本 change 只在 `_common` 增加第一个真实多尺度调用方需要的 resize/smooth 操作。

### REQ-010

summary/metrics records MUST 使用公共 `state_index/stage_index/stage_state_index/scale`，进度 total
为 `core_count * len(scales) * (iterations_per_stage+1)`。

## 5. Scope

### 5.1 In Scope

- CurvMulti config/optimizer、resize/smooth 共享操作、variadic tuple parser。
- adapter/runner/config、测试、文档与报告。

### 5.2 Out of Scope

- 每级不同 iterations/step（属于 Multilevel）；macro 联合场；其他 interpolation 策略。
- MRC/EPE/shot、尺度自动选择、scheduler/checkpoint、shared workflow 重构。

### 5.3 Protected Areas

`00_PAST/layout/geometry/lithography/evaluation/TestReticle GDS` 不得修改。

## 6. Invariants

### INV-001

每个尺度的光刻输入/输出、target 与 ownership 均为完整 256 网格；scale 只描述控制参数网格。

### INV-002

stage optimizer 独立；下一 stage 只读上一 stage 每样本 best，不读最后未选参数或 optimizer state。

### INV-003

context 固定且不回写；ownership 唯一写入契约与 Simple 完全相同。

### INV-004

nominal/process 名称决定 loss 语义，tuple 顺序不能让 process[0] 冒充 nominal。

### INV-005

batch size/reorder 不改变每样本 stage best 和最终输出。

### INV-006

既有 fixed tuple config 与 Simple ILT 共享接口不变。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `ilt._common` | 新增 `resize_image`、`smooth_sigmoid_mask` | stage policy/optimizer |
| `ilt.curvmulti` | scales、SGD stages、wafer curvature、best | workflow/GDS/config file I/O |
| `_curvmulti_ilt_workflow` | METHOD 描述 | 算法复制 |
| `run_curvmulti_ilt` | 直接入口 | 业务逻辑 |

### 7.2 Dependency Direction

```text
curvmulti -> ilt._common + lithography
adapter -> _ilt_workflow + curvmulti
runner -> adapter
```

不得 import `simple.py`/`levelset.py`，不得让 common 依赖具体方法。

### 7.3 Data Flow

```text
core target/ownership [B,256,256]
 -> initial=target
 -> for scale in scales:
      downsample reference/ownership/control initial
      independent SGD
      for stage state 0..N:
         smooth sigmoid control
         nearest restore full grid
         mix fixed full context
         full-grid lithography/loss/wafer curvature
         per-sample stage best
      carry stage best control to next scale
 -> final scale=1 ILTBatchResult
 -> unchanged workflow
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| method init | target | fixed full mask | immutable reference | 每尺度不重算 full fixed |
| scale init | prev best/reference | resize + new SGD | control params | 不带旧 optimizer |
| scale states | control | N+1 full optics | stage records/best | 不用粗 grid 光刻 |
| final | scale1 best | shared result | workflow | 不跨 stage 比不可比状态选输出 |

### 7.5 Planned Call Graph

```text
run_curvmulti_ilt.main
└─ _curvmulti_ilt_workflow.run_curvmulti_ilt
   └─ _ilt_workflow.run_ilt_workflow(METHOD,...)
      └─ curvmulti.optimize_curvmulti_batch
         ├─ _common.resize_image/smooth_sigmoid_mask
         ├─ per scale: SGD
         │  └─ per state: model.forward_many + shared losses/curvature
         └─ ILTBatchResult
```

## 8. Data Contracts

### `CurvMultiILTConfig`

| Field | dtype | meaning |
|---|---|---|
| `scales` | `tuple[int,...]` | strict descending, final 1, each divides 256 |
| `iterations_per_stage` | strict int | 每级更新数 `>=1` |
| `step_size` | float | 每级独立 SGD lr `>0` |
| `smoothing_kernel` | strict int | positive odd；不得大于最粗控制网格边长 |
| `sigmoid_steepness` | float | finite `>0` |
| `sigmoid_offset` | float | finite `[0,1]` |
| `weight_process_l2` | float | finite `>=0` |
| `weight_pvband` | float | finite `>=0` |
| `curvature_weight` | float | finite `>=0`，OpenILT 基线配置 200 |
| `mask_threshold` | float | finite `(0,1)` |
| `batch_size` | strict int | core batch `>=1` |

### Multi-scale state

- control params：float32 `[B,256/scale,256/scale]`，device，one stage。
- full mask/printed：float32 `[B,256,256]`，device，one state。
- previous stage best：float32 control grid，device，仅到下一 stage 初始化。
- record：公共 `ILTStateRecord`；global state 单调，stage state 从 0 重启，scale 为当前值。
- final `ILTBatchResult.best_parameters` 必须为 `[B,256,256]`（final scale=1）。

### 8.1 Configuration Contract

新增 `[curvmulti_ilt]`，上述十一字段全部 required、无默认。`scales` TOML 示例 `[4,2,1]`。
共享 layout/partition/lithography/output 不变，不要求 edge。

### 8.2 Persisted Artifact Contract

复用 pixel plan/problem/final；方法文件 `curvmulti_ilt_result.npz`；metrics records 必须持久化 stage/
scale 坐标；summary method=`curvmulti_ilt` 且 config 保留 tuple 为 JSON list。

## 9. Interface Changes

### IF-001：common 新增多尺度操作

```python
resize_image(image: torch.Tensor, shape: tuple[int,int], mode: str) -> torch.Tensor
smooth_sigmoid_mask(parameters: torch.Tensor, kernel: int,
                    steepness: float, offset: float) -> torch.Tensor
```

只接受共享内部 `[B,H,W]`；resize 使用 `torch.nn.functional.interpolate`，area/nearest 由调用方显式传。

### IF-002：variadic tuple parser

Current：`tuple[int,...]` 被当作固定两项且会解析 Ellipsis。

Target：当 `get_args(annotation)==(T, Ellipsis)` 时接受非空 TOML list 并逐项按 T 严格解析；其他 tuple
仍要求精确长度。现有 `macro_grid` 输入输出和错误路径保持。

### IF-003：solver/entry

```python
optimize_curvmulti_batch(target, ownership, model, config, *,
                         on_states_completed=None) -> ILTBatchResult
run_curvmulti_ilt(config_path: str | Path) -> dict
```

### IF-004：共享接口

`ILTMethod`、`ILTBatchResult`、PixelMacroProblem、`run_ilt_workflow`：No change required。

## 10. Algorithm

### 10.1 Initialization

`initial=target.detach()`；`fixed_full=smooth_sigmoid_mask(initial,...)`。对每 scale，reference 用 area
缩到 control shape；第一 stage initial=reference，后续 initial=nearest(previous_best)。ownership 缩放
用 nearest，并在 full grid 再次混合保证最终固定区逐像素不变。

### 10.2 Stage state

```text
effective = params*stage_movable + stage_reference*(1-stage_movable)
stage_soft = smooth_sigmoid_mask(effective)
optimized_full = nearest(stage_soft, 256x256)
mask = optimized_full*full_ownership + fixed_full*(1-full_ownership)
printed = model.forward_many(mask, named conditions)
loss = nominal target + process target + PV + curvature(nominal wafer), all owned
per-sample best update
if not final state: backward(sum); SGD.step
```

### 10.3 Boundary Conditions

| Condition | Behavior |
|---|---|
| scale 非整数/不递减/末项非1/不整除256 | config `ValueError` |
| smoothing kernel 偶数/大于最粗 grid | config `ValueError` |
| curvature=0 | 不构造 conv graph |
| stage transition | 丢弃旧 SGD；只带 per-sample best params |
| final stage | 只取 scale=1 best 输出 |
| target gray | area resize保留覆盖率，初值仍 target |
| nonfinite | `FloatingPointError`，无 completed macro |

### 10.4 State Transition

每 stage 有 state 0..N；global state 连续编号。不同 stage 的 loss 可记录但不用于跨 stage 选最终输出；
只有 final stage best 是权威结果。

## 11. Ownership and State

workflow/problem ownership 不变。stage params/SGD 由 optimizer 独占；fixed target/context 只读；旧
stage optimizer 在同步点释放；batch 完成后 shared workflow 回写 owned final mask。失败不发布 macro。

## 12. Error Handling

- Config/tuple/scale/kernel/shape/range：`ValueError`，消息含 section/field/scale。
- 非有限 loss/params：`FloatingPointError`，不 retry/降 lr。
- I/O/CUDA/model：原样传播；entry 不伪造 completed。

## 13. Performance and Memory Constraints

- 每 stage 只常驻一个 control params + 一个 full batch graph；旧 stage optimizer/graph 必须释放。
- 每 state 一次三条件 full-grid forward；绝不在粗 grid 调模型。
- resize/smooth 全 torch batch 操作；禁止逐 pixel/sample Python hot loop。
- GPU 上界仍按 `[B,256,256]` optics，而非 reticle/macro；CPU problem/output 复用 shared contract。
- smoke 记录各 stage/state、总耗时/RSS/CUDA peak；不设阈值。

## 14. File-Level Change Plan

| File / Symbol | Type | Action | Contract change | Reason |
|---|---|---|---|---|
| `opc/iteration/ilt/_common.py` | 业务代码 | modify | add resize/smooth | REQ-002..004/009 |
| `opc/iteration/ilt/curvmulti.py` | 业务代码 | add | config + staged solver | REQ-002..007 |
| `opc/iteration/ilt/__init__.py` | 业务代码 | modify | exports | REQ-001 |
| `main/configuration.py::_parse_scalar/CONFIG_SECTIONS` | 业务代码 | modify | variadic tuple + section | REQ-008 |
| `main/_curvmulti_ilt_workflow.py` | adapter | add | METHOD/thin run | REQ-009/010 |
| `main/run_curvmulti_ilt.py` | entry | add | direct Python | REQ-001 |
| `config/curvmulti_ilt.toml` | config | add | smoke | §8.1 |
| `tests/opc/iteration/test_curvmulti_ilt.py` | tests | add | formula/stages/batch/real model | TEST-001..007 |
| `tests/main/test_curvmulti_ilt_runner.py` | tests | add | adapter/CLI/artifacts/progress | TEST-008/009 |
| `tests/main/test_configuration.py` | tests | modify | variadic/fixed tuple matrix | TEST-010 |
| `doc/contracts/ilt.md` | 接口文档 | modify | 增加 CurvMulti API/限制 | 交付 |
| `doc/architecture/system.md`、`doc/architecture/dataflow.md` | 架构文档 | modify | 增加多尺度组件/数据流 | 交付 |
| `doc/development_manual.md`、`doc/test_manual.md` | 手册 | modify | 使用与测试 | 交付 |
| `doc/changes/active/CHG-20260818-curvmulti-ilt/implementation_spec.md` → `doc/changes/completed/CHG-20260818-curvmulti-ilt/implementation_spec.md` | 规格 | move | baseline/status/revision/evidence | 交付 |
| `doc/changes/completed/CHG-20260818-curvmulti-ilt/development_report.md` | 开发报告 | add | 实施、偏差、审计 | 交付 |
| `doc/changes/completed/CHG-20260818-curvmulti-ilt/test_report.md` | 测试报告 | add | 环境、命令、结果 | 交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` | 项目记录 | modify | 同步 | AGENTS |

`main/_ilt_workflow.py`、`opc/input/pixel/**` 不得修改。

## 15. Test Specification

### TEST-001：smooth sigmoid 公式

手算 7×7/3×3、2D batch；forward 与 `avg_pool2d+sigmoid` 逐值。

### TEST-002：scale/config

合法 `[4,2,1]`；拒绝空、bool、0、重复/升序、末项非1、不整除256、非法 kernel。

### TEST-003：full physical optical grid

spy model 在所有 stage/state 只收到 `[B,256,256]`；粗控制 shape/nearest warm-start 精确。

### TEST-004：具名 nominal 与 curvature

给 nominal/max 不同的 stub，证明 nominal L2 不取 max；curvature 取 nominal wafer 且 owned-only。

### TEST-005：stage/state/best

每 stage N+1、SGD 独立、只带 per-sample stage best、final scale1 output；global/stage record 坐标精确。

### TEST-006：context/batch invariance

平滑前后 context 固定；batch 1/B/reorder 同一 core 输出一致。

### TEST-007：真实 ICCAD13 CPU/CUDA

至少 `[2,1]`、每级一更新；finite backward/输出，CPU/CUDA 容差记录。

### TEST-008：runner/artifacts/progress

generated multi-core GDS；method/config/stage records/GDS/summary 完整；total 精确。

### TEST-009：直接 main

仓库外 cwd 运行 config，完成最终 merge；异常条 finally close。

### TEST-010：tuple parser 回归

variadic int/float list、空/非法元素；既有 macro_grid 两项通过、一/三项继续失败；所有旧 config tests 绿。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected |
|---|---|---|
| Scales | [1], [2,1], [4,2,1], invalid | shape/order/error |
| Geometry | rect/hole/diagonal/dense/empty | finite/full-grid |
| Boundary | ownership/context/core seam | fixed context |
| State | N=1/2, different sample best | stage sync/batch invariant |
| Device | CPU/CUDA | full-grid backward |
| Config | fixed/variadic tuple/errors | no old regression |

### 15.2 Verification Commands

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_curvmulti_ilt.py tests/main/test_curvmulti_ilt_runner.py tests/main/test_configuration.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_simple_ilt.py tests/opc/input/test_pixel_problem.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc main tests
D:\app\miniforge\envs\myopc\python.exe -m compileall -q opc main tests
D:\app\miniforge\envs\myopc\python.exe main/run_curvmulti_ilt.py config/curvmulti_ilt.toml
git diff --check
```

## 16. Requirement Traceability

| Requirement/Invariant | Implementation | Tests | AC |
|---|---|---|---|
| REQ-001/009/010 | adapter/runner | TEST-008/009 | AC-001/005 |
| REQ-002/003 | common/config/solver | TEST-001..003 | AC-002 |
| REQ-004/006 | solver | TEST-003/004/007 | AC-003 |
| REQ-005/007 | solver | TEST-005/006 | AC-004 |
| REQ-008 | parser | TEST-010 | AC-006 |
| INV-001..005 | solver/common | TEST-003..007 | AC-003/004 |
| INV-006 | no-change/full regression | TEST-010/full | AC-005/006 |

## 17. Acceptance Criteria

- [ ] **AC-001**：direct main 完成 method=`curvmulti_ilt`，产物/merge 与共享 contract 一致。
- [ ] **AC-002**：smooth/scale/control-grid/warm-start 公式测试通过。
- [ ] **AC-003**：所有 stage 光刻固定 256，nominal bug 未复制，wafer curvature 正确。
- [ ] **AC-004**：N+1 stage、独立 SGD、per-sample best、context/batch 不变量通过。
- [ ] **AC-005**：共享 workflow/problem/result 未改，Simple 与全量回归绿，资源基线记录。
- [ ] **AC-006**：variadic tuple 新测试和既有 fixed tuple 全绿。
- [ ] **AC-007**：文档/报告/规划/简化审计完成。

## 18. Compatibility and Migration

- API：新增方法与两个 `_common` 操作；共享 workflow/result/problem 不变。
- Config：扩展 tuple parser 支持 variadic，fixed tuple 保持；新增 section。
- Data：复用 pixel problem/result v1，method/config/record 区分。
- CLI：新增 direct entry，无旧 argparse 兼容。
- Numerical：保留 CurvMulti 核心，明确修复 printedMax bug并采用当前模型/ownership/N+1/per-sample best；
  不承诺 OpenILT 逐值。

## 19. Decisions

### DEC-001：粗尺度只控制参数，光学始终 full grid

Reason：Hopkins kernel 的 pixel 物理含义固定；粗图 padding 会缩小物理图形。

### DEC-002：修复具名 nominal bug

Reason：上游变量与算法意图冲突；复制会重复计算 process[0] 而漏掉 nominal。

### DEC-003：CurvMulti 用 uniform SGD，Multilevel 留给下一 change

Reason：保持两个方法真正差异，避免一个万能配置/solver。

### DEC-004：多尺度 helper 到首个真实调用方才加入 common

Reason：避免 Simple change 一次性预建；此时 CurvMulti 为当前调用方，后续 Multilevel 可复用。

## 20. Open Questions

### 20.1 Blocking

None.

### 20.2 Non-blocking

- 自动 scale、不同 interpolation、mask curvature 与 wafer curvature 对比另立 change。

## 21. Implementation Freedom

允许等价 batch resize 与局部变量组织；不得改变初值、SGD、stage best、full-grid optics、wafer
curvature、tuple/record/shared contract。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Verify | Commit |
|---|---|---|---|---|
| A | tuple parser + multi helpers | configuration/common/tests | TEST-001/002/010 | `feat(config): 支持ILT多尺度元组` |
| B | CurvMulti optimizer | curvmulti/tests | TEST-003..007 | `feat(ilt): 实现CurvMulti优化器` |
| C | adapter/runner/artifacts | main/config/tests | TEST-008/009/direct | `feat(main): 接入CurvMultiILT` |
| D | full/smoke/docs/audit | doc/planning | all | `docs(ilt): 完成CurvMulti迁移报告` |

只本地 commit，未经授权不得 push。

## 23. Delivery and Final Audit

同步 contracts/architecture/manual/reports/planning；记录性能和数值偏差；审计 printedMax bug 未残留、
无粗网格模型调用、无复制 resize/loss、无一次性抽象/吞异常、中文 docstring 完整；protected paths 未改。

## 24. Known Limitations and Future Work

- 继承 tile-independent seam；pixel stair-step/MRC/shot 不处理。
- uniform stage iterations/step，scale 必须整除 256。
- 曲率是 resist 图离散代理，可能与最终二值轮廓质量不一致。

## 25. Specification Approval Gate

- [ ] Simple change 完成且 baseline 更新；
- [ ] 用户确认修复 printedMax bug并保留 wafer curvature；
- [ ] scales/SGD/N+1/stage best/config 确定；
- [ ] shared workflow/problem/result 不修改；
- [ ] tests/AC/file list 完整，Blocking None。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-18 | draft | 首版；冻结 full-grid optics、具名 nominal 修复、uniform SGD 与 variadic tuple 扩展 | 待用户审核 |
