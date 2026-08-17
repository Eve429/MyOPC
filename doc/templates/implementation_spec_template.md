---
id: CHG-YYYYMMDD-short-name
title: <功能名称>
type: implementation-spec
status: draft
baseline_commit: <完整 Git commit hash>
baseline_worktree: clean
baseline_dirty_paths: []
scope:
  - <受影响的模块或目录>
depends_on: []
supersedes: []
---

# <功能名称>

> 使用说明：复制本文件创建 change 规格，不直接修改模板。删除所有占位文本并填写事实；
> 不适用的必填章节写明 `Not applicable — <原因>`，不得静默删除。实际 `status` 只能填写
> `draft`、`approved`、`implementing`、`completed` 或 `superseded` 中的一个值。

## 0. Document Contract

本文档是该 change 不依赖聊天上下文的唯一实现规格。仓库根 `AGENTS.md`、本文记录的基线源码、
测试以及 `depends_on` 文档仍是必须读取的上位约束和事实来源；本文不得覆盖 `AGENTS.md`。

实现 AI MUST：

- 以本文档的 Target Behavior、Invariants、Interfaces 和 Acceptance Criteria 为实现目标；
- 在开始前确认 `baseline_commit`，并核对与本 change 有关的工作树差异；
- 只修改 File-Level Change Plan 列出的文件和符号；发现必须扩大范围时停止并请求修订规格；
- 不依赖生成本文档时的聊天上下文，不自行补充未定义的产品或算法需求；
- 对无法从基线代码、测试、依赖文档或本规格确认的事实明确报告不确定性；
- 若 Blocking Open Question 非空、状态不是 `approved`，或基线发生实质漂移，不得开始实施；
- 实现结束后逐项提供 requirement、test 和 acceptance evidence。

实现 AI MUST NOT：

- 为未来假设需求增加未被当前 requirement 使用的抽象、兼容层或防御分支；
- 静默 fallback、忽略异常、改变数据 ownership、坐标约定或同步语义；
- 修改 `00_PAST/`、用户数据或未授权的 `layout/`、`geometry/`。

规范词含义：

- **MUST / MUST NOT**：验收所需的强制要求；
- **SHOULD / SHOULD NOT**：除非在 Decisions 中记录可验证理由，否则必须遵守；
- **MAY**：不改变其他 contract 时允许采用的实现选择。

状态流转：

```text
draft -> approved -> implementing -> completed
   \-----------> superseded <-------------/
```

已批准规格的任何 contract 变更 MUST 将状态退回 `draft` 并重新审核。只修正错字且不改变语义
时 MAY 保持状态，但 MUST 记录在 Revision History。

## 1. Objective

用一句话说明为什么要修改。

完成以后系统应该具有什么可观察能力；不要在这里描述实现步骤。

## 2. Baseline and Evidence

### 2.1 Baseline

- Commit：`<与 front matter 一致的完整 commit hash>`
- Worktree：`clean | dirty`
- 与本 change 有关的未提交文件：`None | <逐项列出路径和影响>`
- 验证环境：`<Python、主要依赖、CPU/GPU、操作系统；不相关项可省略>`

`baseline_worktree: dirty` 时，所有会影响 Current Behavior 或接口判断的文件 MUST 写入
`baseline_dirty_paths`。无法取得稳定基线时，本规格 MUST 保持 `draft`。

### 2.2 Confirmed Facts

只记录从基线源码、测试或依赖文档核对出的事实。代码证据使用稳定符号：
`path/to/file.py::Class.method`；测试证据使用
`tests/path/test_file.py::test_name`。不得使用“之前实现的”“现有逻辑”“我们讨论过的方案”。

| Fact ID | Confirmed fact | Evidence | Verification method |
|---|---|---|---|
| FACT-001 | <当前事实> | `path/to/file.py::symbol` | `<静态阅读或命令>` |

### 2.3 Uncertainty Boundary

- 已确认的不确定性：`None | ...`
- 不能从当前证据推出的结论：`None | ...`

不得把本节内容写成当前能力。影响实现选择的问题必须移入 Blocking Open Questions。

### 2.4 External and Archive References

外部仓库和 `00_PAST/` 只能作为参考，不能作为 Current Behavior 的证据。迁移任务必须说明取舍：

| Reference | Role | Adopt | Explicitly reject | Reason |
|---|---|---|---|---|
| `<只读路径/仓库/论文>` | `<算法/数值/测试参考>` | `<采用的语义>` | `<不复制的设计>` | `<与本项目 contract 的关系>` |

引用外部代码时还必须确认许可证、数值差异和本项目依赖方向；`00_PAST/` MUST 保持只读。

## 3. Current Behavior

只描述基线源码已经存在、并由第 2 节证据支持的行为：

1. `<当前输入如何进入系统>`；
2. `<当前状态和输出>`；
3. `<当前错误或限制>`。

涉及代码时 MUST 使用 `path/to/file.py::Class.method` 或 `path/to/file.py::function`。

## 4. Target Behavior

每项 requirement 只表达一个可验证行为。不得使用“尽量”“适当”“高性能”等无法验收的词，
除非同时给出测量方法和阈值。

### REQ-001

系统 MUST `<明确行为>`。

Rationale：`<为什么需要，不能只重复 requirement>`

### REQ-002

系统 MUST NOT `<明确禁止行为>`。

Rationale：`<原因>`

## 5. Scope

### 5.1 In Scope

- `<本 change 必须完成的内容>`

### 5.2 Out of Scope

- `<明确不做的相邻需求>`

### 5.3 Protected Areas

- `00_PAST/**`：MUST NOT 修改；
- `layout/**`、`geometry/**`：未经用户针对本 change 的明确批准 MUST NOT 修改；
- `<其他不得改变的路径、数值行为或用户产物>`。

Out of Scope 和 Protected Areas 不是“以后顺便实现”的列表。实现 AI 不得以重构、修 bug 或
兼容未来方法为由扩大范围。

## 6. Invariants

### INV-001

`<例如：一个 segment MUST 有且只有一个 owner。>`

Enforced by：`<构造函数、校验边界或测试>`

### INV-002

`<例如：halo/context MUST 是只读的。>`

Enforced by：`<位置>`

### INV-003

`<例如：本轮所有 tile MUST 读取同一个 d_current。>`

Enforced by：`<同步点>`

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `<module>` | `<唯一职责>` | `<明确禁止反向承担的职责>` |

### 7.2 Dependency Direction

允许：

```text
A -> B -> C
```

禁止：

```text
C -X-> A
```

项目默认依赖方向是：

```text
layout -> geometry -> opc.input -> opc.input.edge
opc.iteration.<method> -> opc.input / lithography / evaluation
main -> application modules
```

如果本 change 需要不同方向，MUST 在 Decisions 中明确批准理由。

### 7.3 Data Flow

```text
Input
  -> validation
  -> preprocessing
  -> persistent state
  -> batch computation
  -> synchronized update
  -> authoritative output
```

对每个箭头说明调用符号、输入、输出，以及是否跨 Python/KLayout、CPU/GPU 或文件边界。

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| 0 | `<...>` | `<...>` | `<...>` | `<前一阶段昂贵计算>` |

### 7.5 Planned Call Graph

使用最终计划中的真实路径和符号，不能用 `manager()`、`handler()` 等占位名：

```text
main/run_x.py::main()
└─ application.workflow::run(config)
   ├─ input.prepare(...)
   ├─ method.solve(...)
   │  ├─ lithography.forward(...)
   │  └─ evaluation.metric(...)
   └─ output.write(...)
```

调用图必须标出 batch/iteration/macro 循环、同步点、持久化边界和昂贵计算发生次数。

## 8. Data Contracts

对每个重要数据结构定义 owner、lifetime、mutability、dtype、shape、unit 和 coordinate system。

### `<DataType>`

- Owner：`<模块、对象或文件>`
- Lifetime：`<batch / iteration / prepared problem / process / persisted>`
- Mutability：`immutable | mutable by <symbol>`
- Resident location：`CPU | GPU | disk | transient transfer`
- Coordinate system：`<全局 DBU、局部像素、数组 [y,x] 等>`

| Field | dtype | shape | unit | mutability | meaning |
|---|---|---|---|---|---|
| `<field>` | `<dtype>` | `<shape>` | `<unit>` | `<规则>` | `<唯一含义>` |

必须明确数组长度符号，例如 `S=segment_count`、`T=tile_count`，以及空数组是否合法。

### 8.1 Configuration Contract

| Key | Type | Unit | Required | Default | Validation | Consumer |
|---|---|---|---|---|---|---|
| `<section.key>` | `<type>` | `<unit>` | yes/no | `<value/None>` | `<约束与异常>` | `path.py::symbol` |

没有默认值时写 `None`，不得让实现 AI自行选择默认参数。互斥字段和跨字段约束必须写明。

### 8.2 Persisted Artifact Contract

| Artifact | Producer | Consumer | Format/version | Atomicity | Required content |
|---|---|---|---|---|---|
| `<path pattern>` | `path.py::symbol` | `<symbol/user>` | `<NPZ/GDS/JSON...>` | `<规则>` | `<字段/层/cell>` |

若有 NPZ/JSON/GDS 等输出，必须定义字段、dtype、shape、层/cell、路径命名、覆盖策略和
`allow_pickle` 等安全约束。没有持久化产物时写明 `Not applicable`。

## 9. Interface Changes

### IF-001

Current：

```python
foo(a: X) -> Y
```

Evidence：`path/to/file.py::foo`

Target：

```python
foo(a: X, b: Z) -> Y
```

- Behavior：`<输入输出语义>`
- Caller migration：`<所有调用点及改法>`
- Exceptions：`<条件 -> 精确异常类型与消息要求>`
- Compatibility：`preserved | intentionally broken；原因见 COMP-xxx`

新增接口的每个参数必须有当前调用方；删除或移动接口必须列出所有调用点。

## 10. Algorithm

伪代码必须明确初始化、正常路径、边界条件、状态更新时机、同步点和 failure behavior。

```text
initialize immutable reference state
initialize current state

for iteration:
    for batch:
        read current only
        compute temporary result
        accumulate metrics
    synchronize
    validate proposed next state
    publish next state only after validation succeeds

write authoritative output
```

### 10.1 Boundary Conditions

| Condition | Required behavior | Requirement |
|---|---|---|
| `<空输入/跨边界/最后不足 batch...>` | `<行为>` | `REQ-xxx` |

### 10.2 State Transition

```text
S0 --evaluate--> metrics(S0) + proposal(S1)
S1 --evaluate--> metrics(S1) + proposal(S2)
```

指标属于哪个已评价状态、提案何时成为已发布状态，MUST 明确，二者不得混写。

## 11. Ownership and State

| State/data | Owner | Writers | Readers | Publish point | Lifetime |
|---|---|---|---|---|---|
| `<state>` | `<owner>` | `<唯一写者>` | `<只读方>` | `<同步/校验后>` | `<范围>` |

必须明确：

- 谁拥有数据，谁可以修改，谁只能读取；
- 哪些状态跨 iteration，哪些只是 batch 临时量；
- macro/core/tile/segment 的 ownership 边界；
- context/halo 是否只读；
- 失败时保留哪个最后合法状态；
- 单机顺序、并行或未来分布式执行是否改变语义。

## 12. Error Handling

### ERR-001

- Condition：`<精确触发条件>`
- Detection boundary：`path/to/file.py::symbol`
- Behavior：`raise <ExceptionType>(<消息必须包含的信息>)`
- State after failure：`<未发布/保留最后合法状态/无输出>`
- MUST NOT：`静默 fallback / ignore / retry / 写半份权威输出`

仅把属于算法正常控制流的失败转换成领域状态；I/O、依赖库和未知异常 MUST 原样传播或携带
明确上下文传播，不得用宽泛 `except` 降级。

## 13. Performance and Memory Constraints

### PERF-001

`<可测量的性能或内存 contract>`

- Measurement workload：`<固定输入、规模和配置>`
- Measurement command：`<命令>`
- Pass threshold：`<数值；若本 change 只要求记录基线，明确写“记录，不设阈值”>`

必须明确：

- 主要路径的时间复杂度和随 `S/T/B/H/W` 的增长关系；
- CPU/GPU resident state、峰值内存上界及 cache lifetime；
- 是否允许全局 materialization；
- 哪些数据必须流式处理或 batch 后立即释放；
- 禁止逐 polygon/edge/segment 的 Python/KLayout hot loop；
- 重复迭代中哪些昂贵结果必须缓存，哪些不得跨 iteration 保存；
- 性能统计位置、单位和报告产物。

无法在设计阶段给出可靠阈值时，MUST 把它写成待测基线，不能声称“已提速”。

## 14. File-Level Change Plan

只列本 change 必须新增、修改、移动或删除的文件。文件外的改动必须先修订规格。

| File / Symbol | File type | Action | Contract change | Reason / Requirement |
|---|---|---|---|---|
| `opc/...py::foo` | 业务代码 | modify | `<签名/行为>` | `REQ-001` |
| `tests/...py::test_x` | 测试代码 | add | `<新增回归>` | `TEST-001` |
| `doc/...md` | 手册/报告 | modify | `<同步内容>` | `<交付要求>` |

Action 只能是 `add`、`modify`、`move` 或 `delete`。移动文件必须同时列出旧路径、新路径和 import
迁移；删除文件必须证明调用点为零。

## 15. Test Specification

每个 MUST/MUST NOT、invariant、接口异常、关键边界和性能 contract 都必须至少被一个测试或
明确的静态检查覆盖。测试必须写明层级，不得只列场景名称。

### TEST-001

- Level：`unit | integration | end-to-end | performance | static audit`
- File/function：`tests/path/test_file.py::test_name`
- Given：`<输入、fixture、设备、配置>`
- When：`<调用或命令>`
- Then：`<精确可断言结果>`
- Artifacts：`None | <临时 PNG/GDS/JSON/报告路径>`
- Covers：`REQ-001, INV-002, ERR-001`

### 15.1 Required Test Matrix

| Dimension | Cases | Expected distinction |
|---|---|---|
| Geometry | `<矩形/hole/斜边/跨边界...>` | `<各自断言>` |
| Scale | `<最小/典型/压力>` | `<内存与行为>` |
| Device | `<CPU/CUDA>` | `<一致性或容差>` |
| Failure | `<非法输入/非法候选/I/O>` | `<异常和状态>` |

### 15.2 Verification Commands

```powershell
<精确 pytest 命令>
<精确 ruff 命令>
<精确 compileall 命令>
<直接运行 main 的 smoke 命令>
```

硬件或资产缺失时允许 skip 的测试必须逐项列出 skip 条件；核心 CPU contract 不得用环境原因
整体跳过。

## 16. Requirement Traceability

| Requirement / Invariant | Implementation symbol | Tests | Acceptance criterion |
|---|---|---|---|
| `REQ-001` | `path.py::symbol` | `TEST-001` | `AC-001` |
| `INV-001` | `path.py::symbol` | `TEST-002` | `AC-002` |

不得存在没有实现符号或验证证据的 MUST/MUST NOT；纯文档 requirement 的 Implementation symbol
可以写对应文档路径。

## 17. Acceptance Criteria

- [ ] **AC-001**：`<执行哪条命令，在什么输入上，得到什么精确结果>`；
- [ ] **AC-002**：`<产物存在且字段/几何/数值满足什么检查>`；
- [ ] **AC-003**：`<性能或峰值内存通过指定测量，或已如实记录无阈值基线>`；
- [ ] **AC-004**：`<全量回归、静态检查和直接 Python 入口通过>`；
- [ ] **AC-005**：`<开发报告、测试报告、手册和项目记录已同步>`。

所有 AC 必须可客观判断 pass/fail，且能追溯到 requirement/test。不得把“代码质量良好”作为
独立 AC；应改成可执行的未调用函数、重复实现、异常入口和差异审计。

## 18. Compatibility and Migration

### COMP-001

- API compatibility：`preserved | broken；范围和调用方迁移`
- Data compatibility：`preserved | versioned migration | not required`
- Archive compatibility：`00_PAST read-only；是否读取旧产物`
- CLI compatibility：`命令、参数、默认值和退出码变化`
- Numerical compatibility：`逐值相同 | 容差 | 有意改变及依据`

如果不要求任何向后兼容，明确写：`No backward compatibility required.`

## 19. Decisions

### DEC-001

- Decision：`<已经确定、实现不得改动的选择>`
- Reason：`<证据、性能或架构原因>`
- Rejected alternatives：`<替代方案及拒绝理由>`
- Consequences：`<明确代价和已知限制>`

不能把仍需用户决定的问题写进 Decisions。

## 20. Open Questions

### 20.1 Blocking

None.

只要本节不是 `None`，文档 MUST 保持 `draft`，实现不得开始。每个问题必须写清需要谁决定、
可选项，以及各选项会改变哪些 requirement/interface/file。

### 20.2 Non-blocking

- `None | <不影响当前实现的研究问题或明确在 Out of Scope 的未来问题>`

如果某个问题会改变本次代码路径、默认值、接口、数据格式或验收结果，它就不是 Non-blocking。

## 21. Implementation Freedom

实现 AI 可以自行决定：

- 局部变量名；
- 由多个当前调用点共享、或具有独立领域含义的私有 helper 的具体组织；
- 不影响 public contract、数据 ownership、性能约束和测试观察结果的内部实现。

实现 AI 不得自行决定：

- 产品行为和算法语义；
- 数据 ownership、lifetime、mutability 或持久化格式；
- public API、配置默认值和 CLI；
- coordinate convention、单位、极性和 rounding；
- synchronization、publish、rollback 和 best-state 语义；
- architecture dependency；
- File-Level Change Plan 之外的重构或兼容层。

只有一个调用点且没有独立领域含义的逻辑 SHOULD 留在调用函数内，避免一次性抽象。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Required verification | Suggested local commit |
|---|---|---|---|---|
| A | `<最小可独立验证目标>` | `<路径>` | `<命令>` | `<type(scope): message>` |

每阶段只在验证通过后进行本地 commit。未经用户明确授权 MUST NOT 推送远端。用户 GDS、图片和
无关工作树修改 MUST 保留并排除在提交之外。

## 23. Delivery and Final Audit

实现完成后 MUST：

- 更新开发手册、测试手册、专项开发报告和专项测试报告；
- 同步 `task_plan.md`、`findings.md` 和 `progress.md`；
- 记录实际文件与本规格偏差；任何 contract 偏差必须先获批准，不能在报告中事后合理化；
- 记录测试环境、命令、通过/失败/skip、耗时、RSS/CUDA peak 和产物路径；
- 搜索调用点，删除仅服务于旧错误的 helper、包装层、分支和变量；
- 审计未调用函数、重复实现、异常吞噬、一次性抽象、过度文件拆分和无需求字段；
- 检查所有第一方 Python 文件、函数、方法和测试函数的中文 docstring，以及关键逻辑中文注释；
- 提供 `git diff --check`、目标测试、全量测试、ruff、compileall 和直接 main smoke 的结果；
- 说明是否修改 `layout/`、`geometry/`、`00_PAST/` 或用户数据，正常答案应为否；
- 列出本地 commit，明确未推送远端。

## 24. Known Limitations and Future Work

- `<已经确认但本 change 不解决的限制；不得描述为当前能力>`
- `<未来优化方向；不得为它预留无当前调用方的空接口或字段>`

## 25. Specification Approval Gate

只有以下各项全部满足，用户才应把 `status` 从 `draft` 改为 `approved`：

- [ ] front matter 没有占位符，`baseline_commit` 和相关脏文件已核对；
- [ ] Current Behavior 的每项关键事实都有源码、测试或依赖文档证据；
- [ ] Target Behavior 没有“尽量”“适当”等不可验收要求；
- [ ] Blocking Open Questions 为 `None`；
- [ ] public API、配置默认值、数据格式、坐标、ownership 和同步语义均已确定；
- [ ] 每个 MUST/MUST NOT 和 invariant 都映射到实现符号、测试与 AC；
- [ ] File-Level Change Plan 覆盖全部必要代码、测试、配置、手册和报告，且没有无需求文件；
- [ ] 性能/内存要求有测量工作负载与命令，无法设阈值的项目已明确只记录基线；
- [ ] Out of Scope、Protected Areas、已知限制和参考项目取舍均已写清；
- [ ] 实施阶段可独立验证，提交边界不会包含用户数据或无关工作树修改。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | `<YYYY-MM-DD>` | draft | 初始规格 | `<待审核>` |
