# MyOPC AI开发文档体系重构计划

## 1. 目标

重构 MyOPC 当前开发文档体系，使其适合 AI Agent 跨会话、跨模型协作开发。一个可能的流程为：**AI A 分析代码并编写 ****`implementation_spec.md`**** → AI B 不依赖历史聊天，根据正式文档完成实现和测试 → AI B 输出 ****`development_report.md`**** 和 ****`test_report.md`**** → 验证通过后归档为 Completed Change。**此外，目录也可以辅助AI快速了解项目；

本次重构只调整文档目录、文档职责、事实源和 AI 读取规则，不修改现有业务代码和算法。核心要求：

- 当前系统事实必须有明确且唯一的权威来源。
- 当前架构、当前接口、待实现变更、实施结果、测试结果、历史资料必须分离。
- 历史文档不得被 AI 当作当前需求或当前架构。
- 所有 AI 必须通过统一入口读取文档，不允许自行遍历文档后判断哪些文件有效。
- 新功能设计统一通过 `implementation_spec_template.md` 生成 `implementation_spec.md`，本计划不定义其内部细节。
- AI B 完成实现后必须生成 `development_report.md` 和 `test_report.md`。
- 所有正式文档必须独立于聊天上下文，可以直接交给其他 AI 使用。

## 2. 最终目录结构

```text
MyOPC/
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   ├── INDEX.md
│   ├── glossary.md
│   ├── implementation_spec_template.md
│   ├── architecture/
│   │   ├── system.md
│   │   ├── dataflow.md
│   │   └── data_model.md
│   ├── contracts/
│   │   ├── layout.md
│   │   ├── geometry.md
│   │   ├── opc_input.md
│   │   ├── edge.md
│   │   ├── lithography.md
│   │   ├── mbopc.md
│   │   ├── ilt.md
│   │   └── evaluation.md
│   ├── adr/
│   │   ├── ADR-001-xxx.md
│   │   └── ADR-002-xxx.md
│   ├── changes/
│   │   ├── active/
│   │   │   └── CHG-xxx-feature-name/
│   │   │       └── implementation_spec.md
│   │   └── completed/
│   │       └── CHG-xxx-feature-name/
│   │           ├── implementation_spec.md
│   │           ├── development_report.md
│   │           └── test_report.md
│   └── archive/
│       ├── reports/
│       ├── worklogs/
│       └── manuals/
├── layout/
├── geometry/
├── opc/
├── lithography/
├── evaluation/
└── tests/
```

现有文档按以下原则迁移：

- `development_manual.md`：有效内容拆入 `architecture/` 和 `contracts/`，原文件迁入 `archive/manuals/`。
- `module_interface_reference.md`：按模块拆入 `contracts/`，原文件迁入 `archive/manuals/`。
- `function_call_architecture.md`：系统级调用关系迁入 `architecture/dataflow.md`，接口内容迁入对应 `contracts/`，原文件归档。
- `current_architecture_review.md`：已确认结论迁入 `architecture/` 或 `adr/`，原文件迁入 `archive/reports/`。
- `项目开发手册.md`：迁入 `archive/manuals/`。
- `task_plan.md`：仍有效任务转为 `changes/active/CHG-xxx/implementation_spec.md`，历史内容迁入 `archive/worklogs/`。
- `findings.md`：有效架构结论迁入 `architecture/`，接口结论迁入 `contracts/`，关键决策迁入 `adr/`，其余迁入 `archive/worklogs/`。
- `progress.md`：迁入 `archive/worklogs/`。
- 旧体系中的 `*_development_report.md`、`*_test_report.md`：迁入 `archive/reports/`；新体系下每个 Change 的正式开发报告和测试报告保存在对应 `changes/completed/CHG-xxx/` 中。
- 尚未完成的设计方案：转为 `changes/active/CHG-xxx/implementation_spec.md`；已经完成且能够确认实施和测试结果的变更整理到 `changes/completed/`。

## 3. 各目录职责

| 文件/目录                                  | 职责                                                                                                   |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `AGENTS.md`                            | 所有 AI 必须遵守的长期开发规则，包括文档读取规则、模块依赖、代码修改原则、测试要求、错误处理、性能原则和禁止行为。不得保存具体功能设计、项目进度或历史信息。                     |
| `CLAUDE.md`                            | Claude Code 入口适配文件，只负责引导读取 `AGENTS.md`、`docs/INDEX.md` 和当前任务，不保存架构、API、算法或项目状态。                      |
| `docs/INDEX.md`                        | AI 文档总入口和导航表，说明不同任务应该读取哪些文档以及哪些目录不能作为当前事实源。                                                          |
| `docs/glossary.md`                     | 项目统一术语定义，Macro、Tile、Core、Halo、Owner、Segment、Round、Iteration 等术语只定义一次。                                |
| `docs/implementation_spec_template.md` | AI A 创建新 Change 时必须使用的统一实现规格模板。                                                                      |
| `docs/architecture/`                   | 当前系统架构，只描述系统现在如何组织，不保存未来方案。                                                                          |
| `architecture/system.md`               | 当前模块划分、模块职责、模块边界、依赖方向和禁止依赖。                                                                          |
| `architecture/dataflow.md`             | 当前程序端到端数据流及 MB-OPC、ILT 等主要流程的数据流。                                                                    |
| `architecture/data_model.md`           | 跨模块核心数据对象、生命周期、ownership、mutability、引用关系和 CPU/GPU 所在位置。                                              |
| `docs/contracts/`                      | 当前模块稳定契约，包括接口、数据结构、输入输出、单位、坐标、ownership、异常和 invariant。                                               |
| `contracts/layout.md`                  | GDS/OASIS、cell hierarchy、layer、DBU、区域查询等 Layout 契约。                                                  |
| `contracts/geometry.md`                | Polygon、Region、Box、Boolean、clip、merge、transform、空间索引等 Geometry 契约。                                   |
| `contracts/opc_input.md`               | Macro、Tile、Core、Halo、simulation region、update region、rasterization input 等 OPC 输入契约。                 |
| `contracts/edge.md`                    | Segment、边段切分、owner、movable/read-only、reference geometry、displacement、reconstruction 等契约。             |
| `contracts/lithography.md`             | 光刻模型输入输出、tensor、pixel size、kernel、dose、focus、dtype、device、gradient 等契约。                              |
| `contracts/mbopc.md`                   | MB-OPC problem/state、iteration、round、tile execution、barrier、displacement、convergence 等契约。            |
| `contracts/ilt.md`                     | ILT 输入、mask parameterization、loss、gradient、optimization state 和输出契约。                                 |
| `contracts/evaluation.md`              | EPE、PVBand、L2、采样规则和评价接口契约。                                                                           |
| `docs/adr/`                            | 已确定的重要架构决策及原因。架构决策改变时新增 ADR，不直接覆盖历史 ADR。                                                             |
| `docs/changes/active/`                 | 当前尚未完成的开发任务，每个 Change 独立目录，开发前只保存正式 `implementation_spec.md`。                                        |
| `docs/changes/completed/`              | 已完成 Change 的正式交付记录，每个 Change 至少保存 `implementation_spec.md`、`development_report.md`、`test_report.md`。 |
| `development_report.md`                | AI B 完成开发后记录“实际实现了什么”，包括实际修改文件和 symbol、需求实现位置、关键实现说明、与 Spec 的偏差、未完成项、architecture/contracts 更新情况。    |
| `test_report.md`                       | AI B 完成测试后记录“实际验证结果”，包括测试环境、执行测试、测试结果、Acceptance Criteria、失败项、未验证项和最终测试结论。                           |
| `docs/archive/`                        | 旧文档体系和历史资料，AI 默认不得将其内容视为当前需求、接口或架构。                                                                  |
| `archive/reports/`                     | 旧体系遗留的开发报告、测试报告、架构评审等历史资料。                                                                           |
| `archive/worklogs/`                    | `progress.md`、`findings.md`、`task_plan.md` 等过程性记录。                                                   |
| `archive/manuals/`                     | 已废弃或被新体系替代的开发手册和接口手册。                                                                                |

Change 生命周期固定为：

```text
用户需求
→ AI A
→ changes/active/CHG-xxx/implementation_spec.md
→ AI B 实现
→ AI B 测试
→ 更新必要的 architecture/contracts
→ development_report.md
→ test_report.md
→ 验证通过
→ 整个 CHG-xxx 从 active/ 移入 completed/
```

## 4. 文档事实源规则

当前事实和变更事实只能存在于以下位置：

```text
AGENTS.md
→ AI 应该如何工作

docs/architecture/
→ 当前系统如何组织

docs/contracts/
→ 当前系统现在对外保证什么

docs/changes/active/*/implementation_spec.md
→ 当前准备修改成什么

docs/changes/completed/*/development_report.md
→ 某次 Change 实际实现了什么

docs/changes/completed/*/test_report.md
→ 某次 Change 实际验证结果是什么
```

必须遵守以下规则：

- `architecture/` 只写当前已经存在的系统结构，不写未来设计。
- `contracts/` 只写当前已经成立的接口和数据契约，不写待实现方案。
- `changes/active/` 描述目标状态，不代表当前代码已经实现。
- `implementation_spec.md` 是实施依据，`development_report.md` 是实施事实，`test_report.md` 是验证证据，三者不得混用。
- Change 完成后，如果其实现改变了长期架构或契约，必须同步更新 `architecture/` 或 `contracts/`，不能要求后续 AI 通过读取 completed report 才知道当前系统。
- `changes/completed/` 用于变更追溯，不作为当前架构或当前 API 的主要事实源。
- `adr/` 只解释关键设计决策及原因。
- `archive/` 只保存旧体系历史信息，不得覆盖当前事实。
- `CLAUDE.md` 等 Agent 专用文件不得保存独立的项目事实。
- 同一事实不得在多个文档中独立维护，需要使用时引用其正式事实源。

## 5. AI读取文档顺序

AI A 设计新 Change 时：

```text
AGENTS.md
→ docs/INDEX.md
→ 相关 architecture
→ 相关 contracts
→ 当前源码
→ 当前 tests
→ implementation_spec_template.md
→ 创建 changes/active/CHG-xxx/implementation_spec.md
```

AI B 实现 Change 时：

```text
AGENTS.md
→ docs/INDEX.md
→ 当前 implementation_spec.md
→ implementation_spec 引用的 architecture
→ implementation_spec 引用的 contracts
→ 相关源码
→ 相关 tests
→ 实现代码
→ 执行测试
→ 更新必要的 architecture/contracts
→ 生成 development_report.md
→ 生成 test_report.md
```

额外规则：

- AI 不得默认递归读取整个 `docs/`。
- AI 不得自行从 `archive/` 中寻找当前需求。
- AI B 不得依赖 AI A 的聊天上下文。
- AI B 如果需要了解某个旧 Change 的实现原因，可以按需读取 `changes/completed/`，但不得用 completed report 覆盖当前源码、contracts 或 architecture。
- 新 Change 必须基于 `implementation_spec_template.md` 创建。
- 如果没有对应 active implementation spec，不得将旧 `plan`、`report`、`findings` 当作替代实现规格。

## 6. 文档优先级

对于“目标行为”，优先级：

```text
1. 用户当前明确指令
2. Approved active implementation_spec.md
3. AGENTS.md
4. docs/contracts/
5. docs/architecture/
```

对于“当前已经实现的行为”，优先级：

```text
1. 当前源码
2. 当前测试
3. docs/contracts/
4. docs/architecture/
```

对于“某次历史 Change 实际做了什么”，优先级：

```text
1. 对应版本源码/Git记录
2. development_report.md
3. implementation_spec.md
```

对于“某次历史 Change 测试结果”，优先级：

```text
1. test_report.md
2. 对应测试代码及可保存的测试结果
```

其他规则：

- `implementation_spec.md` 描述目标，不得用于证明功能已经实现。
- `development_report.md` 描述实现结果，不得反向修改原始需求定义。
- `test_report.md` 描述验证结果，不得代替当前 contracts。
- `archive/` 永远不能覆盖当前事实源。
- `CLAUDE.md` 等入口文件不得覆盖 `AGENTS.md`。
- Active Change 只允许覆盖其明确 scope 内的目标行为。
- 无法通过优先级解决的冲突必须明确记录，不得由实现 AI 自行猜测。

## 7. 验证

完成文档体系重构后必须验证：

1. 完全不知道历史聊天的 AI，只读取 `AGENTS.md`、`docs/INDEX.md`、一个 active `implementation_spec.md` 及其引用文档后，可以理解并实现任务。
2. AI 可以明确区分：当前系统是什么、当前接口是什么、本次准备修改什么、最终实际修改了什么、实际测试结果是什么。
3. 仓库中不存在两个文档同时作为 current architecture、current API 或 current implementation plan 的事实源。
4. `CLAUDE.md` 不保存可能过期的架构、模块状态或 API 信息。
5. `archive/` 不被 `INDEX.md` 作为正常开发输入引用。
6. 所有 `architecture/` 和 `contracts/` 内容与当前源码和测试一致。
7. 所有 active Change 位于独立 `CHG-xxx-*` 目录并使用 `implementation_spec_template.md`。
8. 已完成 Change 必须至少包含 `implementation_spec.md`、`development_report.md`、`test_report.md`。
9. `development_report.md` 必须能够追踪重要需求实际实现到哪些文件或 symbol。
10. `test_report.md` 必须能够追踪测试项目、测试结果和验收结果。
11. Change 完成后，其对长期架构和接口造成的变化已经同步更新到 `architecture/` 和 `contracts/`。
12. 所有内部 Markdown 路径有效，不存在指向失效旧目录的引用。
13. 原有重要文档中的有效信息完成迁移后才能归档。
14. 最终提供旧文件迁移清单：`Old File → New Location → Action → Notes`。

## 8. 限制

本次任务允许：

- 创建新的文档目录。
- 重构、拆分、迁移、合并和归档现有 Markdown。
- 根据当前源码和测试修正文档中的过期描述。
- 创建 `INDEX.md`、`glossary.md`、architecture、contracts、ADR、changes、archive 目录。
- 修复 Markdown 内部引用。
- 将旧体系报告迁入 `archive/reports/`。

本次任务禁止：

- 修改 OPC 算法。
- 修改业务代码。
- 修改现有公共 API。
- 重构源代码目录。
- 实现尚未完成的新功能。
- 因文档与代码不一致而擅自修改代码。
- 为填充文档自行发明不存在的接口、数据结构或算法行为。
- 删除尚未完成信息迁移的旧文档。
- 将旧 `*_development_report.md`、`*_test_report.md` 直接混入新 Change；新体系报告必须由对应 Change 的实际实施过程生成。

如果旧文档与源码冲突，应以当前源码和测试确认 Current Behavior 并修正文档；如果无法确定，则记录冲突，不修改业务实现。

## 9. 最终原则

```text
AGENTS.md
= How AI should work

docs/INDEX.md
= Where AI should read

docs/glossary.md
= What project terms mean

docs/architecture/
= How the current system is structured

docs/contracts/
= What the current system guarantees

docs/adr/
= Why important architecture decisions were made

changes/active/*/implementation_spec.md
= What is going to change

changes/completed/*/implementation_spec.md
= What the approved change specification was

changes/completed/*/development_report.md
= What was actually implemented

changes/completed/*/test_report.md
= What was actually verified

docs/archive/
= Historical information from the old documentation system

implementation_spec_template.md
= How AI A must describe a new change
```

一个完整 Change 必须形成：

```text
implementation_spec.md
        ↓
代码实现
        ↓
测试
        ↓
development_report.md
        +
test_report.md
        ↓
更新 architecture/contracts
        ↓
completed
```

任何文档只承担一个主要职责。以后禁止重新建立 `development_manual.md`、`global_findings.md`、`global_progress.md`、`global_task_plan.md` 等混合多种职责的全局文档。最终目标是：**任意 AI Agent 在没有历史聊天上下文的情况下，可以通过结构化 Markdown 准确理解当前系统和当前任务；任务完成后，又可以通过开发报告和测试报告准确追溯实际实现与验证结果。**
