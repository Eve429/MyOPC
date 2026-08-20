# INDEX — AI 文档总入口

> 本目录（`doc/`）是项目文档体系的正式实例（2026-08-18 由 `doc_/` 试行
> 版切换而来，依据 `archive/manuals/MyOPC AI开发文档体系重构计划.md`
> 构建；切换记录见 `migration_map.md` 末节）。根目录 `AGENTS.md` 与
> `CLAUDE.md` 是工作规则入口，与本目录互补。

## 按任务读取

| 任务 | 必读 | 顺带参考 |
|---|---|---|
| 设计新 Change（AI A） | `AGENTS.md` → 本文件 → 相关 `contracts/*.md` → 相关 `architecture/*.md` → 当前源码与 tests → `implementation_spec_template.md` | 相关 `adr/*.md`（决策背景） |
| 实现 Change（AI B） | `AGENTS.md` → 本文件 → 对应 `changes/active/CHG-xxx/implementation_spec.md` → spec 引用的 contracts/architecture → 相关源码与 tests | — |
| 了解术语 | `glossary.md` | — |
| 了解系统现状 | `architecture/system.md` → `architecture/dataflow/index.md` | `architecture/data_model.md` |
| 查某模块接口保证 | 对应 `contracts/<module>.md` | — |
| 查开发/测试操作 | `development_manual.md`（根） → `test_manual.md`（根） | — |
| 追溯某次变更 | `changes/completed/CHG-xxx/`（spec + 两报告） | 对应版本 git 记录 |
| 理解关键决策原因 | `adr/` | — |

## 事实源规则（摘要）

```text
AGENTS.md                              → AI 应该如何工作
doc/architecture/                      → 当前系统如何组织
doc/contracts/                         → 当前系统对外保证什么
doc/development_manual.md / test_manual.md     → 开发/测试操作手册
doc/changes/active/*/implementation_spec.md    → 当前准备修改成什么
doc/changes/completed/*/development_report.md  → 某次 Change 实际实现了什么
doc/changes/completed/*/test_report.md         → 某次 Change 实际验证结果
```

- `architecture/`、`contracts/` 只写当前已存在的事实，不写未来方案；
- `implementation_spec.md`（目标）、`development_report.md`（实施事实）、
  `test_report.md`（验证证据）三者不得混用；
- Change 完成后必须同步更新受影响的 `architecture/` 与 `contracts/`；
- **不得递归遍历本目录**，不得从 `archive/` 寻找当前需求，不得依赖生成
  文档时的聊天上下文。

## 禁止作为当前事实源的目录

| 目录 | 用途 | 禁止事项 |
|---|---|---|
| `archive/` | 旧体系历史资料 | 不得当作当前需求、接口或架构 |
| `changes/completed/` | 变更追溯 | 不得覆盖当前源码、contracts 或 architecture |

## 冲突优先级

目标行为：用户当前指令 > approved spec > AGENTS.md > contracts > architecture。
当前已实现行为：源码 > 测试 > contracts > architecture。
无法通过优先级解决的冲突必须明确记录，不得由实现 AI 猜测。

## 目录一览

```text
INDEX.md / glossary.md / implementation_spec_template.md
development_manual.md · test_manual.md   （活跃手册）
architecture/  system.md · dataflow/（index + 四工作流文件） · data_model.md
contracts/     layout · geometry · opc_input · edge · lithography · mbopc · ilt · evaluation
adr/           ADR-001 .. ADR-006
changes/       completed/（active/ 于下一个进行中 CHG 时自建）
archive/       reports/ · worklogs/ · manuals/
migration_map.md   旧文件 → 新位置迁移清单
```
