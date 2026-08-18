# Migration Map — 旧文件迁移清单

依据重构计划 §7.14：`Old File → New Location → Action → Notes`。
试行期全部为**复制**，原文件未移动未删除；正式切换时按本表执行归档/删除。

## 手册与计划

| Old File | New Location | Action | Notes |
|---|---|---|---|
| `doc/development_manual.md` | `doc_/architecture/` + `doc_/contracts/` | 拆分迁移 → 原件复制至 `archive/manuals/` | 有效内容拆入 system/dataflow/契约；环境命令类内容由 AGENTS.md（切换时）承接 |
| `doc/test_manual.md` | 各 completed CHG 的 test_report + `contracts/` | 拆分迁移 → 原件复制至 `archive/manuals/` | 套件职责表由 INDEX 导航替代；smoke 命令进各 CHG 验证记录 |
| `doc/MyOPC AI开发文档体系重构计划.md` | `doc_/`（本体系） | 已执行 → 原件复制至 `archive/manuals/` | 本清单即其 §7.14 交付物 |
| `doc/templates/implementation_spec_template.md` | `doc_/implementation_spec_template.md` | 复制复用 | 未跟踪用户文件，原件不动 |

## 设计与报告（→ changes/）

| Old File | New Location | Action | Notes |
|---|---|---|---|
| `doc/macro_core/macro_core_pipeline_design.md` | `changes/completed/CHG-20260815-macro-core-pipeline/implementation_spec.md` | 整理迁移 → 原件复制至 `archive/reports/` | 按 approved spec 收录 |
| `doc/macro_core/macro_core_pipeline_development_report.md` | 同上 `development_report.md` | 整理迁移 → 归档 | 事实内容对齐当前源码 |
| `doc/macro_core/macro_core_pipeline_test_report.md` | 同上 `test_report.md` | 整理迁移 → 归档 | 同上 |
| `doc/macro_core/macro_core_pipeline_review_issues.md` | 同上 `review_issues.md` | 复制收录 | 审查清单是该 CHG 的一部分 |
| `doc/single_pass_bias_design.md` | `changes/completed/CHG-20260815-single-pass-bias/implementation_spec.md` | 整理迁移 → 归档 | 该批无正式报告，两报告从源码/测试/task_plan 记录整理，缺失处如实标注 |
| `doc/lithography/lithography_migration_design.md` | `changes/completed/CHG-20260816-lithography-iccad13/implementation_spec.md` | 整理迁移 → 归档 | — |
| `doc/lithography/development_report.md` | 同上 `development_report.md` | 整理迁移 → 归档 | 原文件名无前缀，归档副本已加 lithography_ 前缀 |
| `doc/lithography/test_report.md` | 同上 `test_report.md` | 整理迁移 → 归档 | 同上 |
| `doc/opc/mbopc_migration_design.md` | `changes/completed/CHG-20260816-simple-mbopc/implementation_spec.md` | 整理迁移 → 归档 | — |
| `doc/opc/mbopc_development_report.md` | 同上 `development_report.md` | 整理迁移 → 归档 | 含审查修复轮 |
| `doc/opc/mbopc_test_report.md` | 同上 `test_report.md` | 整理迁移 → 归档 | 同上 |
| 审查结论（用户 `.planning/lithography_mbopc_review/findings.md`） | `changes/completed/CHG-20260816-mbopc-review-fixes/implementation_spec.md` | 整理为新 spec | 审查项即需求；两报告从修复提交 3725c0e/acfcab0/e289f2c 记录整理 |
| `doc/opc/gradient_mbopc_migration_design.md` | `changes/active/CHG-20260816-gradient-mbopc/implementation_spec.md` | 复制（未跟踪用户文件） | status: draft；原件不动 |

## 过程记录（→ archive/worklogs/）

| Old File | New Location | Action | Notes |
|---|---|---|---|
| `task_plan.md` | `doc_/archive/worklogs/task_plan.md` | 复制 | layout/geometry 手迁批次的过程记录保留于此，不造 CHG（无正式 spec） |
| `findings.md` | `doc_/archive/worklogs/findings.md` | 复制 | 架构/接口结论已蒸馏入 architecture/contracts/adr |
| `progress.md` | `doc_/archive/worklogs/progress.md` | 复制 | 会话日志 |

## 切换阶段遗留（本次不做）

- `CLAUDE.md` 瘦身为纯入口（不保存架构/API/状态）；
- `AGENTS.md` 按重构计划 §3 重写（长期规则，剥离项目状态）；
- 原 `doc/` 按本表删除已迁移文件；`doc_` 更名/合并为 `docs/`；
- 根目录规划三文件的日常更新职责移交 changes 体系。

## 切换执行记录（2026-08-18）

doc_/ 试行版正式切换为 doc/：12 个 doc/ 旧文件（archive 已有同源副本）
删除；8 个增量迁移——两手册至根（活跃位）、gradient design Revision
0.2 归位 completed CHG（active 清空）、gradient 两报告原件入
archive/reports、mbopc design 用户新版覆盖 archive 副本、config_refactor
两报告入新 CHG-20260818-config-system-refactor（附摘要版 spec）。
INDEX 撤试行说明；CLAUDE/AGENTS/task_plan 路径引用同步。
