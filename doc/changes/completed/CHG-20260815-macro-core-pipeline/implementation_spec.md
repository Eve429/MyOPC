---
id: CHG-20260815-macro-core-pipeline
title: Macro–Core 两级网格管线重构
type: implementation-spec
status: completed
---

# Macro–Core 两级网格管线重构

> 本文件是新体系下的整理版规格（事实来自原始设计与实施记录）；完整原始
> 文本见 `doc_/archive/reports/macro_core_pipeline_design.md` 与
> `review_issues.md`（同目录 `development_report.md`/`test_report.md` 为
> 新整理的实施/验证记录）。

## Objective

把 opc.input 重构为 Macro–Core 两级网格 + 持久化 MacroProblem + 双轮 ±2nm
验证管线 + ownership 权威覆盖最终合并，取代旧 MBOPCProblem/MacroPreparation
重复结构。

## 关键需求（摘要）

- REQ 两级网格：macro ownership 半开不重叠、面积和 == 层 bbox；macro 尺寸
  严格大于 core；context ≥ max_displacement；canvas 容量预检。
- REQ 持久化问题：每 macro 一次构造（提边/分段/切线分裂/owner/CSR），
  NPZ format v1，全部成功后才写 plan.json。
- REQ 双轮验证：±2nm 冻结值、读旧写新、owner 恰写一次守卫、逐 core
  transmission sum、第二轮回零 XOR == 0。
- REQ 最终权威覆盖：各 macro 候选只贡献自身 ownership，single_cell /
  macro_cells 双模式物理覆盖一致，回读面积守恒验证。
- REQ 审查轮（review_issues.md 逐项）：契约冻结、空 membership 不变量、
  复杂几何矩阵、正逆序双轮对照、未处理层对照、coverage 审计。

## Out of Scope（当时明确不做）

全局同层几何合并/规范化（tile seam 碎片治理，记 AGENTS 未来优化）；
ILT/像素级方法。

## 批准与实施

用户批准 `doc/macro_core/macro_core_pipeline_design.md` 后实施五批提交
+ 审查轮；详见同目录 development_report.md。
