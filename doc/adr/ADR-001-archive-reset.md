# ADR-001 — 旧库整体归档与从零重建

- Date：2026-08-15
- Status：accepted

## Decision

旧 AI 代码库整体移入 `00_PAST/`（只读归档：源码、tests、doc、config、
.learnlearnings、规划文件），仓库根从零重建（分支 `migration`）；迁移按
依赖顺序逐模块进行，旧库不原样照搬而是过滤/重写。

## Reason

旧库为 AI 一次性生成，所有者对多数模块缺乏理解与信任；混合演进会让
未理解代码持续累积。归档+重建使每个迁入模块都被审查、理解并拥有。

## Rejected alternatives

- 在原仓库上继续演进：无法区分可信与不可信代码。
- 只归档部分：依赖链交织，无法切割。

## Consequences

- `00_PAST/` 只读纪律：复制出来改写允许，改归档须用户明示批准。
- 迁移期的规格书 = 旧 tests + 旧 doc（对照移植）；过程记录曾集中在
  task_plan/findings/progress（已随文档体系重构迁入 archive/worklogs）。
- 旧库规模 ~11k 生产代码 + 4k 测试，逐模块成本高但可控。
