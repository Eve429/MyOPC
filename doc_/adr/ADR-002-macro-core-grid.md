# ADR-002 — Macro–Core 两级网格与持久化 MacroProblem

- Date：2026-08-15（macro_core 管线重构）
- Status：accepted

## Decision

废弃旧「全局 core 网格反向组合 macro」与 MBOPCProblem/MacroPreparation
重复结构，改为两级规划：`plan_macros` 先切 ownership 半开不重叠的 macro，
macro 内再切 core；每 macro 一次构造完整参考输入并持久化为
`MacroProblem` NPZ（顶点/边段/owner/CSR/网格契约）。

## Reason

- 单一网格无法同时表达"独立可持久化的求解单元"（macro）与"唯一计分/
  可回写单元"（core）两种正交关切；
- 旧结构里 ownership、分段、缓存职责混杂，重建成本高。

## Rejected alternatives

- 保持单级 core 网格 + 全局迭代：跨 macro 同步开销与状态所有权不清。
- 以 polygon 为单元：数量动态、无稳定 ID，与位移迭代模型冲突。

## Consequences

- 契约冻结点：macro 尺寸严格大于 core；ownership 面积和恰等于层 bbox；
  own ⊆ membership；段中点归属即 owner。
- MacroProblem 是跨会话/跨进程的稳定求解输入（format v1，不含 dbu_um——
  GDS 写出函数由调用方传 DBU）。
- 切线分裂需处理斜边交点参数的整数一致性（两次真实 bug 均源于此）。
