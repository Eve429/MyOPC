---
id: CHG-20260816-mbopc-review-fixes
title: MB-OPC 审查问题修复（3 P1 + P2 组）
type: implementation-spec
status: completed
---

# MB-OPC 审查问题修复

> 规格 = 用户独立只读审查结论（2026-08-16，`.planning/lithography_mbopc_review/`）
> 逐条核实后的修复范围。审查结论 3 项 P1 全部属实、P2 七项属实、
> 两项有保留。

## 需求（审查项即需求，摘要）

- REQ-P1.1 无有效探针不得报告 zero_epe → 新增 insufficient_probes 停止
  状态（政策：不拒绝 Problem、不自适应距离）；
- REQ-P1.2 最终几何流式（save_final 逐 tile 窗口物化、merge 验证窗口化）
  + 五处 ±2^30 魔法框 → layer_bbox；
- REQ-P1.3 TOML 整数配置拒绝 1.5/true 静默截断（_as_int）；
- REQ-P2 组：参考几何整迭代物化一次、EPE 回切整 batch 化、无变化提案
  跳过重复评价、末轮纯评价、macro_grid 前置校验、tqdm finally、
  §16.3 缺失测试场景真构造补齐、差异断言加上界。

## 有依据不采纳

- except 收窄到 ReconstructionError：实测几何退化（共线 ring 少于三顶点）
  以 ValueError 从 KLayout 冒出；包装需改 reconstruction.py（当时不修改
  清单）。维持宽捕获并在代码注释记录证据。

## 明确不做

- ProcessCondition 绑定 focus/defocus：设计选择（ADR-004 已记录）；
- PatchWriter 流式化：geometry/ 用户领地（记为已知上界）。
