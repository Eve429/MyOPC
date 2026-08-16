---
id: CHG-20260816-simple-mbopc
title: 最简 MB-OPC（固定步长 EPE 驱动离散边移动）
type: implementation-spec
status: completed
---

# 最简 MB-OPC

> 原始设计：`doc_/archive/reports/mbopc_migration_design.md`（用户
> 2026-08-16 批准，1151 行自含规格）。

## Objective

从 GDS 直接可运行的固定步长、EPE 驱动离散边移动求解器 + evaluation
最小子集 + LithographyModel 契约 + points_to_canvas + 共享宏管线生命周期
+ 单/多 macro 两入口；每个 macro 独立完成全部迭代，最终一次合并。

## 关键需求（摘要）

- REQ EPE 方向表与探针有效性四条件（evaluation/metrics.py，threshold
  0.499 保留）；
- REQ 探针坐标必须经 points_to_canvas（居中 padding + float64）；
- REQ 同步 Jacobi：同轮只读同一 current，方向写 next，owner 恰写一次；
- REQ Round N 指标属于第 N 次位移后状态；EPE 选 best、L2/PVBand 只诊断；
- REQ 独立 macro + 恰一次显式映射 merge；差异量化不宣称全局同步；
- REQ target uint8 有界 LRU；GPU 批后释放；±max_displacement 裁剪；
- REQ 候选非法终止该 macro 保留 best，原因不吞；
- REQ 明确不做：梯度/DiffOPC/ILT、SRAF、逐轮全局合并、注册器/工厂。

## 批准与实施

用户批准后六批实施（A evaluation+契约 → B 坐标 → C 共享生命周期重构 →
D 求解器 → E 两入口 → F 验证报告）；后续审查修复轮见
CHG-20260816-mbopc-review-fixes。
