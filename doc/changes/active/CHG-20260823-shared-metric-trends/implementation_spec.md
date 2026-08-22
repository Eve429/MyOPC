---
id: CHG-20260823-shared-metric-trends
title: 公共迭代指标趋势图组件
type: implementation-spec
status: completed
baseline_commit: working-tree
scope:
  - common/metric_trends
  - main/mbopc-runners
depends_on:
  - docs/contracts/mbopc.md
supersedes: []
---

# 公共迭代指标趋势图组件

## 目标

将 Simple MB-OPC 的趋势图逻辑移动到算法无关的 `common.metric_trends`，供
Simple/Gradient MB-OPC 立即使用，并为后续 ILT 提供直接调用接口。

## 接口

`save_metric_trends(metrics_files, output_dir, metric_fields, *, best_state_indices=None, overview_mode="mean")`

组件只读取包含 `records` 的 JSON，不依赖 MB-OPC、ILT 或具体 workflow。输出使用
`series_pngs` 和 `series_<id>.png`，不保留旧的 `macro_pngs` 命名。

## 当前边界

- Simple 默认四项指标；Gradient 默认六项指标。
- 指标列表分别由 `[mbopc]`、`[gradient]` 配置。
- ILT runner 本次不自动接入，但 ILT 风格 loss records 已由公共测试验证可绘制。
- 不修改 layout、geometry、ILT 求解器和最终光刻输出。
