---
id: CHG-20260815-single-pass-bias
title: 单遍偏置扩张入口 run_single_pass
type: implementation-spec
status: completed
---

# 单遍偏置扩张入口 run_single_pass

> 原始设计：`doc_/archive/reports/single_pass_bias_design.md`（用户审查后
> 放行，含两项决定：displacement_nm 做成配置项、[lithography] 段保留）。

## Objective

提供不迭代、单次全局偏置（默认 +5nm 环双向扩张）的验证入口，复用验证管线
全部核心（exact_dbu/plan_macros/prepare_macro_problem/reconstruct_region/
write_macro_results），用于快速检查重建与合并链路。

## 关键需求（摘要）

- REQ 环双向扩张语义：外环外移、hole 环内移（同一位移值）；
- REQ displacement_nm 为配置项；[lithography] 段仅为网格契约校验保留
  （两套网格合法性标准不可分叉）；
- REQ 产物唯一（每 macro 一个 GDS + 最终合并），不写逐轮状态。

## 已知几何边界（实施中发现并记录）

- 孔闭合：+d 双向收缩孔，两维都必须 > 2d 才不闭合；
- 边压内部 macro 切线退化：切线分裂只保证段不跨越切线，边整条归一侧时
  两侧拼合出现一位移宽台阶（XOR = 2×d²），测试几何须避开切线重合。
