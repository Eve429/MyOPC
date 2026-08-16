---
id: CHG-20260816-lithography-iccad13
title: ICCAD13 Hopkins 光刻模型迁移
type: implementation-spec
status: completed
---

# ICCAD13 Hopkins 光刻模型迁移

> 原始设计：`doc_/archive/reports/lithography_migration_design.md`
> （用户 2026-08-16 批准）。

## Objective

只迁 ICCAD13 Hopkins 模型：一个具体类 + 四资产 + 原生 autograd 批量前向 +
main 验证入口；CPU 数值与 OpenILT 同资产基线逐位一致。

## 关键需求（摘要）

- REQ 资产身份：四 .pt 的 SHA-256 硬断言；布局只接受 [H,W,K]+[K]；
- REQ 前向：pad→fft2(norm="forward")→四象限核相乘→ifft2→scale 加权
  |field|²→dose²→sigmoid→crop，全原生可微；
- REQ 批量：一次 forward_many = 1 次 mask FFT + 每 bank 一次传播；
- REQ 透光率契约：1=透光、行 0=最低 Y、范围 (0,1)、与 raster canvas
  padding 契约逐位一致（256 直传不二次移动）；
- REQ 明确不迁：Protocol（当时无消费者）、手写 backward/CT/combo/
  TorchLitho、resize 分支、资产 shape 猜测。

## 批准与实施

用户批准后四批实施（A 配置/资产 → B 前向 → C main → D 报告）；
后续追加阶段 6 matplotlib 可视化（用户需求）。
