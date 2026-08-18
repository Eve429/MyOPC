# ADR-005 — EPE 单目标驱动与 insufficient_probes 语义

- Date：2026-08-16（审查修复轮补全语义）
- Status：accepted

## Decision

最简 MB-OPC 的优化目标只有 nominal EPE：方向由 nominal 探针违规产生，
best 只按 EPE 严格改善选择（平局保留较早轮）；L2/PVBand 仅累计诊断，
不进入决策。`valid_probes == 0` 且存在 owner 段时报告
`insufficient_probes`（无法评价，保留 baseline），不冒充 zero_epe；
空 macro（零段）的 zero_epe 语义正确（无违规对象）。

## Reason

- 单目标最简闭环是第一版定位；多目标（工艺窗）需要连续位移上的联合
  loss，属梯度方法范畴；
- EPE 平衡点 ≈ 工艺角分歧最大点，因此 PVBand 会随 EPE 优化变宽——这是
  目标函数的结构性后果（gcd_45nm 实测 +36%），不是缺陷；
- 探针全无效时 epe 恒 0（violation 只在有效探针上累计），与真收敛必须
  区分，否则 2nm 窄壁 + 8nm 探针会被误判为已收敛（独立审查 P1 复现）。

## Rejected alternatives

- 拒绝这类 Problem：窄壁跨 macro，构造期无法预检；
- 自适应探针距离：改变评价语义，超出最简范围。

## Consequences

- 循环内的 insufficient 检查必须先于 best 比较（valid==0 的 epe=0 会被
  误当改善状态）；
- PVBand/L2 的改善诉求指向梯度 MB-OPC（联合 loss），simple 版是单目标
  基线；
- 停止状态共五种：zero_epe / no_update / invalid_geometry /
  insufficient_probes / iteration_limit。
