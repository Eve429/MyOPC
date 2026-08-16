# ADR-003 — 独立 macro 迭代与最终一次合并

- Date：2026-08-16（simple MB-OPC 设计）
- Status：accepted

## Decision

每个 macro 独立完成自己的全部迭代（baseline 到停止），macro 之间不设逐轮
屏障、不交换中间位移；全部 macro 完成后只调用一次
`merge_macro_results(plan, {macro_id: best.gds}, ...)`（显式映射）。
边界 core 的 context 固定为邻区**参考几何**（零位移副本）。

## Reason

逐轮全局 merge/I/O 的开销与复杂度高；第一版优先建立可运行的完整闭环。
merge 函数设计为不关心调用时机——未来升级为逐轮同步时只改编排层调用
方式与 context 来源，不改几何合并实现。

## Rejected alternatives

- 每轮全局同步（Jacobi 跨 macro）：上下文最准但每轮 merge 成本高；
- 以 polygon 为单位交换：无稳定 ID（见 ADR-002）。

## Consequences

- macro 边界使用邻区参考几何，**不是**全局同步最优——差异必须量化
  （gcd_45nm 实测：single 全 ROI 总 EPE 比 multi 2×2 之和小 236 段 ≈1%，
  最终覆盖 XOR 34650860 DBU²），不得在报告/文档中宣称等价；
- macro 正逆序求解结果相同（独立性测试锁定）；
- 相邻 macro 同边段可能得到不同方向形成真实 jog（不平滑，已知限制）。
