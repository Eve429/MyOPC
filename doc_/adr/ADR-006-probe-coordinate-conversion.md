# ADR-006 — 探针坐标统一走 points_to_canvas

- Date：2026-08-16
- Status：accepted

## Decision

全局 DBU 点到画布像素坐标的换算唯一入口是
`opc/input/raster.py::points_to_canvas`（float64 连续坐标，含居中
padding 项）；任何探针/采样坐标禁止手写 `(x−left)/pixel−0.5` 公式。

## Reason

- 旧管线契约是 tile+2×halo 恰满 256 画布、无 padding 概念（旧公式与旧
  raster 自洽）；新 Macro–Core 的 context 任意（如 400nm→228px+14px
  padding），探针换算必须补 `+low_x/+low_y`，否则整体偏移 14 像素；
- 精度必须 float64：坐标是全局 int32 DBU 域，超 2²⁴ 后 float32 连整数
  都无法精确表示（大数相减再除法放大误差），round 边界会翻错像素；
- 三个公开栅格函数共享 `_center_padding`，`points_to_canvas` 与
  `ownership_canvas` 互为反函数（全部 True 像素中心整数回映，批量测试
  锁定），换算与掩码天然对齐。

## Rejected alternatives

- 各调用点内联公式：口径分叉（历史上 solver 与 raster 已分叉过一次）；
- 换算时直接 round 成 int 索引：越界判定与 half-to-even 语义属于评价层
  （evaluate_edge_probes），提前取整丢失可逆性与可测试性。

## Consequences

- 未来梯度 MB-OPC 复用同一换算（midpoint 采样坐标同源）；
- 任何新的"DBU 点 → 画布"需求先扩展/复用该函数，不再新增公式。
