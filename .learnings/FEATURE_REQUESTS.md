# Feature Requests

## [FEAT-20260809-001] planner_roi_png_render

**Logged**: 2026-08-09T05:10:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Requested Capability
提供一个可直接调用的函数，把 planner 给出的版图区域提取为像素图，并选择直接展示或保存为 PNG。

### User Context
用户需要检查 `gcd_45nm.gds` 等真实版图的局部切分结果，并让同一结果可用于后续 OPC 像素计算。

### Complexity Estimate
medium

### Suggested Implementation
复用只读 LayoutDB 和单层 ROI 查询，使用 KLayout 原生面积覆盖栅格化生成灰度数组，再由 Pillow 展示或原子保存 PNG；通过条带处理限制峰值内存。

### Metadata
- Frequency: first_time
- Related Features: hierarchical_roi_query, patch_output

### Resolution
- **Resolved**: 2026-08-09T06:05:00+08:00
- **Notes**: 已提供可复用数据库和已提取批次两种函数入口、原生灰度覆盖栅格、PNG 原子保存、系统查看器以及直接 CLI，并通过真实版图与严格性能验证。

---
