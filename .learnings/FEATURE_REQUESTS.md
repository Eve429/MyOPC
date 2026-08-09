# Feature Requests

## [FEAT-20260809-002] mbopc_shared_frontend

**Logged**: 2026-08-09T14:30:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Requested Capability
开发可与 ILT 等方法共享物理 mask/core/采样基础的高性能 MB-OPC 前端，完成分段、归属、更新、重建、直接运行入口、多图形测试和标注图片。

### User Context
用户需要后续构建不同 OPC 方法，并通过一个主 Python 文件验证本次全部功能，同时获得开发手册、测试手册和详细图形测试文档。

### Complexity Estimate
complex

### Suggested Implementation
建立 `opc.common` 与 `opc.mbopc` 单向依赖，使用紧凑参数化 segment 数组、一次性缓存、稳定 key 和稀疏 core membership，并在完成后执行严格性能与简化审计。

### Resolution
- **Resolved**: 2026-08-09T19:10:00+08:00
- **Notes**: 已交付 `opc.common`/`opc.mbopc`、紧凑分段与稳定 key、稀疏 core 归属、owner-only 更新、重建、无安装主入口、严格基准、多图形标注图集和中文手册/报告；81 项全仓库测试与真实 `gcd_45nm.gds` 验证通过。

### Metadata
- Frequency: recurring
- Related Features: hierarchical_roi_query, planner_roi_png_render

---

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
