# Learnings

## [LRN-20260809-002] correction

**Logged**: 2026-08-09T03:05:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
本项目要求所有源码注释和 docstring 使用中文，并且必须支持不安装项目包的直接 Python 文件入口。

### Details
首批代码使用了英文 docstring，且验证流程通过 editable install 运行。用户明确要求改为中文注释、细化关键路径说明，并在最终交付前审查过度设计和错误修复造成的逻辑混乱。

### Suggested Action
统一翻译源码与测试注释，提供根目录 CLI，最终执行独立的简化与加固审查。

### Resolution
- **Resolved**: 2026-08-09T04:30:00+08:00
- **Notes**: 注释与 docstring 已统一为中文；根目录入口在卸载项目包后通过外部工作目录测试；最终简化审查已完成。

### Metadata
- Source: user_feedback
- Related Files: layout, geometry, tests, run_layout_geometry.py
- Tags: comments, cli, maintainability
- Pattern-Key: myopc.chinese_comments_direct_entry
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09

---

## [LRN-20260809-001] best_practice

**Logged**: 2026-08-09T01:00:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
For hierarchical OPC layouts, module separation remains fast when the boundary is an ROI/layer batch instead of individual polygons.

### Details
KLayout can retain the hierarchy and materialize only shapes intersecting a local ROI. A one-million-instance array prototype returned 25 shapes without full expansion.

### Suggested Action
Keep LayoutDB hierarchical and read-only, use lazy ROI queries, and convert to NumPy only once per local work area.

### Resolution
- **Resolved**: 2026-08-09T04:30:00+08:00
- **Notes**: 已落实为只读 LayoutDB、惰性 ROI 查询、原生 Region 批处理和显式 NumPy 转换边界，并通过百万逻辑实例基准。

### Metadata
- Source: conversation
- Related Files: layout, geometry
- Tags: opc, klayout, performance, hierarchy
- Pattern-Key: opc.batch_roi_boundary
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09

---
