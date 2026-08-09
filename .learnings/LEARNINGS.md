# Learnings

## [LRN-20260809-005] correction

**Logged**: 2026-08-09T14:35:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: backend

### Summary
MB-OPC 开发不得修改现有 layout 和 geometry；接口不足时必须先停下由用户确认。

### Details
新公共 OPC 层只能依赖已经存在的公开接口。即使最小改动看似合理，也不能把它作为普通实现步骤直接执行。

### Suggested Action
把边界写入 AGENTS.md 和任务计划；提交前用 Git 路径检查确认两个目录没有差异。

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, layout, geometry
- Tags: scope, preservation, approval
- Pattern-Key: myopc.no_layout_geometry_changes_without_approval
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09
- Promoted: AGENTS.md

---

## [LRN-20260809-004] correction

**Logged**: 2026-08-09T14:30:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: backend

### Summary
项目永久要求在 bug 修复后清理遗留逻辑，并为所有文件、函数和关键函数内部路径提供详细紧凑的中文注释，同时同步维护开发与测试手册。

### Details
这些要求适用于后续全部功能，不是本次 MB-OPC 的临时验收项。注释必须解释坐标、性能、不变量和边界原因；bug 修复不能留下只为旧错误存在的函数、包装层或分支。

### Suggested Action
把规则写入 AGENTS.md，并在每次功能交付中执行 AST 注释扫描、回归测试、调用点搜索和手册同步检查。

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, doc
- Tags: comments, simplify, manuals, quality-gate
- Pattern-Key: myopc.project_delivery_core_rules
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09
- Promoted: AGENTS.md

---

## [LRN-20260809-003] correction

**Logged**: 2026-08-09T06:40:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
KLayout 的 `Shapes.SProperties` 是“仅选择带属性 Shape”的过滤器，不是属性保留标志。

### Details
原实现把 `SProperties` 与 Box/Path/Polygon 类型做按位或，导致 `preserve_properties=True` 时没有属性的有效几何被静默丢弃。真正的“保留全部几何并导入已有属性”只需要在类型选择不变时调用 `RecursiveShapeIterator.enable_properties()`。

### Suggested Action
移除 `SProperties`，保留 `enable_properties()`，并用带属性和无属性图形混合的 GDS 同时验证数量、面积和属性键值。

### Metadata
- Source: user_feedback
- Related Files: layout/query.py, tests/layout/test_query.py
- Tags: klayout, properties, iterator, geometry-loss
- Pattern-Key: klayout.sproperties_is_filter
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09

### Resolution
- **Resolved**: 2026-08-09T06:50:00+08:00
- **Notes**: 已移除 `SProperties` 过滤器，仅按需启用属性导入；混合属性测试确认几何数量不变且键值正确保留。

---

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
