# Learnings

## [LRN-20260809-001] best_practice

**Logged**: 2026-08-09T01:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
For hierarchical OPC layouts, module separation remains fast when the boundary is an ROI/layer batch instead of individual polygons.

### Details
KLayout can retain the hierarchy and materialize only shapes intersecting a local ROI. A one-million-instance array prototype returned 25 shapes without full expansion.

### Suggested Action
Keep LayoutDB hierarchical and read-only, use lazy ROI queries, and convert to NumPy only once per local work area.

### Metadata
- Source: conversation
- Related Files: layout, geometry
- Tags: opc, klayout, performance, hierarchy
- Pattern-Key: opc.batch_roi_boundary
- Recurrence-Count: 1
- First-Seen: 2026-08-09
- Last-Seen: 2026-08-09

---
