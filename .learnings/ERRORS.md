# Errors

## [ERR-20260809-003] klayout_roi_region_shape_filter

**Logged**: 2026-08-09T01:30:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
KLayout ROI Region materialization counted 11 shapes while the optional iterator diagnostic classified only 10 polygon-like shapes and one text.

### Error
`assert 11 == 10` in `test_simple_materialization_ignores_text_and_reports_it_on_demand`.

### Context
- Full recursive Region materialization previously produced 10 polygons.
- The new implementation uses the explicit ROI RecursiveShapeIterator constructor.
- The result must be understood before setting mask-shape policy.

### Suggested Fix
Inspect delivered native shapes and Region polygons, then explicitly filter shape classes at the native iterator level if KLayout supports it without per-shape Python transfer.

### Metadata
- Reproducible: yes
- Related Files: layout/query.py, tests/layout/test_query.py

### Resolution
- **Resolved**: 2026-08-09T01:45:00+08:00
- **Notes**: Set `RecursiveShapeIterator.shape_flags` to native Box/Path/Polygon flags, adding the properties flag only when requested. Region count and polygon iteration now agree without Python-side filtering.

---

## [ERR-20260809-001] repository_detection

**Logged**: 2026-08-09T01:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: config

### Summary
An earlier Git status probe failed because the repository had not yet been initialized.

### Error
`fatal: not a git repository (or any of the parent directories): .git`

### Context
- The project was initially only a directory skeleton.
- The user subsequently initialized Git.

### Suggested Fix
Probe repository state without assuming Git exists; once initialized, inspect dirty files before each commit.

### Metadata
- Reproducible: no
- Related Files: task_plan.md

### Resolution
- **Resolved**: 2026-08-09T01:00:00+08:00
- **Notes**: Repository now exists and is on branch master.

---

## [ERR-20260809-002] unicode_subprocess_path

**Logged**: 2026-08-09T01:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: config

### Summary
A Chinese local path was corrupted when embedded in a Python script piped through PowerShell stdin.

### Error
`OSError: [Errno 22] Invalid argument: 'C:\\Users\\23158\\Desktop\\OPC??????.html'`

### Context
- Python source was provided over a PowerShell pipeline.
- The file itself was valid and readable.

### Suggested Fix
Resolve Unicode paths in PowerShell and pass them through an environment variable or command argument.

### Metadata
- Reproducible: yes
- Related Files: findings.md

### Resolution
- **Resolved**: 2026-08-09T01:00:00+08:00
- **Notes**: Environment-variable path passing succeeded.

---
