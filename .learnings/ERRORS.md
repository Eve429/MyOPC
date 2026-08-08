# Errors

## [ERR-20260809-008] cli_negative_box_parsing

**Logged**: 2026-08-09T03:25:00+08:00
**Priority**: low
**Status**: resolved
**Area**: config

### Summary
逗号连接的负坐标 Box 参数被 argparse 识别成新的命令行选项。

### Error
直接入口测试以退出码 2 结束，`--box -2500,-600,500,1600` 未被接受。

### Context
- 自定义逗号解析器让负坐标字符串与 argparse 选项语法冲突。

### Suggested Fix
使用 `nargs=4, type=int` 接收四个独立坐标，删除自定义解析器。

### Metadata
- Reproducible: yes
- Related Files: run_layout_geometry.py, tests/test_cli.py

### Resolution
- **Resolved**: 2026-08-09T03:25:00+08:00
- **Notes**: CLI 改为 `--box LEFT BOTTOM RIGHT TOP`，实现更简单且原生支持负数。

---

## [ERR-20260809-007] hierarchical_text_roi_diagnostics

**Logged**: 2026-08-09T02:55:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: backend

### Summary
Area-overlap ROI iterators excluded all zero-area Text points inside hierarchical instances.

### Error
Generated fixture diagnostics reported zero Text objects instead of eight.

### Context
- Production polygon materialization should use `overlapping=True`.
- Text diagnostics require the iterator's touching-or-overlapping mode because Text bbox can be a point.

### Suggested Fix
Use touching semantics only in the optional diagnostic pass; retain area overlap for the performance path.

### Metadata
- Reproducible: yes
- Related Files: layout/query.py, tests/layout/test_generated_layout.py

### Resolution
- **Resolved**: 2026-08-09T02:55:00+08:00
- **Notes**: Diagnostic and production iterator semantics are now deliberately separate and documented.

---

## [ERR-20260809-006] generated_layout_expectations

**Logged**: 2026-08-09T02:40:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Generated hierarchy tests underestimated the full layout bbox and rotated ROI shape count.

### Error
Expected bbox right/top 700/2650 instead of 1000/2700; expected two ROI shapes instead of three.

### Context
- Layout bbox includes Text even though Region materialization filters Text.
- The rotated LEAF ROI intersects its Box, Path, and holed Polygon.

### Suggested Fix
Keep separate assertions for layout-database semantics and polygon-materialization policy.

### Metadata
- Reproducible: yes
- Related Files: tests/layout/test_generated_layout.py

### Resolution
- **Resolved**: 2026-08-09T02:40:00+08:00
- **Notes**: Corrected expectations without changing production behavior.

---

## [ERR-20260809-005] klayout_region_area_semantics

**Logged**: 2026-08-09T02:20:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
A combine test expected KLayout Region.area() to sum overlapping raw polygon areas.

### Error
`assert 15000 == 20000`

### Context
- Region `+` retained two raw polygons and did not merge them.
- With default merged semantics, area reports geometric set area rather than a raw per-polygon sum.

### Suggested Fix
Test raw preservation with count/is_merged and test geometric coverage with set area separately.

### Metadata
- Reproducible: yes
- Related Files: tests/geometry/test_region.py

### Resolution
- **Resolved**: 2026-08-09T02:20:00+08:00
- **Notes**: Corrected the expectation; production combine behavior was unchanged.

---

## [ERR-20260809-004] pytest_relative_test_import

**Logged**: 2026-08-09T02:10:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Geometry tests using relative imports failed collection because the test directories were not Python packages.

### Error
`ImportError: attempted relative import with no known parent package`

### Context
- `test_region.py` and `test_contour_edge.py` import `.helpers`.

### Suggested Fix
Keep shared helpers local and add package markers to the relevant test directories.

### Metadata
- Reproducible: yes
- Related Files: tests/__init__.py, tests/geometry/__init__.py

### Resolution
- **Resolved**: 2026-08-09T02:10:00+08:00
- **Notes**: Added minimal package markers without changing production imports.

---

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
