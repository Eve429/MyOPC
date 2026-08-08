# Errors

## [ERR-20260809-013] chinese_comment_prefix_scan

**Logged**: 2026-08-09T04:45:00+08:00
**Priority**: low
**Status**: resolved
**Area**: backend

### Summary
中文注释扫描仍发现五条以英文 API 名称开头的行注释。

### Error
`KLayout`、`Python`、`Region.count()`、`DbuBox`、`Layout` 位于注释起始位置。

### Context
- 注释的语法和解释主体已经是中文，但用户要求所有注释使用中文。
- 技术名词可以保留在句中，无需成为起句主语。

### Suggested Fix
用中文描述起句，把必要 API 名称放到句中。

### Metadata
- Reproducible: yes
- Related Files: layout/query.py, layout/writer.py, geometry/spatial.py, tests/layout/test_generated_layout.py

### Resolution
- **Resolved**: 2026-08-09T04:45:00+08:00
- **Notes**: 五条注释均已调整，并重新执行扫描。

---

## [ERR-20260809-012] temporary_directory_cleanup_policy

**Logged**: 2026-08-09T04:40:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
仓库外直接入口复验脚本因包含动态临时目录的递归删除而在执行前被安全策略拒绝。

### Error
`rejected: blocked by policy`

### Context
- 失败发生在命令审批阶段，基准、扫描和入口测试均未开始执行。
- 验证目标只要求从仓库外工作目录运行，不需要创建或删除临时目录。

### Suggested Fix
直接使用已有的仓库父目录作为工作目录，不执行任何清理命令。

### Metadata
- Reproducible: yes
- Related Files: run_layout_geometry.py

### Resolution
- **Resolved**: 2026-08-09T04:40:00+08:00
- **Notes**: 改用现有仓库外目录复验，避免不必要的文件系统操作。

---

## [ERR-20260809-011] formatter_style_mismatch

**Logged**: 2026-08-09T04:10:00+08:00
**Priority**: low
**Status**: resolved
**Area**: backend

### Summary
Ruff 规则检查通过，但格式检查要求把用户偏好的紧凑代码大幅展开。

### Error
`25 files would be reformatted`

### Context
- 用户明确偏好紧凑式代码格式。
- Ruff 提出的不是语义或静态规则问题，只是其 Black 风格布局差异。

### Suggested Fix
保留手工紧凑排版，以 Ruff rule check、compileall 和测试作为质量门槛。

### Metadata
- Reproducible: yes
- Related Files: layout, geometry, tests, benchmarks, run_layout_geometry.py

### Resolution
- **Resolved**: 2026-08-09T04:10:00+08:00
- **Notes**: 不运行自动格式化；静态规则检查已全通过。

---

## [ERR-20260809-010] static_audit_findings

**Logged**: 2026-08-09T03:55:00+08:00
**Priority**: low
**Status**: resolved
**Area**: backend

### Summary
Ruff 静态审查返回 10 个维护性问题。

### Error
包括旧式 Mapping/Callable 导入、多余生成器与整数转换、未标注类常量、测试默认对象和嵌套上下文。

### Context
- 功能测试已经通过，这些问题属于最终简化与加固阶段。

### Suggested Fix
手工逐项简化并复查差异，不使用可能改变紧凑格式的自动修复。

### Metadata
- Reproducible: yes
- Related Files: layout, geometry, tests, benchmarks, run_layout_geometry.py

### Resolution
- **Resolved**: 2026-08-09T03:55:00+08:00
- **Notes**: 10 项均已手工修正，公共接口和行为不变。

---

## [ERR-20260809-009] missing_ruff_auditor

**Logged**: 2026-08-09T03:45:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
最终静态审查命令无法运行，因为 myopc 环境尚未安装 Ruff。

### Error
`No module named ruff`

### Context
- Ruff 只用于开发期静态审查，不属于运行时依赖。

### Suggested Fix
将 Ruff 放入可选 dev 依赖并单独安装，不要求用户安装项目包。

### Metadata
- Reproducible: yes
- Related Files: pyproject.toml

### Resolution
- **Resolved**: 2026-08-09T03:45:00+08:00
- **Notes**: 已加入可选开发依赖，运行时依赖没有增加。

---

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
