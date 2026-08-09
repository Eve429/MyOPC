# Errors

## [ERR-20260809-019] common_mask_edge_expectation

**Logged**: 2026-08-09T14:50:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
公共物理 mask 测试把两个矩形组件的总外边数误写为 10。

### Error
`assert 8 == 10`

### Context
- 两个重叠矩形合并为一个四边矩形。
- 仅角点接触的第三个矩形在 minimum-coherence 下独立保留，同样有四条边。

### Suggested Fix
把预期改为 8，并保留 Polygon 数量、面积和内部 cut-line 消失的独立断言。

### Metadata
- Reproducible: yes
- Related Files: tests/opc/test_common.py

### Resolution
- **Resolved**: 2026-08-09T14:51:00+08:00
- **Notes**: 仅修正测试预期；生产几何逻辑不变。

---

## [ERR-20260809-018] multi_file_patch_hunk_format

**Logged**: 2026-08-09T14:35:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
固化用户范围规则的多文件补丁因在 hunk 内直接开始下一文件而被整体拒绝。

### Error
`apply_patch verification failed: invalid hunk`

### Context
- 补丁没有写入任何部分内容。
- 下一次按每个文件的完整 hunk 边界重新组织。

### Suggested Fix
每个 `Update File` 段都使用独立的 `@@` 上下文，不在未结束的 hunk 中切换文件。

### Metadata
- Reproducible: yes
- Related Files: AGENTS.md, task_plan.md, progress.md

### Resolution
- **Resolved**: 2026-08-09T14:36:00+08:00
- **Notes**: 使用独立 hunk 重写补丁。

---

## [ERR-20260809-019] property_test_after_database_close

**Logged**: 2026-08-09T06:45:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
属性回归测试在关闭 LayoutDB 后才读取由层级迭代器构造的 Region，得到错误的零计数。

### Error
`assert plain.count() == preserved.count() == 2` 实际得到 0。

### Context
- 生产查询在数据库打开期间物化正确。
- Region 仍可能引用原生层级数据，不能把关闭数据库后的访问作为属性语义验证。

### Suggested Fix
把 Region 数量和属性断言放在 `with LayoutDB.open(...)` 生命周期内。

### Metadata
- Reproducible: yes
- Related Files: tests/layout/test_query.py

### Resolution
- **Resolved**: 2026-08-09T06:45:00+08:00
- **Notes**: 测试断言已移入数据库上下文，不改变生产实现。

---

## [ERR-20260809-018] planning_checker_table_format

**Logged**: 2026-08-09T06:20:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
planning-with-files 完整性脚本不能识别项目现有的阶段状态表。

### Error
`Task in progress (0/0 phases complete)`，但计划表中 11 个阶段均为 complete。

### Context
- 脚本退出码为 0，只是解析结果不适用于当前表格格式。
- 为工具改写已经稳定使用的项目规划格式没有实际价值。

### Suggested Fix
人工核对阶段表和验收门禁；保留当前文档格式。

### Metadata
- Reproducible: yes
- Related Files: task_plan.md

### Resolution
- **Resolved**: 2026-08-09T06:20:00+08:00
- **Notes**: 已人工确认阶段 1-11 全部 complete，最终门禁均通过。

---

## [ERR-20260809-017] raster_comment_prefix_scan

**Logged**: 2026-08-09T06:10:00+08:00
**Priority**: low
**Status**: resolved
**Area**: backend

### Summary
最终中文扫描发现两条新增注释以英文技术名词开头。

### Error
`planner` 和 `rasterize` 位于注释起始位置。

### Context
- 说明主体已经是中文，但项目约定要求中文起句。
- 其他并行门禁输出因扫描返回非零而未汇总，必须完整重跑。

### Suggested Fix
分别改为“规划器”和“原生栅格接口”起句，再执行全部最终门禁。

### Metadata
- Reproducible: yes
- Related Files: geometry/raster.py, tests/geometry/test_raster.py

### Resolution
- **Resolved**: 2026-08-09T06:10:00+08:00
- **Notes**: 两条起句均已改为中文。

---

## [ERR-20260809-016] mutable_user_fixture_baseline

**Logged**: 2026-08-09T05:50:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
全量回归有两项失败，因为自动化测试把用户可编辑的 `simple.gds` 内容硬编码为固定基线。

### Error
版图 bbox 从 `(-2400,-500;400,1500)` 变为 `(-2000,-1100;-200,2200)`，旧 ROI 中 Polygon 数从 10 变为 11。

### Context
- 用户确认版图变化是主动修改，不能恢复文件。
- 生产栅格代码和生成式专项测试均通过。

### Suggested Fix
把精确坐标、计数和 CLI 断言迁移到测试时生成的确定性 GDS；用户样例只用于手工只读验证。

### Metadata
- Reproducible: yes
- Related Files: tests/layout/test_database.py, tests/layout/test_query.py, tests/test_cli.py

### Resolution
- **Resolved**: 2026-08-09T05:50:00+08:00
- **Notes**: 所有依赖 `simple.gds` 精确内容的断言均改用生成式版图，用户文件未修改。

---

## [ERR-20260809-015] learning_resolution_patch_context

**Logged**: 2026-08-09T05:47:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
更新 GDS 调查记录时，把 `task_plan.md` 的错误表行放进了 `.learnings/ERRORS.md` 补丁上下文。

### Error
`apply_patch verification failed: Failed to find expected lines`

### Context
- 补丁在写入前完整失败，没有产生部分修改。
- 两个文件的目标位置已经分别读取确认。

### Suggested Fix
按文件分别更新状态、解决说明和计划错误表，不复用跨文件上下文。

### Metadata
- Reproducible: yes
- Related Files: .learnings/ERRORS.md, task_plan.md

### Resolution
- **Resolved**: 2026-08-09T05:47:00+08:00
- **Notes**: 已按实际文件位置拆分补丁。

---

## [ERR-20260809-014] tracked_gds_fixture_changed

**Logged**: 2026-08-09T05:40:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
全量测试后，原本干净的跟踪文件 `TestReticle/simple.gds` 被 Git 标记为二进制变化。

### Error
工作树对象哈希 `33ef637a...` 与 HEAD 对象哈希 `83acea5c...` 不一致，文件大小仍为 2,190 bytes。

### Context
- 本功能只应读取该文件；CLI 的 Patch 和 PNG 输出均指向 pytest 临时目录。
- 必须先确认语义差异和写入来源，再决定是否恢复，不能把无关二进制变化纳入提交。
- 第一次尝试使用 `git show --output` 导出 blob，但该参数不生成纯 blob 文件，KLayout 报告未知流格式；后续改用 `git archive`，不重复失败方法。

### Suggested Fix
导出 HEAD 版本到忽略目录，比较两个 GDS 的结构、图层、图形和头信息；定位写入路径后增加防回归检查。

### Metadata
- Reproducible: unknown
- Related Files: TestReticle/simple.gds, tests/test_cli.py, layout/database.py

### Resolution
- **Resolved**: 2026-08-09T05:45:00+08:00
- **Notes**: 用户确认 `simple.gds` 是其主动修改；该文件保持原样并从本次暂存与提交范围中排除。

---

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
