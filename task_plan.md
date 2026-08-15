# MyOPC 亲手迁移任务计划

## Goal

把全 AI 编写的旧代码库（归档于 `00_PAST/`，只读参照）按依赖顺序亲手迁移/过滤到新树，
使每个模块都被项目所有者理解并拥有；每个批次闭环 = 迁移 → 清理 → 测试 → 演示 → 本地提交。

## Next Step

**Phase 5（lithography + evaluation）**：318 + 153 行纯消费者层迁移。注意 opc.input
已重构为 Macro–Core 管线（见 Phase 4），lithography 的 ICCAD13 画布契约
（canvas 256、核 35×35×24、norm="forward"）已由 `opc/input/raster.py` 居中
canvas 对齐，迁移时以 `doc/macro_core_pipeline_design.md` §6 为准。

## Phases

### Phase 1: 归档重置 — Status: complete
- 旧库整体移入 `00_PAST/`，根目录清零，分支 `migration`（commit `a0cacb6`）。
- 规则落地：`00_PAST/` 只读，复制出来改写允许，改归档须请示（AGENTS.md 迁移期规则）。

### Phase 2: layout 批次 — Status: complete
- 迁移 + API 精简（详见 findings.md「API 变更记录」）。
- 交付：`tests/layout/` 27 用例、`main/main_test_layout.py` 演示、`pyproject.toml`。
- Commit `84b1bef`（2026-08-15）。

### Phase 3: geometry 批次 — Status: complete
- 自 00_PAST 迁移，API 零变化（contour.py 三字段加中文行尾注释）。
- 交付：`tests/geometry/` 22 用例、`main/main_test_geometry.py` 演示；全量 49 passed。
- Commit `02f45c9`（2026-08-15）。

### Phase 4: opc.input 重构为 Macro–Core 管线 — Status: complete
- 依据 `doc/macro_core_pipeline_design.md`（用户批准实施），实施 A–E 五批本地提交：
  A 两级网格 + 居中 canvas → B 持久化 MacroProblem（删旧六文件）→
  C 双轮 ±2nm 迭代 → D 最终权威覆盖 + 双模式写出 → E 报告与简化审计。
- 交付：`tests/opc/input/` 55 例 + `tests/main/` 26 例 + `PatchWriter.write_macro_results`；
  gcd_45nm 2×2 smoke：343018 段 / 8 macro GDS / 最终 XOR == 0 / 10.6s。
- **审查轮（2026-08-16）**：用户审查清单 `doc/macro_core_pipeline_review_issues.md`
  逐项核实全部成立，commit `fb80a4e` 落实契约冻结（macro>core、±2nm）、空
  membership 不变量、复杂几何矩阵（11 新用例）、正逆序双轮对照、未处理层
  对照、coverage 审计（84%），并连带修复两个新暴露 bug（切线交点重复分裂
  点、空 macro 崩溃）。审查后 §21 完成标准逐项通过；细节见两份报告。

### Phase 5: lithography + evaluation — Status: pending
- 318 + 153 行，纯消费者层；ICCAD13 画布契约（canvas 256、核 35×35×24、norm="forward"）。

### Phase 6: opc.iteration — Status: pending
- 1670 行，三种求解器（mbopc/diffopc/ilt）；须遵守 Macro–Core 全局轮次屏障
  （设计文档 §20.3）。

### Phase 7: main 入口 + 收尾审计 — Status: pending
- 旧 3357 行接线层已被 `main/run_macro_pipeline.py` 取代大半；剩余入口待评审。
- 最终全量回归 + 交付审计（未用函数/重复实现/异常入口）。

## Decisions Made

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-08-15 | 删除 CellRef 类型，全链路用 str | 相对 str 增量价值薄；顺带删 RegionBatch.cell 死字段与 db.cell() 冗余方法 |
| 2026-08-15 | source 拆分 read_layout / read_glp，分派上移 LayoutDB.open | 单一职责；唯一调用方持有格式选择 |
| 2026-08-15 | 删除 layout/hierarchy.py（HierarchySummary 全家），新增 LayoutDB.cell_hierarchy() 邻接表 | 旧结构零生产调用方；新 API 有真实调用方与测试 |
| 2026-08-15 | 测试全生成式，不迁 TestReticle 依赖 | 遵循 ERR-20260809-016（不硬编码用户 GDS） |
| 2026-08-15 | geometry 本体零修改直接迁移 | 与新 layout API 完全兼容，无过滤必要 |
| 2026-08-15 | opc.input 废弃「全局 core 反向组合 macro」，改为 Macro–Core 两级网格 + 持久化 MacroProblem | 设计文档 §1（用户批准）；消除 MBOPCProblem/MacroPreparation 重复结构 |
| 2026-08-15 | `_write_macro_gds` 较设计文档 §13 增加 dbu_um 参数 | GDS 写出必需源 DBU 而 NPZ 格式不含该字段（最小必要偏差，已记开发报告） |
| 2026-08-15 | 新增注释规则：main/ 每行中文短注释，其他目录文件/函数/分段注释 | 用户明令（2026-08-15），已写入 AGENTS.md 并约束本次全部新代码 |

## Errors Encountered

| Error | 尝试 | 解决 |
|---|---|---|
| 断言脚本「关闭守卫未抛 ClosedLayoutError」 | 反复调试 | 根因：lambda 漏调用括号（`db.cell_hierarchy` 非 `db.cell_hierarchy()`）；已随重写消除 |
| read_glp 收到 tuple 层映射 AttributeError | 1 | 测试违反契约；tuple 规范化只在 LayoutDB.open 入口做，修正测试传 LayerSpec |
| 切线分裂后 SegmentBatch 校验失败（非一维） | 逐层 spy 定位 | 新批次 edge_ids 误传原始段号而非数学边号；修正为 `segments.edge_ids[boundary_seg[~last]]`（回归用例：test_grid/problem 系列） |
| np.where 全分支求值致穿越索引越界 | 1 | last/first 行索引先夹回有效范围；被夹位置的值不会被选中 |
