# MyOPC 亲手迁移任务计划

## Goal

把全 AI 编写的旧代码库（归档于 `00_PAST/`，只读参照）按依赖顺序亲手迁移/过滤到新树，
使每个模块都被项目所有者理解并拥有；每个批次闭环 = 迁移 → 清理 → 测试 → 演示 → 本地提交。

## Next Step

**Phase 4（opc.input + opc.input.edge）**：用户已把 `opc/` 复制到新树（未适配）。
先逐文件 diff 00_PAST 确认改动面，再适配 layout/geometry 新 API（RegionBatch 三参、
top_cell_name、无 CellRef），然后按惯例补 main 演示 + tests 用例。

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

### Phase 4: opc.input + opc.input.edge — Status: in_progress
- 约 2073 行（1315 + 758），架构枢纽 `prepare_problem()` 在此层。
- 用户已复制 opc/；核心链（opc.input.edge）导入新 layout/geometry 正常，
  已知遗留：`opc/diagnostics.py` 仍用 CellRef + 旧 RegionBatch 四参（适配点）。
- 已完成：`opc/input/edge/ownership.py` 教学级注释加厚（代码零改动），
  并用 2×2 网格横条示例实证 owner 唯一 / own⊆membership / CSR 一致三不变量。
- 不变量断言重点：零位移 XOR==0、segment key 唯一、长度≤配置、法向单位向量、owner 唯一。

### Phase 5: lithography + evaluation — Status: pending
- 318 + 153 行，纯消费者层；ICCAD13 画布契约（canvas 256、核 35×35×24、norm="forward"）。

### Phase 6: opc.iteration — Status: pending
- 1670 行，三种求解器（mbopc/diffopc/ilt）。

### Phase 7: main 入口 + 收尾审计 — Status: pending
- 3357 行接线层；最终全量回归 + 交付审计（未用函数/重复实现/异常入口）。

## Decisions Made

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-08-15 | 删除 CellRef 类型，全链路用 str | 相对 str 增量价值薄；顺带删 RegionBatch.cell 死字段与 db.cell() 冗余方法 |
| 2026-08-15 | source 拆分 read_layout / read_glp，分派上移 LayoutDB.open | 单一职责；唯一调用方持有格式选择 |
| 2026-08-15 | 删除 layout/hierarchy.py（HierarchySummary 全家），新增 LayoutDB.cell_hierarchy() 邻接表 | 旧结构零生产调用方；新 API 有真实调用方与测试 |
| 2026-08-15 | 测试全生成式，不迁 TestReticle 依赖 | 遵循 ERR-20260809-016（不硬编码用户 GDS） |
| 2026-08-15 | geometry 本体零修改直接迁移 | 与新 layout API 完全兼容，无过滤必要 |

## Errors Encountered

| Error | 尝试 | 解决 |
|---|---|---|
| 断言脚本「关闭守卫未抛 ClosedLayoutError」 | 反复调试 | 根因：lambda 漏调用括号（`db.cell_hierarchy` 非 `db.cell_hierarchy()`）；已随重写消除 |
| read_glp 收到 tuple 层映射 AttributeError | 1 | 测试违反契约；tuple 规范化只在 LayoutDB.open 入口做，修正测试传 LayerSpec |
