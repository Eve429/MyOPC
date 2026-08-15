# MyOPC 迁移进度日志

## 会话记录

### 2026-08-15（会话 3：Macro–Core 管线重构，Phase 4 完成）

- 用户批准实施 `doc/macro_core_pipeline_design.md` 全部内容，并新增注释规则
  （main/ 每行中文短注释；其他目录文件/函数/分段注释），规则已入 AGENTS.md。
- 实施 A–E 五批本地提交：两级网格 + 居中 canvas（22 例）→ 持久化 MacroProblem
  （删旧六文件，18 例）→ 双轮 ±2nm 迭代（23 例）→ 最终权威覆盖双模式写出
  （+3 patch 例）→ 开发/测试报告 + 双手册 + 简化审计（旧符号/术语/投机抽象零残留）。
- 开发中修两个自引入 bug：切线分裂 edge_ids 误传段号、np.where 越界索引。
- 测试基线 49 → **115 passed**；gcd_45nm 2×2 smoke：8 macro GDS、
  343018 段、总 10.6s、**最终 XOR == 0**。产物在 output/（不提交）。
- 最小必要偏差（已记开发报告）：`_write_macro_gds` 增 dbu_um 参数；
  merge RSS 为完成后即时采样。
- Phase 4 置 complete；Next Step → Phase 5（lithography + evaluation）。

## 会话记录

### 2026-08-15（会话 2：opc 批次开始）

- 规划文件落地（task_plan / findings / progress），用户明令此后多步任务必须走 planning-with-files。
- `opc/input/edge/ownership.py` 注释加厚（模块头三数组契约 + 七个阶段块注释；代码零改动）。
- ownership 具体示例跑通：2×2 网格 + halo 30 + 跨切线横条，实证
  owner 唯一 / own⊆membership / membership 总数==CSR 终点三条不变量。
- `reconstruction.py` 拐角块逐行注释加入文件（miter/bevel 逻辑）。
- 发现并修复用户注释重构引入的 bug：SegmentBatch 字段重排后位置传参错位
  （fragmentation.py:243 改关键字传参）；验证零位移 XOR==0、+3 DBU 重建 3276。
- opc 首次 ruff 检查：5 个导入排序问题已 --fix；compileall / pytest 49 全绿。
- task_plan Phase 4 置 in_progress。

## 会话记录

### 2026-08-15（会话 1：归档 + layout + geometry）

- 建立迁移工作模式：`00_PAST/` 归档（只读纪律写入 AGENTS.md/CLAUDE.md）、分支 `migration`、
  重写 CLAUDE.md 反映迁移现实、写入持久记忆。
- **layout 批次**：用户手迁 database/types/query/source 并做 str 化改造；Claude 协作清理
  （`_native_cell` 的 `cell.name` AttributeError 真 bug、`cell()` 冗余方法、三分支分派、
  `__init__`/query.py 连带断点）；交付 27 用例 + 演示入口。Commit `84b1bef`。
- **geometry 批次**：用户迁移（diff 确认 API 零变化）；Claude 交付 22 用例 + 演示入口，
  helpers 适配新 RegionBatch 签名。Commit `02f45c9`。
- 测试基线：`pytest -q tests` → 49 passed；ruff / compileall 全绿。
- 工作树遗留：`AGENTS.md`（迁移期规则 + 未来优化条目）、`CLAUDE.md`（重写）未提交，
  待用户决定归属批次；`TestReticle/M1_test10.glp` 用户数据不提交；`opc/` 已复制待迁移。

## 测试结果速查

| 日期 | 范围 | 结果 |
|---|---|---|
| 2026-08-15 | tests/layout | 27 passed |
| 2026-08-15 | tests/layout + tests/geometry | 49 passed |
| 2026-08-15 | 全量（layout 27 + geometry 25 + opc/input 40 + main 23） | 115 passed |
| 2026-08-15 | gcd_45nm 2×2 smoke | 总 10.6s，最终 XOR 面积 = 0 |
