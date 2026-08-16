# MyOPC 迁移进度日志

## 会话记录

### 2026-08-16（会话 6：Phase 5A lithography 迁移）

- 用户批准 `doc/lithography/lithography_migration_design.md` 实施（先做三路事实
  核对：§3 全部吻合、无设计错误；基线实测 143 passed）。
- 实施四批 A–D：配置/资产 → 可微批量前向 → main 验证入口 → 报告与简化审计。
- 提交：`6338710`（配置+资产+43 例）→ `8773e37`（前向+27 例）→
  `5f0747a`（main 入口+3 例）→ 报告批次（补 11 个防御分支用例 + requirements
  + 手册 + 两报告 + 规划同步）。
- **数值验收**：CPU 三工艺角 sums 与 OpenILT 基线逐位相等（差 0.0）；
  有限差分/批量一致性/CUDA parity（GTX 1650，1e-4）全过。
- **实施中实际复现 Windows DLL 缺失**（nvrtc-builtins64_124.dll），按设计
  §11.7 预设授权加回旧版最小修复（模块级 DLL 目录注册）。
- 测试基线 143 → **224 passed**；lithography coverage **100%**（204/204）。
- gcd_45nm smoke：不适用（本批无管线改动）；main 直跑 172.4ms / 32MiB。

## 会话记录

### 2026-08-16（会话 5：run_single_pass 单遍偏置扩张入口）

- 用户先审查设计文档 `doc/single_pass_bias_design.md` 后放行（含两项决定：
  displacement_nm 做成配置项、[lithography] 段保留）。
- 交付：`main/run_single_pass.py`（每行注释）、`config/single_pass.toml`、
  `tests/main/test_single_pass.py` 8 用例（环双向扩张正/负、产物唯一、
  未处理层、macro_cells 一致、配置校验 3）。
- 实施中确认两个测试几何陷阱并记 findings：孔两维都必须 >2d 才不闭合；
  **边压内部 macro 切线的退化**（切线分裂不处理边落在切线上，两侧拐角
  重建不一致 → 拼合台阶，对验证管线同样成立，属已知边界情形）。
- gcd_45nm 单遍 +5nm smoke：0.80s、唯一产物 output/single_pass/。
- 测试基线 135 → **143 passed**；ruff / compileall 全绿。

## 会话记录

### 2026-08-16（会话 4：审查清单处理 + layer_bbox 收尾）

- `243a5fd`：`LayoutDB.layer_bbox`（原生 `bbox_per_layer`）取代 main 流式
  bbox 扫描（用户授权修改 layout/）；用户修正 config final_layout 路径。
- **审查清单逐项评估**（`doc/macro_core_pipeline_review_issues.md`）：全部
  实质指控成立、无一误报；§5 已被 243a5fd 先行修复。
- `fb80a4e`：契约冻结（macro_size>core、双轮 ±2nm）、空 membership 不变量
  无条件检查、复杂几何矩阵 11 新用例、正逆序双轮完整对照、2/0 对照层、
  阶段 0 零逐 shape 遍历守卫、三个嵌套函数 docstring。
- **新测试暴露并修复两个真 bug**：斜边穿 x/y 切线交点 → 等值穿越点未去重
  产生零长碎段；空 macro（查询框不接触图形）→ 切线分裂在空数组上崩溃。
- 测试基线 116 → **135 passed**；coverage 84%（关键文件 88–95%，未命中以
  防御守卫为主）；smoke 仍 XOR=0。文档/报告/规划同步（本次提交）。

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
- **收尾修复（用户审查后提交 `107fb68`）**：`LayoutDB.layer_bbox` 用原生
  `bbox_per_layer` 取代 main 里的流式 bbox 扫描（用户授权修改 layout/）；
  用户修正 config 的 final_layout 相对路径（最终版图落回 output/）。
  验证：pytest 116 passed，smoke 与改前逐位一致。

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
