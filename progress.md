# MyOPC 迁移进度日志

## 会话记录

### 2026-08-16（会话 8：MB-OPC 审查问题修复）

- 用户提交独立只读审查结论（3 P1 + 5 组 P2 + 测试缺口）；Claude 逐条对照
  代码原文核实：P1 全部属实、P2 七项属实、两项有保留（ProcessCondition
  为设计选择、PatchWriter 受 geometry/ 领地限制）。
- 两批修复：`3725c0e`（P1：insufficient_probes 新停止状态——「无法评价」
  不再冒充「零违规」；save_final 逐 tile 窗口物化 + merge 验证窗口化 +
  五处 ±2^30 魔法框换 layer_bbox；_as_int 拒绝 1.5/true 静默截断）→
  `acfcab0`（P2：reference 复用、EPE 整 batch 回切、无变化提案跳过、
  末轮纯评价、前置校验、tqdm finally、真构造越界用例与差异上界）。
- **实测推翻审查一项建议**：except 收窄到 ReconstructionError——几何退化
  （共线 ring 少于三顶点）以 ValueError 从 KLayout 冒出（−20 位移实测），
  包装需改 reconstruction.py（不修改清单），维持宽捕获并记录证据；
  同时发现 −25/−30 翻转会被 miter 解析成反向合法 ring（守卫不触发），
  测试注释如实记录。
- 验证：全量 341 passed；gcd_45nm smoke 四 macro best_epe 迁移/P1/P2
  三版本逐位一致（7263/5904/5625/4884）。
- 手册与两报告同步审查修复轮；task_plan/findings/progress 更新。

### 2026-08-16（会话 7：Phase 6A 最简 MB-OPC 迁移）

- 用户批准 `doc/opc/mbopc_migration_design.md` 实施（三路事实核对 + 四处原文
  精读：无算法/接口错误；两项裁决——EPE threshold 保留旧值 0.499、tqdm 实测
  已在环境；§3.4 光刻未实施已过时但 API 吻合无需对齐）。
- 实施六批 A–F：evaluation+契约（2 提交）→ points_to_canvas → 共享 macro
  生命周期重构（±2/-2 与 gcd XOR 验证不变）→ 求解器 simple.py → 单/多
  macro 两入口 → 端到端验证与报告。
- 提交：`2b9194a`/`a5509bc`（A）→ `c596d70`（B）→ `71d42ba`（C）→
  `986cbfd`（D）→ `84407e5`（E）→ 报告批次（F，本批）。
- **实施中真实 bug 一枚**：方向写入漏乘步长（±1 DBU 而非 ±step），测试
  `values==2.0` 拦截后修正为 `next += directions*step`；连同 stub 直通模型
  的像素量化陷阱、invalid 测试的参考重建计数陷阱一并记 findings。
- **端到端**（gcd_45nm CUDA 870 tile）：multi 126.0s（4 macro EPE
  37743→7263 等逐轮单调降）、single 126.6s（128227→23440）；独立 macro
  代价量化：single 比 multi 之和小 236 段 EPE、覆盖 XOR 34650860 DBU²。
- 测试基线 224 → **330 passed**；evaluation coverage 100%、simple.py 99%
  （仅两行不可构造的防御 RuntimeError）；两份报告 + 手册 + 规划同步。

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

## 会话记录

### 2026-08-16（会话：梯度 MB-OPC 实施，Phase 6A-G）

- 事实核对规格 `doc/opc/gradient_mbopc_migration_design.md`（1200 行）：autograd 前提
  （forward_many 全链可微 + 6 个既有 backward 测试）、全部接口签名、极性/坐标不变量
  均成立；发现规格 §10.4「只捕 ReconstructionError」与 simple 轮实测（KLayout
  ValueError 穿透）冲突，经用户批准裁决为宽捕获 (ReconstructionError, ValueError)。
- 另三项裁决：P=0 空问题直接 no_owned_segments（不跑 forward）；段法向常驻
  float64[S,2]；doc_/changes/active 副本不动。
- 实施批次 A–D（共享 cache → 梯度求解器 → 配置与入口 → 全量验证与报告）。

## 测试结果速查

| 日期 | 范围 | 结果 |
|---|---|---|
| 2026-08-15 | tests/layout | 27 passed |
| 2026-08-15 | tests/layout + tests/geometry | 49 passed |
| 2026-08-15 | 全量（layout 27 + geometry 25 + opc/input 40 + main 23） | 115 passed |
| 2026-08-15 | gcd_45nm 2×2 smoke | 总 10.6s，最终 XOR 面积 = 0 |
| 2026-08-16 | 全量（mbopc 迁移 + 审查修复轮后） | 341 passed |
| 2026-08-17 | 全量（梯度 MB-OPC 后） | 410 passed |
| 2026-08-17 | gcd_45nm 梯度 smoke（CUDA，iterations=1） | 41.61s，四 macro best_state=1，loss −10.1% |
