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
| 2026-08-17 | 全量（P1-1 修复后）+ 梯度 smoke 复跑 | 411 passed；state0 逐位不变、state1 loss −0.12% |
| 2026-08-17 | 全量（P1-3 修复后）+ 单遍 smoke | 422 passed；单遍 0.78s 产物照常 |

### 2026-08-17（会话：全项目审查 + P1-1 修复）

- 三视角独立审查（架构/既有方法/梯度批判复读）：依赖方向、内存契约、管线
  验证、梯度核心数学全部独立验证通过；发现 4 P1（梯度 owner-only 采样、
  空 macro merge 崩溃 LayerNotFoundError、run_single_pass 校验漂移、simple
  loader NaN 击穿）+ 6 P2。审查报告已交付用户。
- P1-1 修复（用户修复规格 + 用户批准计划）：梯度采样改 membership 制
  （40 条 vs 24 条）、EPE slots 拆分、测试 helper 同步 + 计数断言 + SUM
  累加断言（Spy Adam 捕获 grad、上下半 core 线性叠加）。
- 验证：定向 45+54 绿；全量 411 passed；smoke state0 逐位不变（前向零
  漂移）、state1 全指标一致小幅改善（loss −0.12%、EPE −60 段）。

### 2026-08-17（会话：P1-3 修复）

- 用户裁决本次只修 P1-3（P1-2/P1-4 暂不修）。按用户修复规格 §4 实施：
  _macro_pipeline 抽 MacroCommonConfig + load_macro_common_config（公共
  五段唯一权威解析层）；run_single_pass 删除约 80 行复制的校验，改共享层
  + [iteration] 专属解析；exact_dbu 改从 _macro_pipeline 导入（消除
  入口→入口依赖）。
- 途中两枚 dataclass 组装陷阱（replace 按基类构造、asdict 递归转 dict）
  改用 fields() 浅拷贝，记 findings。
- 新增 11 例回归：layer/datatype/canvas_pixels × float/bool/string 严格
  拒绝、work_dir 未知键拒绝、跨 loader 一致拒绝。
- 验证：定向 49 绿；全量 422 passed；gcd_45nm 单遍 smoke 0.78s 产物照常；
  load_macro_config 重构零行为漂移（macro_pipeline 30 例全绿）。

### 2026-08-17（会话：审查问题 1/2/3/5 修复）

- 批 1（cdd62ad）：CUDA 峰值显式统计设备（cuda_stats_device 传入 reset/
  max，不 set_device）+ _run_mbopc macro_grid 检查真正前置（prepare/模型
  构造之前；plan 后兜底保留）。spy 真跑测试 + 两条"被调用即 AssertionError"
  前置证明。
- 批 2：EPE 阈值统一（metrics 默认 0.5 + simple/gradient 显式传模型
  PrintThresh + 三指标传播测试 ×2）+ lr 超限 UserWarning（合法集合不变、
  参数原样；pytest.warns + 无警告对照）。
- 双 smoke：gradient best loss 逐位不变；simple best_epe 漂移 ±1~15 段
  （nominal 连续值在 [0.499,0.5) 带的探针判定翻转 → 方向序列微调）——
  指标一致化的预期变化，机制与幅度已记 findings。
- 全量 422 → **429 passed**；ruff/compileall 绿。

### 2026-08-17（会话：TestReticle 版图集构建）

- 用户批准计划（含修订：正负板成对 + 100µm 压力级）。批 A：
  build_reticles.py（10 场景 ×2=20 份）+ 回读核对；实施修正两处——
  sparse 需角标记撑 bbox（纯空白不进网格域）、bench 格框改母题平铺
  （30µm=3×2=32×21µm/672 core，100µm=10×7=109×76µm/8025 core），
  文档 §4/§5/§5.7/§5.9/§5.10 同步实测。
- 批 B 四验证全过：① 单元级 2.9s loss −9%；② sparse 3.1s 精确失败于
  merge（P1-2 复现）；③ 30µm 16s 4/4 改善；④ 100µm 176s 16/16 改善，
  CUDA 峰 495MiB 与 30µm 相同（显存与 core 数解耦实证）。
- 批 C：20 GDS + 脚本 + 计划文档提交；test_manual smoke 数据源句更新。

### 2026-08-18（会话：配置系统重构，四批+批 0）

- 用户规格评估：核心原则全部合理；7 项按现状调整获批（GradientConfig
  解读=算法段、ILTConfig 不建、final_cell_mode 保留、以仓库为准、补
  Decimal、补单遍/验证管线、solver DBU 包豁免保留）。
- 批 0（94cd621）：上轮已批的入口合并落地（run_mbopc 单入口、删数量
  约束，用户已自删大半，本批收尾）。
- 批 1（1db593d）：configuration.py（8→最终 9 Config + load_config/
  parse_scalar 声明式映射）+ 20 例 loader 行为测试。
- 批 2+3（8f71b5a）：实施中补 EdgeConfig（两轮遗漏）与 ValidationConfig
  （[iteration] 同名冲突）；四流程全迁移、四旧 Config+五旧 loader 删除、
  5 toml 段名键位迁移（layout 切 bench_30um——gcd_45nm 已删）；四测试
  文件模板同步；途中四枚陷阱（模板正则、切片错位、plan str、幽灵调用）
  记 findings。
- 批 4：全量 444 passed；四 smoke 全绿（simple 47.6s EPE 至 497、
  gradient 用户参数 205s loss −50%、管线 XOR=0、单遍 0.12s）；两报告
  + 手册同步。

### 2026-08-18（会话：common 包集中）

- 用户裁决：lithography 完全不动 → casting.py 取消（as_integer/
  as_finite_float 留 iccad13 嵌套闭包）；用户补充验证要求（四组旧符号
  残留 grep 零命中 + 切断 _mbopc_workflow→_macro_pipeline 的
  atomic_write_json/exact_dbu 旧依赖）并入计划。
- 交付：common/{arrays,io,units,runtime}（五文件含 __init__）；六项
  迁移（_arrays 整体三函数、双原子写、内联 NPZ 归一、exact_dbu、
  resolve_device）；grid.py 相对导入漏检补修（as_points 复活）；
  _macro_pipeline 锚点反转重做。
- 验证：全量 444 passed；smoke bench_30um 基线逐位复现
  （1011/820/497）；四组残留检查零命中；定义仅存 common/。

- common 收口补遗（f8723c1 后续）：MacroProblem.save 的内联 NPZ 原子写
  改 common.io.atomic_write_npz；全量 444 passed；单遍 smoke 0.12s 照常。

- _mbopc_workflow 按算法拆分（用户方案）：simple/gradient 两 workflow +
  save_final_lithography 归 _macro_pipeline + 原文件删除；两 runner 测试
  仅改 import 行（monkeypatch 随别名跟随）；test_macro_pipeline 新增
  save_final stub 直测。全量 445 passed；双 smoke 基线复现。**新纪律
  生效：commit 后直接 push 远端**（用户 2026-08-18 明示）。

- doc_ 正式切换为 doc：8 增量迁移 + 12 副本删除 + 目录改名；
  INDEX 撤试行说明、migration_map 记切换；CLAUDE/task_plan 路径更新。
  全量 445 passed 复跑确认零代码影响。
