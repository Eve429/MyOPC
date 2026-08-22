# MyOPC 迁移进度日志

## 2026-08-21（会话续：环带恒暗修订）

- 用户裁定后批次 1/2/3 交付：ILT 双边界参数（17e9e2c）→ MB-OPC 暗界
  贯通与搁置缺陷关闭（0e7d0c9）→ 文档与双极性验证（本批）。

## 2026-08-21（会话：处理框 field_box/field_size）

- 用户报告版图 2.048² 大于 layer 1² 的规划问题并给出方案雏形；三轮
  AskUserQuestion 裁定（环带极性外推/双写法配置/两路径同批）+ 负板边缘
  语义对账后计划批准。
- 批次 1/2/3：config+resolve+12 单测 → 双路径接入+4 e2e（双极性环带
  target 三层判别、输出留 field、MB-OPC 段数和一致）→ 文档（contracts
  场边界节扩写/development_manual 8b/dataflow 两文件）。

## 2026-08-21（会话：ILT 三方法审查 + P3 骨架合并）

- 用户要求三方法审查（正确性/架构/性能），产出报告（cProfile 实测：
  backward ~60-65%、forward ~30%、参数化 <5%；推翻画布冗余与 np.add.at
  两个疑似热点）。用户裁定执行 P3（骨架合并）。
- 批次 0/A/B/C/D：golden 基线 29 case → _skeleton + Simple → LevelSet →
  CurvMulti → states_total + 文档；每批 golden 逐位对比 + 门禁 + 独立
  commit（3d4c99c/14e0373/2b56baf/本批）。

## 2026-08-21（会话：CurvMulti ILT 审查与交付）

- 用户要求审查 CurvMulti 规格；按既定模式产出审查报告（3 事实错误/5 契约
  缺口/2 决策问题），AskUserQuestion 三项裁定后批次 0 代笔修订规格。
- 批次 A/B/C 依次交付求解器、入口配置、文档归档；每批测试先行、门禁全绿、
  独立 commit；用户 WIP（_ilt_workflow 排版、pyproject）以显式 pathspec
  排除。全量 660 passed + 1 skipped。

## 2026-08-20（会话：入口与适配器合并）

- 用户裁定四对入口+适配器全并 + CLAUDE.md 过期入口名同步（AskUserQuestion
  两问）。合并为纯符号搬移；4 个测试文件各改 1 行 import；monkeypatch 宿主
  全在共享模块，无需额外改动。四入口 smoke 直跑 + 全量门禁后单 commit。

## 2026-08-20（会话：负板透光环修复）

- 用户报告负板 macro 扩张最外圈变透光；经查为新树丢失 00_PAST field_box
  契约的迁移回归（详见 findings 2026-08-20 节）。
- 计划模式评审：两问（修法/范围）经用户裁定——transmission 层置零、仅 ILT
  像素路径；MB-OPC edge 路径同缺陷另行立 CHG。
- 实施：回归测试先行（26 红含 TypeError 证明契约缺口）→ prepare 增必填
  layout_bounds + 置零 + 两类校验 → _ilt_workflow 传 bounds → 三个测试
  helper 补参数；contracts/ilt.md 契约行、test_manual 609。全量 609
  passed + 1 skipped。

## 2026-08-19（会话：Gradient MB-OPC EPE loss 更新设计）

- 用户要求参考 DiffOPC，为现有 gradient MB-OPC 设计新增 EPE loss 的更新方案。
- 本轮先做源码、测试、现有报告与 DiffOPC 证据核对，只更新设计文档和必要工作记录，
  不修改 gradient 算法实现。
- 已启用 planning-with-files 维护设计证据与阶段状态，并按 self-improvement 复核已有
  midpoint、DBU 单位和跨 core membership 梯度经验。
- 已定位当前梯度求解器、测试、配置和只读 DiffOPC 归档；确认现有 EPE 是参考边
  inner/outer 二值探针诊断，训练梯度则走当前重构中点的 midpoint STE。
- 已提取归档连续 EPE：有效参考段的 inner 欠曝、outer 过曝分别用 squared ReLU；
  owner 唯一计分、与其他 batch loss 同次 backward。归档分母和 H/V/斜边适用性仍需对照
  官方 DiffOPC 主源确认，不能直接采用。
- 已定位 NVlabs/DiffOPC 官方仓库与 ICCAD 2024 论文；开始以一手源码核对 EPE surrogate，
  不把旧归档适配报告当官方契约。
- 官方仓库浅克隆完成（commit `bdc6e72`）；确认官方 EPE 是 H/V 中点法向线上的
  squared target-wafer error 经 sigmoid 聚合，corner/非 H/V 被忽略，半径 15 pixel 硬编码。
  该实现只作为设计来源，不原样复制。
- 已从作者主页论文 PDF 提取 eq. (6)-(16)，确认官方 loss/梯度链路；由于系统缺
  `pdftotext`，在 `/tmp` 安装独立 `pypdf` 后完成只读提取，没有改项目依赖。
- 已完成与当前 owner/membership/macro 屏障的语义映射：设计采用唯一 owner 段的
  reference 法向 profile、任意角度向量化采样、归一化零基线 penalty，并复用同次 backward。
- 已核对配置解析、metrics/summary、runner 与结果持久化调用点：新增字段可用 dataclass 默认值
  保持旧 TOML 兼容，NPZ 无需改版；EPE 对齐约束应在求解器入口校验，避免扩大配置层依赖。
- 已新增独立 draft 规格
  `doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md`，完整冻结
  EPE 公式、profile 坐标、owner/membership 梯度、归一化、状态/best、接口、异常、性能、文件
  计划、测试矩阵、分阶段本地提交和 approval gate。
- 终审按当前模板补齐 architecture/data ownership/persisted artifact/verification/delivery 章节；修正
  四权重 EPE-only 校验、边界 pixel center、公开 record 默认兼容及真实 completed report 路径。
- `git diff --check` 与新规格独立 whitespace check 通过。本轮只设计、未修改生产代码和测试，
  因此未重复运行 pytest；沿用本日 `450 passed, 8 skipped` 的生产基线。未提交、未推送。

## 2026-08-19（会话：Simple ILT 规格语义修订）

- 用户指出草案把 core 错当独立优化问题，并要求改为 macro 唯一 pixel 参数、
  ownership 唯一计 loss、simulation context 内同 macro 参数全可微、跨 core
  梯度累加、macro 同步 step 与 macro 级 best。
- 同时要求修正 fractional coverage 的 sigmoid 反函数初始化，并确认/冻结
  core 尺寸必须为 pixel 整数倍，删除 partial-core-pixel 处理语义。
- 本轮只更新 Simple ILT 规格及必要工作记录，不修改生产实现。
- 已完成规格前半部最小修订：补当前网格边缘事实；REQ-005..009 改为三域
  语义、logit coverage 初始化、macro 同步梯度屏障、macro best 与 CPU/GPU
  分块状态；INV-001/002/004 同步改写。
- 已同步 Architecture/Data Contracts/Interface：数据流和 call graph 改为
  `optimize_simple_macro` 内按 state 遍历 core batch；`ILTBatchResult` 改为
  CPU `ILTMacroResult`，best state 为 macro 标量；PixelMacroProblem 改提供
  macro trainable index canvas，不再逐 core place best。
- 已重写 §10.2 的状态伪码但保留章节结构：state0 用 logit 恢复 coverage；
  每个 macro state 遍历全部 core batch、scatter-add 梯度、屏障后一次 CPU
  SGD；§10.3/§11/§12/§13 同步到 macro best、实际 cuts 整除和批量显存上界。
- 已同步文件计划、TEST-002/005..010、测试矩阵、traceability、AC、兼容性、
  DEC-001/004、开放问题、实施自由度、限制与 approval gate；Revision 增至
  0.2 draft，保留 0.1 历史记录。
- 第一轮差异审读后补齐 fractional transmission 文字、trainable index 的
  row-major 同像素同索引契约，并修正 Scope/PERF 排版残留。
- 已刷新规格 baseline 为当前 `540a012` 和 Linux CPU 回归
  `450 passed, 8 skipped`；生产源码与测试相对 HEAD 无差异。
- 终审补充 state N 只 forward/loss/best、不 backward/step，明确 core canvas
  使用 `target_u8/255`；旧语义残留仅存在于 0.1 历史或 rejected 说明。
- `git diff --check` 通过；本轮未运行新测试，因为只修改规格与工作记录，沿用本日
  已完成的全量 CPU 基线。未修改生产代码、`layout/`、`geometry/`、`00_PAST/`，
  未提交、未推送。

## 2026-08-19（会话：项目现状复核）

- 用户要求先理解当前项目，为后续方案设计或开发建立基线。
- 已确认工作树开始时无未提交改动；现有规划记录显示迁移主线已完成至
  simple/gradient MB-OPC，下一规划阶段为 ILT 独立设计评审。
- 本轮只读检查当前实现、测试、配置与文档，不修改 `00_PAST/`、`layout/`、
  `geometry/` 或生产代码。
- 直接执行 `pytest -q` 因 WSL PATH 无 pytest 失败；这不是用例失败。已按项目
  手册定位 `/mnt/d/app/miniforge/envs/myopc/python.exe`，后续改用该解释器。
- 指定 Windows 解释器在当前 Linux 执行器报 `Exec format error`，说明工具环境
  无 PE 互操作；正在查找可用的 Linux 项目环境。
- 找到 `/home/wzh/miniconda3/envs/myopc312/bin/python` 并完成全量 pytest：
  **450 passed, 8 skipped in 87.86s**；跳过项均为当前环境无 CUDA。
- ruff 显式范围检查与 compileall 均通过；当前 CPU 基线已建立。
- 已精读共享 macro/MB-OPC workflow 与 simple/gradient 求解器，并用 AST
  import 图核对依赖方向；实现与核心架构文档总体一致。
- 项目基线复核完成：已确认现有能力、运行入口、关键数据/状态不变量、ILT
  草案依赖顺序、已知局限与两处文档/契约漂移；本轮未改生产代码、测试、
  `layout/`、`geometry/`、`00_PAST/` 或用户 GDS，未提交、未推送。

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

- resolve_*_config 三函数集中（用户方案 + 第 4 消费方单遍）：四文件
  换算/校验/构造全收口 configuration.py；跨段校验时机统一后移到
  prepare 后（行为变化记 findings）。全量 445 passed；三 smoke 基线
  逐位复现。commit 后 push。

- MB-OPC 公共 workflow 上提（用户 P1 方案，两点修正获计划批准：不建
  MacroSolveOutput、资源统计上提公共层）：批 1 gradient 先行（f4b472a）
  + 批 2 simple 切换（1f1ef96）+ 批 3 文档与记录。MBOPCMethod 七字段
  注入、两 adapter 各 ~125 行、入口零改动；测试仅迁 merge patch 宿主
  （×2）+ simple 新键断言。全量 445 passed ×2；双 smoke 逐位复现
  （gradient 0.069138/501MiB；simple 1596/1011/820/497）。
  发现既存损坏：mbopc_single_macro.toml context_nm=1024 超画布上限
  （用户 7b3ca1e 改动，未动待裁决）。

- 用户 P2 两项：solve 包装上提公共层（6dcd08f，MBOPCMethod 字段改
  optimize_macro，测试注入改 dataclasses.replace+METHOD 重绑定）与
  外层进度条 try/finally 收尾（053886e，回归测试双向验证）。两 adapter
  ~85 行纯差异钩子；446 passed；双 smoke 逐位复现。

- 注释整改四批（2a1fa87/f156377/3ead527/批 4）：gradient 样板 →
  opc+common+litho 清理 → main 全目录 → AGENTS.md 规则改写 + iccad13
  §引用清理。tokenize 逻辑行口径 + AST 等价校验 + 残留归零；446
  passed；双 smoke 逐位（1596/1011/820/497、0.069138/501MiB）。

- TestReticle 负板重制（用户发现正负同文件）：_opaque 改为包围盒补区，
  脚本/计划文档同步，20 份重生成（clear 几何等价恢复原字节），逐对
  XOR==0 验证 + 负板单遍管线消费性 smoke 通过；Region 惰性挂接陷阱
  记 findings。注释整改四批同日完成（见上）。

- gradient 采样中点 P1 修复三批（ddf6a4a/04741d4/记录批）：
  reconstruction 下沉产出实际中点（几何单测 4 例解析验证）→ gradient
  消费已发布中点（spy 成员关系 + 非均匀 FD 回归测试；FD 不判别旧新
  如实记录）→ findings 记录。gradient smoke 新基线 0.134467/
  invalid_geometry（守卫按设计工作）；simple 逐位不变；452 passed。

- pyproject 落档 A（[tool.ruff] extend-exclude 固化 00_PAST 与
  geometry/contour.py 豁免，刻意不配 format 段）；requirements 补
  pytest 9.1.1/ruff 0.16.2 门禁工具并复核运行时版本零漂移。裸
  ruff 门禁命令（无 --exclude）经配置直接通过；452 passed。

- gradient 结构重构（用户三段方案，收口 midpoint 修复时立案的结构待办）：
  optimize_gradient_macro 由 308 行拆为 _prepare_macro_context（静态
  上下文 + _GradientMacroContext）/_evaluate_state（批评价 +
  _GradientStateEvaluation，只 backward）/ _take_optimizer_step（step +
  clamp + 候选重构，返回 (Region, midpoints) | None，异常上抛）+ 主函数
  ~110 行编排；_GradientCandidate 按用户裁定取消。逻辑逐字迁移零算法
  变化（详见 findings「gradient 结构重构事实」节）；既有 45 例零改动，
  新增 TestStructuralSplit 4 例。
  验证（WSL ~/miniconda3/envs/myopc312，修正早前"会话内无法跑门禁"的
  误判）：compileall/ruff 通过（ruff 抓出 _evaluate_state 漏传 model 的
  F821 真 bug，补参数修复）；458 passed；CPU A/B（gcd_30um 单 macro、
  iterations=6、870 core）重构前后逐 state loss/l2/pv/epe/disp 精确相等
  （loss 0.149844393→0.120088223）、best_state=6、stop=iteration_limit、
  best_displacements npz 逐位一致。


- Simple ILT 全量实施（CHG-20260818-simple-ilt，规格 Rev 0.2→0.3→1.0）：
  事实核对发现 §14 漏 run_single_pass（Rev 0.3 补录）；五阶段 commit
  5ad8ac0/54ab866/1539b6f/fefaea8/bdf86ac + 文档报告批；共享层修复 merge
  空 macro 候选容忍；测试 67 新增（pixel 20 + ilt 30 + runner 12 +
  GridRuntime 5），全量 525 passed；smoke corners_unit CUDA 1.90s
  best_state=1。全部在 WSL myopc312 自跑（CUDA 可见）；push 因会话网络/
  凭据不可用延后（本地 6 commit 待推）。开发/测试报告见
  doc/changes/completed/CHG-20260818-simple-ilt/。


- Simple ILT P1-1 修复（用户裁决：OpenILT 2T−1 初始化取代 logit+eps）：
  新 change CHG-20260819-simple-ilt-openilt-init（supersedes REQ-006 初始化
  子句）；simple.py init 块替换 + 六处测试同步 + 纯对齐几何真模型回归
  （max|Δp|≈3.9e-4）；smoke 重调 step_size=1.0（7952→6233，−21.6%）；
  全量 526 passed；commit ebce389 + 记录批。WSL myopc312 自跑；push 仍
  待网络可用（累计 8 commit 待推）。


- Simple ILT P1-1 关闭三件套（用户审查结论）：① context 统一修复（4c5f5f1，
  固定 context = σ(β(2T−1))，跨宏 seam 测试锁定，smoke 7880.69→6162.49/
  binaryL2 2875）；② P1-3 性能修复（aa583a5，trainable_index_canvas 索引块
  窗口化，O(宏像素)/调用 → O(core 窗口)）；③ 文档同步（CHG-20260818 规格/
  测试报告加【已取代】标注、CHG-20260819 Rev 1.1 补 REQ-D 与两报告、
  contracts/ilt.md transmission 单一定义、测试旧注释清理）。全量 527
  passed；三个独立 commit。


- Simple ILT P1-1 Rev 1.2（用户边界审查：padding 不得 sigmoid）：新增
  PixelMacroProblem.context_valid_canvas()，训练/终评/镜像统一三值语义，
  padding 判别测试（window 12px core）；smoke 6162.49→6162.66 为缩短
  core 环消除的语义变化；双真源文档清理（Decisions 旧条目 +
  CHG-20260818 §10.2/TEST-008 标注）。全量 528 passed；fix + docs 两
  commit。P1-3 按用户复核结论关闭。


- Simple ILT P1-4/P2-1/契约措辞（用户算法审查三项）：索引域全链 int64
  （P1-4，防 2^31 构造期溢出）+ curvature×context≥1px 入口联合约束（P2-1）
  + REQ-B 二值一致性限定 threshold=0.5；测试 +1（约束双向）+ dtype 断言；
  全量 529 passed；fix + docs 两 commit。


- Gradient EPE loss 规格修正（用户 P1/P2）：profile 聚合 mean→sum（Q 不变性）+
  segment 归约等权→参考长度加权（切段不变性）；DEC-002 反转 + DEC-007 新增 +
  TEST-013/014 + AC-011，基线刷新至 08f4866；Rev 0.2，仍 draft 待批。


- Gradient EPE loss 阶段 C 收尾（本会话接手）：核实 A/B 实现完整（定向
  123 passed）→ PERF-004 对照 smoke（ON 183.2s vs OFF 177.1s，全部指标
  改善、资源增量符合 O(O·Q) 上界）→ 迭代算法自审（无饱和、连续/离散
  同降、EPE 占比 85% 记录）→ contracts/mbopc.md Gradient 节 + 两手册 +
  规格 Rev 1.1 移 completed + 两报告 + 三记录。全量 545 passed。


- 数据流文档重构（用户方案）：dataflow.md → dataflow/ 目录五文件
  （index + 总管线/simple mb/gradient mb/simple ilt），每文件函数级流向
  + 伪代码双表示；顺带修正旧文档 load_macro_config 过期事实与补齐
  gradient 求解层；INDEX/active 规格引用同步。doc-only，全量门禁照跑。

- LevelSet ILT（CHG-20260818-levelset-ilt）四批次交付：A 终评 context
  策略化解耦（Simple 零回归 bit-identical）→ B SciPy SDF/halo STE/宏 Adam
  求解器（42 测试，含 float64 镜像与 Adam 逐位复现）→ C 入口/配置/smoke
  （15 测试，两 runner 直跑）→ D contracts/system/dataflow/两手册/两报告/
  active→completed。审查三观察（scipy 前置、门禁 scope、smoke 值）随批
  处理；步长与 0 等值线事实入 contract。全量 603 passed + 1 skipped。

- 50nm 定尺寸测试掩膜组（计划评审通过，两项用户裁定 + 边框内缩）：
  扩展 build_reticles.py（档位表 dense/mid/loose、结构族 helper、组 1
  综合采样与组 2 复刻式 builder、设计区框成对写出、--p50 CLI、写前
  包络/互不接触 + 写后读回/互补终检自检），生成 p50_1024/p50_2048 ×
  3 档 × 正负板 12 份；双板 target_u8 逐位等价、环带两极性恒 0、ILT
  单宏 + MB-OPC 四宏双极性 smoke。规格入 reticle_build_plan.md §10，
  CLAUDE.md 测试与数据节同步（含 gcd_45nm→gcd_30um 过期项修正）。

- 入口改名同步（用户 5f2a964 改名四入口后补全）：4 个 runner 测试 import
  与直跑路径、4 个入口 usage 自引用、CLAUDE.md 命令块（gradient 改新名
  并补三个 ILT 入口）、10 份活跃文档（dataflow 五文件/system/data_model/
  contracts.ilt/两手册）；doc/changes、doc/archive、.planning 与 findings
  历史条目按规则保留旧名。全量门禁恢复 683 passed + 1 skipped。

- MB-OPC（simple+gradient）全链路三角度审查（用户指派，范围=整条调用
  链）：入口/编排/配置/输入层/双求解器/评价/光刻消费面/缓存约 3700 行
  逐文件通读 + 微基准 + sparse_6um 实测，报告入 doc/review/（13 项不变
  量核对表、C/A/P 分级发现与决策清单、250 例测试覆盖盘点）；审查 C1
  环带旧注释单独修复（c948166）。随后全库 doc 过期信息清扫：计数/迁移
  进度/门禁范围/gcd_45nm 历史标注/MacroProblem v2 与 dark_box 契约/
  reticle 计划 P1-2 与 config 提醒关闭。全量 683+1 门禁绿。

- MB-OPC 审查 A 项全修（用户指派）：A1+A2 骨架化 _batching（钩子注入保
  monkeypatch 锚点，golden A/B 12 例逐位一致 + 全量 683+1 + 双入口 smoke）
  → A3 CLI 资源打印对齐 + A4 分段公式模块归位 → 文档同步（contracts
  pack 参数、dataflow 双文件含 np.add.at 旧描述修正、审查报告处理状态
  标注、两手册）。A5 维持观察项判定并说明理由。

- 环带几何方案（用户裁定"去除现在的方案"，两批交付）：批次 1
  （8e349f3）——输入层/求解器/编排全链切换 + 三处版本 bump + 测试
  重构（-4 旧例 +7 新例）+ golden A/A 新基线 + 用户场景双极性 e2e
  （负板输出带真实铬框）；批次 2——contracts（edge v3/opc_input/ilt
  场边界节）、development_manual §8b、CLAUDE.md 状态行、test_manual
  （686 + 补铬组描述）、reticle plan §10.5、dataflow 同步。全量
  686 passed + 1 skipped。

- 双求解器整改（用户发起：两 optimize 函数同义异名 + pack/ctx 差异
  + optimize_macro 名不达意；三决策经 AskUserQuestion 确认全取推荐项），
  三批交付：批次 1（94b9507）结构收敛——组批单源 iter_core_batches/
  upload_eval_batch + _GradientContext 瘦身以 pack 为唯一共享源（行为
  不变，686+1skip 前后一致）；批次 2（9b70750）命名统一——
  optimize_simple_macro/evaluate_state/SimpleMBOPCIterationRecord/
  state_index/best_state_index + _prepare_gradient_context，产物键对齐
  gradient 入口（result.npz v2）；批次 3 文档——contracts/mbopc 新增
  静态分层小节、dataflow 双文件、development_manual、data_model（含
  环带批漏网的 v2/dark_box 陈旧行）、test_manual、contracts/edge。
  ADR-004 为点时决策记录保持原文。全量 686 passed + 1 skipped。

## 2026-08-22（会话：光刻 PNG 留档三改）

用户审查 gradient MB-OPC p50 产物时提出三件事：保存的光刻 PNG 上下
颠倒；希望输出时同时保存原始（未 OPC）光刻对照；main_test_lithography
改为 GDS→光刻结果工具。四批落地：
1. d7aa0a6 fix——save_final_lithography 补 I/O 边界 flipud（非对称 GDS
   stub 回归，修复前必红）；
2. 47b12ae refactor——抽参数显式内核 save_lithography_pngs（行为不变，
   687+1 等价证明）；
3. 8b9777b feat——两工作流 save=true 时对源版图再留档一套
   （final_lithography_source，显式 top_cell；summary 记
   source_lithography_tiles；双顶层 GDS 回归守卫）；
4. 本批 feat——main_test_lithography 重写为 argparse CLI（<gds>+9 flag，
   校验便宜前置/模型最后，main(argv) 进程内可测；TestMainEntry 原地
   重写 + TestEntryValidation 4 例）。全量 695 passed + 1 skipped。
   文档：CLAUDE.md 入口行与计数、test_manual §4/§5/套件表、
   contracts/mbopc、development_manual、data_model、dataflow×5。

## 2026-08-22（会话：五条风格规则全仓迁移）

用户审查本会话代码后定五条风格规则（导入无注释/函数注释简明/步骤级
注释/关键算法加注/ruff format 120 列），范围"除 00_PAST 无豁免"。两批：
15b8382 全仓机械迁移（68 文件 format、161 处导入注释、contour.py 豁免
移除、pyproject/AGENTS/CLAUDE 规则与四件套门禁同步）；429e52d 本会话
代码按规则 2/3/5 精修（9 文件中 5 个有实质改动）。695 passed + 1
skipped 前后一致。
