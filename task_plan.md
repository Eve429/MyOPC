# MyOPC 亲手迁移任务计划

## Goal

把全 AI 编写的旧代码库（归档于 `00_PAST/`，只读参照）按依赖顺序亲手迁移/过滤到新树，
使每个模块都被项目所有者理解并拥有；每个批次闭环 = 迁移 → 清理 → 测试 → 演示 → 本地提交。

## Next Step

**Phase 6 剩余（ilt 后三方法）**：Simple ILT 与像素管线完成；Gradient
EPE loss（CHG-20260819）完成；LevelSet ILT 完成（2026-08-20，
CHG-20260818-levelset-ilt：SciPy SDF/halo STE/宏 Adam）；CurvMulti ILT
完成（2026-08-21，CHG-20260818-curvmulti-ilt：多尺度控制网格/wafer
曲率，660 passed）；Multilevel 规格待批。
随后 Phase 7（旧 main 入口评审 + 收尾审计）。

## 2026-08-22 补充：光刻留档工具化（非阶段项）

- [completed] 修复最终光刻 PNG 上下颠倒（I/O 边界翻转缺失）+ 回归测试。
- [completed] 迭代工作流 save=true 时同批留档源版图对照
  （final_lithography_source/，summary 记 source_lithography_tiles）。
- [completed] main_test_lithography.py 由合成演示改写为 GDS→光刻结果
  CLI（逐 tile PNG+manifest，与管线同一内核；argparse 首例）。
  全量 695 passed + 1 skipped。

## 当前会话：项目现状复核（2026-08-19）

- [completed] 复核仓库结构、入口、配置、依赖与测试组织。
- [completed] 对照开发手册、专项报告和当前实现，区分已确认能力与待确认事项。
- [completed] 形成后续方案设计与开发可直接使用的项目基线摘要。

## 当前会话：Simple ILT 规格语义修订（2026-08-19）

- [completed] 全文核对现有规格与用户指出的 core/context、coverage 初始化、
  macro 同步迭代、macro best 和像素整除问题。
- [completed] 核实现有 grid/raster 对 core 与 pixel 对齐的真实约束。
- [completed] 最小范围更新 Simple ILT implementation spec 的关联章节。
- [completed] 搜索旧语义残留并完成文档差异审计。

## 当前会话：Gradient MB-OPC EPE loss 更新设计（2026-08-19）

- [completed] 核对当前 gradient MB-OPC 的参数、采样、loss、梯度、状态和测试契约。
- [completed] 定位并精读 DiffOPC 的 EPE loss 参考实现与论文/文档证据。
- [completed] 评估 EPE loss 与现有 midpoint STE、membership、ownership、单位和批处理的兼容语义。
- [completed] 编写最小更新规格，明确配置、公式、接口、性能、异常和回归测试。
- [completed] 完成差异、依赖、旧契约残留和未决问题审计，提交设计结论供用户评审。

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

### Phase 5A: lithography — Status: complete
- 依据 `doc/lithography/lithography_migration_design.md`（用户 2026-08-16 批准）：
  只迁 ICCAD13 Hopkins 模型（一个具体类 + 四资产 + 原生 autograd 前向 +
  main 验证入口）；不迁 Protocol/手写 backward/CT/TorchLitho/resize 分支。
- 实施四批提交：`6338710`（配置/资产）→ `8773e37`（可微批量前向）→
  `5f0747a`（main 入口）→ 报告批次（D）。
- 交付：`lithography/`（三公共类型）+ `main/main_test_lithography.py` +
  `tests/lithography/` 81 例（coverage 100%）；CPU 三工艺角 sums 与
  OpenILT 基线**逐位复现**；CUDA（GTX 1650）parity 1e-4 通过；
  DLL 缺失实际复现后按设计 §11.7 授权加回最小 Windows 修复。
- 全量 143 → **224 passed**。

### Phase 5B: evaluation — Status: complete（并入 Phase 6A 实施）
- 153 行纯消费者层；由 mbopc 迁移设计 §8.2 定义最小子集（L2/PVBand/EPE，
  不迁 shot），随 Phase 6A 阶段 A 迁移并配 25 用例（coverage 100%）。

### Phase 6A: opc.iteration.mbopc（最简 MB-OPC）— Status: complete
- 依据 `doc/opc/mbopc_migration_design.md`（用户 2026-08-16 批准）：
  固定步长、EPE 驱动离散边移动；evaluation + LithographyModel 契约 +
  points_to_canvas + 共享 macro 生命周期 + 求解器 + 单/多 macro 两入口。
- 实施六批提交：`2b9194a`（契约）→ `a5509bc`（evaluation）→ `c596d70`
  （居中坐标）→ `71d42ba`（共享生命周期重构，±2/-2 与 gcd XOR 不变）→
  `986cbfd`（求解器）→ `84407e5`（两入口）→ 报告批次（F）。
- 交付：evaluation/（25 例 100%）+ opc/iteration/mbopc/（53 例，simple.py
  99%）+ 两入口（23 例）；gcd_45nm CUDA 实测两入口各 ~126s，EPE 逐轮单调
  下降；独立 macro 边界代价量化（single 比 multi 之和小 236 段 EPE，
  覆盖 XOR 34650860 DBU²）。
- 全量 224 → 330 → **341 passed**；偏差与取舍见
  `doc/opc/mbopc_development_report.md`。
- **审查修复轮（2026-08-16）**：用户独立只读审查提出 3 P1 + 多项 P2，
  逐条核实全部属实后两批修复——`3725c0e`（insufficient_probes 停止状态 /
  几何流式与 layer_bbox 替换 ±2^30 魔法框 / _as_int 严格整数校验）→
  `acfcab0`（reference 复用、EPE 整 batch 回切、无变化提案跳过重复评价、
  末轮纯评价、前置数量校验、tqdm finally、真构造越界用例）。审查一项
  建议不采纳（except 收窄）有实测反证并记录。gcd_45nm smoke 三版本
  best_epe 逐位一致。

### Phase 6A-G: 梯度 MB-OPC（CHG-20260816-gradient-mbopc）— Status: complete
- 依据 `doc/changes/completed/CHG-20260816-gradient-mbopc/implementation_spec.md`（用户 2026-08-16 批准计划，
  含四项裁决：几何退化宽捕获 ReconstructionError+ValueError、P=0 空问题直接
  no_owned_segments、段法向常驻 [S,2]、doc_ 副本不动；规格 Revision 0.2 同步）。
- 实施四批提交：`42bf6f3`（共享 TargetCanvasCache 抽出）→ `17ff75c`
  （gradient.py：midpoint STE 2·g_mid + owner-only Adam + 三项连续 loss，
  44 例）→ `c3e59bc`（load_gradient_config/solve/run + 单入口任意 macro 数
  + summary RSS/CUDA，25 例）→ 报告批次（D）。
- 交付：gcd_45nm CUDA smoke 41.6s（2×2、iterations=1：四 macro 全部一轮
  更新即改善，loss −10.1%、EPE −9%、L2 −10%；PVBand 连续分量 +2.9% 与
  simple 轮结构性观察一致；CUDA 峰值 496 MiB、RSS 1244 MiB）。
- 全量 341 → **410 passed**；state0 EPE 与 simple baseline 逐位一致。
- 偏差与裁决记录：`doc/opc/gradient_mbopc_{development,test}_report.md`。

### Phase 6B: opc.iteration 剩余（ilt）— Status: in_progress
- **Simple ILT + 像素管线完成（2026-08-19，CHG-20260818-simple-ilt）**：
  五阶段实施（GridRuntime/writer 解耦 → opc/input/pixel → ilt/_common+simple
  → _ilt_workflow/入口/配置 → 文档报告）；525 passed；新增共享层修复
  （merge 空 macro 候选容忍）；**P1-1 修复（2026-08-19，ebce389）**：初始
  化切换 OpenILT 2T−1 方案，恢复内部 0/1 像素可优化性（斜率 4.8e-7 →
  0.0707），smoke 重调 step_size=1.0；Rev 1.1 收尾：固定 context 统一
  σ(β(2T−1))（跨宏 seam 一致性测试）、trainable_index_canvas 窗口化
  （P1-3）、历史文档【已取代】标注；Rev 1.2：数值 padding 三值语义
  （context_valid_canvas，padding 恒 0）；Rev 1.3：索引域 int64（P1-4）
  + curvature×context 联合约束（P2-1）+ REQ-B 阈值措辞限定。终基线
  6162.66，529 passed。LevelSet/CurvMulti/
  Multilevel 规格已在 changes/active 待续。

### Phase 7: main 入口 + 收尾审计 — Status: pending
- 旧 3357 行接线层已被 `main/run_macro_pipeline.py` + MB-OPC 两入口取代大半；
  剩余入口待评审。最终全量回归 + 交付审计（未用函数/重复实现/异常入口）。

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
| 当前 WSL `PATH` 无 `pytest` | 3 | 找到 Linux conda 环境 `myopc312`，全量 450 passed、8 CUDA skipped |
| 更新记录的补丁上下文不匹配 | 1 | 读取实际中文原文后使用精确补丁更新 |
| 断言脚本「关闭守卫未抛 ClosedLayoutError」 | 反复调试 | 根因：lambda 漏调用括号（`db.cell_hierarchy` 非 `db.cell_hierarchy()`）；已随重写消除 |
| read_glp 收到 tuple 层映射 AttributeError | 1 | 测试违反契约；tuple 规范化只在 LayoutDB.open 入口做，修正测试传 LayerSpec |
| 切线分裂后 SegmentBatch 校验失败（非一维） | 逐层 spy 定位 | 新批次 edge_ids 误传原始段号而非数学边号；修正为 `segments.edge_ids[boundary_seg[~last]]`（回归用例：test_grid/problem 系列） |
| np.where 全分支求值致穿越索引越界 | 1 | last/first 行索引先夹回有效范围；被夹位置的值不会被选中 |
