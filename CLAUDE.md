# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

MyOPC：面向 OPC（光学邻近效应校正，半导体光刻）的层级版图与几何基础库，含完整可运行管线（GDS → Macro–Core 两级网格 → ICCAD13 光刻仿真 → EPE 驱动迭代 → 最终 GDS 合并）。旧系统（AI 编写）已整体归档。

## 当前工作状态：用户自迁移（每个会话必读）

**旧代码库整体归档于 `00_PAST/`（只读参照）；仓库根从零重建，分支 `migration`，全量 683 用例绿。**

- **迁移进度**（细节见根目录 `task_plan.md`）：layout ✅、geometry ✅、opc.input(+edge) 重构为 Macro–Core 管线 ✅、lithography（ICCAD13，数值与 OpenILT 逐位一致）✅、evaluation（最小子集）✅、opc.iteration.mbopc（simple/gradient）✅、opc.iteration.ilt（Simple/LevelSet/CurvMulti 三方法 + 公共骨架）✅、main 编排层（验证管线/单遍/MB-OPC 双入口/三 ILT 入口）✅、配置统一 + field 处理框（field_box/field_size）+ 环带恒暗语义 ✅。**待迁移**：diffopc（需独立设计评审，不建空目录）、收尾审计。
- **`00_PAST/` 只读纪律（用户明令）**：不修改归档内任何内容；允许复制出来到新结构改写；确需修改归档必须先请示并获明确批准。
- **既定工作模式**（lithography 与 mbopc 两轮先例）：用户写设计文档 → Claude 事实核对（对照 00_PAST 原文与当前代码，报告偏差与疑点）→ 用户批准 → Claude 按设计分批本地实施（每批测试先行、门禁全绿、独立 commit）。日常答疑与验证照旧；未被指派时不主动写新模块。
- **权威参照文档**：旧系统调用图 `00_PAST/doc/function_call_architecture.md`（第 2–4、10 节）；新系统各批设计与报告在 `doc/`（macro_core/、lithography/、opc/）。
- 早期过滤决策点已全部处理：hierarchy.py 已删；`render_layout_region` 保留（有直接回归与演示）。

## 绑定规则

`AGENTS.md`（2026-08 重写版）是最高开发手册，与其他规则冲突时以它为准。要点：

- 所有 docstring 与注释用中文且以中文词开头，注释解释 why（坐标方向、数据不变量、性能路径、内存上界、边界归属、异常原因），置于紧凑逻辑块之前。
- 未经用户逐次确认不得修改 `layout/` 与 `geometry/`——**对用户新迁移出的同名目录同样适用，它们是用户手工迁移的领地**。
- 禁止自动格式化，保持紧凑排版；质量门禁 = ruff 规则检查 + compileall + pytest。
- 每个 bug 修复必须带可复现回归测试；修复后搜索调用点，删除仅服务旧 bug 的函数/包装/分支/变量。
- 禁止投机抽象：新接口必须有当前调用方。
- 本地 commit 为主，未经明确授权不 push；`TestReticle/*.gds` 用户回归数据与 `Test/klayout.ipynb` 不在修改范围。

## 环境与命令

- 解释器固定为 myopc conda env：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12；依赖 klayout / numpy / pillow / psutil / torch / matplotlib / tqdm，版本见 `requirements.txt`）。
- Bash on Windows：路径用正斜杠。
- 门禁（范围必须显式，绝不 `ruff check .`；当前全量 683 用例）：

```bash
python -m compileall -q common layout geometry opc lithography evaluation main tests
python -m ruff check common layout geometry opc lithography evaluation main tests
python -m pytest -q tests
```

- 新树可运行入口（均免安装、可从仓库外直跑）：

```bash
python main/run_macro_pipeline.py config/macro_pipeline.toml        # 双轮 ±2nm 验证管线
python main/run_single_pass.py config/single_pass.toml              # 单遍偏置扩张
python main/run_mbopc.py config/mbopc_single_macro.toml        # simple MB-OPC（macro 数由网格决定）
python main/run_mbopc_gradient.py config/gradient_mbopc.toml  # gradient MB-OPC
python main/run_ilt_simple.py config/simple_ilt.toml           # Simple ILT（像素型）
python main/run_ilt_levelset.py config/levelset_ilt.toml       # LevelSet ILT
python main/run_ilt_curvmulti.py config/curvmulti_ilt.toml     # CurvMulti ILT
python main/main_test_lithography.py                                # 光刻模型演示
```

- 参照运行旧管线（完整旧仓库，可独立执行）：

```bash
cd 00_PAST && python main/run_mbopc_frontend.py        # 合成冒烟
cd 00_PAST && python main/run_layout_geometry.py TestReticle/simple.gds --layer 1/0 --arrays
```

## 迁移中必须保持的核心不变量（从旧库与新管线蒸馏，评审新代码的对照基准）

| 不变量 | 内容 |
|---|---|
| 左下原点 | 所有版图/模型数组第 0 行 = 最低 Y；PNG/显示仅在 I/O 边界翻转 |
| Region 生命周期 | `materialize()` 与 `prepare_macro_problem()` 必须在 `with LayoutDB.open(...)` 内执行，否则得到空 Region |
| 固定参考 vs 迭代态 | `problem.*` 参考数组只读；迭代状态只有一维 `displacements` 数组（context 段恒 0）；重建仅用于候选校验与最终输出 |
| owner 唯一写 | 每个 segment 唯一 owner 写入，halo/context 只读提供光学上下文，同轮全部 core 评价完（Jacobi 屏障）后才发布下一状态 |
| 光刻画布契约 | canvas 固定 256²、Hopkins 核 35×35×24；core context 任意（如 400nm）按 `_center_padding` 差值均分、奇数余量归高侧居中入画布；FFT 循环卷积污染由 context 吸收，仅 ownership 像素计分 |
| 探针坐标 | DBU 点→canvas 必须经 `opc.input.points_to_canvas`（含居中 padding 项），禁止手写 `(x-left)/pixel-0.5`；与 ownership_canvas 互为反函数 |
| 独立 macro 语义 | macro 间不交换中间状态，边界 core 的 context 固定为邻区参考几何；全部 macro 完成后恰一次 `merge_macro_results`（显式 macro_id→GDS 映射，不猜路径）；这不是全局同步最优，差异须量化 |
| EPE 驱动 | direction ∈ {-1,0,+1}×step（clip 到 ±max_displacement）；EPE 严格更小才更新 best，平局保留较早轮，L2/PVBand 只诊断；records[0]=baseline、records[N]=第 N 次位移后评价 |
| 极性约定 | 透光率 1 恒为透光；opaque = field − coverage，法向经翻转统一为"透光→不透光"（求解器无极性分支） |
| 几何断言 | 零位移 XOR == 0、segment key 唯一、segment 长度 ≤ 配置、法向单位向量、owner 唯一、候选经 reconstruct 守卫后才发布（非法即终止该 macro 并留 stop_detail，不吞错） |
| nm→DBU 换算 | 严格整数换算（exact_dbu，Decimal 无浮点误差），tile/context 必须是 pixel 整数倍 |
| 内存上界 | target 用有界 uint8 LRU（key 含 macro id）；GPU 每 batch 只保留当前张量，批后释放再报进度；不保存整张 reticle tensor |

## 测试与数据

- `TestReticle/`：`build_reticles.py` 参数化生成测试版图集（10 场景 × 正负板，2026-08-17 起；50nm 定尺寸组 `p50_1024/`、`p50_2048/` 各 3 间距档 × 正负板，2026-08-21 起），规格与再生成依据见同目录 `reticle_build_plan.md`；simple / JustPoly / test1 / gcd_30um 为用户可编辑回归数据。测试不得硬编码其坐标/计数，新测试用生成式 GDS；gcd_30um 供 smoke（layer 11/0、TOP）。
- 旧测试套件（`00_PAST/tests/`）是迁移的规格书：实现迁移时对照移植测试，测试先行或同行。
- 新几何逻辑必须成组断言：零位移 XOR == 0、segment key 唯一、法向单位向量、owner 唯一；阶段边界行为用 monkeypatch 调用计数证明，不用注释或口头约定。
- 套件职责表与 smoke 验收标准见 `doc/test_manual.md`（当前全量 683）。

## Where to look

| 需求 | 位置 |
|---|---|
| 绑定规则 | `AGENTS.md` |
| 新系统开发/测试手册 | `doc/development_manual.md`、`doc/test_manual.md`（doc 根，2026-08-18 起为 doc_ 体系正式实例） |
| 新系统各批设计与报告 | `doc/changes/completed/CHG-*/`（spec + 两报告三件套）；文档体系总入口 `doc/INDEX.md` |
| 新树规划三文件 | 根目录 `task_plan.md`（阶段与状态）、`findings.md`（批次事实）、`progress.md`（会话日志） |
| 旧系统调用图 / 数据流 / 求解器骨架 | `00_PAST/doc/function_call_architecture.md`（第 2–4、10 节） |
| 历史错误 / 经验 / 需求 | `00_PAST/.learnings/`（ERR- / LRN- / FEAT- 编号） |
