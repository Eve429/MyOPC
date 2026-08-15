# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

MyOPC：面向 OPC（光学邻近效应校正，半导体光刻）的层级版图与几何基础库。旧系统（AI 编写）包含 MB-OPC 前端、三种求解器（mbopc / diffopc / ilt）、ICCAD13 Hopkins 光刻模型与评价指标。

## 当前工作状态：用户自迁移（每个会话必读）

**整个旧代码库已整体归档到 `00_PAST/`（只读参照，内含完整旧仓库：源码、tests/、doc/、config/、.learnings/、规划文件）。仓库根目录从零开始，当前分支 `migration`。**

- **迁移方式**：用户（项目所有者）本人逐模块手工迁移/过滤，按依赖顺序 `layout → geometry → lithography → evaluation → opc.input → opc.input.edge → opc.iteration → main`。旧库规模约 11k 行生产代码 + 4k 行测试。
- **`00_PAST/` 只读纪律（用户明令）**：正常情况下不修改 `00_PAST/` 内任何内容；允许把代码复制出来到新结构中改写；若认为必须修改归档内部，必须先向用户请示说明必要性，获明确批准后才可动手。
- **Claude 的角色**：答疑与验证——解释 `00_PAST/` 旧实现的契约与历史决策、按需运行验证门禁、评审用户迁移后的代码。**默认不主动替用户编写新模块**；仅当用户对某个具体任务明确要求时才动手写代码。
- **权威参照文档**：`00_PAST/doc/function_call_architecture.md`（旧系统调用图与数据流，重点第 2–4、10 节）。
- **已知过滤决策点**（迁移到时提醒用户决策，勿自行处理）：
  - `00_PAST/layout/hierarchy.py`（HierarchySummary/CellInfo/build_hierarchy_summary）——零生产调用方的死代码；
  - `00_PAST/geometry/raster.py` 的 `render_layout_region`——零生产引用但有直接回归测试。

## 绑定规则

`AGENTS.md`（2026-08 重写版）是最高开发手册，与其他规则冲突时以它为准。要点：

- 所有 docstring 与注释用中文且以中文词开头，注释解释 why（坐标方向、数据不变量、性能路径、内存上界、边界归属、异常原因），置于紧凑逻辑块之前。
- 未经用户逐次确认不得修改 `layout/` 与 `geometry/`——**对用户新迁移出的同名目录同样适用，它们是用户手工迁移的领地**。
- 禁止自动格式化，保持紧凑排版；质量门禁 = ruff 规则检查 + compileall + pytest。
- 每个 bug 修复必须带可复现回归测试；修复后搜索调用点，删除仅服务旧 bug 的函数/包装/分支/变量。
- 禁止投机抽象：新接口必须有当前调用方。
- 本地 commit 为主，未经明确授权不 push；`TestReticle/*.gds` 用户回归数据与 `Test/klayout.ipynb` 不在修改范围。

## 环境与命令

- 解释器固定为 myopc conda env：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12；依赖 klayout / numpy / pillow / torch）。
- Bash on Windows：路径用正斜杠。
- 新树尚未建立完整包结构前，门禁按已迁移模块子集运行：

```bash
python -m compileall -q <已迁移模块>
python -m ruff check <已迁移模块>        # 绝不 ruff check .（范围必须显式）
python -m pytest -q tests/<对应子集>
```

- 参照运行旧管线（完整旧仓库，可独立执行）：

```bash
cd 00_PAST && python main/run_mbopc_frontend.py        # 合成冒烟
cd 00_PAST && python main/run_layout_geometry.py TestReticle/simple.gds --layer 1/0 --arrays
```

## 迁移中必须保持的核心不变量（从旧库蒸馏，评审新代码的对照基准）

| 不变量 | 内容 |
|---|---|
| 左下原点 | 所有版图/模型数组第 0 行 = 最低 Y；PNG/显示仅在 I/O 边界 flipud |
| Region 生命周期 | `materialize()` 与 `prepare_problem()` 必须在 `with LayoutDB.open(...)` 内执行，否则得到空 Region |
| 固定参考 vs 迭代态 | `problem.*` 参考数组只读；迭代状态只有一维 `displacements` 数组；重建仅用于最终输出，不进热路径 |
| owner 唯一写 | 每个 segment 唯一 owner 写入，halo 只读提供上下文，全部 core 评价完经屏障后才发布 |
| 光刻画布契约 | canvas 256²、Hopkins 核 35×35×24、tile 1024nm + halo 512nm + pixel 8nm 恰满画布；FFT 循环卷积的边界污染由 halo 吸收，仅 core 像素计分（ownership_canvas） |
| 极性约定 | 透光率 1 恒为透光；opaque = field − coverage，外法向随之翻转 |
| 几何断言 | 零位移 XOR == 0、segment key 唯一、segment 长度 ≤ 配置、法向单位向量、owner 唯一 |
| nm→DBU 换算 | 严格整数换算（exact_dbu），tile/halo 必须是 pixel 整数倍 |

## 测试与数据

- `00_PAST/TestReticle/*.gds`（simple / gcd_45nm / JustPoly / test1）是用户可编辑回归数据：测试不得硬编码其坐标/计数，新测试用生成式 GDS。
- 旧测试套件（`00_PAST/tests/`）是迁移的规格书：实现迁移时对照移植测试，测试先行或同行。
- 新几何逻辑必须成组断言：零位移 XOR == 0、segment key 唯一、法向单位向量、owner 唯一。

## Where to look

| 需求 | 位置 |
|---|---|
| 绑定规则 | `AGENTS.md` |
| 旧系统调用图 / 数据流 / 求解器骨架 | `00_PAST/doc/function_call_architecture.md` |
| 旧系统开发/测试手册与专项报告 | `00_PAST/doc/`（development_manual、test_manual、*_report.md） |
| 历史错误 / 经验 / 需求 | `00_PAST/.learnings/`（ERR- / LRN- / FEAT- 编号） |
| 旧规划文件 | `00_PAST/task_plan.md`、`00_PAST/findings.md`、`00_PAST/progress.md` |
