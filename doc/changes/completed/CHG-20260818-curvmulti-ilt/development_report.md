# CHG-20260818-curvmulti-ilt 开发报告

## 0. 实施基线

- 批次 A 开工 HEAD：`31a4c19`（规格修订提交）；工作树含用户 WIP
  （`main/_ilt_workflow.py` 排版、`pyproject.toml`），各批 commit 用显式
  pathspec 排除，未裹挟。
- 批次 A 门禁基线：全量 650 passed + 1 skipped（617 基数 + 本批 33）；
  批次 B 门禁：660 passed + 1 skipped（+10 runner）。
- 审查基线与规格修订（阶段 0，commit `31a4c19`）：修正 3 处事实错误、
  补 5 处契约缺口、DEC 三项经用户 AskUserQuestion 裁定。

## 1. 交付批次

| 批次 | commit | 内容 |
|---|---|---|
| 0 | `31a4c19` | 规格修订（事实错误/契约缺口/DEC/算法细节钉死） |
| A | `721be5a` | `_common` resize_image/smooth_sigmoid_mask + `curvmulti.py` 求解器 + 33 测试 |
| B | `8eebc3c` | CONFIG_SECTIONS 注册 + `run_curvmulti_ilt.py` 入口 + smoke toml + 10 runner 测试 |
| C | 本提交 | contracts/dataflow/两手册/两报告/归档 |

## 2. 实施要点

- **现有函数零修改承诺的执行情况**：`run_ilt_workflow`、`PixelMacroProblem`、
  `_common` 现有函数、Simple/LevelSet 求解器全部未动；CurvMulti 沿用
  `owned_continuous_losses` 固定三条件。**一处例外（已在批次 B commit
  声明）**：`main/configuration.py::_parse_scalar` 元组分支通用扩展变长
  `tuple[X,...]`——TOML 无元组类型，`scales` 必须列表→元组；定长路径
  （macro_grid）逐字不变，属类型驱动解析器的通用扩展而非 CurvMulti 分支。
- **求解器结构**（`opc/iteration/ilt/curvmulti.py`）：per-stage 控制网格
  torch 叶子 + SGD；可微链 控制网格 → smooth_sigmoid_mask →
  resize(nearest) → trainable 索引 gather → 三值画布；梯度经 autograd
  链路跨批自然累加进 control.grad（与 Simple 的 numpy scatter-add 等价，
  链路内 gather/upsample 自动完成散布）。
- **实施中修复的三处自查缺陷**（均在批次 A 提交前）：
  ① state_index 在 best 赋值后递增导致 best_state_index 记后继编号；
  ② 每个 stage 都计算 area stage_reference（后续 stage 为死代码）；
  ③ REQ-006 best_parameters"末尺度"表述与全局 best 语义矛盾（规格随
  批次 A 同步修正为全局 best 上采样语义）。
- **runner 测试几何**：bbox 160×80（三角形上移避免 bbox 顶到 y=96 造成
  2px 短行 macro）；每宏 ownership 20×10px 整除 scales=[2,1]。

## 3. 对照 smoke（corners_unit，GTX 1650，CUDA）

| 方法 | 状态数 | total | RSS 峰值 | CUDA 峰值 | best_state/loss/binaryL2 |
|---|---|---|---|---|---|
| CurvMulti（scales=[5,1]，每 stage 1 更新，kernel 7，step 0.5） | 4 | 15.84s | 1325 MiB | 513 MiB | 2 / 8500.68 / 3675 |
| Simple（iterations 1，step 1.0，β=4） | 2 | 3.01s | 948 MiB | 508 MiB | 1 / 6162.66 / 2876 |

损失数值不跨参数化直接比较（状态数、初值、平滑参数化不同）；本表仅
记录资源与规模事实。CurvMulti 4 态 15.84s vs Simple 2 态 3.01s 的
每态成本差异主要来自每 state 两次额外 resize（粗→全分辨率上采样）与
CPU↔GPU 控制网格往返，属预期。

## 4. 交付审计（AGENTS 最终交付条款）

- 未用函数扫描：无（`_evaluated_states`/`build_curvmulti_final_context_canvas`
  均有 ILTMethod 消费方；测试 spy 仅为观察点）。
- 重复实现检查：终评 context helper 与 Simple 刻意代码重复（REQ-012
  先例：训练 GPU torch / 终评 CPU numpy，避免 round-trip），已注释声明。
- 异常入口：配置构造/求解入口/加载层三类 ValueError 均有测试；
  非有限 FloatingPointError 路径与 Simple 同构。
- 文档四件套：contracts/ilt.md、dataflow/curvmulti_ilt.md + index、
  development_manual §9b、test_manual 计数与行目均已更新。
