# Macro–Core 两级网格与双轮迭代重构 · 开发报告

日期：2026-08-15 ｜ 分支：`migration` ｜ 依据：`doc/macro_core_pipeline_design.md`（用户批准版）

## 1. 交付概览

| 实施批次 | 提交 | 内容 |
|---|---|---|
| A 配置、两级网格、居中 canvas | `refactor(opc-input): 重建两级网格与居中光刻画布` | `config/macro_pipeline.toml`、`grid.py` 重写（`MacroSpec`/`plan_macros`/三个切分函数）、`raster.py` 重写（居中 canvas） |
| B 持久化 MacroProblem | `refactor(opc-input): 建立持久化 MacroProblem` | `edge/problem.py` 新增（含 ownership 切线分裂）、`mask.py` 简化、`reconstruction.py` 消费 MacroProblem、删除旧路线六文件 |
| C 双轮迭代 | `feat(main): 完成双轮 macro-core 迭代验证` | `main/run_macro_pipeline.py` 阶段 0–2（每行注释） |
| D 最终 merge 与双模式输出 | `feat(geometry): 完成 macro 结果全局合并与双模式写出` | `PatchWriter.write_macro_results` + 阶段 3 `merge_final` + `run`/`main` 入口 |
| E 报告与审计 | `docs: 完成 macro-core 重构开发与测试报告` | 本报告、测试报告、开发/测试手册、规划三文件同步 |

删除清单（rg 确认新树零外部调用方后才删除）：`opc/input/macro.py`、
`opc/input/preflight.py`、`opc/input/edge/builder.py`、`opc/input/edge/macro.py`、
`opc/input/edge/ownership.py`、`opc/diagnostics.py`。

## 2. 关键实现决策

- **ownership 切线分裂**（`problem.py::_split_segments_at_ownership_cuts`）：
  x/y 切线穿越参数 t 由原始整数端点与全局整数切线计算，共享 macro 边界两侧
  得到逐位一致的分裂参数（测试实证无 33/34 DBU 分歧）；与片段边界重合的
  穿越（±1e-12 容差）不产生新段。
- **context membership 精确区间**：候选列/行由 searchsorted 直接给出「扩张
  bbox 与 core ownership 区间的精确交集」，越出 macro 的远端段得到空范围，
  不会被误裁进边界 core（旧实现的 clip 语义在单 macro 场景是错误的）。
- **居中 canvas**：`_center_padding` 奇数余量归高坐标侧；opaque 极性只反转
  局部窗口（背景 1 − coverage），canvas 外围 padding 恒为 0；全局
  DBU→canvas 映射公式固定在 `ownership_canvas` 注释中供后续 EPE 复用。
- **权威覆盖 + 全局 merge**：每轮 macro GDS 保存完整候选 polygon；最终合并
  按 macro ownership 精确裁剪消除重复写入，`single_cell` 在写出端做一次
  `min_coherence + merged()` 消除表示层 seam，并回读验证覆盖面积不变。

## 3. 与设计文档的偏差（均为最小必要，逐条说明）

| 偏差 | 原因 |
|---|---|
| `_write_macro_gds(problem, region, path, dbu_um)` 较 §13 多一个 `dbu_um` 参数 | GDS 写出必须知道源版图 DBU，而 MacroProblem/NPZ 格式（§10.1）不含该字段；不增加参数必然写出错误物理尺寸 |
| `merge_peak_rss_bytes` 为合并完成后的即时 RSS 采样 | psutil 无法回溯进程历史峰值；口径已在代码注释与本报告如实说明 |
| §15.5-7 的「无效 polygon」异常路径为运行时守卫、未单测 | 无法通过公开接口把无效 polygon 写入 GDS（KLayout 拒绝写出）；守卫代码存在于 `merge_final` 读取与回读两处 |
| §15.1 的 DBU/位移校验用例放在 `tests/main` 而非 `tests/opc/input` | `exact_dbu` 与 context≥位移校验发生在管线阶段 0，归属 main 测试 |

## 4. 开发中发现的实现 bug（已修，均有测试兜底）

1. 切线分裂后新批次 `edge_ids` 误传原始**段号**而非**数学边号**（矩形 4 边
   26 段时直接触发 `SegmentBatch` 校验失败）；修正为
   `segments.edge_ids[boundary_seg[~last]]`。
2. `np.where` 全分支求值使 `last` 位置的穿越索引越界（IndexError）；先夹回
   有效范围，被夹位置的值不会被选中。

## 5. 简化审计结果（§12 阶段 4 第 7 条）

- 旧符号（RectilinearCoreGrid / MBOPCProblem / MacroPreparation / PhysicalMask /
  prepare_problem / preflight / macro_boxes / diagnostics）：**零残留**。
- 术语（`optical_range` / `macro_mode` / `input_box`）：**零残留**（含 TOML）。
- 投机抽象（Worker / Protocol / 注册器 / 抽象基类）：**零存在**。
- 重复 raster：**一套**——`geometry.iter_region_coverage_tiles` 是共享底层，
  `opc/input/raster.py` 三个函数全部基于它，无第二套裁剪/合并/归一化逻辑。
- 兼容包装 / 异常吞噬：未发现（所有 `except` 均显式转型为带上下文错误或测试
  内的预期捕获）。
- 有意保留的无当前生产消费方符号：`edge_probe_points`（`sampling.py` 为
  §14.4 明确不修改文件，未来评价层消费）；`reconstruct_contours`
  （`reconstruct_region` 的公共中间入口）；`rasterize_region_window`
  （`rasterize_mask_canvas` 的公共底层，测试直接消费）。
- 导出消费统计：除上述三项外，`opc.input` / `opc.input.edge` 全部公开符号
  均有 ≥1 个生产或测试消费点。

## 6. 新增项目规则

- `AGENTS.md`：`main/` 下文件每一行必须有中文短注释；其他目录维持文件级
  docstring、函数 docstring 与分段注释。本重构全部新增代码已按此执行
  （`run_macro_pipeline.py` 约 540 行逐行注释）。

## 7. 已知限制（继承设计文档 §20）

- 巨大单 polygon 可能使单个 problem 超内存（本阶段不实现 polygon shard）；
- `single_cell` 全局 merge 是最终内存峰值，超过物理内存时显式失败而非静默
  降级；
- ±2 nm 双轮只验证状态机与几何管线，不是真实 OPC；
- context 充分性（光学收敛）待后续收敛测试。
