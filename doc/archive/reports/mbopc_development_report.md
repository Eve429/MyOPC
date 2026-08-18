# 最简 MB-OPC 迁移开发报告

> 依据：`doc/opc/mbopc_migration_design.md`（用户批准实施，2026-08-16）。
> 实施日期：2026-08-16；提交 `2b9194a`（光刻契约）→ `a5509bc`（evaluation）→
> `c596d70`（居中坐标）→ `71d42ba`（共享生命周期）→ `986cbfd`（求解器）→
> `84407e5`（两个入口）→ 本报告批次（F）。

## 1. 实际交付文件

**新增**：`lithography/contracts.py`、`evaluation/`（metrics + __init__）、
`opc/iteration/mbopc/simple.py`（含 Config/Cache/Step/Record/Result 与两个算法
函数）、`main/_macro_pipeline.py`、`main/_mbopc_workflow.py`、
`main/run_mbopc_single_macro.py`、`main/run_mbopc_multi_macro.py`、
`config/mbopc_{single,multi}_macro.toml`、`opc/input/raster.py::points_to_canvas`、
`tests/evaluation/`（25 例）、`tests/opc/iteration/`（51 例）、
`tests/main/test_mbopc_runners.py`（21 例）。

**修改**：`lithography/__init__.py`（导出 LithographyModel）、
`opc/input/__init__.py`（导出 points_to_canvas）、
`main/run_macro_pipeline.py`（改为共享模块消费方）、
`tests/main/test_macro_pipeline.py`（适配重构）、`requirements.txt`（tqdm）。

**零改动**：`00_PAST/**`、`layout/**`、`geometry/**`、
`opc/input/edge/{problem,fragmentation,reconstruction}.py`（设计 §15.3）、
用户 GDS、`.vscode/`、用户工作树未提交内容（`doc/opc/mbopc_migration_design.md`
的后续编辑、`main/main_test_lithography.py`）。

## 2. 六阶段实施与验证

| 阶段 | 提交 | 验证 |
|---|---|---|
| A evaluation + 契约 | `2b9194a`/`a5509bc` | 25 例；全量 249 |
| B points_to_canvas | `c596d70` | 9 例回归；已有栅格逐值不变 |
| C 共享生命周期 | `71d42ba` | 全量 258；gcd_45nm XOR==0 / 10.67s 与基线一致 |
| D 求解器 | `986cbfd` | 50 例（后补 1 例循环内 zero_epe）；全量 308 |
| E 两入口 | `84407e5` | 21 例；全量 329 |
| F 端到端 + 报告 | 本批次 | 全量 **330 passed**；两入口 gcd_45nm CUDA 实测（§6） |

阶段 C 是唯一触及既有行为的重构：plan.json 去掉 `round_deltas_dbu` 键（该值
改由验证 runner 的 `load_validation_deltas` 持有）、`load_config` 拆为
`load_macro_config(extra_sections=...)` + `load_validation_deltas`、
`merge_final` 拆为 `collect_round_macro_gds`（轮次校验）+
`merge_macro_results`（显式映射）。±2/-2 文件数、result NPZ 字段、错误语义与
gcd_45nm 最终 XOR 全部不变（TestTwoRounds/TestFinalMerge 全绿 + smoke 回零）。

## 3. 与设计文档的偏差清单

| 偏差 | 原因 |
|---|---|
| `evaluate_edge_probes` 默认 threshold 保留旧值 **0.499**（设计 §8.2 写 0.5） | 实施计划批准的裁决：忠实旧数值行为；L2/PVBand 仍 0.5 |
| `solve_macro()` 较 §12.4 签名补 `dbu_um` 参数 | MacroProblem 不含源 DBU（NPZ 无该字段），先例同 `_write_macro_gds` |
| `save_final_lithography` 用独立规整 tile 网格（单 macro 全 ROI 按 core 切分），不复刻迭代期 macro 边界 | plan.json 不存 macro 切线；可视化网格参数全部写入 manifest 供对账 |
| §3.4“光刻迁移尚未实施”已过时 | lithography 已于同日完成（224 基线）；ICCAD13 API 满足 §8.1 契约，未触发停下对齐 |
| §7“旧求解器漏掉了居中 padding”措辞不精确 | 旧契约 tile 1024+2×halo 512 恰满 256 画布、无 padding 概念，旧公式与旧 raster 自洽；新 Macro–Core 的 context 400nm 产生 228px+14px padding，故新管线必须加 padding 项（`points_to_canvas`）——技术决策正确，定性按此精确化 |
| OpenILT `opc/simpleopc.py` 不在 00_PAST（grep 零匹配） | 它是设计研究期的外部思想参照；实施依据 = 设计规格 + `00_PAST/opc/iteration/mbopc/solver.py` 语义 |
| tqdm 无需安装 | 实测环境已有 4.70.0（探索期误报未安装）；requirements.txt 已声明 |
| plan.json 去掉 `round_deltas_dbu` 键 | prepare_problems 不读 [iteration]（设计 §11.2 规则）；该值唯一来源改为验证 runner |

其余接口、算法、资产、目录、测试矩阵、提交划分与设计文档一致。

## 4. 关键取舍记录

- **独立 macro 策略与边界代价**：macro 间不交换中间状态，边界 core 的 context
  固定为邻区参考几何。gcd_45nm 实测量化（§6）：single（全 ROI 一个 macro）总
  EPE 23440，multi（2×2）四 macro 之和 23676——独立 context 代价 236 段
  （约 1.0%）；最终覆盖 XOR 面积 34650860 DBU²。两者都不是“全局同步最优”，
  差异如实记录，不宣称等价。
- **旧差分快路未迁**：`_subset_contours`/`_polygon_ids_for_core`/`_current_tile`
  依赖已删除的 `MBOPCProblem.physical_mask` 与旧 raster 签名；第一版每 macro
  直接重建 Region 栅格化（KLayout 逐 tile），这是明确基线，优化等基准证明
  瓶颈后再做。
- **Protocol 引入理由**：`LithographyModel` 在 `evaluate_and_propose()` 获得
  第一个真实求解器调用方（ICCAD13 是当前唯一实现），满足“新接口必须有当前
  调用方”纪律；测试用 `_PhaseModel` 假模型同时验证了结构化消费（不依赖
  ICCAD13 具体类型）。
- **五结构调用方**：Config/Cache 由 workflow 与 solver 构造；Step 是
  evaluate_and_propose 的返回；Record 由 optimize_macro 逐轮生成并经 asdict 写
  metrics.json；Result 是 optimize_macro 的返回且 workflow 消费其全部字段。
- **merge 的独立边界**：`merge_macro_results(plan, mapping, out, cell_mode)`
  不读 result NPZ、不猜路径、键集必须与 plan 一致；调用时机（最终一次或未来
  逐轮同步）完全由编排层决定。
- **异常边界**：候选非法（ReconstructionError/ValueError）终止当前 macro 并把
  原因写入 stop_detail 与 summary（对比旧版静默回滚继续——有意的不吞错变更）；
  I/O、Torch、KLayout 异常原样传播，不拿半份结果拼最终 GDS。

## 5. 测试与 coverage

- 全量 **330 passed**（基线 224 + 106 新增）；ruff（显式范围）与 compileall 全绿。
- coverage：`evaluation/metrics.py` 100%；`simple.py` 99%（228 语句，
  未覆盖仅 266/269 两行防御 RuntimeError——触发需要破坏 MacroProblem 构造期
  已保证的 owner 唯一不变量，正常与注入路径均无法构造，如实记录不豁免）。
- 关键分支命中：三种 EPE 方向 + ambiguous、cache 命中/替换/LRU 驱逐/单项超限/
  0 禁用/跨 macro key、baseline 与移动后指标、每批恰一次三条件 forward_many
  （monkeypatch 计数）、cache 命中免重栅格（栅格调用计数 8→4）、进度回调
  =（iterations+1）×core、batch 不变性、macro 正逆序覆盖 XOR==0、
  invalid_geometry 保留 best 继续、恰一次 merge、像素对齐步长的循环内 zero_epe。

## 6. gcd_45nm 端到端实测（2026-08-16，CUDA GTX 1650）

- 规模：870 core、343018 段；`config/mbopc_*.toml` 默认参数（迭代 8、
  步长 8nm 衰减 4、EPE 距离 16nm、batch 8、target cache 512MiB）。
- **multi（2×2 macro）**：126.0s；EPE 逐轮单调下降：
  mr0c0 37743→7263、mr0c1 32053→5904、mr1c0 30642→5625、mr1c1 27789→4884
  （全部 iteration_limit，best_round=8）；merge 0.64s；870 tile 光刻 PNG。
- **single（全 ROI 1 macro）**：126.6s；EPE 128227→23440。
- 差异量化：single 总 EPE 比 multi 之和小 236 段；最终覆盖 XOR 34650860 DBU²。
- 回读大记录警告（`Record length larger than 0x8000`）来自 KLayout 读回
  OPC_RESULT 大 polygon 的 GDS 格式提示，无害。

## 7. 简化审计

- 无未调用函数：simple.py 全部公开结构/函数与 workflow 全部函数均有调用方；
  旧 `merge_final`/`load_config` 已删除而非保留包装（测试改用新入口）。
- 无重复实现：坐标换算唯一在 `points_to_canvas`（`ownership_canvas` 内联公式
  与之互为反函数且有批量一致性测试）；GDS 写出/merge 唯一在 `_macro_pipeline`。
- 无吞错：invalid_geometry 保留 stop_detail；防御分支 raise 而非 log。
- 无 Protocol 泛滥：仅 LithographyModel/LithographyConfigView 两个
  runtime_checkable Protocol，无注册器/工厂/插件/Worker/队列。
- 无第二套 raster/求解器入口复用问题：两入口只 import workflow。
- `00_PAST/`、`layout/`、`geometry/`、`opc/input/edge/` 三文件零改动。

## 8. 审查修复轮（2026-08-16，用户独立审查后）

用户以只读审查（`.planning/lithography_mbopc_review/`）提出 3 项 P1 与多项
P2，逐条对照代码原文核实后实施两个修复提交：

- **`3725c0e`（P1 三项）**：
  1. **insufficient_probes 停止状态**（新算法政策，用户批准）：评价返回
     `valid_probes==0` 且存在 owner 段时终止并保留 baseline，不再把
     "无法评价"报成 zero_epe（2nm 壁 + 8nm 探针实测：修复前 zero_epe
     误报 → 修复后 insufficient_probes，stop_detail 写明探针/段数）；
     循环内检查先于 best 比较终止（valid==0 时 epe 恒 0 会被误当改善）；
     空 macro（零段）维持 zero_epe。不选"拒绝 Problem"（窄壁跨 macro
     无法预检）与"自适应探针距离"（改变评价语义）。
  2. **几何流式与真实 bounds**：save_final_lithography 改 with 内逐 tile
     窗口 materialize_intersecting 就地栅格（峰值 O(reticle)→O(tile)，
     PNG 逐位不变）；merge 回读面积验证改逐 macro ownership 窗口累加
     （显式裁回 ownership 防跨界重复计数；消除第二个全量 Region，失败可
     定位 macro）；五处 ±2^30 魔法框 → layer_bbox（GDS int32 域外图形
     不再静默丢失）。
  3. **`_as_int` 严格整数校验**：workflow 四字段与 macro_pipeline 的
     layer/datatype/canvas_pixels 拒绝 TOML 浮点/布尔（1.5/true 原先被
     int() 静默截断）。
- **`acfcab0`（P2）**：参考几何整迭代物化一次（reference 参数）；EPE 回切
  整 batch 化（每张张量一次 .cpu()）；无变化提案直接 no_update 不再重复
  评价（**行为变化**：no_update 时 records 只含 baseline）；末轮纯评价不
  生成被丢弃的提案；macro_grid 数量模式配置层前置校验；tqdm try/finally；
  补齐设计 §16.3 两个真构造越界用例与差异上界断言。
- **审查建议不采纳一项（有实测证据）**：except 只捕 ReconstructionError——
  实测几何退化（四边共线，ring 少于三顶点）以 ValueError 从 KLayout 数组
  校验冒出，包装它需改 `reconstruction.py`（设计 §15.3 不修改清单），故
  维持 `except (ValueError, ReconstructionError)` 并在代码注释记录依据；
  更大幅度翻转（−25/−30）会被 miter 解析成反向合法 ring 而不触发守卫，
  该行为已用测试注释如实记录。
- **验证**：全量 330 → **341 passed**；gcd_45nm multi smoke 四 macro
  best_epe 三个版本（迁移后/P1 修复后/P2 修复后）逐位一致
  （7263/5904/5625/4884），几何与算法路径零漂移。
- **仍开放（记录不改）**：merge patches 列表持有全部 clipped Region——
  PatchWriter 接口在 geometry/（用户领地）；ProcessCondition 绑定
  focus/defocus——设计 §8.1 原文如此，等真实第二模型。

## 9. 已知限制（承设计 §22）

context 光学充分性未证明；离散 EPE 是启发式（gcd_45nm 8 轮未收敛到 0，EPE
下降但放缓）；相邻 macro 同边段可能形成真实 jog（未平滑）；SREF/AREF 展开
由 layout 读取层保证（runner 测试走生成式平坦 GDS，层级版图端到端由
`run_macro_pipeline` 的 gcd_45nm smoke 与 layout 测试覆盖）；梯度 MB-OPC 与
ILT 的目录已预留但不建空实现。
