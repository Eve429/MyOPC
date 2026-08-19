---
id: CHG-20260818-simple-ilt
title: Simple ILT 与像素型 ILT 基础管线迁移
type: implementation-spec
status: approved
baseline_commit: 540a0121eb06904bdc44ae7fe3bd491aeff22fb5
baseline_worktree: dirty
baseline_dirty_paths:
  - doc/changes/active/CHG-20260818-simple-ilt/implementation_spec.md
  - findings.md
  - progress.md
  - task_plan.md
  - .learnings/ERRORS.md
  - .learnings/LEARNINGS.md
scope:
  - opc/input/pixel
  - opc/iteration/ilt
  - main
  - config
  - tests
  - doc
depends_on:
  - doc/architecture/system.md
  - doc/architecture/dataflow.md
  - doc/contracts/opc_input.md
  - doc/contracts/lithography.md
  - doc/contracts/evaluation.md
  - doc/contracts/ilt.md
supersedes: []
---

# Simple ILT 与像素型 ILT 基础管线迁移

## 0. Document Contract

本文档是本 change 不依赖聊天上下文的唯一实现规格。实现 AI 必须先读取仓库根
`AGENTS.md`、本规格依赖的 architecture/contracts、基线源码与测试。

实现 AI MUST：

- 以本文的 Target Behavior、Invariants、Interfaces 和 Acceptance Criteria 为目标；
- 开始前核对基线与相关工作树差异；
- 只修改 §14 列出的路径；必须扩大范围时停止并请求修订规格；
- 不依赖生成本文档时的聊天上下文，不自行补充算法或产品行为；
- `status` 不是 `approved`、基线实质漂移或 Blocking Open Question 非空时不得实施；
- 实施结束逐项提供 requirement、test 和 acceptance evidence。

实现 AI MUST NOT：

- 修改 `00_PAST/**`、`layout/**`、`geometry/**` 或用户 GDS；
- 恢复旧 `run_ilt.py` 的 method 字符串大分派、argparse 参数长表或 offline input；
- 让像素 ILT 依赖 `SegmentBatch`、`MacroProblem`、edge ownership 或边段重建；
- 为尚未实施的方法建立 solver 基类、注册器或空模块；
- 静默 fallback、吞异常或在失败后发布半份最终版图。

规范词 MUST / MUST NOT / SHOULD / MAY 含义遵循
`doc/implementation_spec_template.md`。已批准后的 contract 变更必须退回 `draft`。

## 1. Objective

迁移 OpenILT Simple ILT，并建立后续 LevelSet、CurvMulti、Multilevel 可直接复用的最小像素
problem、结果和 macro/core 执行边界，使用户可以直接运行一个 Python 文件，从 GDS 输入完成
macro 同步优化、core-batched 光刻计算、指标计算、逐 macro 结果、最终 GDS 合并和可选最终光刻图保存。

## 2. Baseline and Evidence

### 2.1 Baseline

- Commit：`540a0121eb06904bdc44ae7fe3bd491aeff22fb5`
- Worktree：本次修订仅修改本规格、根目录工作记录与 `.learnings/` 记录；生产源码和测试无未提交差异。
- 验证环境：Linux、Python `/home/wzh/miniconda3/envs/myopc312/bin/python` 3.12.0、CPU；当前环境
  CUDA 不可用。
- 基线回归：`450 passed, 8 skipped in 87.86s`；8 项跳过均为 CUDA 测试。

### 2.2 Confirmed Facts

| Fact ID | Confirmed fact | Evidence | Verification method |
|---|---|---|---|
| FACT-001 | 当前没有 ILT 生产实现 | `doc/contracts/ilt.md`、`opc/iteration/` | 静态阅读 |
| FACT-002 | 光刻接口已支持 batch、具名工艺条件与 autograd | `lithography/contracts.py::LithographyModel`、`lithography/iccad13.py::ICCAD13Lithography.forward_many` | 源码与既有 backward 测试 |
| FACT-003 | Macro/Core 网格已冻结 256 canvas、pixel 对齐和 context 上限 | `opc/input/grid.py::plan_macros` | 源码与 grid 测试 |
| FACT-004 | mask canvas 约定为 `[y,x]`、低 Y 在 row 0、`1=透光`、外围 padding=0 | `opc/input/raster.py::rasterize_mask_canvas` | 源码与 raster 测试 |
| FACT-005 | L2/PVBand 可用 ownership mask 排除 context | `evaluation/metrics.py::evaluate_binary_l2`、`evaluate_pvband` | 源码与 evaluation 测试 |
| FACT-006 | 当前 GDS merge 和最终光刻保存不依赖具体 MB-OPC 算法 | `main/_macro_pipeline.py::merge_macro_results`、`save_final_lithography` | 静态阅读 |
| FACT-007 | `write_macro_gds` 当前仅有两个生产调用点且类型绑定 edge `MacroProblem` | `main/_macro_pipeline.py::write_macro_gds` | `rg "write_macro_gds\\(" main tests` |
| FACT-008 | 配置由 section→dataclass 单入口严格解析 | `main/configuration.py::load_config`、`CONFIG_SECTIONS` | 源码与 config 测试 |
| FACT-009 | 旧迁移 Simple ILT 是 sigmoid 参数化、SGD、连续 nominal/process/PV loss 和可选 mask curvature | `00_PAST/opc/iteration/ilt/simple.py::optimize` | 只读静态阅读 |
| FACT-010 | OpenILT 上游按中央 filter 固定 context，最后一次 SGD 更新未评价，batch 共用一个 best | `OpenILT/pyilt/simpleilt.py::SimpleILT.solve` | 只读静态阅读 |
| FACT-011 | OpenILT 根许可证是 MIT | `OpenILT/LICENSE` | 只读静态阅读 |
| FACT-012 | 名义 core/context 已强制为 pixel 整数倍，但版图 bbox 末端缩短 core 仍可不是 pixel 整数倍 | `opc/input/grid.py::plan_macros/_core_cuts`、`tests/opc/input/test_grid.py::TestGridValidation` | 源码与测试静态阅读 |

### 2.3 Uncertainty Boundary

- 本 change 不宣称逐 macro 独立求解等价于全 reticle 联合 ILT；相邻 macro 不交换已优化参数。
- 本 change 不宣称像素 GDS 满足 MRC、shot count、最小线宽或全局最优。
- 当前没有可核验的性能阈值；本 change 只记录固定 smoke 的时间/RSS/CUDA peak，不声称提速。

### 2.4 External and Archive References

| Reference | Role | Adopt | Explicitly reject | Reason |
|---|---|---|---|---|
| `OpenILT/pyilt/simpleilt.py` | 原始算法 | sigmoid、SGD、三连续损失、best snapshot | 全局变量、DataParallel、assert 配置、固定 core context、最后一步不评价、batch 共用 best | 与 macro 同步状态和 batch 不变量冲突 |
| `OpenILT/SIMPLEILT_IMPLEMENTATION_GUIDE.md` | 算法解释与实测 | loss 公式、ROI/filter、EPE 仅评价的事实 | 旧 2048 benchmark runner 与 shot 依赖 | 当前模型固定 256 且工程管线不同 |
| `00_PAST/opc/iteration/ilt/simple.py` | 首次适配参考 | 具名 ProcessCondition、结构化 record/result、真实模型测试 | `SimpleILTResult` 被其他方法反向引用、单图 runner | 本次从首个方法建立中性公共结果契约 |
| `00_PAST/main/run_ilt.py` | 旧接线反例与产物参考 | NPZ/PNG/资源统计字段意图 | method 大分派、旧 offline_inputs/artifacts、无 macro、无 GDS | 不符合当前 main 结构和大版图路线 |

算法迁移源码须保留 MIT 来源说明；`00_PAST/**` 保持只读。

## 3. Current Behavior

1. `main/_macro_pipeline.py::prepare_problems` 只生成 edge `MacroProblem`，会提边、分段和构造 segment ownership；像素 ILT 无可消费 problem。
2. `opc/input/raster.py::rasterize_mask_canvas` 可以把一个 Region 变成单 core canvas，但没有持久化 macro 像素输入，也没有 binary pixel→Region 的反向路径。
3. `opc/iteration/` 只有 MB-OPC；`main/` 没有可直接运行的 ILT 入口。
4. `doc/contracts/ilt.md` 明确声明当前无 ILT 能力。

## 4. Target Behavior

### REQ-001

系统 MUST 提供 `python main/run_simple_ilt.py <config.toml>` 直接入口，从 GDS/OASIS/GLP
目标层运行 Simple ILT，无需安装项目包。

### REQ-002

ILT 输入 MUST 复用 `plan_macros`，但 MUST NOT 调用 edge 提取、分段、segment ownership 或
`prepare_macro_problem`。

### REQ-003

每个 macro MUST 只物化和栅格化一次 `macro.query_box`，持久化一张 `uint8` transmission
target；迭代阶段 MUST 从该数组切 core canvas，不重复调用 KLayout。

### REQ-004

所有数组 MUST 使用 `[y,x]`，row 0=最低全局 Y，`1=透光/0=不透光`，中间值保留面积覆盖率；
clear/opaque 只在 problem 构造与最终 GDS 反变换边界处理，求解器内部 MUST 无极性分支。

### REQ-005

core ownership 只定义当前 core 唯一统计 loss 的区域；macro trainable domain 是当前 macro
ownership 内全部可修改 pixel；simulation context 是该 core 完整 256×256 光刻画布。处理一个 core
时，simulation context 内属于 macro trainable domain 的全部 pixel MUST 保持可微，只有 macro 外部
context 与 padding 固定。最终写回仍只发布 macro ownership，不能从 context 重复写回。

ILT prepare 还 MUST 在栅格化前确认每个实际 macro/core ownership box 的宽高都是 `pixel_dbu`
整数倍；最外侧缩短 core 不满足时必须 `ValueError`，不得创建 partial ownership pixel。

### REQ-006

> 【2026-08-19 已取代】本条初始化契约已被 `CHG-20260819-simple-ilt-openilt-init` 修订：现行为 `params = 2·T − 1`，state0 soft = σ(β(2T−1))，不再要求 1e-6 恢复；固定 context 同为 σ(β(2T−1))（监督目标仍 raw T）。本节以下文字为当时历史记录。

Simple ILT MUST 保留 `target_u8/255` 的 fractional transmission coverage。设 `T=target_u8/255`，
macro 参数初值 MUST 为 `logit(clamp(T, eps, 1-eps))/beta`，其中 `eps` 是仅用于避免严格 0/1
产生无穷值的固定 float32 小量，不增加配置项；state 0 的 soft mask MUST 在绝对误差 `1e-6`
内恢复 T。macro 外部 context 直接使用输入 transmission，不参数化。

训练仍使用 sigmoid、SGD、nominal L2、process-corner L2、连续 PV loss 和可选 mask curvature；
每个 core 的各项 loss 只在其 ownership 像素求和，不做 overlap loss averaging。

### REQ-007

`iterations=N` MUST 表示 macro 级发布并评价 N 次同步 SGD 更新：同一 state 的全部 core/batch
读取同一 macro parameter snapshot，各自反向后把 local gradient scatter-add 到唯一
macro gradient；全部 core 完成后才允许执行一次 step 并发布下一 state。记录 state 0 baseline
与 state 1..N；state N 只执行 forward/loss/best 评价，不再 backward 或 step。

### REQ-008

每个 state 完成全部 core 评价后，MUST 把各 core ownership loss 求和为 `macro_total_loss`，并按
严格更低值选择唯一 `best_macro_parameters`。不得把不同 core 的不同 best state 拼成最终 mask。
改变 `batch_size` 或 core 遍历顺序 MUST NOT 改变宏级 best 与最终输出（允许浮点约定容差）。

### REQ-009

CPU MUST 只常驻当前 macro 的 target、parameters、accumulated gradient 与 best parameters；GPU
一次只处理一个 core batch。每批 backward 后只把 local gradient scatter-add 回 CPU macro
gradient 并释放 autograd graph；同一物理 pixel 只有一个 macro 参数，可以累加来自多个 core loss
的梯度，不得建立重叠参数副本或对重叠梯度取平均。全部 batch 完成后才更新 CPU macro parameters；
不得把整个 macro 或 reticle tensor 常驻 GPU。

### REQ-010

系统 MUST 在最终 best binary mask 上额外执行一次 nominal/dose_max/defocus_min 前向，并按
ownership 统计 binary L2 与 PVBand；EPE 和 shot count MUST NOT 作为本 change 的训练项或指标。

### REQ-011

每个 macro MUST 保存 result NPZ、metrics JSON 与 best GDS；全部 macro 完成后 MUST 仅调用一次
现有 ownership merge，并可复用现有最终光刻 PNG/manifest 保存。

### REQ-012

binary transmission→GDS MUST 使用行程合并而不是逐像素插入 polygon；clear 输出透光像素，
opaque 输出 ownership 内的不透光像素，所有矩形 MUST 裁到 macro ownership 后统一 Region merge。

### REQ-013

首个实现 MUST 提供仅含当前真实调用方的 `ILTMethod`、通用 macro result/record 和公共 workflow；
MUST NOT 建 solver 基类、注册器或未使用方法模块。

### REQ-014

进度条 total MUST 为 `core_count*(iterations+1)`，每完成一个已评价 batch state 按真实 batch
core 数增加；异常路径 MUST 在 `finally` 关闭内外层进度条。

### REQ-015

summary MUST 记录输入/准备/优化/合并/总耗时、RSS 起点/准备后/峰值、显式 CUDA device peak、
macro/core/pixel 数、loss/指标、产物路径和已知 macro seam 策略。

### REQ-016

现有 MB-OPC、验证管线、layout、geometry、lithography 数值和 public import MUST 保持兼容；
只允许 §9 明确列出的内部接口迁移。

### REQ-017

统一配置解析 MUST 使用 `typing.get_type_hints(config_type)` 解析注册 dataclass 的字段注解，使定义在
`opc.iteration.ilt` 且启用 postponed annotations 的算法 Config 可直接注册；现有九种 Config 的解析
输入、输出和异常 MUST 保持不变，不得为用户配置和 solver 再定义一份重复结构。

## 5. Scope

### 5.1 In Scope

- 像素 MacroProblem 的构造、NPZ、core canvas/loss ownership/trainable index 映射和 pixel→Region。
- Simple ILT macro optimizer、公共 ILT record/result、公共 ILT workflow。
- TOML 配置、一个薄适配器、一个直接入口、逐 macro/最终产物。
- CPU 单元/集成测试、可用时 CUDA parity、生成式多图形与真实小 GDS smoke。
- 当前文档、开发/测试报告与项目规划记录同步。

### 5.2 Out of Scope

- LevelSet、CurvMulti、Multilevel 的代码实现；本 change 只冻结它们可复用的接口。
- 跨 macro 联合参数、更新交换、Schwarz/overlap reconciliation。
- EPE loss、EPE 像素轮廓评价、shot、MRC、SRAF、mask rule regularization。
- NPZ 直接离线入口、训练 checkpoint/resume、分布式 worker、多 GPU DataParallel。
- 全局多边形最小化或非像素格矢量拟合。

### 5.3 Protected Areas

- `00_PAST/**`、`layout/**`、`geometry/**`、`TestReticle/*.gds`：MUST NOT 修改。
- `lithography/**`、`evaluation/**`：行为和数值 MUST NOT 修改。
- 用户输出和无关工作树改动必须保留且不得纳入提交。

## 6. Invariants

### INV-001

一个 macro ownership 像素 MUST 有且只有一个最终 writer，并且只属于一个 core 的 loss ownership；
macro 外部 context 和 padding MUST 无 writer。

Enforced by：core ownership 计数、macro trainable index 映射与最终写出测试。

### INV-002

同一 state 的全部 core/batch MUST 读取同一 macro parameter snapshot；core ownership 只限制 loss
统计，不截断 simulation context 内 macro-trainable pixel 的梯度。全部 core 梯度完成前参数不可更新。

Enforced by：跨 core 梯度求和、state 屏障与反序/batch size 对照。

### INV-003

同一 state 的指标 MUST 描述 step 前已评价状态；proposal/step 完成后只在下一 state 发布记录。

Enforced by：N+1 record、手算一阶更新与 best snapshot 测试。

### INV-004

best state MUST 是完整 macro 已评价状态；batch 重排或拆分不能改变 macro best。

Enforced by：batch size 1/2、core 正逆序和“不同 core 局部最优轮不同”对照。

### INV-005

target、mask、wafer 和 GDS 映射必须遵守统一 transmission/坐标约定，不允许 y 翻转进入算法层。

Enforced by：非对称 L 图案和左下/右上标记测试。

### INV-006

最终 macro Region 只能贡献自身 ownership；所有 macro merge 后不得有正面积重叠或遗漏像素格。

Enforced by：clear/opaque、core/macro seam 与回读 XOR/面积测试。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `opc.input.pixel.problem` | macro target 一次栅格、NPZ、core 映射、owned raster→Region | 光刻、loss、optimizer、CLI |
| `opc.iteration.ilt._common` | 通用 macro record/result、连续 loss/curvature | 方法分派、GDS、进度 |
| `opc.iteration.ilt.simple` | macro 参数、core-batched 梯度累加、同步 SGD、macro state/best | layout、macro merge、tqdm |
| `main._ilt_workflow` | 配置、problem 准备、指标、产物、macro merge | 具体参数化和 optimizer 数学 |
| `main._simple_ilt_workflow` | Simple 方法描述对象与薄调用 | 公共生命周期复制 |
| `main.run_simple_ilt` | CLI config path、调用、摘要打印 | 算法与业务逻辑 |

### 7.2 Dependency Direction

允许：

```text
layout -> geometry -> opc.input -> opc.input.pixel
opc.iteration.ilt -> opc.input.pixel + lithography
main._ilt_workflow -> layout/geometry/opc.input.pixel/opc.iteration.ilt/lithography/evaluation
run_simple_ilt -> _simple_ilt_workflow -> _ilt_workflow
```

禁止：

```text
opc.input.pixel -X-> opc.iteration/lithography/evaluation/main
lithography/evaluation -X-> opc.iteration/main
opc.iteration.ilt -X-> layout/geometry/main/opc.input.edge
```

### 7.3 Data Flow

```text
GDS + TOML
 -> load_config
 -> LayoutDB.open once
 -> layer_bbox + plan_macros
 -> per macro query(query_box).materialize_intersecting
 -> prepare_pixel_macro_problem -> pixel_problems/<macro>.npz
 -> per macro load once
 -> optimize_simple_macro: initialize one macro parameter state from coverage
    -> per state 0..N
       -> per core batch: snapshot context canvas + ownership loss mask
          -> forward/backward; scatter-add local gradient to macro gradient
          -> release batch GPU graph
       -> aggregate macro loss; update macro best
       -> if not state N: one synchronized CPU SGD step
    -> final best binary per-core forward + owned L2/PV
 -> reconstruct_pixel_region -> best.gds + result/metrics
 -> merge_macro_results exactly once
 -> optional save_final_lithography
 -> summary.json
```

边界：KLayout 只在 prepare、pixel→Region、GDS 写/merge；CPU/GPU 只按 batch 搬运；NPZ
使用 `allow_pickle=False`；JSON/NPZ/GDS 原子写出。

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| 0 | TOML+GDS | 配置/DBU/网格 | macros | 不物化 geometry |
| 1 | 一个 macro query | 一次物化/栅格 | PixelMacroProblem NPZ | 不提边；不按 core 调 KLayout |
| 2 | 一个 problem | 加载 target、初始化 macro 参数/梯度/best | CPU macro state | 不重读 GDS |
| 3 | 一个 macro state | 全部 core batch 评价/反向、梯度累加、同步 step | 下一 macro state 或 final best | 屏障前不更新参数 |
| 4 | macro best | final binary 指标、pixel→Region、NPZ/JSON/GDS | macro artifacts | 不逐 core 写 GDS |
| 5 | 全部 macro GDS | ownership merge、可选最终光刻 | final artifacts | 全局 merge 恰一次 |

### 7.5 Planned Call Graph

```text
main/run_simple_ilt.py::main
└─ main/_simple_ilt_workflow.py::run_simple_ilt
   └─ main/_ilt_workflow.py::run_ilt_workflow(METHOD, config)
      ├─ prepare_pixel_problems                         [macro loop; KLayout once open]
      │  └─ opc.input.pixel.prepare_pixel_macro_problem
      ├─ ICCAD13Lithography
      ├─ per macro
      │  ├─ PixelMacroProblem.load
      │  ├─ simple.optimize_simple_macro                [macro state loop N+1]
      │  │  └─ per core batch                           [same snapshot]
      │  │     ├─ problem.target_canvas/trainable_index_canvas/ownership_canvas
      │  │     ├─ model.forward_many                    [once per batch state]
      │  │     └─ scatter-add local gradient            [CPU macro gradient]
      │  ├─ model.forward_many(best binary)             [per core batch final evaluation]
      │  ├─ reconstruct_pixel_region
      │  └─ write_macro_gds + NPZ/JSON
      ├─ merge_macro_results                           [once]
      ├─ save_final_lithography                        [optional]
      └─ summary.json
```

## 8. Data Contracts

### `PixelMacroProblem`

- Owner：`opc.input.pixel.problem`
- Lifetime：one prepared macro；disk persisted，solve 时一次加载。
- Mutability：frozen；数组构造期规范化为只读语义。
- Resident：disk + 当前 macro CPU。
- Coordinate：全局 DBU 对应的局部 `[y,x]` pixel grid，origin=`macro.query_box.left/bottom`。

| Field | dtype | shape | unit | meaning |
|---|---|---|---|---|
| `macro` | `MacroSpec` | scalar | DBU | 既有 macro/core 网格 |
| `layer` | `LayerSpec` | scalar | - | 输出目标层 |
| `polarity` | `MaskPolarity` | scalar | - | GDS polygon→transmission 解释 |
| `target_u8` | NumPy `uint8` | `[Hq,Wq]` | 1/255 | query box 覆盖率 transmission，0..255 |

ILT prepare 已保证 query/macro/core box 尺寸整除 pixel，因此
`Hq=query_box.height/pixel_dbu`，`Wq` 同理；不得存每 core 的重复 256 canvas。

Public operations：

```python
prepare_pixel_macro_problem(batch, layer, polarity, macro) -> PixelMacroProblem
PixelMacroProblem.save(path) -> Path
PixelMacroProblem.load(path) -> PixelMacroProblem
PixelMacroProblem.target_canvas(core_index) -> np.ndarray[np.uint8]       # [256,256]
PixelMacroProblem.ownership_canvas(core_index) -> np.ndarray[np.bool_]   # [256,256]
PixelMacroProblem.trainable_index_canvas(core_index) -> np.ndarray[np.int32]  # [256,256]，macro 外=-1
reconstruct_pixel_region(problem, binary_ownership) -> kdb.Region
```

`trainable_index_canvas` 的非负值是 `[Hm,Wm]` macro parameter 的 row-major 扁平索引；同一物理
pixel 在不同 core context 中必须返回同一个索引。

### `ILTStateRecord`

- Owner：`opc.iteration.ilt._common`
- Lifetime：one macro result；每条记录对应完整 macro 已评价 state。

| Field | dtype | unit | meaning |
|---|---|---|---|
| `state_index` | int | state | 全方法单调状态编号，0 起 |
| `stage_index` | int | stage | Simple 固定 0；为后续多尺度保留真实通用坐标 |
| `stage_state_index` | int | state | Simple 等于 state_index |
| `scale` | int | pixel ratio | Simple 固定 1 |
| `total_loss` | float | 1 | 全部 core ownership 加权 loss 之和 |
| `nominal_l2` | float | 1 | ownership 连续 nominal L2 |
| `process_l2` | float | 1 | ownership 全 process conditions 对 target L2 |
| `pvband_loss` | float | 1 | ownership process max-min 连续平方差 |
| `curvature_loss` | float | 1 | ownership mask 曲率平方和 |
| `elapsed_seconds` | float | second | 本 macro state 全部 core wall time |

### `ILTMacroResult`

- Owner：具体 ILT optimizer；workflow 只读。
- Lifetime：one macro solve。
- Resident：CPU；GPU batch graph 在每批 backward 后释放。

| Field | dtype | shape | meaning |
|---|---|---|---|
| `best_parameters` | NumPy float32 | `[Hm,Wm]` | 唯一 macro best 已评价参数 |
| `soft_mask` | NumPy float32 | `[Hm,Wm]` | 对应 macro ownership transmission |
| `binary_mask` | NumPy bool | `[Hm,Wm]` | 按方法定义二值化的 macro best mask |
| `best_state_index` | int | scalar | 唯一 macro best state |
| `records` | tuple[`ILTStateRecord`,...] | `[N+1]` | macro 状态记录 |

`Hm/Wm` 是 macro ownership 的整像素形状。结果不携带 autograd graph。

### `ILTMethod`

`main._ilt_workflow` 的 frozen 描述对象，仅含当前 workflow 必须调用的四个差异点：

| Field | meaning |
|---|---|
| `method_name` | summary/artifact 稳定标识 `simple_ilt` |
| `config_type` | `load_config` 请求的算法 dataclass |
| `optimize_macro` | 统一 solver 签名 |
| `evaluated_states` | 从具体配置返回每 macro 评价状态数；Simple=`iterations+1` |

不得增加 save/summary 等没有真实差异的 hook。

### 8.1 Configuration Contract

新增 `[simple_ilt]`：

| Key | Type | Required | Default | Validation |
|---|---|---|---|---|
| `iterations` | strict int | yes | None | `>=1` |
| `step_size` | float | yes | None | finite `>0` |
| `sigmoid_steepness` | float | yes | None | finite `>0` |
| `weight_process_l2` | float | yes | None | finite `>=0` |
| `weight_pvband` | float | yes | None | finite `>=0` |
| `curvature_weight` | float | yes | None | finite `>=0` |
| `mask_threshold` | float | yes | None | finite `(0,1)` |
| `batch_size` | strict int | yes | None | `>=1` |

nominal L2 权重固定 1.0，与 OpenILT Simple 公式一致；不得增加未使用 WeightEPE。
配置还必须包含既有 `[layout]`、`[partition]`、`[lithography]`、`[output]`，MUST NOT 要求 `[edge]`。

### 8.2 Persisted Artifact Contract

| Artifact | Format/version | Required content |
|---|---|---|
| `ilt_plan.json` | JSON v1 | layout/top/layer/polarity/dbu/grid、macro entries、pixel/problem bytes、准备时间/RSS |
| `pixel_problems/<macro>.npz` | `myopc.pixel-ilt-problem` v1，uncompressed，allow_pickle=False | macro cuts/boxes/grid、layer/polarity、`target_u8[Hq,Wq]` |
| `macros/<macro>/simple_ilt_result.npz` | v1 | ownership origin/shape、best_parameters float32、soft_mask float32、binary_mask uint8、best_state_index int32 scalar |
| `macros/<macro>/metrics.json` | JSON v1 | aggregated records、binary L2/PVBand、core count、耗时 |
| `macros/<macro>/best.gds` | GDS | `RESULT` cell、目标 layer、完整 macro 候选 |
| final layout | GDS | 现有 ownership merge/cell_mode contract |
| `final_lithography/` | 现有 manifest v1 + PNG | 仅 `save_final_lithography=true` |
| `summary.json` | JSON | method、规模、配置、逐 macro 指标/产物、资源与 seam 策略 |

所有写出使用既有原子 I/O；失败不得留下宣称 completed 的 plan/summary。

## 9. Interface Changes

### IF-001：抽出算法无关网格 DBU 解析

Current：

```python
resolve_prepare_config(partition, litho, edge, dbu_nm) -> PrepareRuntime
```

Target：

```python
resolve_grid_config(partition, litho, dbu_nm) -> GridRuntime
resolve_prepare_config(partition, litho, edge, dbu_nm) -> PrepareRuntime
```

`GridRuntime` 只保存 `core_dbu/context_dbu/pixel_dbu/macro_size_dbu`；`PrepareRuntime` 改为
`grid: GridRuntime` + `fragmentation`。现有 `_macro_pipeline.prepare_problems` 调用迁到
`runtime.grid.*`；数值和异常保持不变。ILT workflow 直接调用 `resolve_grid_config`，不读取 EdgeConfig。

### IF-002：GDS writer 显式接收 layer

Current：

```python
write_macro_gds(problem: MacroProblem, region: kdb.Region, path: Path, dbu_um: float) -> Path
```

Target：

```python
write_macro_gds(layer: LayerSpec, region: kdb.Region, path: Path, dbu_um: float) -> Path
```

迁移全部两个生产调用点和测试。行为、cell 名和原子写不变；不为 edge/pixel 建 Protocol。

### IF-003：配置解析支持外部 postponed annotations dataclass

Current：`_parse_config` 直接读取 `dataclasses.Field.type`，只验证过本模块即时类型注解。

Target：`_parse_config` 对每个 config type 调用一次 `get_type_hints(config_type)`，逐字段把解析后的
真实类型交给 `_parse_scalar`。无法解析的注解异常必须传播，不得退回字符串猜测。

### IF-004：新增统一 optimizer 签名

```python
optimize_simple_macro(
    problem: PixelMacroProblem,
    model: LithographyModel,
    config: SimpleILTConfig,
    *,
    on_tiles_completed: Callable[[int], None] | None = None,
) -> ILTMacroResult
```

solver 内部按 `config.batch_size` 提取 `[B,256,256]` simulation canvas；非法 problem/config
抛 `ValueError`，模型/CUDA 异常原样传播。

### IF-005：新增公共 workflow 与直接入口

```python
run_ilt_workflow(method: ILTMethod, config_path: str | Path) -> dict
run_simple_ilt(config_path: str | Path) -> dict
main/run_simple_ilt.py::main() -> int
```

CLI 只接受一个可选位置参数 config，默认 `config/simple_ilt.toml`；业务异常不在 workflow 吞掉。

## 10. Algorithm

### 10.1 Pixel problem 准备

```text
open layout once
resolve grid DBU; plan macros
for macro in stable row-major order:
    materialize_intersecting(macro.query_box) once
    normalize physical region
    coverage = rasterize_region_window(region, query_box, pixel)
    transmission = coverage if clear else 1-coverage
    target_u8 = round(clamp(transmission,0,1)*255)
    save one PixelMacroProblem
write ilt_plan only after all macros succeed
```

### 10.2 Macro 同步状态与 core batch 梯度

```text
T = macro ownership target_u8 / 255
T_clamped = clamp(T, float32_eps, 1-float32_eps)
macro_parameters = log(T_clamped/(1-T_clamped)) / beta
assert abs(sigmoid(beta*macro_parameters)-T) <= 1e-6  # 【已取代：见 REQ-006 标记】
best_macro_loss = +inf

for state in 0..N:
    build_gradient = state < N
    macro_gradient = zeros_like(macro_parameters) if build_gradient else None
    macro loss components = 0
    for core batch in stable order:
        extract full 256x256 target transmission canvas as target_u8 / 255
        gather current macro parameters for every trainable pixel in each simulation context
        mask = target canvas, overwritten by sigmoid(beta*local_parameters) where trainable_index >= 0
        printed = forward_many(mask, nominal+dose_max+defocus_min)
        compute each core loss only on its core ownership
        if build_gradient:
            backward(sum(core losses))
            scatter-add local parameter gradients into macro_gradient by trainable_index
        release batch tensors/autograd graph; notify progress(B)
    sum all core components into one macro state record
    update best_macro_parameters only if macro_total_loss strictly decreases
    if state == N: break
    macro_parameters -= step_size * macro_gradient
```

同一 state 的 batch 只读同一 `macro_parameters` snapshot；scatter-add 是求和，禁止按一个 pixel
出现的 core 数量做平均。macro 外部 context/padding 从 target canvas 直通且无参数。严格 0/1
target 的 epsilon clamp 只改变 state0 的数值近似，不改变 target、loss ownership 或最终二值阈值。

曲率使用固定 3×3 零和核；只对卷积有效区对应的 ownership 像素求和。`curvature_weight=0`
时不得执行 conv2d。process 条件为空不作为配置能力暴露；当前固定两个既有 process conditions。

### 10.3 最终评价与回写

```text
best_soft = sigmoid(beta*best_macro_parameters)
binary_macro = best_soft >= mask_threshold
for each core batch:
    compose full simulation canvas from binary_macro inside this macro and initial target outside
    forward_many(canvas, three conditions) under no_grad
    evaluate binary L2/PVBand only on each core ownership
reconstruct row-run Region; polarity inverse; clip ownership; merge
```

### 10.4 Boundary Conditions

| Condition | Required behavior |
|---|---|
| 空目标层 | prepare 前 `ValueError`，无 plan |
| 最后 core 小于名义 core | 允许缩短，但实际宽高必须为 pixel 整数倍；否则 raster 前失败 |
| core seam | 同一 macro 参数与梯度跨 core 共享；loss ownership 唯一，不平均重叠梯度 |
| macro seam | macro 外 context 固定初始 target；最终仅写自身 ownership，记录独立 macro 限制 |
| clear/opaque | solver 输入均为 transmission；最终 Region 按 polarity 逆变换 |
| hole/concave/diagonal | 像素域允许；GDS 为合并的 pixel stair-step geometry |
| batch 尾部不足 B | 真实 batch 大小，进度按真实数 |
| loss/parameter 非有限 | `FloatingPointError`，不写 macro completed artifact |
| `curvature_weight>0` 且 canvas 小于 3 | 当前 canvas=256，不另加不可达分支 |

### 10.5 State Transition

```text
S0 --evaluate/save best--> metrics(S0) --backward/step--> S1
S1 --evaluate/save best--> metrics(S1) --backward/step--> S2
...
SN --forward/evaluate/save best--> metrics(SN) --no backward/step--> result
```

## 11. Ownership and State

| State/data | Owner | Writers | Readers | Publish point | Lifetime |
|---|---|---|---|---|---|
| PixelMacroProblem | input layer | prepare only | workflow | NPZ atomic replace | persisted macro |
| target/context | problem | none | optimizer/model | problem load | one macro |
| macro parameters | optimizer | synchronized SGD step | all core batches | all core gradients complete | one macro |
| macro gradient | optimizer | local gradient scatter-add | synchronized SGD step | all core batches complete | one state |
| macro best | optimizer | strict lower macro total loss | workflow/artifact writer | evaluated macro state | one macro |
| macro result arrays | optimizer | best state materialization | artifact writer | solve complete | one macro |
| macro GDS | workflow | writer | final merge | macro complete | disk |
| final GDS | merge | merge only | user/final litho | all macros complete | disk |

macro 外 context 是只读近似边界；core context 内的 macro-trainable pixel 不是只读。单机顺序、
不同 batch 切法和未来并行不得改变 loss ownership、梯度求和或 macro best 语义。异常时不发布
当前 macro；既有已完成 macro 文件可留作诊断，但不得写最终 summary。

## 12. Error Handling

### ERR-001：配置/网格非法

- Detection：`load_config`、`resolve_grid_config`、`plan_macros`、ILT 实际 macro/core cuts 对齐校验
- Behavior：`ValueError`，消息包含 section/key 或尺寸原因。
- MUST NOT：自动改 pixel/core/context、保留 partial ownership pixel 或 fallback CPU。

### ERR-002：problem 损坏

- Detection：`PixelMacroProblem.load/__post_init__`
- Behavior：`ValueError`；版本、shape、dtype、长度或范围不符立即失败。
- MUST NOT：`allow_pickle=True`、补零修复、重读源 GDS。

### ERR-003：优化非有限

- Detection：每个已评价 state 的 macro loss、local gradient、累积 gradient 与 step 后 parameters。
- Behavior：`FloatingPointError`，不发布当前 macro。
- MUST NOT：跳过样本、clamp NaN、降低学习率重试。

### ERR-004：I/O、KLayout、CUDA、PNG

- Behavior：原异常传播或增加明确路径上下文后链式传播。
- MUST NOT：宽泛捕获后返回 completed/partial summary。

## 13. Performance and Memory Constraints

### PERF-001

prepare 对每个 macro 恰好一次 KLayout materialization 和一次 query-box raster；不得逐 core KLayout。

### PERF-002

CPU 常驻上界为一个 `uint8[Hq,Wq]` problem、当前/梯度/best 三个 macro ownership float32 数组、
final soft/binary 输出和一个 core batch；不得常驻全 reticle raster。

### PERF-003

GPU 常驻为一个 `[B,256,256]` batch 的 local parameters、mask 与光刻 autograd；batch backward 后
只回传 local gradient 并释放。macro parameters/gradient/best 常驻 CPU，Simple SGD 不得分配 Adam
两份状态，也不得把整个 macro 复制到 GPU。

### PERF-004

同一 macro state 每 batch 必须一次 `forward_many` 产生三条件；final binary 评价每 batch 再一次。
不得按 condition 重复 mask FFT。`curvature_weight=0` 时不得构建 curvature kernel/conv graph。

### PERF-005

pixel→Region 允许按行/连续 run 的 Python 循环，不允许逐 pixel KLayout insert；每个 macro 只 merge
一次 Region。复杂度目标 `O(Hq*Wq + C*(N+2)*256²)`，内存不随 reticle macro 数累积。

### PERF-006

使用 `TestReticle/corners_unit_clear.gds`、`iterations=1`、CPU 与可用 CUDA 记录 prepare/solve/merge/
total、RSS/CUDA peak；本 change 只记录基线，不设硬阈值。

## 14. File-Level Change Plan

| File / Symbol | File type | Action | Contract change | Reason |
|---|---|---|---|---|
| `opc/input/pixel/__init__.py` | 业务代码 | add | 导出像素 problem API | REQ-002/003 |
| `opc/input/pixel/problem.py` | 业务代码 | add | `PixelMacroProblem`、实际 cuts 整像素校验、core mapping/reconstruct | REQ-003..005/012 |
| `opc/iteration/ilt/__init__.py` | 业务代码 | add | 导出公共结果与 Simple API | REQ-013 |
| `opc/iteration/ilt/_common.py` | 业务代码 | add | `ILTStateRecord/ILTMacroResult`、loss/curvature | REQ-006/008/013 |
| `opc/iteration/ilt/simple.py` | 业务代码 | add | `SimpleILTConfig/optimize_simple_macro` | REQ-006..009 |
| `main/configuration.py::GridRuntime/resolve_grid_config/PrepareRuntime/_parse_config/CONFIG_SECTIONS` | 业务代码 | modify | 抽出无 edge 网格解析；解析外部 dataclass 注解；注册 `[simple_ilt]` | IF-001/003、REQ-017 |
| `main/_macro_pipeline.py::prepare_problems/write_macro_gds` | 业务代码 | modify | 适配 GridRuntime 和显式 layer writer | IF-001/002 |
| `main/run_single_pass.py` | 业务代码 | modify | 适配 GridRuntime 字段结构（`runtime.grid.*`） | IF-001 |
| `main/run_macro_pipeline.py::run_round` | 业务代码 | modify | writer 调用迁移 | IF-002 |
| `main/_mbopc_workflow.py::_solve_macro` | 业务代码 | modify | writer 调用迁移 | IF-002 |
| `main/_ilt_workflow.py` | 业务代码 | add | `ILTMethod`、pixel prepare、macro/output 生命周期 | REQ-001/009..015 |
| `main/_simple_ilt_workflow.py` | 方法适配器 | add | METHOD + `run_simple_ilt` | REQ-013 |
| `main/run_simple_ilt.py` | 运行入口 | add | 直接 Python 入口 | REQ-001 |
| `config/simple_ilt.toml` | 配置文件 | add | 完整 smoke 配置 | §8.1 |
| `tests/opc/input/test_pixel_problem.py` | 测试代码 | add | persistence/mapping/polarity/GDS | TEST-001..004 |
| `tests/opc/iteration/test_simple_ilt.py` | 测试代码 | add | 数学/state/best/batch/real model | TEST-005..010 |
| `tests/main/test_simple_ilt_runner.py` | 测试代码 | add | prepare/CLI/artifact/merge/progress | TEST-011..015 |
| `tests/main/test_configuration.py` | 测试代码 | modify | GridRuntime、Simple config、旧配置回归 | IF-001 |
| `tests/main/test_macro_pipeline.py`、`test_mbopc_runners.py` | 测试代码 | modify | writer 签名与行为零回归 | IF-002/REQ-016 |
| `doc/contracts/ilt.md` | 接口文档 | modify | 从“无实现”更新 Simple/共享契约和限制 | 交付 |
| `doc/contracts/opc_input.md` | 接口文档 | modify | pixel problem contract | 交付 |
| `doc/architecture/system.md`、`dataflow.md`、`data_model.md` | 架构文档 | modify | 当前 ILT 模块/数据流 | 交付 |
| `doc/development_manual.md`、`doc/test_manual.md` | 手册 | modify | 运行与测试方式 | 交付 |
| `doc/changes/active/CHG-20260818-simple-ilt/implementation_spec.md` → `doc/changes/completed/CHG-20260818-simple-ilt/implementation_spec.md` | 规格 | move | 状态、revision、实施证据 | 审批/交付 |
| `doc/changes/completed/CHG-20260818-simple-ilt/development_report.md` | 开发报告 | add | 实际实现、偏差、性能、清理审计 | 交付 |
| `doc/changes/completed/CHG-20260818-simple-ilt/test_report.md` | 测试报告 | add | 环境、命令、结果、产物 | 交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` | 项目记录 | modify | 同步实施 | AGENTS |

不得修改清单外文件；发现需要 `layout/geometry/lithography/evaluation` 改动时停止。

## 15. Test Specification

### TEST-001：pixel problem 一次栅格与持久化

- Level：unit/integration
- Given：非对称 clear/opaque 生成式 GDS，2×2 core。
- Then：每 macro 一次 raster；NPZ round-trip 逐值；无 edge API 调用。
- Covers：REQ-002..004。

### TEST-002：core ownership 完整唯一

- Given：常规 core、跨 core 图形、整像素缩短末端 core、非整像素 bbox、2×2 macro。
- Then：合法场景所有 macro ownership pixel/core loss owner count 恰 1；非整像素实际 core 在 raster
  前明确失败，不生成 partial ownership pixel。
- Covers：REQ-005、INV-001/006。

### TEST-003：pixel→Region 极性与几何矩阵

- Given：矩形、孔、凹形、斜边阶梯、多岛、全 0/全 1，clear/opaque。
- Then：回栅格 binary 与输入 owned raster 一致；Region 有效且裁到 ownership。
- Covers：REQ-004/012、INV-005/006。

### TEST-004：problem 错误与格式

- Then：版本/dtype/shape/cuts 损坏明确失败；`allow_pickle=False`。
- Covers：ERR-002。

### TEST-005：Simple 参数化与 loss 精确公式

- Given：由 0/0.25/0.5/0.75/1 coverage 量化得到的 `target_u8`、identity differentiable model 与手算小图。
- Then：~~logit 初始化使 state0 soft 在 `1e-6` 内恢复 `target_u8/255`~~【已取代：state0 soft = σ(β(2T−1))，公式复算锁定】；四项 loss、ownership 选择、
  curvature on/off 逐值一致，严格 0/1 参数有限。
- Covers：REQ-006。

### TEST-006：N 次更新/N+1 已评价状态

- Given：N=1/2，已知 SGD 梯度。
- Then：全部 core 同 state snapshot、宏级 state 编号、屏障后 step、最终评价和 best snapshot 精确；
  无 core 提前更新，state N 不 backward 且无多余 step。
- Covers：REQ-007、INV-003。

### TEST-007：macro best 与 batch 不变量

- Given：两个 core 的局部最优轮不同；batch=1、2、core 正序/反序。
- Then：只按完整 macro total loss 选择一个 best state；不得拼接局部 best；batch 切分/顺序下
  macro records、best 和输出在约定容差内一致。
- Covers：REQ-008、INV-004。

### TEST-008：跨 core context 梯度累加

- Given：一个 macro 内相邻 core，共享光学 context；某个 macro pixel 同时影响两个 core loss。
- Then：两个 core 只统计各自 ownership loss，但该 pixel 获得两份 local gradient 之和且只对应一个
  参数；去掉任一 core loss 时梯度相应减少。macro 外 context/padding 始终等于初始 target 且无梯度。
- Covers：REQ-005/009、INV-002。

### TEST-009：真实 ICCAD13 backward

- Given：CPU 小 batch，一次更新；CUDA 可用时同输入。
- Then：loss/参数/累积梯度有限，至少一个 macro-trainable pixel 改变；CPU/CUDA loss 容差
  `1e-4 relative/absolute`。
- Covers：REQ-006/016。

### TEST-010：调用次数与释放边界

- Then：训练 forward=`(N+1)*core_batch_count`，其中仅前 N 个 state backward；final binary
  forward=`core_batch_count`。前 N 个 state 各恰一次 macro gradient 清零与 step 屏障，三条件共享，
  curvature=0 无 conv，batch graph 及时释放，callback 真实。
- Covers：REQ-009/014、PERF-003/004。

### TEST-011：配置与直接入口

- Given：合法/缺键/未知键/NaN/float 冒充 int；仓库外 cwd 直接运行。
- Then：合法完成，非法在 prepare 前失败，入口无需 pip install。
- Covers：REQ-001/015、ERR-001。

该测试还必须定义一个启用 `from __future__ import annotations` 的临时/fixture Config，证明
`get_type_hints` 后 int/float/tuple/Path 可解析，并对全部既有 Config 做参数化零回归。

### TEST-012：端到端产物

- Given：生成式跨 core clear/opaque GDS，一轮。
- Then：plan/problem/result/metrics/best/final/summary 字段、dtype、shape、路径完整。
- Covers：REQ-003/010/011/015。

### TEST-013：多 macro merge 与 seam ownership

- Given：2×2 macro×多 core，图形跨 core/macro。
- Then：每 macro 独立完成；merge 调用恰一次；最终 raster 等于所有 owned 输出拼接，无重叠。
- Covers：REQ-011、INV-006。

### TEST-014：进度与异常收尾

- Then：total/updates 精确；第二 macro 中途异常时内外条均 close，final summary 不存在。
- Covers：REQ-014、ERR-004。

### TEST-015：现有流程零回归

- Then：macro pipeline、simple/gradient MB-OPC 定向测试与全量测试全绿；writer 数值不变。
- Covers：REQ-016。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected distinction |
|---|---|---|
| Geometry | rectangle/hole/concave/diagonal/multi-island/empty/full | 栅格和 Region 逐类断言 |
| Polarity | clear/opaque | solver 统一 transmission，GDS 逆变换不同 |
| Boundary | non-aligned final core、cross core、cross macro | 前置拒绝、唯一 loss/write、merge |
| State | N=1/2、不同 core 局部 best | macro N+1、同步屏障、batch invariance |
| Scale | one core/multi core/2×2 macro | 内存不累计、产物完整 |
| Device | CPU/CUDA | backward 与容差 |
| Failure | config/problem/nonfinite/I/O | 明确异常、无 completed summary |

### 15.2 Verification Commands

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/input/test_pixel_problem.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/opc/iteration/test_simple_ilt.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests/main/test_simple_ilt_runner.py tests/main/test_macro_pipeline.py tests/main/test_mbopc_runners.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check common layout geometry opc lithography evaluation main tests
D:\app\miniforge\envs\myopc\python.exe -m compileall -q common layout geometry opc lithography evaluation main tests
D:\app\miniforge\envs\myopc\python.exe main/run_simple_ilt.py config/simple_ilt.toml
git diff --check
```

CUDA 测试仅在 `torch.cuda.is_available()` 为 false 时 skip；CPU 核心 contract 不得 skip。

## 16. Requirement Traceability

| Requirement / Invariant | Implementation | Tests | Acceptance |
|---|---|---|---|
| REQ-001/013 | runner/adapter/workflow | TEST-011 | AC-001 |
| REQ-002..005 | pixel problem + prepare | TEST-001..004 | AC-002 |
| REQ-006..009 | simple/common | TEST-005..010 | AC-003 |
| REQ-010/011/014/015 | workflow/output | TEST-012..014 | AC-004/005 |
| REQ-012 | reconstruct_pixel_region | TEST-003/013 | AC-002/004 |
| REQ-016 | existing callers/tests | TEST-015 | AC-006 |
| REQ-017 | `configuration._parse_config` | TEST-011/015 | AC-006 |
| INV-001/002 | ownership/trainable mapping/gradient sum | TEST-002/008 | AC-002/003 |
| INV-003/004 | macro state/best | TEST-006/007 | AC-003 |
| INV-005/006 | coordinate/polarity/merge | TEST-001/003/013 | AC-002/004 |

## 17. Acceptance Criteria

- [ ] **AC-001**：直接 Python 入口从仓库外 cwd 成功运行，summary method=`simple_ilt`。
- [ ] **AC-002**：pixel problem、坐标、极性、实际 core 整像素限制与 pixel→GDS 全矩阵通过。
- [ ] **AC-003**：coverage 初始化、跨 core 梯度和、macro 同步 state/best、batch invariance 与真实 backward 全通过。
- [ ] **AC-004**：2×2 macro 跨界端到端产物完整，merge 恰一次，拼接 raster 无重叠遗漏。
- [ ] **AC-005**：smoke 记录时间、RSS/CUDA peak、规模与 macro-independent seam 限制。
- [ ] **AC-006**：全量 pytest、ruff、compileall、diff check 通过，现有 writer 数值零变化。
- [ ] **AC-007**：开发/测试手册、contracts/architecture、两报告和规划记录同步。
- [ ] **AC-008**：最终审计无未调用函数、重复 raster/atomic writer、吞异常、一次性抽象或清单外改动。

## 18. Compatibility and Migration

### COMP-001

- API：新增 ILT public API；`write_macro_gds` 和 `PrepareRuntime` 是 main 内部有意迁移，全部调用点同批修改。
- Data：新增 pixel problem/result v1；不读取 `00_PAST` NPZ，无旧数据兼容要求。
- Archive：`00_PAST` 只读。
- CLI：新增 `run_simple_ilt.py`，不恢复旧 `run_simpleilt.py` 参数兼容。
- Numerical：Simple 核心 loss 与 OpenILT/PAST 同源；~~coverage-preserving logit 初始化~~【已取代：OpenILT 2T−1 初始化，见 CHG-20260819】、macro 同步
  N+1 状态、macro best、ownership-only loss 统计与 context gradient、具名条件是明确工程修正，
  不承诺与 OpenILT benchmark 逐值相同。

## 19. Decisions

### DEC-001：首个 ILT 使用 macro 同步状态与 core-batched 光刻

- Decision：每个 macro 维护唯一参数/梯度/best；core 只负责 simulation canvas 与唯一 loss ownership，
  同一 state 全部 core 梯度求和后统一 step。
- Reason：保留跨 core 光学耦合的梯度，并保证被选 best 是实际评价过的完整 macro mask；GPU 仍只常驻
  core batch，显存不随 macro/reticle 尺寸增长。
- Rejected：core 独立参数/固定 context/per-core best，会丢梯度并拼出未整体评价的 mask；overlap
  loss averaging 会改变全局 loss 的真实梯度。
- Consequence：core seam 不再来自独立 state；macro 之间仍固定外部 context，跨 macro 协调另立 change。

### DEC-002：建立 pixel input 子包，不复用 edge problem

- Reason：像素 target 无 segment/owner CSR；复用会制造空字段和错误依赖。

### DEC-003：建立最小 `ILTMethod`，不建基类/注册器

- Reason：共享 workflow 有当前 Simple 调用方，且后三份已规划方法只需相同三处差异；无运行时发现需求。

### DEC-004：macro best，评价最后更新

- Reason：修复 OpenILT batch-size 依赖、无效末步和 per-core best 拼接；与当前 MB-OPC “全部 core
  读取同一状态、指标属于已评价状态、屏障后发布更新”一致。

### DEC-005：最终输出 GDS

- Reason：当前项目端到端权威产物是版图；仅 NPZ/PNG 不能接入 merge 与后续光刻验证。
- Rejected：逐像素 polygon；性能不可接受。采用 row-run+Region merge。

### DEC-006：EPE/shot 不迁

- Reason：EPE 当前依赖边段探针，shot 当前无实现；二者不是 Simple 训练核心，顺带迁移违反最小范围。

### DEC-007：算法 Config 直接注册，不复制 main/runtime 两份字段

- Reason：配置字段不含 nm→DBU 派生值，复制同构 dataclass 会制造双真源；`get_type_hints` 是让统一
  parser 支持外部 dataclass 所需的最小改动，并有当前 Simple 调用方和全部既有配置回归。

## 20. Open Questions

### 20.1 Blocking

None.

### 20.2 Non-blocking

- 未来是否采用跨 macro 同步 ILT、overlap Schwarz 或 seam refinement，必须另立 change。
- pixel stair-step GDS 的 MRC/shot 优化不属于本 change。

## 21. Implementation Freedom

实现 AI 可以决定局部变量名、行程扫描的等价向量化写法、私有 helper 排布；不得改变 public API、
字段、坐标/极性、macro 同步语义、N+1 状态、macro best、产物格式、依赖方向或文件清单。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Required verification | Suggested local commit |
|---|---|---|---|---|
| A | GridRuntime 与 writer 最小迁移，既有流程零变化 | configuration/_macro_pipeline/两个调用方/tests | TEST-015 + full targeted | `refactor(main): 解耦网格解析与宏版图写出` |
| B | pixel problem、NPZ、mapping、Region | opc/input/pixel + tests | TEST-001..004 | `feat(opc-input): 增加像素宏问题` |
| C | common + Simple optimizer | opc/iteration/ilt + tests | TEST-005..010 | `feat(ilt): 实现Simple像素优化` |
| D | workflow/adapter/runner/config/artifacts | main/config/tests | TEST-011..014 + direct main | `feat(main): 接入SimpleILT宏流程` |
| E | 全量、smoke、文档、报告、审计 | doc/planning | §15.2 全部 | `docs(ilt): 完成SimpleILT迁移报告` |

每阶段验证后只做本地 commit；未经用户明确授权不得 push。

## 23. Delivery and Final Audit

完成后必须更新开发/测试手册、专项开发/测试报告、contracts/architecture、任务规划；记录实际
环境、命令、pass/fail/skip、耗时、RSS/CUDA peak、产物和规格偏差。必须搜索并清理未调用函数、
重复实现、异常吞噬、一次性抽象、旧符号和无需求字段；核查全部新增 Python 文件/函数/测试的
中文 docstring 与关键坐标/性能/ownership 注释。正常应报告：未修改 layout/geometry/00_PAST/
用户数据，未推送远端。

## 24. Known Limitations and Future Work

- macro 之间不交换优化后参数，macro seam 仍可能存在。
- 输出是 pixel-grid stair-step 几何，不做全局轮廓平滑、MRC 或 shot 优化。
- 只支持现有 ICCAD13 256 canvas contract；模型可经 `LithographyModel` 替换，但网格尺寸仍由配置校验。
- 不支持 checkpoint/resume、offline NPZ 输入和分布式 worker。

## 25. Specification Approval Gate

- [ ] baseline 与相关工作树已核对；
- [ ] 用户确认 DEC-001 的 macro 同步/core-batched 梯度与跨 macro 独立取舍；
- [ ] public API、配置、pixel problem/result、坐标/极性和 GDS 输出已确认；
- [ ] 每个 MUST/invariant 映射到测试与 AC；
- [ ] Blocking Open Questions 为 None；
- [ ] 文件清单无 layout/geometry/00_PAST 改动；
- [ ] 实施阶段与本地 commit 边界可独立验证。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-18 | draft | 首版；选择 tile 独立大版图执行，冻结 pixel problem、N+1 状态、per-sample best 与最终 GDS | 待用户审核 |
| 0.2 | 2026-08-19 | draft | 修正为 macro 唯一参数、跨 core 梯度累加、同步 N+1、macro best、coverage-preserving 初始化与实际 core 整像素限制 | 待用户审核 |
| 0.3 | 2026-08-19 | approved | §14 补录 `main/run_single_pass.py`（PrepareRuntime 字段消费方，IF-001 迁移牵连，事实核对发现）；ILTStateRecord 的 stage/scale 字段经用户裁定保留；ilt_plan.json 须携带 merge/final-litho 兼容键记入实施约束 | 用户批准开发计划 |
| 1.0 | 2026-08-19 | completed | 五阶段实施完成（5ad8ac0/54ab866/1539b6f/fefaea8/bdf86ac + 本报告批）；偏差与裁决见 development_report.md（含 merge 空 macro 候选容忍修复）；525 passed；smoke 基线见 test_report.md | 开发/测试报告 |
| 1.1 | 2026-08-19 | completed | 标注 REQ-006/§10.2/TEST-005/COMP-001 的初始化描述已被 CHG-20260819 取代（正文保留历史） | 用户文档同步指令 |
