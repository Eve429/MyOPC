---
id: CHG-20260816-gradient-mbopc
title: 基于梯度的 MB-OPC 迁移
type: implementation-spec
status: approved
baseline_commit: e289f2c60c3db6302c687bdf30c6977f108c47f0
implementation_base_commit: fbc059b（e289f2c 后仅规格入库与文档提交，业务代码零漂移）
baseline_worktree: dirty
baseline_dirty_paths:
  - doc/opc/mbopc_migration_design.md
scope:
  - opc/iteration/mbopc
  - main
  - config
  - tests/opc/iteration
  - tests/main
  - doc
depends_on:
  - doc/macro_core/macro_core_pipeline_design.md
  - doc/lithography/lithography_migration_design.md
  - doc/opc/mbopc_migration_design.md
supersedes: []
---

# 基于梯度的 MB-OPC 迁移

## 0. Document Contract

本文档是该 change 不依赖聊天上下文的唯一实现规格。仓库根 `AGENTS.md`、本文记录的基线源码、
测试以及 `depends_on` 文档仍是必须读取的上位约束和事实来源；本文不得覆盖 `AGENTS.md`。

实现 AI MUST：

- 以本文档的 Target Behavior、Invariants、Interfaces 和 Acceptance Criteria 为实现目标；
- 开始前确认基线 commit，并核对 `doc/opc/mbopc_migration_design.md` 的未提交修改；
- 只修改 File-Level Change Plan 列出的文件和符号；必须扩大范围时停止并请求修订规格；
- 不依赖生成本文档时的聊天上下文，不自行补充未定义的产品或算法需求；
- 若本文状态不是 `approved`、Blocking Open Question 非空或基线实质漂移，不得实施；
- 逐项提供 requirement、test、acceptance 和性能证据。

实现 AI MUST NOT：

- 把本 change 称为完整 DiffOPC、MRC-clean OPC、SRAF OPC 或全局同步 reticle OPC；
- 为未来 ILT/SRAF/其他 optimizer 增加注册器、统一 Solver 基类或空接口；
- 静默 fallback、吞掉异常、改变 ownership、坐标、极性或 macro 同步语义；
- 修改 `layout/`、`geometry/`、`00_PAST/` 或用户 GDS。

规范词：MUST/MUST NOT 为验收强制项；SHOULD 除非 Decisions 有可验证理由否则必须遵守；
MAY 是不改变 contract 时的实现自由。

## 1. Objective

在现有 `MacroProblem`、ICCAD13 可微光刻模型和 macro/core 流水线上新增一个最小、可验证的
梯度 MB-OPC：使用精确面积覆盖率 mask 前向、DiffOPC midpoint edge-gradient surrogate 和
同步 Adam 优化固定边段法向位移，并从 GDS/OASIS 直接产出最佳合法版图、指标与最终光刻图。

## 2. Baseline and Evidence

### 2.1 Baseline

- Commit：`e289f2c60c3db6302c687bdf30c6977f108c47f0`
- Worktree：dirty
- 与本 change 有关的未提交文件：`doc/opc/mbopc_migration_design.md`；只作为用户正在修改的
  依赖文档读取，不得覆盖或纳入功能提交。
- 其他未提交项：`AGENTS.md`、`CLAUDE.md`、`main/main_test_lithography.py`、`.planning/`、
  `doc/templates/`；全部保留并排除在功能提交之外。
- 验证环境：Windows、Python 3.12、KLayout 0.30.10、NumPy 2.5.1、PyTorch 2.5.1；
  基线相关测试 249 项通过。

### 2.2 Confirmed Facts

| Fact ID | Confirmed fact | Evidence | Verification method |
|---|---|---|---|
| FACT-001 | `MacroProblem` 已保存固定边段、唯一 owner 和 core membership CSR | `opc/input/edge/problem.py::MacroProblem` | 静态阅读 + input 测试 |
| FACT-002 | 正位移的公共法向语义在 clear/opaque 下都扩大透光区域 | `opc/input/edge/fragmentation.py::fragment_edges` | 源码与 polarity 测试 |
| FACT-003 | 候选重建会检查位移上限、环翻转、hole 越界和 Polygon 合法性 | `opc/input/edge/reconstruction.py::reconstruct_region` | 源码与 simple 回归 |
| FACT-004 | 光刻模型保留 mask autograd，且多工艺条件共享一次 FFT | `lithography/iccad13.py::ICCAD13Lithography.forward_many` | 单元测试 |
| FACT-005 | `points_to_canvas()` 是 DBU 到居中 canvas `(x,y)` 的唯一换算 | `opc/input/raster.py::points_to_canvas` | 坐标测试 |
| FACT-006 | mask raster 是精确面积覆盖率，padding 恒为零 | `opc/input/raster.py::rasterize_mask_canvas` | raster 测试 |
| FACT-007 | simple MB-OPC 已按 macro 独立求解、全部完成后一次 merge | `main/_mbopc_workflow.py::_run_mbopc` | runner 测试 |
| FACT-008 | 当前光刻协议已满足梯度方法，不需要新模型接口 | `lithography/contracts.py::LithographyModel` | 静态阅读 |
| FACT-009 | 相关基线测试通过 249 项 | 本文 §15.2 的基线命令 | 实际执行，60.40 s |
| FACT-010 | 真实 ICCAD13 下，目标扩大时四边 `dL/dMask<0`、目标缩小时 `dL/dMask>0` | 调研期 256² 矩形数值实验 | CPU forward/backward，四边符号一致 |

### 2.3 Uncertainty Boundary

- 本规格不能证明任意版图或任意超参数下 loss 单调下降或达到工艺最优；验收只验证算法契约、
  方向、同步、合法发布和固定 smoke workload。
- 本项目采用 KLayout 面积覆盖率 forward，而参考代码采用 Manhattan ray casting；两者不会逐
  像素复现官方 benchmark，故官方 L2/PV/EPE 数字不是本 change 的 acceptance oracle。
- 当前没有 foundry MRC rule deck；本 change 不能实现或宣称论文的 MRC-aware velocity。
- 独立 macro 在迭代中不交换邻 macro 的最新位移，边界 context 仍是参考几何。

上述不确定性均已通过 Scope、Known Limitations 和测试边界固定，不改变本次实现路径。

### 2.4 External and Archive References

| Reference | Role | Adopt | Explicitly reject | Reason |
|---|---|---|---|---|
| NVIDIA DiffOPC 论文，DOI `10.1145/3676536.3676764`，Algorithm 4 | 算法依据 | STE hard forward、segment midpoint mask gradient、velocity、Adam 思路 | MRC、SRAF、完整 benchmark 等价声明 | 本次只迁移最小梯度边段方法 |
| `https://github.com/NVlabs/DiffOPC` | 源码语义参考 | Binarize backward、best evaluated snapshot、连续 L2/PV loss | Hydra、OpenCV、逐 polygon Python 循环、日志框架 | 不符合本项目依赖与性能契约 |
| `00_PAST/opc/iteration/diffopc/` | 只读历史参考 | owner-only 计分、逐 batch backward、轮次屏障 | sigmoid occupancy-delta raster、旧 Problem/API、宽泛异常捕获 | 软边公式不是官方算法且未证明多 polygon 正确 |

官方仓库使用带非商业限制的 NVIDIA License。本 change MUST 依据论文公式和本项目接口重新实现，
MUST NOT 逐段复制官方代码；`00_PAST/` 保持只读。

## 3. Current Behavior

1. `opc/iteration/mbopc/simple.py::optimize_macro` 只提供固定步长、EPE 驱动的离散方法；
2. `opc/iteration/mbopc/simple.py::TargetCanvasCache` 是固定 target 的 CPU `uint8` 有界 LRU，
   只有 simple 方法消费；
3. `main/_mbopc_workflow.py::run_single_macro` 与 `run_multi_macro` 只调 simple 求解器；
4. `main/_macro_pipeline.py::prepare_problems` 已逐 macro 生成可复用 NPZ，梯度方法无需重做输入；
5. 当前没有 `GradientMBOPCConfig`、梯度 edge backward、梯度结果格式、TOML 或直接入口；
6. `evaluation/metrics.py` 返回离散 Python 指标，只能用于报告，不能作为训练 loss。

## 4. Target Behavior

### REQ-001

系统 MUST 通过 `python main/run_gradient_mbopc.py <config.toml>` 从 GDS/OASIS/GLP 配置直接运行
单 macro 或多 macro 梯度 MB-OPC，不要求 pip install 或离线 NPZ 手工准备。

### REQ-002

梯度方法 MUST 原样消费 `MacroProblem`，MUST NOT 修改其字段、NPZ version、segment ID、owner
或 membership。

### REQ-003

每个已评价状态的 mask forward MUST 来自 `reconstruct_region()` 后的合法 Region 与
`rasterize_mask_canvas()` 面积覆盖率；零位移 forward MUST 与现有精确 raster 逐值一致。

### REQ-004

反向 MUST 实现论文 Algorithm 4 的 midpoint surrogate：对每个参与当前 tile 的可写 segment，
在其当前中点双线性采样 `dL/dMask`；本项目标量位移同时驱动两个 endpoint，故该 tile 对
`d_s` 的梯度 MUST 为 `2 * g_mid`。

### REQ-005

梯度方向 MUST 使用现有单位公共法向，不按 H/V 分支；clear、opaque、hole 和斜边 MUST 使用同一
公式。MUST NOT 引入 sigmoid soft raster、soft temperature 或 segment-length 梯度缩放。

### REQ-006

训练 loss MUST 定义为：

```text
P = 全部 core ownership 像素数
L_nom = ΣΩ (printed_nominal - target)² / P
L_process = ΣΩ [(printed_dose_max - target)²
                + (printed_defocus_min - target)²] / P
L_pv = ΣΩ (printed_dose_max - printed_defocus_min)² / P
L_total = w_nominal * L_nom + w_process * L_process + w_pv * L_pv
```

三个权重来自显式配置、均非负且至少一个为正。离散 L2/PVBand/EPE 只作同状态诊断，MUST NOT
参与 `L_total` 或 best 选择。

### REQ-007

loss 只在 `ownership_canvas` 内累计；context/halo mask 只参与光刻传播。同一 owner segment 在
多个 tile membership 中得到的梯度 MUST 累加到同一个参数；`owner=-1` segment MUST 恒为零且
不得拥有 optimizer state。

### REQ-008

同一 macro 的一个 state 中，全部 tile/batch MUST 读取同一 `d_current`；每 batch MAY 立即
backward 并释放光刻输出，但 Adam `step()` MUST 位于全部 tile 完成后的唯一屏障之后。

### REQ-009

`iterations=N` MUST 表示最多发布 N 次参数更新；记录 MUST 为 baseline `state_index=0` 加最多 N
个移动后已评价状态。最后一次已发布状态 MUST 被评价，不得产生未评价候选作为 best 或输出。

### REQ-010

每次 Adam 候选 MUST 先裁到 `±problem.fragmentation.max_displacement_dbu`，再由
`reconstruct_region()` 验证；只有验证成功才发布为下一状态。候选非法时 MUST 保留最后已评价合法
状态和历史 best，并记录 `invalid_geometry` 及原始 `ReconstructionError` 文本。

### REQ-011

best MUST 只按已评价状态的 `total_loss` 严格小于更新；相同 loss 保留更早状态。结果中的
`best_displacements`、`best_state_index` 与该条 record MUST 属于同一个快照。

### REQ-012

每个 macro MUST 独立完成全部迭代并写局部结果；全部 macro 完成后 MUST 复用
`merge_macro_results()` 恰好合并一次。MUST NOT 增加逐轮 macro merge 或动态 context 交换。

### REQ-013

每个 macro MUST 保存 gradient result NPZ、metrics JSON 和 best GDS；全局 MUST 保存 summary、
final GDS，并按配置复用 `save_final_lithography()` 保存最终 nominal/binary PNG。

### REQ-014

直接入口 MUST 支持 tqdm tile/macro 进度；进度只在 batch backward 完成并释放主要 GPU 输出后更新，
异常路径 MUST 在 `finally` 关闭进度条。

### REQ-015

数值 loss、gradient 或 candidate parameter 非有限时 MUST 抛 `FloatingPointError`；I/O、CUDA OOM、
配置错误及未知程序异常 MUST 原样或带明确上下文传播，MUST NOT 转换成成功结果或静默停止。

### REQ-016

simple MB-OPC 的 public import、TOML、NPZ、CLI 和数值行为 MUST 保持兼容；共享 cache 移动后
`from opc.iteration.mbopc import TargetCanvasCache` MUST 继续有效。

### REQ-017

本 change MUST 不新增 OpenCV、Hydra、自定义 CUDA、SRAF、MRC、EPE training loss、optimizer
注册器、统一 ILT Problem 或 Solver 基类。

## 5. Scope

### 5.1 In Scope

- hard-area-raster forward + midpoint edge-gradient surrogate；
- owner-only Adam 位移、同步 batch 梯度累计、合法候选发布；
- nominal/process/PV 连续 loss 与离散 L2/PVBand/EPE 诊断；
- 单/多 macro 通用直接入口、配置、NPZ/JSON/GDS/PNG；
- CPU/CUDA、clear/opaque、hole、斜边、跨 core/macro 测试；
- target cache 的最小共享移动；
- 开发/测试手册、专项报告与最终简化审计。

### 5.2 Out of Scope

- 论文 MRC-aware velocity、rule deck、SRAF、动态新增/remesh segment；
- 论文官方 ray-casting/CUDA kernel、官方 benchmark 数值复现；
- differentiable EPE loss、curvature/shot loss、学习率 scheduler、optimizer 选择器；
- 逐轮跨 macro 同步、分布式 worker、断点恢复、自动调参；
- ILT 迁移或把 ILT 绑定到 `MacroProblem`；
- 全 reticle 常驻 GPU、全局 polygon normalize 新实现。

### 5.3 Protected Areas

- `00_PAST/**`、`layout/**`、`geometry/**`、用户 GDS：MUST NOT 修改；
- `opc/input/**`、`lithography/**`、`evaluation/**`：当前接口足够，MUST NOT 修改；
- `doc/opc/mbopc_migration_design.md` 的用户未提交修改：MUST 保留并排除提交。

## 6. Invariants

### INV-001

一个 segment 有且只有一个 `owner_indices`；只有 `owner>=0` 的 segment 有可训练参数。

Enforced by：`MacroProblem.__post_init__` 与 `GradientMBOPC` owner mapping 测试。

### INV-002

context segment 和 halo 像素只读；context displacement 恒为 0，halo loss 权重恒为 0。

Enforced by：参数只为 owner segment 分配、`ownership_canvas`、结果断言。

### INV-003

同一 state 的全部 batch 读取同一参数快照，optimizer 每 state 最多 step 一次。

Enforced by：`optimize_gradient_macro()` 的 batch loop 外屏障和调用顺序测试。

### INV-004

record 的 loss/指标、best snapshot 与输出几何只属于已经完成全部 tile 评价的合法状态。

Enforced by：评价后记录、更新前验证、best snapshot 测试。

### INV-005

数组第 0 维是 batch、图像为 `[B,H,W]`，像素索引为 `[y,x]`，点坐标为 `(x,y)`；canvas 行 0
对应最低 Y，外围 padding 为 0。

Enforced by：复用 `points_to_canvas()` 和坐标梯度测试。

### INV-006

正 displacement 在所有极性下扩大透光区域；optimizer 不含 polarity 分支。

Enforced by：现有 normal contract 与 clear/opaque 方向测试。

## 7. Architecture

### 7.1 Components

| Component | Responsibility | MUST NOT own |
|---|---|---|
| `opc.iteration.mbopc._cache` | simple/gradient 共用固定 target LRU | 求解、几何、GPU cache |
| `opc.iteration.mbopc.gradient` | edge surrogate、loss、Adam、state/best | layout 读取、macro merge、CLI |
| `main._mbopc_workflow` | TOML、macro 生命周期、进度、产物 | edge gradient 数学 |
| `main.run_gradient_mbopc` | 直接脚本参数与摘要 | 业务求解实现 |
| `main._macro_pipeline` | 已有 prepare/write/merge | 具体迭代方法 |

### 7.2 Dependency Direction

允许：

```text
main.run_gradient_mbopc
  -> main._mbopc_workflow
  -> opc.iteration.mbopc.gradient
  -> opc.input.edge / opc.input / lithography / evaluation

main._mbopc_workflow -> main._macro_pipeline
opc.iteration.mbopc.simple|gradient -> opc.iteration.mbopc._cache
```

禁止：

```text
layout / geometry / opc.input / lithography / evaluation -X-> gradient
gradient -X-> main
gradient -X-> simple
ILT -X-> MacroProblem（本 change 不涉及 ILT）
```

### 7.3 Data Flow

```text
config TOML
  -> load_gradient_config()
  -> prepare_problems()                         # 每 macro 一次，已持久化 NPZ
  -> MacroProblem.load()
  -> optimize_gradient_macro()
       -> baseline reconstruct_region()         # 当前合法 Region
       -> for state 0..N:
            for tile batch:
              target cache / exact current raster
              _EdgeGradientMask.apply()         # forward 不改 mask
              LithographyModel.forward_many()
              ownership loss + discrete metrics
              backward()                        # 只累计 d_current.grad
              release batch tensors
            state barrier
            record + best snapshot
            Adam step + clamp
            reconstruct_region(candidate)       # 发布前唯一几何屏障
       -> GradientMBOPCResult
  -> write_macro_gds() + result.npz + metrics.json
  -> next macro
  -> merge_macro_results()                      # 全部 macro 后恰一次
  -> optional save_final_lithography()
  -> summary.json
```

### 7.4 Stage Boundaries

| Stage | Input | Work | Output | MUST NOT repeat |
|---|---|---|---|---|
| 0/1 | GDS + common config | 网格、物化、分段、owner/CSR | plan + MacroProblem NPZ | 梯度阶段不得重提边/分段 |
| 2 | 一个 MacroProblem | reference midpoint/owner parameter map | macro immutable state | 每 iteration 不得重建 mapping |
| 3 | `d_current` + Region | tile raster、litho、loss/backward | record + accumulated grad | batch 内不得 step |
| 4 | grad + optimizer | step、clamp、精确重建验证 | published next state | 同一 candidate 不得重复重建 |
| 5 | macro best | NPZ/JSON/GDS | macro artifacts | 不得逐轮全局 merge |
| 6 | 全部 macro GDS | ownership merge + optional PNG | final artifacts | merge 恰一次 |

### 7.5 Planned Call Graph

```text
main/run_gradient_mbopc.py::main()
└─ main/_mbopc_workflow.py::run_gradient_mbopc(config_path)
   ├─ load_gradient_config(config_path)
   ├─ prepare_problems(pipeline)
   ├─ ICCAD13Lithography(device)
   ├─ for macro in plan.macros
   │  ├─ MacroProblem.load(problem_file)
   │  ├─ solve_gradient_macro(...)
   │  │  └─ optimize_gradient_macro(...)
   │  │     ├─ reconstruct_region(baseline)
   │  │     ├─ for state 0..iterations
   │  │     │  ├─ for tile batch
   │  │     │  │  ├─ rasterize_mask_canvas()
   │  │     │  │  ├─ _EdgeGradientMask.apply()
   │  │     │  │  ├─ model.forward_many()
   │  │     │  │  ├─ loss.backward()
   │  │     │  │  └─ progress callback
   │  │     │  └─ Adam step -> reconstruct_region(candidate) -> publish
   │  │     └─ return best evaluated legal state
   │  ├─ write_macro_gds()
   │  └─ atomic NPZ + JSON
   ├─ merge_macro_results()                     # 一次
   ├─ save_final_lithography()                  # 可选
   └─ atomic summary.json
```

## 8. Data Contracts

符号：`S=macro segment_count`，`O=owner segment_count`，`C=core_count`，`B=batch_size`，
`M=当前 batch 中可训练 membership 条目数`，`H=W=canvas_pixels`。

### `GradientMBOPCConfig`

- Owner：调用方；求解器只读
- Lifetime：一个 run
- Mutability：frozen
- Resident：CPU
- Unit：位移相关字段为 DBU

| Field | dtype | shape | unit | meaning |
|---|---|---|---|---|
| `iterations` | Python `int` | scalar | update | 最多发布更新次数，`>=1` |
| `learning_rate_dbu` | `float` | scalar | DBU | Adam learning rate，有限正数，可为小数 |
| `weight_nominal_l2` | `float` | scalar | 1 | nominal loss 权重，非负 |
| `weight_process_l2` | `float` | scalar | 1 | max/min 对 target loss 权重，非负 |
| `weight_pvband` | `float` | scalar | 1 | max-min 连续差 loss 权重，非负 |
| `epe_distance_dbu` | `float` | scalar | DBU | 仅离散诊断探针距离，有限正数 |
| `batch_size` | `int` | scalar | core | 每个 forward batch core 数，`>=1` |
| `target_cache_bytes` | `int` | scalar | byte | CPU uint8 target LRU 上限，`>=0` |

### `GradientMBOPCIterationRecord`

- Owner：`GradientMBOPCResult`
- Lifetime：结果/持久化 JSON
- Mutability：frozen
- Coordinate：不持有坐标

| Field | dtype | unit | meaning |
|---|---|---|---|
| `state_index` | `int` | state | 0=baseline；N=第 N 次更新后状态 |
| `total_loss` | `float` | 1 | 本状态已评价加权连续 loss |
| `nominal_l2_loss` | `float` | 1 | `L_nom` |
| `process_l2_loss` | `float` | 1 | `L_process` |
| `pvband_loss` | `float` | 1 | `L_pv` |
| `l2` / `pvband` / `epe` | `int` | pixel/segment | 本状态离散诊断 |
| `valid_probes` / `ambiguous_probes` | `int` | segment | 探针诊断 |
| `displaced_segments` | `int` | segment | 本状态非零 owner 位移数 |
| `elapsed_seconds` | `float` | second | 本状态全部 tile 评价耗时 |

### `GradientMBOPCResult`

- Owner：调用方
- Lifetime：一个 macro 求解结果
- Mutability：frozen；数组为独立连续副本
- Coordinate：`best_displacements` 按 `MacroProblem.segments` 全局稳定顺序

| Field | dtype | shape | unit | meaning |
|---|---|---|---|---|
| `best_displacements` | NumPy `float64` | `[S]` | DBU | 最佳已评价合法状态；context 恒 0 |
| `records` | tuple | `[K]`, `1<=K<=N+1` | - | baseline + 已发布移动状态 |
| `best_state_index` | `int` | scalar | state | 指向 `records` 中 best |
| `stop_reason` | `str` | scalar | - | `zero_loss/no_update/invalid_geometry/no_owned_segments/iteration_limit` |
| `stop_detail` | `str|None` | scalar | - | 仅需解释的停止原因有文本 |

### 梯度热路径张量

| Name | dtype | shape | resident | lifetime / meaning |
|---|---|---|---|---|
| `parameters` | Torch `float32` | `[O]` | model device | macro 全迭代，唯一 optimizer 参数 |
| Adam state | Torch `float32` | `2×[O]` | model device | macro 全迭代 |
| `segment_to_parameter` | NumPy `int32` | `[S]` | CPU | macro 全迭代；context=-1 |
| `reference_midpoints` | NumPy `float64` | `[S,2]` | CPU | macro 全迭代，全局 DBU `(x,y)` |
| `hard_masks` | Torch `float32` | `[B,H,W]` | device | 单 batch，精确面积覆盖率 |
| `owned_members` | Torch `int64` | `[M]` | device | 单 batch，指向 `[O]` 参数 |
| `midpoints_xy` | Torch `float32` | `[M,2]` | device | 单 batch，居中 canvas `(x,y)` |
| `printed_*` | Torch `float32` | `[B,H,W]` | device | 单 batch，backward 后释放 |

空 macro 允许 `S=O=0`；非空 macro 允许 `S>0,O=0`（纯 context）。这两种情况返回 baseline，
不构造空 Adam optimizer。

### 8.1 Configuration Contract

公共 `[input]/[grid]/[lithography]/[edge]/[output]` 完全复用 `MacroPipelineConfig`。新增段：

| Key | Type | Unit | Required | Default | Validation | Consumer |
|---|---|---|---|---|---|---|
| `gradient_mbopc.iterations` | strict int | update | yes | None | `>=1`，拒绝 bool/float | loader/solver |
| `gradient_mbopc.learning_rate_nm` | number | nm | yes | None | 有限正数；允许非整数 DBU | loader |
| `gradient_mbopc.weight_nominal_l2` | number | 1 | yes | None | 有限非负 | solver |
| `gradient_mbopc.weight_process_l2` | number | 1 | yes | None | 有限非负 | solver |
| `gradient_mbopc.weight_pvband` | number | 1 | yes | None | 有限非负；三权重至少一正 | solver |
| `gradient_mbopc.epe_distance_nm` | number | nm | yes | None | 正且可精确换整数 DBU；`<=context_nm` | diagnostics |
| `gradient_mbopc.batch_size` | strict int | core | yes | None | `>=1` | solver |
| `gradient_mbopc.target_cache_mb` | strict int | MiB | yes | None | `>=0` | cache |
| `gradient_mbopc.device` | string | - | yes | None | `auto/cpu/cuda/cuda:N` | workflow |
| `gradient_mbopc.save_final_lithography` | bool | - | no | `false` | strict bool | workflow |
| `gradient_mbopc.show_progress` | bool | - | no | `false` | strict bool | workflow |

`learning_rate_nm / dbu_nm` 使用 Decimal 后转 float，MUST NOT 调 `exact_dbu()`；它是连续 optimizer
步长。其他网格、pixel、segment、探针仍遵守现有整数 DBU 契约。

Adam 固定使用 PyTorch 参数 `betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`、`amsgrad=False`；
不新增 optimizer 配置。

`config/gradient_mbopc.toml` MUST 使用以下 smoke 内容；实施 AI不得另选隐式默认值：

```toml
# gcd_45nm 多 macro 梯度 MB-OPC smoke；一轮更新可验证完整状态循环。

[input]
layout = "../TestReticle/gcd_45nm.gds"
top_cell = "TOP"
layer = 11
datatype = 0
polarity = "clear"

[grid]
macro_grid = [2, 2]
core_size_nm = 1024
context_nm = 400

[lithography]
pixel_nm = 8
canvas_pixels = 256

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24
miter_limit = 4.0

[gradient_mbopc]
iterations = 1
learning_rate_nm = 1.0
weight_nominal_l2 = 1.0
weight_process_l2 = 0.5
weight_pvband = 0.1
epe_distance_nm = 16
batch_size = 8
target_cache_mb = 512
device = "auto"
save_final_lithography = true
show_progress = true

[output]
work_dir = "../output/gradient_mbopc"
final_layout = "../output/gradient_mbopc/gcd_45nm_gradient_mbopc.gds"
final_cell_mode = "single_cell"
```

### 8.2 Persisted Artifact Contract

| Artifact | Format/version | Required content |
|---|---|---|
| `problems/<macro>.npz` | 现有 MacroProblem v1 | 原样复用，不改字段 |
| `macros/<macro>/gradient_result.npz` | NPZ v1，`allow_pickle=False` | `format_version:int32[1]`、`macro_id:str[1]`、`best_state_index:int32[1]`、`best_displacements:float64[S]`、`stop_reason:str[1]` |
| `macros/<macro>/gradient_metrics.json` | JSON v1 | macro ID、best state、stop reason/detail、records 全字段 |
| `macros/<macro>/best.gds` | GDS | `RESULT` cell、目标 layer、完整最佳候选 Region |
| `summary.json` | JSON | 方法=`gradient_mbopc`、规模、设备、权重、逐 macro best loss/离散指标/耗时、merge/总耗时、产物路径 |
| final layout | GDS | 现有 ownership merge contract |
| `final_lithography/` | 现有 manifest v1 + PNG | 复用当前最终光刻 contract |

NPZ/JSON 必须经现有同目录临时文件原子替换；不得覆盖 simple 的 `result.npz/metrics.json` 名称。

`summary.json` 顶层键 MUST 至少包含：

```text
method = "gradient_mbopc"
macro_count, core_count, segment_count_sum
device, iterations
loss_weights = {nominal_l2, process_l2, pvband}
macros = [{macro_id, best_state_index, stop_reason, stop_detail,
           state_count, best_total_loss, best_l2, best_pvband, best_epe,
           best_gds, result_npz, metrics_json, elapsed_seconds}]
final_layout, final_cell_mode, merge_seconds, total_seconds
rss_start_bytes, rss_after_prepare_bytes, peak_rss_bytes
cuda_peak_bytes                         # CPU 时为 None
final_lithography_tiles                 # 未保存时为 None
```

## 9. Interface Changes

### IF-001：共享 target cache 移动，public import 不变

Current：

```python
opc.iteration.mbopc.simple.TargetCanvasCache
from opc.iteration.mbopc import TargetCanvasCache
```

Target：

```python
opc.iteration.mbopc._cache.TargetCanvasCache
from opc.iteration.mbopc import TargetCanvasCache  # 继续有效
```

- Caller migration：`simple.py` 与新 `gradient.py` 改从 `._cache` 导入；现有外部/package import 不改。
- Compatibility：preserved。

### IF-002：新增梯度求解接口

```python
def optimize_gradient_macro(
    problem: MacroProblem,
    model: LithographyModel,
    config: GradientMBOPCConfig,
    target_cache: TargetCanvasCache,
    *,
    on_tiles_completed: Callable[[int], None] | None = None,
) -> GradientMBOPCResult:
    """优化一个 macro 的 owner 边段法向位移并返回最佳已评价合法状态。"""
```

- `problem/model/config/cache` 只读；函数拥有 optimizer 和临时状态。
- `on_tiles_completed` 在 batch backward/释放后调用实际 batch core 数。
- canvas/探针/参数越界 `ValueError`；非有限数值 `FloatingPointError`；未知异常传播。

### IF-003：新增配置接口

```python
@dataclass(frozen=True, slots=True)
class GradientMBOPCRunConfig:
    pipeline: MacroPipelineConfig
    iterations: int
    learning_rate_nm: Decimal
    weight_nominal_l2: float
    weight_process_l2: float
    weight_pvband: float
    epe_distance_nm: Decimal
    batch_size: int
    target_cache_mb: int
    device: str
    save_final_lithography: bool
    show_progress: bool

def load_gradient_config(path: str | Path) -> GradientMBOPCRunConfig: ...
```

`load_config()` 仍只读取 `[mbopc]`，签名和行为不变。

### IF-004：新增工作流与直接入口

```python
def run_gradient_mbopc(config_path: str | Path) -> dict: ...

# main/run_gradient_mbopc.py
def main() -> int: ...
```

该入口接受任意 `macro_count>=1`；不复制 single/multi 两个脚本，因为算法和 orchestration 完全相同。

### IF-005：包级导出

`opc/iteration/mbopc/__init__.py` 新增导出：

```python
GradientMBOPCConfig
GradientMBOPCIterationRecord
GradientMBOPCResult
optimize_gradient_macro
```

现有 simple 导出全集保持不变。

## 10. Algorithm

### 10.1 初始化

```text
validate config against problem/model
owner_segment_ids = flatnonzero(owner_indices >= 0)
segment_to_parameter[:] = -1
segment_to_parameter[owner_segment_ids] = arange(O)
parameters = zeros(O, float32, device, requires_grad=True)
reference_midpoints = vectorized midpoint from contours + edge_ids + (t0+t1)/2
reference_region = reconstruct_region(problem, zeros[S])
current_region = reference_region
best_owner = zeros(O)
optimizer = Adam(parameters, fixed constants) only if O > 0
precompute P = sum ownership pixel counts without retaining all masks
```

### 10.2 一个已评价状态

```text
optimizer.zero_grad(set_to_none=True) if gradients are needed
current_owner_cpu = parameters.detach().cpu().numpy()
current_full = zeros(S, float64)
current_full[owner_segment_ids] = current_owner_cpu

for core batch in stable order:
    target[B,H,W] = uint8 LRU hit or raster(reference_region)
    hard[B,H,W] = raster(current_region)
    ownership[B,H,W] = ownership_canvas(...)

    flatten all membership entries whose segment_to_parameter >= 0
    for each entry:
        current_midpoint_dbu = reference_midpoint + normal * current displacement
        midpoint_xy = points_to_canvas(current_midpoint_dbu, that core context)
        local_displacement = parameters[parameter_index]

    mask = _EdgeGradientMask.apply(
        hard, local_displacement, batch_index, midpoint_xy)
    printed = model.forward_many(mask, nominal/max/min)
    accumulate normalized continuous loss over ownership
    accumulate discrete L2/PV diagnostics and fixed-reference owner-probe EPE
    diagnostics for this same state
    if another update is allowed and batch has trainable membership:
        batch_loss.backward()       # accumulates into shared parameters.grad
    release printed/mask/target/ownership/autograd graph
    on_tiles_completed(actual_batch_size)

check total loss finite
append record for current_full
update best only on strict lower total_loss
```

### 10.3 `_EdgeGradientMask`

Forward：原样返回 `hard_masks`；保存 `batch_indices` 和当前 `midpoints_xy`。`local_displacements`
只用于建立 autograd 边，不参与 forward 数值，等价于 STE hard geometry。

Backward：

```text
g_mid = bilinear_sample(grad_output[B,H,W], batch_indices[M], midpoints_xy[M,2])
g_mid[outside canvas] = 0
grad_local_displacements = 2 * g_mid
return None, grad_local_displacements, None, None
```

双线性采样必须向量化，使用 `[y,x]` 访问；不得逐 segment Python 循环或 `.item()`。

`2*g_mid` 的来源：论文给两个 endpoint 各 `g_mid*v`；本项目
`endpoint_k = endpoint_ref_k + v*d_s`，单位 `v` 下两端链式求和为 `2*g_mid`。

### 10.4 状态更新与发布

```text
if O == 0:
    stop no_owned_segments
if current total_loss == 0:
    stop zero_loss
if state_index == iterations:
    stop iteration_limit
if gradient is missing or non-finite:
    raise FloatingPointError

before = parameters.detach().clone()
optimizer.step()
clamp parameters to ±max_displacement
if candidate parameters contain non-finite:
    raise FloatingPointError
if candidate exactly equals before:
    stop no_update

candidate_full = expand candidate to float64[S]
try:
    candidate_region = reconstruct_region(problem, candidate_full)
except (ReconstructionError, ValueError) as exc:   # Revision 0.2 宽捕获
    stop invalid_geometry with original message

publish parameters + candidate_region as next current state
continue
```

（Revision 0.2 修订）几何退化（如位移共线使 ring 顶点不足）会以 `ValueError`
从 KLayout 穿透 `reconstruct_region`（simple 轮实测证据，reconstruction.py 无
包装），故候选守卫捕获 `(ReconstructionError, ValueError)` 并在 stop_detail
保留原始文本；不得捕获 `RuntimeError`、CUDA OOM 或未知异常。非法 candidate
未进入 records，也未成为 best。

### 10.5 Boundary Conditions

| Condition | Required behavior | Requirement |
|---|---|---|
| `S=0` 或 `O=0` | 评价一次 baseline，返回 `no_owned_segments`，不建 optimizer | REQ-009 |
| 一个 batch 无可训练 membership | 正常前向/指标，不调用该 batch backward | REQ-007 |
| midpoint 在某 context canvas 外 | 该 membership surrogate 梯度为 0；其他 tile 贡献仍累计 | REQ-004 |
| 全部 EPE probe 无效 | 记录 `valid=0/epe=0`；不视为 loss 收敛 | REQ-006 |
| 参数更新小于 DBU rounding 阈值 | 保留浮点参数，允许后续累计跨过格点 | REQ-003 |
| 最后不足 batch | 使用真实 B，进度增加真实 core 数 | REQ-014 |
| hole/对边越界 | candidate 不发布，返回 `invalid_geometry` | REQ-010 |
| multi macro seam | 各 macro 独立；最终 ownership merge，一次 | REQ-012 |

### 10.6 State Transition

```text
S0(valid baseline) --evaluate/backward--> record(S0) + grad(S0)
S0 --Adam/clamp/reconstruct validate--> publish S1
S1 --evaluate/backward--> record(S1) + grad(S1)
...
SN --evaluate only--> record(SN)
```

`grad(Si)` 不是 `Si+1` 的指标；候选只有发布并完成下一次全部 tile 评价后才有 record。

## 11. Ownership and State

| State/data | Owner | Writers | Readers | Publish point | Lifetime |
|---|---|---|---|---|---|
| `MacroProblem` | input layer/caller | none | solver/workflow | prepare 完成 | persisted problem |
| reference geometry/target | solver/cache | 初始化/cache put | all states | 初始化 | one macro/cache budget |
| owner `parameters[O]` | solver optimizer | Adam + clamp | all batch | candidate 重建成功后 | one macro |
| context displacement | no optimizer owner | none | raster/reconstruct | 固定 0 | one macro |
| batch mask/printed | current batch | litho/autograd | loss/metrics | 不发布 | one batch |
| accumulated gradient | parameters.grad | batch backward | Adam | all tile barrier | one state |
| record | result builder | state evaluator | best/report | all tile 完成 | result |
| candidate Region | solver | reconstruct | next raster | 重建成功 | next state |
| best owner snapshot | solver | strict lower loss | result expansion | state record 后 | one macro |
| macro GDS/result | workflow | atomic writers | final merge/user | macro complete | disk |

任何 batch 不得直接修改 current parameter。单机顺序、batch size 或未来并行 batch 不应改变数学上的
梯度求和与一次 state step 语义；浮点归约允许本文测试定义的容差。

## 12. Error Handling

### ERR-001：配置非法

- Condition：缺字段、未知字段、类型错误、越界、全零权重、未知 device。
- Detection：`main/_mbopc_workflow.py::load_gradient_config`
- Behavior：`ValueError`，消息包含 section/key。
- MUST NOT：截断 float 为 int、把 bool 当 int、补隐式算法默认值。

### ERR-002：模型/Problem 不兼容

- Condition：canvas 不等、探针超 context、无 ownership 像素。
- Detection：`optimize_gradient_macro`
- Behavior：进入 GPU 大分配前 `ValueError`。

### ERR-003：数值失效

- Condition：loss、gradient 或 candidate parameter 出现 NaN/Inf，或需要更新时 grad 缺失。
- Behavior：`FloatingPointError`，消息含 macro ID、state index 和对象名。
- State：不写该 macro 权威 best GDS/result。
- MUST NOT：转为 `invalid_gradient` 成功结果、自动降学习率或 retry。

### ERR-004：候选几何非法

- Condition：`reconstruct_region()` 抛 `ReconstructionError` 或 KLayout 原生
  `ValueError`（几何退化形态，Revision 0.2）。
- Behavior：返回 `invalid_geometry`，`stop_detail` 保留 state 和原错误；输出历史最佳合法状态。
- MUST NOT：捕获 `RuntimeError` 或未知异常（程序错误必须传播）。

### ERR-005：I/O、CUDA 与依赖错误

- Behavior：原样传播或增加路径/macro 上下文后以原异常链传播。
- MUST NOT：跳过 macro、自动改 CPU、写半份 summary/final GDS。

## 13. Performance and Memory Constraints

### PERF-001：GPU 只常驻一个 macro 的参数和一个 batch 图

- `parameters + grad + Adam moments` 上界约 `16*O` bytes（float32 四份）；
- batch 输入基础数组约 `B*H*W*(4 mask + 4 target + 1 ownership)` bytes，光刻 autograd 图另计；
- batch backward 后 printed、mask、target、ownership 和 edge sampling 临时量必须释放；
- 禁止整张 reticle tensor、全部 tile printed image 或所有 state autograd graph 常驻 GPU。

### PERF-002：CPU 常驻与临时量

- 常驻：一个 `MacroProblem`、`segment_to_parameter int32[S]`、`reference_midpoints float64[S,2]`、
  bounded target cache、reference/current KLayout Region、best owner snapshot；
- 不常驻 `SegmentGeometry.starts/ends/normals` 三份全量数组；
- 每状态允许一个 `float64[S]` 展开向量用于重建，使用后释放；
- 每次只加载一个 macro，完成后释放再进入下一个。

### PERF-003：调用次数

- 每个已评价 state、每个 batch 恰一次三工艺 `forward_many()`；
- 每个 candidate 恰一次 `reconstruct_region()`，结果复用为下一 state raster；
- target cache 命中时不得重栅格 target；
- 全部 macro 后 `merge_macro_results()` 恰一次。

### PERF-004：禁止热循环

- `_EdgeGradientMask.backward`、owner mapping、loss 和梯度回写不得逐 segment Python 循环；
- 不得逐 polygon 调官方 ray caster；KLayout 边界只在批量 Region/raster 接口跨越；
- device 热路径不得逐 segment `.item()`/`.cpu()`；每 batch 只允许标量诊断归约和必要的批量传输。

### PERF-005：测量

- 固定 workload：`config/gradient_mbopc.toml` + `TestReticle/gcd_45nm.gds`；
- 记录 prepare/每 macro solve/merge/total、RSS start/prepare/peak、CUDA peak；
- 第一版只记录基线，不设速度阈值；必须报告 24 GiB GPU / 64 GiB RAM 是否实测，未实测不得声称可跑整张 reticle。

## 14. File-Level Change Plan

| File / Symbol | File type | Action | Contract change | Reason / Requirement |
|---|---|---|---|---|
| `opc/iteration/mbopc/_cache.py::TargetCanvasCache` | 业务代码 | add | 从 simple 移入，行为不变 | 两个真实方法共享，REQ-016 |
| `opc/iteration/mbopc/simple.py::TargetCanvasCache` | 业务代码 | modify | 删除类定义，改私有 import | 消除方法间反向依赖 |
| `opc/iteration/mbopc/gradient.py` | 业务代码 | add | 新增 config/record/result、`_EdgeGradientMask`、`optimize_gradient_macro` | REQ-003..011 |
| `opc/iteration/mbopc/__init__.py` | 业务代码 | modify | 保留旧导出，新增 gradient 导出 | IF-001/005 |
| `main/_mbopc_workflow.py` | 业务代码 | modify | 新增 gradient config/solve/run；抽取仅被 simple+gradient 共用的 `_resolve_device` | REQ-001/012..014 |
| `main/run_gradient_mbopc.py` | 运行入口 | add | 一个入口支持任意 macro 数 | REQ-001 |
| `config/gradient_mbopc.toml` | 配置文件 | add | 显式 `[gradient_mbopc]` smoke 配置 | §8.1 |
| `tests/opc/iteration/test_gradient_mbopc.py` | 测试代码 | add | surrogate、状态、几何、CPU/CUDA | TEST-001..012 |
| `tests/main/test_gradient_mbopc_runner.py` | 测试代码 | add | 配置、入口、产物、macro merge | TEST-013..016 |
| `tests/opc/iteration/test_simple_mbopc.py` | 测试代码 | modify | cache 移动回归仅在确有 import-path 断言需要时修改 | REQ-016 |
| `doc/development_manual.md` | 手册 | modify | 模块、接口、运行方式、限制 | 交付 |
| `doc/test_manual.md` | 手册 | modify | 定向/全量/CUDA/smoke 命令 | 交付 |
| `doc/opc/gradient_mbopc_development_report.md` | 开发报告 | add | 实际偏差、性能、清理审计 | 交付 |
| `doc/opc/gradient_mbopc_test_report.md` | 测试报告 | add | 命令、环境、结果、图形矩阵 | 交付 |
| `doc/opc/gradient_mbopc_migration_design.md` | 实施规格 | modify | 状态、revision、实施 evidence | 审批/交付 |
| `task_plan.md`、`findings.md`、`progress.md` 或任务专属 `.planning/` 等价文件 | 项目记录 | modify | 同步实际执行 | AGENTS 交付规则 |

除 `_cache.py` 外不新增 common/contracts/types/rasterizer/optimizer 文件。若实现需要修改表外文件，必须先修订
本文并重新审批。

## 15. Test Specification

### TEST-001：forward 零差异

- Level：unit
- File/function：`test_gradient_mbopc.py::test_edge_gradient_forward_preserves_exact_raster`
- Given：含 0/部分覆盖/1 的 hard mask 与任意局部位移连接。
- When：`_EdgeGradientMask.apply()`。
- Then：输出逐位、dtype、shape、device 与输入相同。
- Covers：REQ-003, INV-005。

### TEST-002：Algorithm 4 backward 精确公式

- File/function：`test_edge_gradient_backward_is_two_times_bilinear_midpoint`
- Given：两 batch 的已知仿射 `grad_output`、整数/半像素/越界 midpoint、重复参数索引。
- Then：单 membership 为 `2*bilinear(g)`；重复参数由 autograd 求和；越界为 0；无 `.item()`。
- Covers：REQ-004, REQ-007。

### TEST-003：真实光刻方向有限差分

- File/function：`test_surrogate_direction_matches_integer_geometry_difference`
- Given：clear/opaque 矩形的水平/垂直边与 45° 斜边，真实 ICCAD13；分别构造扩大/缩小误差。
- When：比较 surrogate gradient 与 `d±1 DBU` 精确重建/raster loss 中心差分。
- Then：对有限差分绝对值大于 `1e-9` 的选定段，乘积严格为正；不比较幅值。
- Covers：REQ-004/005, INV-006。

### TEST-004：loss 公式与 ownership

- File/function：`test_continuous_losses_match_manual_owned_pixel_formula`
- Given：确定性 differentiable fake model、两个 core、非零 halo。
- Then：三个分量和 total 与手算一致；halo 改值不直接增加计分；分母是全局 P。
- Covers：REQ-006/007。

### TEST-005：batch 与同步不变量

- File/function：`test_batch_size_preserves_gradient_and_published_state`
- Given：同一跨 core 图形，batch 1 与全 core batch。
- Then：records loss 在 `rtol=1e-5, atol=1e-7` 内一致，best displacement `atol=1e-5 DBU`；
  instrumentation 证明首个 optimizer step 发生在所有 batch backward 后。
- Covers：REQ-007/008, INV-003。

### TEST-006：state/record/best 语义

- File/function：`test_records_and_best_use_same_evaluated_snapshots`
- Given：两次更新、第二状态最好、第三状态变差的确定性模型。
- Then：state `[0,1,2]`，best 指向 state 1，其 displacement 与 record 同快照；无未评价 step。
- Covers：REQ-009/011, INV-004。

### TEST-007：几何拒绝与异常边界

- File/functions：
  - `test_invalid_reconstruction_keeps_last_legal_best`
  - `test_program_runtime_error_is_not_converted_to_invalid_geometry`
    （Revision 0.2：ValueError 与几何退化不可按类型区分，改用 RuntimeError 验证传播边界）
  - `test_nonfinite_gradient_raises_floating_point_error`
- Then：只捕 `ReconstructionError` 为领域停止；其他错误传播；非法候选无 record。
- Covers：REQ-010/015, ERR-003/004。

### TEST-008：owner/context 与空问题

- File/functions：
  - `test_context_segments_have_no_parameter_or_adam_state`
  - `test_empty_or_context_only_macro_returns_baseline`
- Then：结果 context 全 0；参数数等于 O；空问题一条 record，无空 optimizer 错误。
- Covers：REQ-007, INV-001/002。

### TEST-009：几何矩阵

- File/function：参数化 `test_geometry_matrix_publishes_only_valid_results`
- Cases：实心矩形、2 nm 壁 hole、多 polygon+hole、凹图形、45° 斜边、跨 3 core、跨 macro、opaque。
- Then：每个已发布 state 重建合法；context=0；位移在上限；最终 best 对应真实 record。窄壁 EPE 可为
  `valid=0`，但不能触发 `zero_loss`，除非连续 loss 真为 0。
- Covers：REQ-005/007/010/012。

### TEST-010：性能调用计数

- File/functions：
  - `test_forward_many_once_per_batch_per_state`
  - `test_candidate_region_reconstructed_once_and_reused`
  - `test_target_cache_hit_avoids_target_raster`
  - `test_edge_backward_has_no_segment_python_loop_or_tensor_item`（静态/monkeypatch）
- Covers：PERF-003/004。

### TEST-011：真实 CPU/CUDA 集成

- File/functions：
  - `test_real_iccad13_cpu_runs_nonzero_update_and_valid_best`
  - `test_real_iccad13_cuda_matches_cpu_direction`（无 CUDA 时精确 skip）
- Then：CPU loss/grad/records 有限，至少一个候选非零，best loss 不大于 baseline；CUDA 选定段梯度方向与 CPU
  一致，loss 在 `rtol=2e-4, atol=2e-6` 内。
- Covers：REQ-003..011。

### TEST-012：simple 兼容

- File/function：现有 `tests/opc/iteration/test_simple_mbopc.py` 全量。
- Then：全部通过；包级 `TargetCanvasCache` import 不变；simple 数值快照不变。
- Covers：REQ-016。

### TEST-013：配置与直接入口

- File/functions：`test_gradient_mbopc_runner.py::test_config_contract`、
  `test_direct_script_runs_outside_repository`。
- Then：严格类型/未知键/全零权重失败；合法 config 从仓库外直跑退出 0。
- Covers：REQ-001, ERR-001。

### TEST-014：产物 contract

- File/function：`test_runner_writes_gradient_artifacts_and_final_lithography`
- Then：NPZ 键/dtype/shape、JSON record、best GDS、summary、final GDS、可选 PNG 全部满足 §8.2。
- Covers：REQ-013。

### TEST-015：多 macro 一次 merge

- File/function：`test_multi_macro_solves_independently_and_merges_once`
- Then：每 macro 独立产物；merge 调用一次；macro 顺序反转后最终 ownership 覆盖 XOR=0（固定 mock solver）。
- Covers：REQ-012。

### TEST-016：进度与资源报告

- File/function：`test_progress_counts_completed_tile_batches_and_closes_on_error`
- Then：更新总数等于实际已评价 tile；异常时 bar close；summary 有时间、RSS/CUDA peak 字段。
- Covers：REQ-014, PERF-005。

### 15.1 Required Test Matrix

| Dimension | Cases | Expected distinction |
|---|---|---|
| Geometry | rectangle/hole/concave/multi/diagonal | 方向、拓扑与重建分别断言 |
| Polarity | clear/opaque | 同一正位移=扩大透光，无 solver 分支 |
| Boundary | cross core/3 cores/cross macro | 梯度归属、屏障、最终 merge |
| Probe | normal/2 nm wall + 8 nm probe | valid 与 unavailable 诊断，不控制训练 |
| Scale | empty/one core/multi core/multi macro/gcd smoke | 空状态、streaming、峰值记录 |
| Device | CPU/CUDA | 方向一致与规定容差 |
| Failure | config/nonfinite/reconstruction/I/O | 精确异常或领域停止 |

### 15.2 Verification Commands

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\opc\iteration\test_gradient_mbopc.py tests\main\test_gradient_mbopc_runner.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\opc\iteration\test_simple_mbopc.py tests\main\test_mbopc_runners.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc\iteration\mbopc main tests\opc\iteration tests\main
D:\app\miniforge\envs\myopc\python.exe -m compileall -q opc main tests
D:\app\miniforge\envs\myopc\python.exe main\run_gradient_mbopc.py config\gradient_mbopc.toml
git diff --check
```

CUDA 测试只有 `torch.cuda.is_available()==False` 时可 skip；核心 CPU contract 不得 skip。gcd smoke 若因资源未执行，
必须在测试报告明确写“未验证”，不得用生成式小图替代后声称通过真实版图验收。

## 16. Requirement Traceability

| Requirement / Invariant | Implementation symbol | Tests | Acceptance |
|---|---|---|---|
| REQ-001 | `main/run_gradient_mbopc.py::main`、`run_gradient_mbopc` | TEST-013 | AC-005 |
| REQ-002 | `gradient.py::optimize_gradient_macro`（只读消费） | TEST-008/012 | AC-002/006 |
| REQ-003/004/005 | `gradient.py::_EdgeGradientMask` | TEST-001..003 | AC-001 |
| REQ-006/007 | `gradient.py::optimize_gradient_macro` | TEST-004/005/008 | AC-002 |
| REQ-008/009/011 | `gradient.py::optimize_gradient_macro` | TEST-005/006 | AC-003 |
| REQ-010/015 | `gradient.py::optimize_gradient_macro` | TEST-007/009 | AC-004 |
| REQ-012 | `_mbopc_workflow.py::run_gradient_mbopc` | TEST-009/015 | AC-004/005 |
| REQ-013 | `_mbopc_workflow.py::run_gradient_mbopc` | TEST-014 | AC-005 |
| REQ-014 | `_mbopc_workflow.py::solve_gradient_macro/run_gradient_mbopc` | TEST-016 | AC-005/007 |
| REQ-016 | `_cache.py` + package exports | TEST-012 | AC-006 |
| REQ-017 | 文件清单、依赖静态审计 | TEST-012 + final audit | AC-008 |
| INV-001/002 | owner-only parameter mapping | TEST-008 | AC-002 |
| INV-003/004 | state barrier / record | TEST-005/006 | AC-003 |
| INV-005/006 | points/polarity contract | TEST-002/003/009 | AC-001 |
| PERF-001/002 | `gradient.py::optimize_gradient_macro` | TEST-010 + smoke measurement | AC-007 |
| PERF-003/004 | batch/reconstruct/cache/backward hot path | TEST-010 | AC-008 |
| PERF-005 | `_mbopc_workflow.py::run_gradient_mbopc` | TEST-016 + smoke | AC-007 |

## 17. Acceptance Criteria

- [ ] **AC-001**：精确 forward 逐值一致，Algorithm 4 backward 等于 `2*g_mid`，真实 ICCAD13 的
  clear/opaque/斜边 surrogate 与整数 DBU 有限差分方向一致；
- [ ] **AC-002**：loss 手算、ownership、context=0、跨 tile 梯度合并和空问题测试通过；
- [ ] **AC-003**：batch 不变性、state `[0..N]`、barrier、final evaluation 和 best snapshot 测试通过；
- [ ] **AC-004**：hole/斜边/跨 core/macro/窄壁/对边越界矩阵通过，异常边界无吞错；
- [ ] **AC-005**：直接 main 从仓库外运行，单/多 macro、一次 merge、全部 NPZ/JSON/GDS/PNG 产物通过；
- [ ] **AC-006**：simple 定向与全量回归通过，public cache import 与 simple 配置/产物兼容；
- [ ] **AC-007**：记录 CPU RSS/CUDA peak 和 gcd smoke 实测状态，不夸大整张 reticle 能力；
- [ ] **AC-008**：ruff、compileall、`git diff --check`、中文 docstring/注释、未调用函数、重复实现、
  异常吞噬和过度拆分审计通过；
- [ ] **AC-009**：开发手册、测试手册、开发报告、测试报告和规划记录同步；
- [ ] **AC-010**：本地阶段 commit 完成、未推送，且未包含用户脏文件或 GDS/PNG。

## 18. Compatibility and Migration

### COMP-001

- API compatibility：simple public API preserved；只新增 gradient API；
- Data compatibility：`MacroProblem` v1 preserved；gradient result 使用独立文件名/version；
- Archive compatibility：不读取 `00_PAST` 产物，不修改归档；
- CLI compatibility：现有两个 simple main 和 TOML 不变；新增一个 gradient main；
- Numerical compatibility：simple 数值不变；gradient 是新方法，无旧数值兼容要求；
- Dependency compatibility：不新增依赖。

## 19. Decisions

### DEC-001：使用精确面积覆盖率 forward + midpoint STE backward

- Decision：forward 复用 KLayout Region/raster；backward 依据论文 Algorithm 4。
- Reason：前向几何与项目唯一真源一致，且迁移的是官方 edge gradient 思路。
- Rejected：归档 sigmoid occupancy-delta；未证明多 polygon/hole 相加正确，也非官方方法。
- Consequence：hard forward 的 surrogate gradient 只保证方向性，不承诺普通导数幅值精确。

### DEC-002：优化标量法向位移，不优化四个 endpoint 坐标

- Reason：复用现有 `SegmentBatch`/reconstruct/ownership，天然限制边沿法向并降低参数/Adam 内存。
- Formula consequence：两个 endpoint 梯度投影到同一标量后为 `2*g_mid`。
- Rejected：复制官方 `[S,2,2]` endpoint 参数；会重复几何字段并绕开当前重建真源。

### DEC-003：首版不加入 EPE training loss

- Decision：EPE 只诊断；训练为 nominal/process/PV 连续 loss。
- Reason：官方 EPE 分支是可选且 H/V 专用；本次没有经任意斜边验证的 EPE loss contract。
- Rejected：把离散 EPE integer 当 loss；不可导。归档 inner/outer ReLU 不是本次参考算法。

### DEC-004：只抽取 TargetCanvasCache

- Decision：cache 移到同包一个私有文件；不抽 Solver/Result/Optimizer 接口。
- Reason：迁移后 cache 有 simple+gradient 两个真实调用方；其余共同字段不足以形成稳定抽象。
- Rejected：gradient import simple、复制 cache、统一 ILT/MB-OPC Solver。

### DEC-005：一个 gradient main 处理单/多 macro

- Reason：两种 macro 数的算法和流程完全相同；无需复制两个薄脚本。
- Rejected：复制 single/multi gradient main；增加无领域差异文件。

### DEC-006：macro 独立完成后一次 merge

- Reason：沿用已批准基础 MB-OPC 速度/边界取舍，避免逐轮全局 I/O。
- Consequence：macro boundary 使用邻区参考 geometry，不是全局同步最优。

## 20. Open Questions

### 20.1 Blocking

None.

### 20.2 Non-blocking

- 未来是否增加 MRC-aware velocity、SRAF 或 differentiable EPE loss，均属独立 change；
- 未来是否增加逐轮 macro 同步，属大版图边界精度 change；
- 未来若 profile 证明 KLayout raster 是瓶颈，再单独评审 CUDA ray casting，当前不预留接口。

## 21. Implementation Freedom

实现 AI 可以决定：局部变量名、私有逻辑块顺序、向量化 bilinear gather 的等价 Torch 写法，以及不
改变 contract 的中文注释组织。

实现 AI不得决定：loss 公式、`2*g_mid`、owner 参数化、state 屏障、best/publish 语义、配置字段、
产物键名、异常边界、依赖方向或表外重构。

除 `TargetCanvasCache` 外，只有一个调用点且无独立领域含义的逻辑 MUST 留在 `gradient.py` 或 workflow
调用函数内；不得新增 `types.py`、`contracts.py`、`rasterizer.py`、factory 或 registry。

## 22. Implementation Stages and Local Commits

| Stage | Objective | Files | Required verification | Suggested local commit |
|---|---|---|---|---|
| A | 共享 cache 移动且 simple 零回归 | `_cache.py`、`simple.py`、`__init__.py`、simple tests | simple 定向测试 | `refactor(mbopc): 共享目标画布缓存` |
| B | 梯度数学与单 macro solver | `gradient.py`、gradient unit tests | TEST-001..012、CPU real model | `feat(mbopc): 实现梯度边段优化` |
| C | 配置、main、macro 生命周期与产物 | `_mbopc_workflow.py`、runner、config、main tests | TEST-013..016、直接 main | `feat(main): 接入梯度MBOPC流程` |
| D | 全量验证、手册、报告与简化审计 | doc + planning records | full pytest/ruff/compileall/diff check/smoke | `docs(mbopc): 完成梯度迁移报告` |

每阶段只在验证通过后本地 commit；不得 push。提交必须显式排除 baseline dirty paths、用户 GDS、PNG
和无关输出。

## 23. Delivery and Final Audit

实现完成后 MUST：

- 更新开发手册、测试手册、专项开发/测试报告及规划记录；
- 记录与本规格的实际差异；contract 差异必须先获批准，不能事后合理化；
- 记录测试环境、命令、pass/fail/skip、耗时、RSS/CUDA peak、smoke 产物；
- 搜索调用点并删除仅服务旧错误的 helper、分支和变量；
- 审计未调用函数、重复实现、异常吞噬、一次性抽象、文件过拆分和无需求字段；
- 检查全部新增/修改 Python 文件、函数、方法和测试函数的中文 docstring，以及坐标、梯度、ownership、
  内存与错误原因的详细中文注释；
- 提供 `git diff --check`、目标/全量测试、ruff、compileall、直接 main smoke；
- 明确 `layout/`、`geometry/`、`00_PAST/`、用户 GDS 未修改；
- 列出本地 commit 并明确未推送。

## 24. Known Limitations and Future Work

- 当前方法是 DiffOPC 核心 edge-gradient 的最小子集，不含 MRC、SRAF 和 EPE training loss；
- 任意斜边是基于单位法向/midpoint 的项目扩展，不声称官方 benchmark 已验证；
- 每 macro CPU 仍物化完整边段问题与 reference/current Region；不是磁盘分片式全 reticle 求解；
- 独立 macro 不交换动态 context，最终 merge 不能补偿缺失的跨 macro 光学反馈；
- midpoint surrogate 对长 segment 内空间变化只取代表点；segment 长度应继续由现有 fragmentation 控制；
- 无 rule deck 时只能保证现有拓扑/位移重建守卫，不能保证 foundry MRC-clean。

## 25. Specification Approval Gate

- [ ] front matter 基线和 dirty path 经用户确认；
- [ ] 用户接受本次是最小 gradient MB-OPC，而非完整 MRC/SRAF DiffOPC；
- [ ] 用户接受 `2*g_mid` 标量位移公式、三项 loss 和 EPE 仅诊断；
- [ ] 用户接受一个通用 gradient main 与独立 macro 后一次 merge；
- [ ] public API、配置、产物、坐标、ownership、同步与异常语义已确定；
- [ ] 每个 MUST/invariant 已映射实现、测试和 AC；
- [ ] 文件清单没有无当前调用方的抽象；
- [ ] Blocking Open Questions 为 `None`。

用户批准后把 `status` 从 `draft` 改为 `approved`；否则实现不得开始。

## 26. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-16 | draft | 初始迁移规格；锁定官方 midpoint STE、owner-only Adam 与最小复用边界 | 待用户审核 |
| 0.2 | 2026-08-17 | approved | 用户批准实施计划（含四项裁决：宽捕获 / P=0 空问题直接 no_owned_segments / 段法向常驻 [S,2] / doc_ 副本不动）；§10.4、ERR-004、TEST-007 同步修订；实施基线 fbc059b | 用户 |
