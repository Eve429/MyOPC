---
id: CHG-20260818-curvmulti-ilt
title: CurvMulti ILT 迁移
type: implementation-spec
status: draft
scope:
  - opc/iteration/ilt
  - config
  - main
  - tests
  - doc
depends_on:
  - doc/changes/completed/CHG-20260818-simple-ilt/implementation_spec.md
  - doc/changes/completed/CHG-20260818-levelset-ilt/implementation_spec.md
  - doc/contracts/ilt.md
---

# CurvMulti ILT 迁移

## 0. Document Contract

本文基于 Simple ILT 与 LevelSet ILT 完成后的公共 ILT 框架实现 CurvMulti。
实施前 MUST 确认公共 `ILTMethod`（五字段，含 `build_fixed_context_canvas`）、
`ILTStateRecord`/`ILTMacroResult`、`PixelMacroProblem` 和 `main/_ilt_workflow.py`
已稳定（审查基线 HEAD `5a4bf5f`，2026-08-20）。

CurvMulti MUST 作为 ILT method 实现，不得复制 workflow，不得建立独立 runner
体系，不得让 Simple/LevelSet/CurvMulti 相互依赖。

**现有函数零修改**：`run_ilt_workflow`、`PixelMacroProblem`、`_common` 现有
函数、Simple/LevelSet 求解器均不动；CurvMulti 沿用 `owned_continuous_losses`
固定三条件（nominal/dose_max/defocus_min），不为其泛化任意条件序列。允许的
现有文件触碰仅限三处注册式追加：`_common.py` 加两个 helper、`ilt/__init__.py`
加导出、`CONFIG_SECTIONS` 加一行。

## 1. Objective

实现基于 OpenILT CurvMulti 思路的多尺度 ILT optimizer：采用 coarse-to-fine
control grid、sigmoid differentiable mask、nearest warm-start 和 nominal
wafer curvature regularization。

目标是证明公共 ILT pipeline 可以支持不同 parameterization，而不是新增一套
ILT 流程。

## 2. Design Changes

### 2.1 Architecture

目标架构（与 Simple/LevelSet 同构，适配器内联入口——2026-08-20 合并后拓扑）：

```text
main/run_curvmulti_ilt.py（入口 + CURVMULTI_ILT_METHOD 适配器内联）
    -> main/_ilt_workflow.py::run_ilt_workflow(CURVMULTI_ILT_METHOD)
        -> opc/iteration/ilt/curvmulti.py::optimize_curvmulti_macro
            -> common lithography/loss/evaluation（_common + ICCAD13）
```

ILTMethod 五字段注入：

- `method_name="curvmulti_ilt"`；
- `config_type=CurvMultiConfig`；
- `optimize_macro=optimize_curvmulti_macro`；
- `evaluated_states=lambda config: len(config.scales) *
  (config.iterations_per_stage + 1)`；
- `build_fixed_context_canvas=build_curvmulti_final_context_canvas`
  （Simple 同款 σ(β(2T−1)) 三值语义，见 REQ-007）。

CurvMulti 只负责：

- 多尺度控制网格参数管理；
- 每 stage 独立 SGD；
- sigmoid mask 生成；
- nominal wafer curvature regularization。

不得负责：

- GDS 输入输出；
- macro/core partition；
- workflow 调度；
- result merge。

## 3. Requirements

### REQ-001 Public Workflow

CurvMulti MUST 复用 Simple/LevelSet 已有 workflow、pixel problem、batch 管理
和 result contract；入口文件只含适配器与 CLI 摘要，不复制流程。

### REQ-002 Multi-scale Optimization

`scales` 表示 control parameter grid，不表示 lithography grid。参数域是宏
ownership 全分辨率 `[Hm, Wm]`（与 Simple 的 flat_parameters、LevelSet 的 phi
同域）；scale=s 时控制网格为 `[Hm/s, Wm/s]`。

例如（Hm=Wm=256）：

```text
scale=4: 64x64 control
scale=2: 128x128 control
scale=1: 256x256 control
```

所有 Hopkins/lithography forward MUST 在完整 physical grid 执行：控制网格
mask 经 nearest 上采样回 `[Hm, Wm]` 后才进光刻模型（不得把小图直接补零送入
模型，否则图形物理尺寸缩小）。

### REQ-003 Stage Transition

每个 scale 使用独立 SGD optimizer（`torch.optim.SGD` 写死，不配置化）。

下一 scale 只允许继承上一 scale 的 sample best parameter（nearest warm-start，
不引入新灰度），禁止继承 optimizer state 与 autograd 图。

### REQ-004 Differentiable Mask

控制参数经：

```text
control parameter [Hm/s, Wm/s]
 -> nearest resize 到 [Hm, Wm]
 -> smooth_sigmoid_mask：avg_pool(k×k, stride 1, pad k//2) -> σ(β(x−offset))
 -> 经 trainable_index_canvas 组装进各 core 画布（Simple 同机制）
```

生成 mask。context/reference 只能作为固定约束（三值语义），不得成为优化变量。

### REQ-005 Loss

CurvMulti 使用公共损失并叠加私有曲率：

```text
loss = owned_continuous_losses(nominal, dose_max, defocus_min, target, ownership)
     + curvature_weight * curvature_loss(printed_nominal, ownership)
```

nominal loss MUST 使用 nominal printed result，不允许使用 printedMax 代替
（00_PAST 修正项：OpenILT 源码把 printedMax 误作 nominal）。

曲率 MUST 作用于 nominal **wafer**（printed 图像）而非输入 mask——这是
CurvMulti 与 Simple/LevelSet（mask 曲率）的核心算法差异；沿用现有
`curvature_loss(image, ownership)` 签名，ownership-only 计分，不修改公共
evaluation。

### REQ-006 Result Contract

结果 MUST 是现有 `ILTMacroResult`（best_parameters/soft_mask/binary_mask/
best_state_index/records），workflow 只读消费：

- `best_parameters`：末尺度（scale=1）全分辨率 `[Hm, Wm]` 参数；
- `soft_mask`：全局 best state 的宏 ownership soft mask；
- `binary_mask`：`soft_mask >= mask_threshold`；
- `best_state_index`：全局单调状态编号（跨 stage 连续）；
- `records`：见 REQ-008。

CurvMulti 可在 metrics.json 内额外保存各 stage 内部 control 参数摘要，但
workflow 消费面只认 ILTMacroResult。

### REQ-007 Fixed Context（三值语义）

训练与终评使用同一套 context 定义：真实 context（window 内、macro 外）取
初始版图 state0 soft `σ(β(2T−1))`（由常量 target 推导、无梯度边），
`build_curvmulti_final_context_canvas` 与 Simple 的 helper 逐值一致；window
外的数值 padding 恒 0（`context_valid_canvas` 判据）。

### REQ-008 Records 与进度

`ILTStateRecord` 写真值：`stage_index`（0 起）、`stage_state_index`（stage 内
0..N，0 为初始评价态）、`scale`（本 stage 控制网格缩放比）；`state_index`
全宏单调连续。N 次 update 对应每 stage N+1 个已评价状态（先评价初始态再
更新，与 Simple/LevelSet 的"末状态纯评价"对齐为"首状态纯评价"多尺度版）。

### REQ-009 宏同步屏障

同一 state 全部 core/batch 读同一控制网格快照；梯度经叶子张量 scatter-add
求和（绝不平均）回控制网格；全部 core 完成后恰一次 SGD step。stage 切换时
丢弃 optimizer 与图，只带走 best parameter。

### REQ-010 配置校验

构造即校验（沿 00_PAST `__post_init__` 全套）：

- `scales` 非空、正整数元组、严格递减且以 1 结尾；
- `smoothing_kernel` 正奇数；
- `[Hm, Wm]` 整除全部 scale；最粗尺度 `min(Hm/s, Wm/s) >= smoothing_kernel`；
- 曲率启用时宏边长 >= 3；
- 迭代次数正整数；浮点字段有限且在范围内
  （step_size>0、sigmoid_steepness>0、0<=sigmoid_offset<=1、权重>=0、
  0<mask_threshold<1）；bool 不得冒充 int。

## 4. Configuration

**DEC-配置（用户裁定 2026-08-20）**：自含 `CurvMultiConfig`，不建共享
ILTConfig 两段式（共享层零改动；Multilevel 批次再评估是否上提公共基类）；
optimizer 写死 SGD 不配置化。注册：`CONFIG_SECTIONS[CurvMultiConfig] =
"curvmulti_ilt"`（main/configuration.py）。

| Field | Meaning |
|---|---|
| scales | coarse-to-fine 控制网格缩放比，严格递减以 1 结尾 |
| iterations_per_stage | 每 stage SGD 更新次数 |
| step_size | SGD 学习率 |
| smoothing_kernel | sigmoid 前均值平滑窗口（正奇数） |
| sigmoid_steepness | sigmoid 斜率 β |
| sigmoid_offset | sigmoid 阈值偏移 |
| weight_process_l2 | process L2 权重 |
| weight_pvband | PVBand 权重 |
| curvature_weight | nominal wafer 曲率权重 |
| mask_threshold | 终评二值化阈值 |
| batch_size | 每 core 批大小（workflow 终评共用） |

## 5. Algorithm

### Initialization

```text
target_u8 -> T=target/255（[Hm, Wm] 宏 ownership 参数域）
stage_reference(scale) = resize(T, [Hm/s, Wm/s], mode="area")   # 保覆盖率
stage 0 初始 = stage_reference；后续 stage 初始 = resize(previous_best, shape, "nearest")
```

### Stage loop（每 stage）

```text
state 0：评价初始（无更新）
repeat N=iterations_per_stage：
    全 core/batch：快照 -> 组画布 -> forward -> loss -> backward
    屏障：scatter-add 梯度 -> 恰一次 SGD step
    评价更新后状态
stage best：严格更低才更新，平局保早
stage 结束：丢弃 optimizer/图，best 参数 nearest 带入下一 stage
```

最终输出全局 best（跨 stage 比较 total_loss，严格更低平局保早）的
scale=1 参数与 mask。

### State Management

- 每 stage state 0..N（N 次 update，N+1 个评价态）；
- `state_index` 全宏单调；`stage_index`/`stage_state_index`/`scale` 见
  REQ-008；
- 跨 stage 不共享 optimizer（REQ-003）。

## 6. Common Utilities

`_common.py` 新增（语义逐字迁自 00_PAST/_common.py:30-40，与具体方法无关）：

```python
resize_image(image, shape, mode)      # [B,H,W] 契约；interpolate area/nearest
smooth_sigmoid_mask(parameters, kernel, steepness, offset)
                                      # avg_pool2d(k, stride=1, pad=k//2)
                                      #  -> σ(β(x−offset))
```

## 7. Ownership and Context

CurvMulti 与 Simple/LevelSet 使用相同 ownership contract：

- macro ownership 为可优化域（**DEC-可动域**：不迁移旧 optimization_mask，
  ownership 即可动域，无平滑前/上采样后双层混合）；
- context 只读、不回写（三值语义，REQ-007）；
- 最终 mask 只提交 owned area（reconstruct 走现有 problem 契约）。

## 8. File Change Plan

| File | Change |
|---|---|
| opc/iteration/ilt/curvmulti.py | 新增：CurvMultiConfig、optimize_curvmulti_macro、build_curvmulti_final_context_canvas |
| opc/iteration/ilt/_common.py | 仅追加 resize_image/smooth_sigmoid_mask（现有函数零改动） |
| opc/iteration/ilt/__init__.py | 仅追加导出 |
| main/run_curvmulti_ilt.py | 新增入口（CURVMULTI_ILT_METHOD 适配器内联 + CLI，镜像 run_simple_ilt.py） |
| main/configuration.py | 仅 CONFIG_SECTIONS 追加一行注册 |
| config/curvmulti_ilt.toml | 新增 smoke 配置（corners_unit_clear.gds 同 simple 网格；scales=[4,2,1]；iterations_per_stage 小值演示） |
| tests/opc/iteration/test_curvmulti_ilt.py | 新增求解器测试 |
| tests/main/test_curvmulti_ilt_runner.py | 新增入口/配置/产物测试 |

## 9. Tests

必须覆盖：

1. `scales=[1]` 时退化为单尺度 ILT。
2. scale transition 后参数尺寸和 nearest warm-start 逐位正确；stage 参考
   area 逐位正确。
3. CurvMulti 与 Simple ILT 使用相同 workflow（fake method / 真入口集成）。
4. batch/reorder 不影响最终结果（宏同步屏障事件序）。
5. 非法 scale/kernel/config 正确报错（REQ-010 全套）。
6. 曲率作用于 printed nominal wafer 而非 mask（与 Simple mask 曲率可判别）。
7. stage 间 optimizer 独立（状态不泄漏）。
8. scatter-add 梯度求和（跨 core membership 累加，float64 镜像）。
9. records 写 stage/scale 真值；best 平局保早。
10. context 三值语义与跨宏 seam 初始一致。
11. 真 ICCAD13 CPU 全有限性；CUDA parity（可用时）。

## 10. Out of Scope

不包含：

- MRC/EPE；
- SRAF；
- 自动 scale selection；
- checkpoint；
- scheduler；
- 独立 ILT workflow；
- optimization_mask（ownership 内可动子域，裁定不迁移）；
- 共享 ILTConfig 两段式（裁定不自含以外改造）。

## 11. Final Principle

CurvMulti 是公共 ILT 框架中的一种优化算法，而不是新的 pipeline。所有 ILT
方法共享数据、workflow、evaluation 和输出接口，差异只存在于参数化方式和
优化策略。

## 12. 修订记录

- 2026-08-20：按审查结论修订（commit `5a4bf5f` 基线）——修正 3 处事实
  （ILTBatchResult→ILTMacroResult、workflow.py 注册→入口内联适配器+
  CONFIG_SECTIONS、optimize_curvmulti_batch→optimize_curvmulti_macro）；
  补 5 处契约（REQ-007/008/009/010、入口与 toml 文件计划）；DEC 三项
  （自含配置、wafer 曲率、不迁 optimization_mask）经用户裁定。
