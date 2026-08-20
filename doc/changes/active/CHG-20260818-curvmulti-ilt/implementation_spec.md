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
  - doc/contracts/ilt.md
---

# CurvMulti ILT 迁移

## 0. Document Contract

本文基于 Simple ILT 完成后的公共 ILT 框架实现 CurvMulti。实施前 MUST 确认公共 `ILTMethod`、`ILTBatchResult`、`PixelMacroProblem` 和 `_ilt_workflow` 已稳定。

CurvMulti MUST 作为 ILT method 实现，不得复制 workflow，不得建立独立 runner 体系，不得让 Simple/LevelSet/CurvMulti 相互依赖。

## 1. Objective

实现基于 OpenILT CurvMulti 思路的多尺度 ILT optimizer：采用 coarse-to-fine control grid、sigmoid differentiable mask、nearest warm-start 和 nominal wafer curvature regularization。

目标是证明公共 ILT pipeline 可以支持不同 parameterization，而不是新增一套 ILT 流程。

## 2. Design Changes

### 2.1 Architecture

原方案中的 `_curvmulti_ilt_workflow` 和独立入口删除。

目标架构：

```text
run_ilt_workflow
    -> ILTMethod(curvmulti)
        -> optimize_curvmulti_batch
            -> curvmulti optimizer
            -> common lithography/loss/evaluation
```

CurvMulti 只负责：
- 多尺度参数管理；
- SGD stage 优化；
- sigmoid mask 生成；
- curvature regularization。

不得负责：
- GDS 输入输出；
- macro/core partition；
- workflow 调度；
- result merge。

## 3. Requirements

### REQ-001 Public Workflow

CurvMulti MUST 复用 Simple ILT 已有 workflow、pixel problem、batch 管理和 result contract。

### REQ-002 Multi-scale Optimization

`scales` 表示 control parameter grid，不表示 lithography grid。

例如：

```text
scale=4: 64x64 control
scale=2: 128x128 control
scale=1: 256x256 control
```

所有 Hopkins/lithography forward MUST 在完整 physical grid 执行。

### REQ-003 Stage Transition

每个 scale 使用独立 optimizer。

下一 scale 只允许继承上一 scale 的 sample best parameter，禁止继承 optimizer state。

### REQ-004 Differentiable Mask

控制参数通过：

```text
parameter
 -> optional smoothing
 -> sigmoid(beta*(x-offset))
 -> full resolution mask
```

生成 mask。

context/reference 只能作为固定约束，不得成为优化变量。

### REQ-005 Loss

CurvMulti 使用公共 lithography loss：

```text
loss = common litho loss + curvature regularizer
```

nominal loss MUST 使用 nominal printed result，不允许使用 printedMax 代替。

curvature 仅作为 CurvMulti 私有 regularizer，不修改公共 evaluation。

### REQ-006 Result Contract

ILTBatchResult 不强制所有 ILT 使用同一种 parameter。

统一输出：

```text
final_mask
metrics
records
(optional) parameters
```

CurvMulti 可以保存内部 control parameters，但 workflow 使用 final mask。

## 4. Configuration

CurvMulti 配置拆分为通用 ILT 配置和方法专属配置。

通用字段归属 ILTConfig：

- iterations
- optimizer
- batch_size
- loss weights

CurvMultiConfig：

| Field | Meaning |
|---|---|
| scales | coarse-to-fine control scales |
| smoothing_kernel | sigmoid 前平滑窗口 |
| sigmoid_steepness | sigmoid slope |
| sigmoid_offset | threshold offset |
| curvature_weight | curvature regularization weight |

不要在 CurvMultiConfig 重复定义通用 ILT 参数。

## 5. Algorithm

### Initialization

```text
target mask
 -> control parameter initialization
 -> scale loop
```

每个 scale：

```text
previous best
 -> resize control parameter
 -> independent SGD
 -> generate differentiable mask
 -> restore full resolution
 -> lithography forward
 -> loss backward
 -> update best
```

最终只输出 scale=1 best mask。

### State Management

每个 stage：

- state 0..N；
- state 0 为评价初始状态；
- N 次 update；
- 保存 sample best。

跨 stage 不共享 optimizer。

## 6. Common Utilities

允许 `_common` 提供：

```python
resize_image()
smooth_sigmoid_mask()
```

这些函数必须与具体 ILT method 无关。

## 7. Ownership and Context

CurvMulti 与 Simple/LevelSet 使用相同 ownership contract：

- ownership 区域可优化；
- context 只读；
- context 不回写；
- 最终 mask 只提交 owned area。

## 8. File Change Plan

| File | Change |
|---|---|
| opc/iteration/ilt/curvmulti.py | 新增 CurvMulti optimizer，实现多尺度 SGD |
| opc/iteration/ilt/_common.py | 增加通用 resize/sigmoid helper |
| opc/iteration/ilt/workflow.py | 仅增加 method 注册，不复制流程 |
| config | 增加 CurvMultiConfig |
| tests | 增加 multi-scale、scale=1 regression、workflow integration 测试 |

## 9. Tests

必须覆盖：

1. `scales=[1]` 时退化为单尺度 ILT。
2. scale transition 后参数尺寸和 warm-start 正确。
3. CurvMulti 与 Simple ILT 使用相同 workflow。
4. batch/reorder 不影响最终结果。
5. 非法 scale/kernel/config 正确报错。

## 10. Out of Scope

不包含：

- MRC/EPE；
- SRAF；
- 自动 scale selection；
- checkpoint；
- scheduler；
- 独立 ILT workflow。

## 11. Final Principle

CurvMulti 是公共 ILT 框架中的一种优化算法，而不是新的 pipeline。所有 ILT 方法共享数据、workflow、evaluation 和输出接口，差异只存在于参数化方式和优化策略。
