---
id: CHG-20260819-simple-ilt-openilt-init
title: Simple ILT 初始化切换为 OpenILT 2T−1 方案（P1-1）
type: implementation-spec
status: approved
baseline_commit: 14ab3e8
baseline_worktree: dirty
baseline_dirty_paths:
  - .learnings/
  - doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/
scope:
  - opc/iteration/ilt
  - tests/opc/iteration
  - doc
depends_on:
  - doc/contracts/ilt.md
  - doc/changes/completed/CHG-20260818-simple-ilt/implementation_spec.md
supersedes:
  - CHG-20260818-simple-ilt 的 REQ-006 初始化子句（logit+eps 与 1e-6 恢复契约）
---

# Simple ILT 初始化切换为 OpenILT 2T−1 方案（P1-1）

## 0. Document Contract

本规格是本 change 的唯一实现依据；实现 AI 不得依赖聊天上下文补充行为。
基线为 `14ab3e8`（CHG-20260818-simple-ilt 全部交付后）。不修改
`00_PAST/**`、`layout/**`、`geometry/**` 与用户数据；不回改已完成 CHG
的历史规格/报告（本 change 以 supersedes 取代其 REQ-006 初始化子句）。

## 1. Objective

修复 P1-1：logit+float32-eps 初始化使 0/1 像素 sigmoid 斜率仅 β·eps
（β=4 时 ≈4.8e-7），内部纯 0/1 像素几乎不可优化；ILT 的拓扑变化/内部
开孔/SRAF 形成要求这些区域能被激活。切换为 OpenILT 的
`params = 2·target − 1`，斜率恢复 β·σ(β)σ(−β)（β=4 时 ≈0.0707，
约 1.5×10⁵ 倍）。

## 2. Baseline and Evidence

- `opc/iteration/ilt/simple.py` 现初始化：clip(T, eps, 1−eps) →
  logit/β；含"state0 soft 在 1e-6 内恢复 T"的构造不变量 raise。
- 实测（CHG-20260818 开发报告）：纯 pixel-aligned 几何的一轮参数变化
  低于记录精度，真模型更新只能靠 fractional coverage 格证明。
- `00_PAST/opc/iteration/ilt/simple.py`：`initial = target.mul(2).sub(1)`
  （OpenILT 同式），端点参数 ±1，sigmoid 不饱和。

## 3. Target Behavior

- **REQ-A**：macro 参数初值 = `2·T − 1`（T = ownership target_u8/255 的
  float32，含分数覆盖率格，连续取值；无 clamp）。
- **REQ-B**：废除"state0 soft 1e-6 内恢复 T"契约。新契约：
  state0 soft = `σ(β·(2T−1))`；在 `mask_threshold` 下二值化与
  `T ≥ 0.5` 逐格一致（`σ(β(2T−1)) ≥ 0.5 ⟺ T ≥ 0.5`）。
- **REQ-C**（P1-1 回归）：纯对齐几何（无分数格）+ 真实 ICCAD13 一轮
  更新可观测：records[1].total_loss ≠ records[0]，且 best_parameters
  相对初值 max|Δp| > 1e-3。

## 4. Scope

In：`opc/iteration/ilt/simple.py` 初始化块；`tests/opc/iteration/
test_simple_ilt.py` 受影响用例与新增回归；contracts/手册/记录与本规格。
Out：初始化策略配置开关；context 直通 target 的固定区设计；宏同步/
梯度累加/best 语义；LevelSet/CurvMulti/Multilevel 规格。

## 5. Invariants

- 除初始化与 state0 soft 数值外，求解器全部既有语义不变（N+1 状态、
  屏障 step、macro best 严格更低、scatter-add 求和、异常行为）。
- 二进制掩膜 state0 一致性（REQ-B）由测试锁定。

## 6. Test Specification

1. state0 恒等模型总损失 == numpy 复算 `Σ_own (σ(β(2T−1)) − T)²` 加权和
   （替换原 1e-6 恢复用例）。
2. float64 镜像 init 同步 2T−1（镜像/屏障用例自动跟随）。
3. 常数模型/曲率/patchwork 期望值以 soft₀ = σ(β(2T−1)) 计算。
4. 新增 REQ-C 回归（对齐几何真模型更新可见）。
5. 其余用例（batch 不变/跨 core/计数/runner）零改动预期。

## 7. Acceptance Criteria

- [ ] AC-1：全量 pytest、ruff、compileall、`git diff --check` 通过。
- [ ] AC-2：REQ-A/B/C 三性质各有测试锁定；smoke 重跑记录新基线
      （state0 loss、best_state、binaryL2、资源）。
- [ ] AC-3：contracts/ilt.md、development_manual §9、test_manual、
      规划三文件与本规格 + 两报告同步，规格移 completed。

## 8. Decisions

- 放弃 state0 精确恢复契约是用户裁决（2026-08-19 P1-1）：可优化性优先
  于 state0 损失最小化；OpenILT 原方案即此取舍。
- context 固定区仍直通 target（不采用 OpenILT 的 fixed_soft 参数化），
  维持 CHG-20260818 的既有保真差异。

## 9. Revision History

| Revision | Date | Status | Change | Reviewer |
|---|---|---|---|---|
| 0.1 | 2026-08-19 | approved | 首版（用户 P1-1 裁决 + 批准开发计划） | 用户 |
| 1.0 | 2026-08-19 | completed | 实施完成（ebce389）：526 passed；REQ-C 阈值定 1e-5、smoke step_size 重调 1.0（偏差见 development_report）；smoke 新基线 7952→6233 | 开发/测试报告 |
