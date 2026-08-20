# 测试报告 — CHG-20260818-levelset-ilt

## 总量

- 起点 545 passed → 交付 **603 passed + 1 skipped（604 collected）**。
- 增量：阶段 A +2（helper 等价、fake method 终评独立性）；阶段 B +42
  （TEST-001..008）；阶段 C +15（runner 12 + configuration 3）。
- 唯一 skip：SDF 参数化的全背景单类入口——退化语义由
  `test_all_background_constant_field` 常量场专测覆盖，参数化内 skip 守卫。
- 门禁：ruff / compileall（`common layout geometry opc lithography
  evaluation main tests`）/ `git diff --check` 全绿。

## 规格测试矩阵（TEST-001..010 → 实测落点）

| 规格 | 落点（tests/opc/iteration/test_levelset_ilt.py 除非注明） |
|---|---|
| TEST-001 SDF | `TestSignedDistanceInitialization`：暴力 oracle（散点/矩形/带洞，rtol 1e-6）、127/128 符号逐格、全前景/全背景 ±max 常量场、SciPy 路径 spy（恰 2 次 EDT）、once/macro spy（N=2 恰 1 次） |
| TEST-002 halo | `TestMacroGradientMagnitude`：标量循环手算（含边缘参数用真实 query context）、常数 phi 判别（replicate 实现边缘系数全 0 → 失配）、形状拒绝、调用数恰 iterations |
| TEST-003 跨 core 同一性 | `TestCrossCoreIdentity`：state0 单批捕获全部 core 画布，query 栅格交集带 mask 逐位相等（判别性：交集 >0） |
| TEST-004 STE | `TestLevelSetBinarize`：前向 `(phi<0)`（phi==0→0）、反向 `-mag·grad_output`、mag.grad None、一维张量反传（内部无 pad/差分） |
| TEST-005 跨 core 求和 | `TestGradientSumAndAdam::test_macro_gradient_is_raw_sum`（_LocalAverageModel 耦合 + float64 单图镜像逐 state）+ `test_batch_size_invariance`（1↔全核单批） |
| TEST-006 Adam/屏障 | `test_adam_matches_torch_reference_exactly`（捕获梯度喂独立 Adam，`assert_array_equal` 逐位）、`test_step_only_after_all_cores_of_state`（事件序 f×4+s ×2 + f×4）、`test_two_updates_three_evaluated_states` |
| TEST-007 context/padding | `TestContextAndPadding`：context<pixel 无条件 ValueError、hard context=（T≥0.5）逐位、padding 严格 0、画布不一致拒绝、NaN → FloatingPointError |
| TEST-008 loss/曲率/真模型 | `TestLossesCurvatureAndRealModel`：恒等模型 state0 三损失 numpy 复算、曲率 hard mask 3×3 核复算 + weight0 零调用（状态数×批数恰调）、真 ICCAD13 CPU 全有限 + INV-004（state0 二值对靶）、CUDA parity 1e-4 |
| TEST-009 workflow 独立性 | tests/main：fake method（config 仅 batch_size）走完终评（阶段 A）、LevelSet 终评不重跑 SDF（全程恰 macro 次）、ILTMethod 五字段消费面、Simple helper 等价（test_simple_ilt.py：numpy 逐位 + torch 1e-6 容差） |
| TEST-010 零回归/CLI/产物 | tests/main/test_levelset_ilt_runner.py：合法/缺键/未知键（含 Simple 字段混入）/未知段/bool/段注册/context<pixel 传播无 summary/仓库外直跑；产物 schema（best_parameters=phi、binary==phi<0、soft==sigmoid(-phi)、records=N+1、dtype 全集）；tests/main/test_configuration.py：[levelset_ilt] 严格解析 3 例；Simple 零回归=阶段 A 固定 workload bit-identical + 全量 Simple 套件绿 |

## 测试实现要点（复现性）

- **float64 单图镜像**：逐 core [1,C,C] 单图 autograd + 镜像独立 halo 差分
  + torch Adam 跟踪——与生产完全不同路径复算逐 state 损失/梯度；lr=0.2
  避免 0 等值线跨越的跨精度阈值放大（见 development_report 裁决 6）。
- **Adam 记录器**：monkeypatch `torch.optim.Adam` 为记录子类，捕获每次
  step 前的宏梯度/step 后参数/事件序；镜像与参考优化器必须用 patch 前的
  真类（`recorder.real_cls`），否则记录链叠加/边迭代边增长。
- **真模型**：ICCAD13 CPU（fixture `cpu_model`）+ CUDA（`skipif` 无卡）；
  WSL GTX1650 实测 parity rel 1e-4 通过。

## 入口直跑（规格 §15）

- `python main/run_levelset_ilt.py config/levelset_ilt.toml`：退出码 0，
  1 macro / 16 core / 3 评价态，CUDA 510 MiB（对照表见 development_report）。
- `python main/run_simple_ilt.py config/simple_ilt.toml`：退出码 0（CLI 层
  Simple 零回归复核）。
