# 开发报告 — CHG-20260818-levelset-ilt

## 实施基线（规格 §0 记录）

- 规格基线 `3bd7202`（父提交 `02de825`）；实施实际起点 HEAD `e036825`
  （自 `02de825` 起仅 docs 提交，生产代码与规格基线父提交一致），
  worktree 干净，起点全量 **545 passed**。
- 交付时全量 **603 passed + 1 skipped**（604 collected；skip 为 SDF 参数化
  中全背景单类退化入口，由常量场专测覆盖）。

## 实施概览

四批次本地 commit（对应规格 §21 阶段 A–D）：

| 阶段 | Commit | 内容 |
|---|---|---|
| A | a09414c | `ILTMethod` 增 `build_fixed_context_canvas` 第 5 字段；`_binary_canvas` 策略化；`_evaluate_best_binary` 去除 `sigmoid_steepness` 读取；`simple.py` 仅新增 `build_simple_final_context_canvas`（solver 热路径零改动）。固定 workload 终评数值改造前后 **bit-identical**（四宏 best loss/binary_l2/binary_pvband 全等） |
| B | 8390c93 | `opc/iteration/ilt/levelset.py`：SciPy EDT SDF（once/macro）、macro halo 梯度系数、`_LevelSetBinarize` STE、hard 终评 helper、`optimize_levelset_macro`（CPU phi + 契约超参 Adam + 跨 core raw-sum + 屏障单步）；requirements.txt 增 scipy（TEST-001..008，42 测试） |
| C | 71149c8 | `[levelset_ilt]` 注册、`_levelset_ilt_workflow.py`、`run_levelset_ilt.py` CLI、`config/levelset_ilt.toml` smoke（TEST-009/010，15 测试） |
| D | 本提交 | contracts/ilt.md、system.md、dataflow/levelset_ilt.md + index、两手册、报告三件套、active→completed |

## 与规格的偏差与裁决记录

1. **scipy 环境前置（审查观察 O1）**：WSL myopc312 env 实施前无 scipy，
   `pip install scipy==1.18.0` 后按规格写入 `requirements.txt`（本 change
   唯一新增运行时依赖，§17 授权）。Windows myopc env 若侧跑门禁需同样安装。
2. **门禁 scope 对齐（审查观察 O2）**：规格 §15 门禁含 `common`；
   `doc/development_manual.md` 原缺 `common` 与 `evaluation`。批次 D 将
   manual 门禁行对齐为 `common layout geometry opc lithography evaluation
   main tests`（实测 `common` 两门禁全绿）。
3. **smoke 配置值（审查观察 O3，按批准值实施）**：`config/levelset_ilt.toml`
   取 N=2（演示三状态）/step 0.2（旧 LevelSetConfig 默认）/权重同 simple
   smoke。补充事实：该配置下 mask 恒不变（见裁决 5），smoke 仅验证管线与
   资源口径。
4. **步长与 0 等值线（实施中发现，已入 contract）**：像素中心 SDF 的
   |phi| ≥ 1，Adam 首步 |Δ| = lr·|g|/(|g|+ε) < lr——lr ≤ 1 时边界像素需
   多状态累积才可能越过 0 等值线。同输入 throwaway 探针（lr=1.5、N=6）：
   total loss 8960.09 → 2854.83（state6 best）、binary_l2 3536 → 1720
   （优于 Simple 同输入 2876），优化链路端到端有效。步长提示写入
   `doc/contracts/ilt.md` 与 `dataflow/levelset_ilt.md`。
5. **测试侧修正（无生产影响）**：float64 镜像须以 [1,C,C] 批维调
   `forward_many`（2-D 输入会被 avg_pool2d 当 [N,C,L] 逐行池化——静默
   错误语义，测试镜像 bug 非生产 bug）；Adam 记录器的全局 monkeypatch
   须在镜像/参考优化器运行前规避（二次 patch 叠加记录链、边迭代边
   append 造成失控增长）。
6. **跨精度损失轨迹（测试容差裁决）**：lr=0.5 时个别像素恰落在符号翻转
   阈值，f32/f64 Adam 舍入差异被 hard 阈值放大成损失跳变——非语义属性；
   raw-sum 测试取 lr=0.2（两步内 |Δ|<0.4，无 0 等值线跨越）保证跨精度
   可比。

## 对照 smoke（规格 §11.9，同输入 corners_unit，WSL GTX1650 CUDA）

| 指标 | LevelSet（N=2，3 态） | Simple（N=1，2 态） |
|---|---|---|
| total_seconds | 2.492 | 0.446 |
| prepare_seconds | 0.050 | 0.043 |
| merge_seconds | 0.003 | 0.002 |
| peak_rss_mib | 1100.9 | 1120.7 |
| cuda_peak_mib | 510.0 | 508.0 |
| best_state / state_count | 0 / 3 | 1 / 2 |
| best_total_loss | 8960.091 | 6162.660 |
| binary_l2 / binary_pvband | 3536 / 846 | 2876 / 883 |
| SDF 秒数（调用数） | 0.0407（1） | — |
| macro_grad 秒数（调用数） | 0.0040（2） | — |

口径说明：LevelSet 先行运行含 CUDA 冷启动，总时长差主要由状态数（3 vs 2）
与首跑 warmup 构成；SDF 40.7ms/宏、宏梯度 4ms（恰 iterations 次）均为
热路径外开销，满足 §11.1/11.2（once/macro、iterations/macro，不随
core/batch 增长）。LevelSet 常驻 CPU 侧仅 phi/Adam 态/宏梯度（各 [Hm,Wm]）
与 initial_query_phi（[Hq,Wq]），RSS 与 Simple 相当（+Δ 在噪声内）。

## 最终审计（规格 §22 逐条）

1. ✓ 实施基线/status/测试数已记录（本报告首节；规格 frontmatter 保持
   baseline_commit 不变）。
2. ✓ production SDF 使用 SciPy EDT 且一个 macro 恰一次
   （`test_production_calls_scipy_edt` + `test_sdf_computed_once_per_macro`
   + 端到端 `test_final_context_does_not_rerun_sdf` 全程恰 macro 次）。
3. ✓ `macro_gradient_magnitude` 调用恒为 iterations（`test_called_once_
   per_backward_state`；smoke 实测 2 次 = N）。
4. ✓ 无每 state 完整 `current_query_phi/query_grad_magnitude` query field
   （实现只构造 [Hm+2,Wm+2] halo 与 [Hm,Wm] 系数；代码审查确认）。
5. ✓ 同一参数跨 core 的 phi/系数一致（`test_same_state_mask_agrees_
   across_cores` 重叠带逐位一致 + raw-sum 镜像逐值一致）。
6. ✓ 跨 core 梯度为 raw sum 不平均（float64 单图镜像 + batch 1↔4 不变）。
7. ✓ Adam 每 backward state 恰一次 macro update 且与 PyTorch reference
   逐位一致（捕获梯度喂独立 Adam `assert_array_equal`；事件序
   f×4+s 循环 + 末态纯评价）。
8. ✓ LevelSet final context 不重复 SDF（spy 计数 0）。
9. ✓ `_ilt_workflow` 不访问方法数学字段（fake method 无
   sigmoid_steepness/mask_threshold 走完终评；ILTMethod 恰五字段）。
10. ✓ Simple solver 热路径零改写（阶段 A 仅增模块级 helper；固定 workload
    终评数值 bit-identical；全量 Simple 套件绿）。
11. ✓ 无 workflow 复制、core-local SDF/梯度、梯度平均、padding 伪透光
    （专项测试 + 代码审查）。
12. ✓ `doc/architecture/dataflow/levelset_ilt.md` 与 index 已更新（连同
    contracts/system/两手册）。
13. ✓ full tests（603 passed）、ruff、compileall、两入口直跑、diff-check
    全绿；test_report 见同目录。
