# 测试报告 — CHG-20260818-simple-ilt

## 环境

- WSL2 / `~/miniconda3/envs/myopc312/bin/python`（3.12.0）；torch 2.13.0+cu130，
  CUDA 可用（TEST-009 CPU/CUDA parity 实跑，无 skip）。
- 基线 540a012；全量 **525 passed**（迁移前 458 + 本 change 新增 67：
  pixel problem 20 + simple ILT 30 + runner 12 + GridRuntime 5）。

## 命令与结果

```bash
python3 -m pytest -q tests/opc/input/test_pixel_problem.py        # 20 passed
python3 -m pytest -q tests/opc/iteration/test_simple_ilt.py       # 30 passed
python3 -m pytest -q tests/main/test_simple_ilt_runner.py \
                    tests/main/test_macro_pipeline.py \
                    tests/main/test_mbopc_runners.py              # 12+全过
python3 -m pytest -q tests                                        # 525 passed
python3 -m ruff check common layout geometry opc lithography evaluation main tests
python3 -m compileall -q common layout geometry opc lithography evaluation main tests
python3 main/run_simple_ilt.py config/simple_ilt.toml             # smoke 1.90s
git diff --check
```

全部通过（ruff/compileall 无输出；smoke 摘要见开发报告性能基线节）。

## 规格测试矩阵 → 实际用例映射

| 规格 | 实际用例（文件::类/方法） |
|---|---|
| TEST-001 | test_pixel_problem.py::TestPrepareAndPersistence（一次栅格 spy、edge 零调用守卫、极性/分数覆盖、NPZ 往返） |
| TEST-002 | ::TestCoreMapping（计分恰一次、trainable 索引跨 core 一致 + target 对齐、公共栅格对照）+ ::TestPixelAlignment（整像素缩短合法 / 非整像素栅格前拒绝） |
| TEST-003 | ::TestReconstruct（clear/opaque × 矩形/孔/凹/斜/多岛/全 0/全 1 回环、形状拒绝） |
| TEST-004 | ::TestCorruption（格式名/版本/dtype/shape/切线损坏拒绝） |
| TEST-005 | test_simple_ilt.py::TestParameterizationAndLoss（state0 恢复 <1e-8、常数模型手算、曲率 numpy 参考、0/1 有限、ownership 限定）+ ::TestConfigValidation |
| TEST-006 | ::TestStateAndBarrier（N+1/调用计数、float64 镜像逐 state、屏障无中途 step） |
| TEST-007 | ::TestMacroBestAndBatch（4-core 数值表 macro-best ≠ 材料核局部最优、batch 1/2/4 不变） |
| TEST-008 | ::TestCrossCoreGradient（avgpool 耦合模型 batch1/2 一致 + np.add.at spy、参数域形状） |
| TEST-009 | ::TestRealModel（CPU 有限且更新、CUDA parity 1e-4） |
| TEST-010 | ::TestCallCountsAndProgress（混合批真实回调 [6,6,4]×3、曲率开关、NaN 异常、画布不一致拒绝）+ ::TestStateAndBarrier 调用计数 |
| TEST-011 | test_simple_ilt_runner.py::TestConfigAndEntry（合法/缺键/未知键/未知段/float 冒充、[edge] 非必需、postponed 注解探针、仓库外直跑） |
| TEST-012 | ::TestArtifacts（plan/problem/result/metrics/best/final/summary 字段与 dtype） |
| TEST-013 | ::TestMergeAndSeam（merge 恰一次；最终栅格 == 全宏 ownership 二值精确拼接，无重叠遗漏） |
| TEST-014 | ::TestExceptionCleanup（第二宏异常传播、无 summary、首宏产物留诊断） |
| TEST-015 | 全量 525 passed + 阶段 A 单遍 smoke XOR==0（既有 writer 数值零变化） |

## 已知口径

- float64 镜像对照容差 rel 1e-4（float32 实现对 float64 参考）。
- 跨 core 梯度判别依赖 avgpool 耦合 stub（逐点模型无空间耦合、无法产生
  跨 core 梯度）；batch1/2 结果一致 + spy 求和共同锁定语义。
- logit 饱和性质：纯对齐几何一轮更新的参数变化低于记录精度，TEST-009
  采用非对齐几何（含分数覆盖格）作梯度入口。
