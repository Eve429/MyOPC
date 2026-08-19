# 测试报告 — CHG-20260819-simple-ilt-openilt-init

## 环境

WSL2 / `~/miniconda3/envs/myopc312/bin/python`（3.12.0，CUDA 可见）。
基线 14ab3e8（525 passed）→ 实施 **526 passed**（+1：对齐几何 P1-1 回归）
→ Rev 1.1 **527 passed**（+1：跨宏 seam 初始 transmission 一致性）
→ Rev 1.2 **528 passed**（+1：数值 padding 严格 0 / 物理 T=0 context 为 σ(−β)）
→ Rev 1.3 **529 passed**（+1：curvature>0 且 context=0 入口拒绝、关曲率合法；索引 dtype int64 断言并入一致性用例）。

## 命令与结果

```bash
python3 -m pytest -q tests/opc/iteration/test_simple_ilt.py   # 31 passed
python3 -m pytest -q tests                                    # 526 passed
python3 -m ruff check common layout geometry opc lithography evaluation main tests
python3 -m compileall -q common layout geometry opc lithography evaluation main tests
python3 main/run_simple_ilt.py config/simple_ilt.toml          # smoke 0.99s
git diff --check
```

全部通过。

## 规格测试项 → 实际用例

| 规格 | 用例 |
|---|---|
| REQ-A（2T−1 初始化/state0 soft 公式） | TestParameterizationAndLoss::test_state0_soft_matches_openilt_initialization（恒等模型 numpy 复算 3Σd²） |
| REQ-B（二值一致性） | 同上内联 probe（[0, 64/255, 128/255, 1] → [F,F,T,T]） |
| REQ-C（P1-1 对齐几何回归） | TestRealModel::test_real_cpu_updates_aligned_geometry（loss₁≠loss₀ 且 max\|Δp\|≈3.9e-4 > 1e-5；旧方案 <1e-6） |
| 镜像/屏障/常数/曲率/patchwork 跟随 | _float64_reference init 换 2T−1；常数模型期望回归 T 监督（监督与初始化无关，纠正过一次误改）；曲率参考卷积输入换 soft₀ |
| REQ-D（Rev 1.1，context 统一） | TestMacroSeamConsistency::test_context_matches_neighbor_state0（A context 的 B 区像素与 B 自身 state0 逐位相等，且值为 σ(β(2T−1))≠raw T）；P1-3 重构回归 = test_pixel_problem 全部 20 例一致性用例 |
| REQ-D 补（Rev 1.2，padding 三值） | TestCanvasPaddingSemantics::test_padding_strictly_zero_and_context_soft（window 12px：padding 严格 0；物理 T=0 context 格 = σ(−β)，两侧同时判别） |
| 既有语义零回归 | batch 不变/跨 core 梯度和/计数/runner 12 例零改动通过 |

## 已知口径

- Δp 阈值 1e-5 依据实测（3.9e-4）与饱和区（<1e-6）各隔一个量级。
- 非对齐几何用例保留（分数覆盖格路径回归），docstring 已注明历史语义。
