# 最简 MB-OPC 迁移测试报告

> 环境：Windows 10、myopc conda env（Python 3.12、torch 2.5.1+cu124、
> klayout 0.30.10、numpy 2.5.1、tqdm 4.70.0）；GPU NVIDIA GTX 1650。
> 日期：2026-08-16。

## 1. 命令与结果

```text
pytest -q tests                       341 passed（迁移 330 + 审查修复轮 11 新增）
pytest -q tests/evaluation             25 passed
pytest -q tests/opc/iteration          53 passed
pytest -q tests/main/test_mbopc_runners.py 23 passed
ruff check evaluation lithography opc main tests   All checks passed
compileall -q evaluation lithography opc main tests  通过
coverage --source=evaluation → metrics.py 100%（66/66）
coverage --source=opc/iteration → simple.py 99%（缺两行不可构造防御行）
```

## 2. 套件职责

| 套件 | 数量 | 职责 |
|---|---|---|
| tests/evaluation | 25 | L2/PVBand 计数与 ownership 屏蔽、EPE 四方向/invalid/阈值边界、契约 isinstance |
| tests/opc/iteration | 51 | cache 全路径、Config 校验、入口契约、stub 模型方向与停止路径、batch/进度/cache 计数、真实 ICCAD13 图形矩阵、CUDA 直通 |
| tests/main/test_mbopc_runners | 21 | 单/多入口产物与 records 语义、恰一次 merge、正逆序、batch 不变性、invalid 保留 best、差异量化、subprocess 仓库外直跑、进度开关、配置校验 |

## 3. 图形矩阵覆盖（§16.3）

普通矩形、4nm 窄壁中空（探针距离 4nm）、凹 L 形、多 polygon 各带 hole、
45° 斜边（直角三角形，验证单位外法向支持非 Manhattan 边）、横跨 ≥3 core
横条、斜边跨 macro 边界（2×2 逐 macro）、opaque 极性、空 macro（零段零探针
baseline 即 zero_epe）。全部以真实 ICCAD13 CPU 模型跑 1 轮迭代，断言四种
合法停止原因之一 + 结构不变量（best 是已评价轮、位移有限、context 归零、
|位移|≤上限）。
说明：SREF/AREF 展开属 layout 读取层契约（tests/layout 已覆盖
`test_materialized_region_batch_survives_database_close` 等 + run_macro_pipeline
gcd_45nm smoke 真实层级版图），本套件用平坦 Region 直构。

## 4. 关键机制证据（monkeypatch 计数）

- **每 batch 恰一次三条件 forward_many**：4 core / batch 2 → 调用序列 [3, 3]
  （每次三条件）。
- **cache 命中免重栅格**：同一状态第二次评价的栅格化调用 8→4（target 全命中，
  只剩 mask）。
- **batch 不变性**：batch_size=1 与 4 的指标与位移逐位一致；runner 层
  batch_size=2 与 4 的 result.npz best_displacements 逐位一致。
- **恰一次 merge**：multi 全流程 `merge_macro_results` 计数 == [1]。
- **进度真实计数**：回调总数 = (iterations+1)×core_count，单次 ≤ batch_size；
  提前停止不补 100%。
- **L2 不打破 EPE 平局**：篡改 evaluate_binary_l2 恒 0，EPE 平局仍保留较早轮。

## 5. 停止路径覆盖

| 路径 | 用例 |
|---|---|
| zero_epe（baseline） | 直通模型零位移与 target 全同 |
| zero_epe（循环内） | 步长 4nm（像素整数倍）+ 直通输出两轮归零 |
| no_update | 反相模型全 ambiguous（方向 0、EPE 非零；提案与当前一致直接停止，records 只含 baseline） |
| invalid_geometry | 第二个 macro 候选重建抛错：保留 best、stop_detail 含原因、其余 macro 继续、最终合并照常 |
| invalid_geometry（真构造×2） | 大步长 hull 内移使 hole 越出 hull；一步 20nm 四边共线退化（ValueError 形态，实测记录） |
| insufficient_probes | 2nm 窄壁 + 8nm 探针距离（审查 P1.1 复现场景）：valid=0/epe=0/保留 baseline/原因在案 |
| iteration_limit | 持续全暗输出跑满轮次（EPE 平局保留 baseline） |

## 6. 端到端 smoke（gcd_45nm，CUDA）

见开发报告 §6：两入口默认配置各 ~126s 完成，870 tile，EPE 逐轮单调下降，
multi 总 best_epe 23676 vs single 23440（独立 context 代价 236 段，约 1.0%）、
最终覆盖 XOR 面积 34650860 DBU²——差异已量化，不宣称等价全局同步。
进程内差异量化用例（TestSingleVersusMulti）在生成式小版图上同样计算并输出
差异面积。

## 7. 未覆盖与如实记录

- `simple.py:266/269` 两行防御 RuntimeError（owner 写集被破坏/context 被意外
  修改）：触发需破坏 MacroProblem 构造期已保证的不变量，正常与注入路径均
  无法构造，不强行伪造。
- 循环内 zero_epe 依赖“步长为像素整数倍”的构造（真实 ICCAD13 输出下小图形
  8 轮 EPE 未归零，属启发式方法的已知行为而非缺陷）。
- CUDA parity 数值断言沿用 lithography 套件（1e-4）；本套件 CUDA 用例只断言
  完成与有限性（离散方向对 ±1e-4 级灰度差稳健）。
- merge_peak/iteration RSS：本 workflow 未逐 macro 采样 RSS（summary 记录
  耗时与 tile 数；显存峰值由 lithography 套件的 peak allocated 机制覆盖），
  如实记录口径。
