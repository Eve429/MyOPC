# 梯度 MB-OPC 迁移开发报告（CHG-20260816-gradient-mbopc）

日期：2026-08-17。规格：`doc/opc/gradient_mbopc_migration_design.md`（Revision 0.2）。
实施模式：规格事实核对 → 用户批准（含四项裁决）→ 四批本地实施。

## 提交链

```text
42bf6f3  refactor(mbopc): 抽出共享目标画布缓存（批 A，IF-001）
17ff75c  feat(mbopc): 实现梯度边段优化器（批 B，44 测试）
c3e59bc  feat(main): 接入梯度MBOPC直接入口（批 C，25 测试）
（批 D）docs(mbopc): 完成梯度迁移报告
```

基线核对：规格锁 `e289f2c`，实施起点 `fbc059b`，其间仅规格入库（4b243c3）与
doc_ 文档三提交，业务代码零漂移。

## 事实核对结论（实施前，全部成立）

- `forward_many` 全链可微（pad→FFT→复乘→IFFT→sigmoid，无 no_grad/detach），
  tests/lithography 已有 6 个 backward 测试（含有限差分对照）——梯度方法地基；
- `MacroProblem` CSR / `materialize()` 段级几何 / `rasterize_mask_canvas` 等
  全部接口签名与规格一致；正位移双极性扩大透光（fragmentation.py 法向翻转）；
- autograd.Function 直通返回的输出与输入逐位相等（torch.equal 层面）。

## 四项用户裁决（计划批准即生效）

1. **几何退化宽捕获**：规格 §10.4 原文只捕 `ReconstructionError` 且 MUST NOT
   捕 ValueError；实测（simple 轮 + 本次再次核对 reconstruction.py 无 try/except）
   几何退化以 ValueError 从 KLayout 穿透。裁决为 gradient 捕
   `(ReconstructionError, ValueError)`，注释附实测依据；TEST-007 的"程序错误
   不被吞"用例改用 RuntimeError 验证（ValueError 与几何退化不可按类型区分）。
   Revision 0.2 已修订规格对应条款。
2. **P=0 解读**：S==0 或 O==0 ⇒ 必然 P==0 ⇒ 不跑 forward、单条全零 record +
   `no_owned_segments`（不建 optimizer）；O>0 而 P==0 才是 ERR-002 的数据
   损坏 ValueError（实测不可达，防御保留）。
3. **段法向常驻**：常驻 `segment_normals float64[S,2]`（约 S×16B）；PERF-002
   "不常驻三份全量数组"按内存上界意图理解，starts/ends 提取中点后即释放。
4. **doc_ 副本不动**：`doc_/changes/active/` 保持原样副本；规格修订只改
   `doc/opc/` 原件。

## 关键实现事实

- **surrogate**（`_EdgeGradientMask`）：forward 逐位直通 hard mask；backward
  向量化双线性四角采样（扁平索引一次 gather，每条 membership 只读 4 像素），
  越界中点整体置零，梯度 `2·g_mid`；重复参数索引由 autograd 求和。
- **初始化一次的 immutable 状态**（阶段 2）：owner 映射
  `segment_to_parameter[S]`、参考中点/法向（materialize 提取后释放三份全量
  数组）、`reconstruct_region(zeros)` 参考几何、逐 core 探针 canvas 坐标
  （固定参考，与状态无关）、loss 分母 P；全迭代复用。
- **评价循环**：批组 target（uint8 LRU）/当前 mask/ownership → 需要梯度时经
  `_EdgeGradientMask.apply` 接 autograd 边（末状态与无可训练批直通不建图）→
  一次 `forward_many` 三条件 → 建图版三项 loss `backward()` 累积 → no_grad
  离散诊断（L2/PVBand/EPE，探针坐标预计算复用）→ del 批张量 → 进度回调。
- **状态发布**：全部 tile 屏障后 grad 有限检查 → `optimizer.step()` →
  clamp ±max_displacement → `torch.equal` 判 no_update → candidate 展开
  float64 → reconstruct 守卫（宽捕获）→ 发布为下一状态。Adam 固定
  betas=(0.9,0.999)/eps=1e-8/weight_decay=0/amsgrad=False。
- **坐标纪律**：中点与探针一律经 `points_to_canvas`（仓库核心不变量），不做
  手写仿射预计算；numpy float64 转 float32 上设备。
- **工作流**（`main/_mbopc_workflow.py`）：`load_gradient_config`（[gradient_mbopc]
  严格类型/权重/Decimal）、`solve_gradient_macro`（tqdm try/finally + 三件
  gradient 产物）、`run_gradient_mbopc`（任意 macro 数单入口、merge 恰一次、
  可选最终光刻、summary 含 RSS 三采样与 CUDA 峰值）；`_resolve_device` 抽出
  供 simple/gradient 共用。
- **学习率换算**：`Decimal(nm)/dbu_nm` 转 float（连续步长，不走 exact_dbu）；
  epe_distance 仍走 exact_dbu 精确整数换算（4.5nm 之类非整数 DBU 在运行准备期
  被拒绝，有测试）。

## 与规格偏差（含裁决，全部在案）

| 偏差 | 理由 |
|---|---|
| 宽捕获 ValueError（裁决 1） | KLayout 几何退化以 ValueError 穿透，收窄需改 reconstruction.py（超出本 change 清单） |
| S==0/O==0 直接 no_owned_segments 不跑 forward（裁决 2） | O=0 ⇒ P=0，任何评价都是 0/0；§10.5"评价一次 baseline"在 P>0 时才有意义 |
| 常驻 normals float64[S,2]（裁决 3） | §10.2 中点位移需要法向；PERF-002 字面清单遗漏 |
| 未加 simple 的人工入口约束（macro 数/每 macro tile 数） | 规格 IF-004 接受任意 macro_count≥1，§0 禁止自行补充需求；测试验证 [1,1] 与 [2,2] 均可运行 |
| TEST-007 程序错误用例改用 RuntimeError | ValueError 与几何退化不可按类型区分（裁决 1 的连带） |

## gcd_45nm smoke 实测（CUDA，2026-08-17）

`config/gradient_mbopc.toml`（2×2 macro、870 core、iterations=1、lr=1nm）：

| macro | state0 loss | state1 loss（best） | L2 | EPE | 非零位移段 |
|---|---|---|---|---|---|
| mr0c0 | 0.167069 | 0.150141（−10.1%） | 436871→393807 | 37743→34255 | 66924 |
| mr0c1 | 0.130610（best） | — | — | — | — |
| mr1c0 | 0.134347（best） | — | — | — | — |
| mr1c1 | 0.137427 | 0.123579（−10.1%） | 328409→295002 | 27789→24497 | 49313 |

- 全部 macro `best_state=1`、`stop=iteration_limit`（一轮更新即严格改善）；
  连续 loss / 离散 L2 / EPE 同向下降；PVBand 连续分量微升（+2.9%，EPE 平衡
  点即工艺角分歧最大点的结构性后果，与 simple 轮观察一致）。
- 耗时：总计 41.61s（prepare 外每 macro ~5-7s；state0 含 target 首栅格化
  4.61s，state1 cache 命中 2.02s——缓存生效）；合并 0.29s。
- 资源：峰值 RSS 1244 MiB、CUDA 峰值 496 MiB（GTX 1650 4GB，远低于上界）。
- 产物：4×{gradient_result.npz, gradient_metrics.json, best.gds} + final.gds
  （single_cell）+ 870 tile 光刻 PNG + manifest + summary.json。
- 对照：state0 EPE 37743（mr0c0）与 simple 轮 baseline 完全一致（同一参考
  几何与评价路径）；simple 8 轮（8nm 步长）EPE 至 7263，梯度 1 轮（1nm 步长）
  至 34255——步长量级差异下的预期对比，非算法优劣结论。
- 24 GiB GPU / 64 GiB RAM 未实测；本 smoke 仅证明 4GB 显存 + ~1.2GiB RSS 可跑
  2×2 macro workload，不宣称整张 reticle。

## 清理与审计（§23）

- 调用点检查：`_resolve_device`/`_as_number` 均有两个真实调用方；未引入
  无调用方的函数；无重复实现（梯度批循环与 simple 保持独立，公共部分只有
  cache——符合 DEC-004）。
- `00_PAST/`、`layout/`、`geometry/`、`opc/input/`、`lithography/`、
  `evaluation/` 零修改（git diff 确认）。
- 异常路径无吞错：FloatingPointError 三处（loss/grad/candidate）、
  invalid_geometry 保留原错误文本、RuntimeError 传播有测试。
- 未新增 OpenCV/Hydra/CUDA/Solver 基类（REQ-017）；新增文件仅
  `_cache.py`、`gradient.py`、入口、配置、两测试文件（§14 清单内）。
- 全部新增代码中文 docstring/注释（why 导向）；`main/` 下每行中文短注释。
