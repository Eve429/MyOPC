# 开发报告 — CHG-20260819-gradient-mbopc-epe-loss

## 实施概览

三阶段（基线 08f4866，规格 Rev 0.2→1.0 用户批准）：

| 阶段 | Commit | 内容 |
|---|---|---|
| A | 3689a5f | gradient.py：GradientMBOPCConfig/IterationRecord 尾部新增 weight_epe/epe_steepness/epe_loss；`_profile_d_s` 向量化双线性 sampler；profile 静态预计算（含画布闭区间校验、owner 唯一、L_sum 分母）；四项 loss 同一次 backward；单元测试 |
| B | 08a057d | configuration.py 同名字段与四权重校验；workflow summary/metrics 增量字段；CLI 四权重打印；示例 TOML 显式启用；runner/config 测试 |
| C | 本批 | 全量验证 + 对照 smoke + 迭代算法自审 + 手册/契约/报告/记录 |

## 与规格的一致性

无偏差。要点核对：profile 坐标每 macro 预计算一次（REQ-007/PERF-001）；
`D` 复用 nominal 平方误差张量（PERF-002）；owner core 唯一计分、membership
梯度仍经 midpoint STE 累加（INV-001/002，TEST-003/004）；sum 聚合 + 长度
加权两条 Rev 0.2 契约由 TEST-013/014 判别锁定；关闭路径逐值兼容
（TEST-006）；result NPZ 不改版。

## 对照 smoke（PERF-004 / AC-005 / AC-008 / TEST-012）

gcd_30um、单 macro 870 core、11 状态、CUDA；同配置仅 `weight_epe`
1.0 vs 0.0（临时对照 config，已删）：

| 指标（state10 对比） | EPE ON | EPE OFF | 增量 |
|---|---|---|---|
| nominal_l2 | 0.047267 | 0.051858 | **−8.9%** |
| process_l2 | 0.100158 | 0.108536 | −7.7% |
| 离散 l2 | 981505 | 1089227 | **−9.8%** |
| 离散 epe | 89526 | 94719 | **−5.5%** |
| total 耗时 | 183.2s | 177.1s | +3.5% |
| 峰值 RSS | 1574 MiB | 1552 MiB | +1.4% |
| CUDA 峰值 | 502 MiB | 502 MiB | 0 |

两组 state0 逐值相同（nominal 0.073487 / l2 1506516 / epe 128137）——
同状态评价路径不受 EPE 开关扰动。`forward_many` 次数两组均为
11×⌈870/8⌉=1199（构造相同 + TEST-011 spy 锁定实现不增量）。

## 迭代算法自审（按 Rev 1.0 用户要求）

1. **饱和性（规格 §24 待验证项）**：ON 组 state0 `epe_loss=0.8318` 反推
   平均 `d_s≈0.60` pixel（σ(γd)=0.916 → γd≈2.38），σ′≈0.077——处于活跃
   窗口但已衰减 ~13×；10 个状态 d_s 降至 ≈0.43（γd≈1.74）。**γ=4/R=2
   在该 workload 无深饱和**；规格"EPE 是 L2 拉边后的局部精修 loss"的
   定位得到数据支持（d_s 始终 < 1 pixel）。
2. **连续/离散 EPE 相关性**：ON 组 11 个状态连续 epe_loss（0.832→0.702）
   与离散 epe（128137→89526）均严格单调同降。按 §2.4 边界，只记录本
   workload 事实，不宣称普适单调。
3. **权重占比（非阻塞问题的数据回答）**：`weight_epe=1.0` 时 EPE 项占
   state0 总目标 85%（0.832/0.982）——梯度方向由 EPE 主导，但本 workload
   同时改善全部 L2 指标；示例值可用但偏激进，实际使用建议按 workload
   降权（如 0.1–0.5 量级起步），不在本 change 内自动调参。
4. **资源增量与上界一致**：RSS +22 MiB ≈ CPU profile 数据
   （229127 段 × Q=4 × 2 坐标 × 8B ≈ 14.7 MB + 段长数组），符合
   PERF-003 的 O(O·Q) 上界；CUDA 峰值无增量。

## 清理与审计

- ruff（含 F401/F841 未用检查）全绿；无 EPE 专用重复实现（sampler 单点
  `_profile_d_s`，graph/no-graph 两调用方）；无吞异常（越界 ValueError、
  非有限 FloatingPointError 均带上下文上抛）。
- 未修改 Protected Areas（layout/geometry/00_PAST/opc.input/lithography/
  evaluation）；`git diff --check` 干净；临时对照 config 已删除。
- 文档：contracts/mbopc.md Gradient 节、development_manual §8 增补、
  test_manual 套件行、本报告 + test_report、规格移 completed（Rev 1.1）。
