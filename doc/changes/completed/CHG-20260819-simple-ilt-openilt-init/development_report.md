# 开发报告 — CHG-20260819-simple-ilt-openilt-init

## 实施概览

单实施批 `ebce389`（基线 14ab3e8）：`opc/iteration/ilt/simple.py` 初始化块
替换为 `params = 2·T − 1`；测试六处同步 + 新增 P1-1 回归；smoke 步长重调。

## 数值行为变化（有意）

| 项 | logit+eps（旧） | 2T−1（新） |
|---|---|---|
| 0/1 像素 sigmoid 斜率（β=4） | ≈4.8e-7（饱和） | ≈0.0707（≈1.5×10⁵ 倍） |
| state0 soft | = T（1e-6 内） | σ(β(2T−1)) ∈ {0.018, 0.982, …} |
| state0 二值掩膜（thr 0.5） | = T≥0.5 | = T≥0.5（不变，σ(β(2T−1))≥0.5 ⟺ T≥0.5） |
| 纯对齐几何一轮更新 | 不可见（<1e-6） | 可见（实测 max\|Δp\|≈3.9e-4） |

## 与规格/前 change 的偏差记录

1. **REQ-C 阈值定 1e-5 而非计划草拟的 1e-3**：真实光刻链路 dprinted/dmask
   低于线性 stub 估计，实测位移 3.9e-4；1e-5 同时远离饱和区（<1e-6）与
   实测值一个量级，判别性充分。
2. **smoke step_size 10 → 1.0**：梯度尺度随初始化增大 ~10⁵，旧步长严重
   过冲（state1 损失反升至 43400）。扫参 1.0/0.5/0.1 后取 1.0（改善最大）。
   教训入 findings：参数化方案与步长尺度耦合。
3. **测试修正一处自误**：常数/逐调用 stub 模型的损失监督是 T（printed 与
   mask 无关），期望值与初始化无关——曾误改为 soft₀ 监督，纠正并记录。
4. OpenILT 差异保留点：context 固定区仍直通 target（非 OpenILT 的
   fixed_soft 参数化），维持 CHG-20260818 的保真差异。

## 新 smoke 基线（corners_unit，16 core / 225,625 像素 / iterations=1 / step 1.0）

records 7952.12 → 6233.46（state1 较 state0 **−21.6%**，best_state=1）；
binary L2 = 2893（旧基线 2896）；总 0.99s；RSS 936 MiB；CUDA peak 503 MiB。

## 清理与审计

- 旧初始化的 eps/clamp/1e-6 守卫整块删除，无残留符号；`_EPS32` 测试常量
  随镜像改写移除；ruff/compileall 全绿。
- 未修改 00_PAST/layout/geometry/用户数据；Rev 1.0 时未回改已完成 CHG
  文档，Rev 1.1 起按用户指令为其过期初始化描述加【已取代】标注
  （历史正文保留，见下节第 3 条）。

## Rev 1.1 追加（2026-08-19，用户 P1/P2 审查 + P1-3）

1. **context 统一（4c5f5f1）**：P1-1 切换后 trainable 用 σ(β(2T−1)) 而
   固定 context 直通 raw T——macro seam 出现 ~1.8% 人为 transmission 跳变，
   且 A 宏看到的邻居不是 B 的真实 state0。修复：solver 与终评画布的固定
   context 统一为 σ(β(2T−1))（无梯度；监督/指标目标仍 raw T），与
   OpenILT fixed 区 sigmoid(β·backup) 同构。新增跨宏 seam 测试：A context
   中属于 B 的像素与 B 自身 state0 逐位相等且值为 σ(β(2T−1))。
   ILTMethod 鸭子契约补充 sigmoid_steepness（终评 context 定义）。
2. **P1-3 性能修复（aa583a5）**：trainable_index_canvas 每次调用分配
   O(宏像素) 全宏 arange 索引表（每 state × 每 core 热路径）；改为行基址
   + 窗口 arange 外积，只分配 core 工作集。数值不变，既有像素一致性
   用例为回归。
3. **文档同步**：CHG-20260818 规格与测试报告的初始化描述加【已取代】
   标注（历史正文保留）；本 change 规格补 REQ-D。
4. **新基线**：全量 527 passed；smoke 7880.69→6162.49（−21.8%）、
   binaryL2 2875、binaryPVBand 883——context 统一消除 seam 光学不一致后
   终评指标较 Rev 1.0（2893）进一步改善。
