# Hopkins 光刻成像方法（TCC 与本征核分解）

- 代码入口：`lithography/torchlitho/tcc.py`（TCC 构造 + randomized SVD +
  网格还原）、`lithography/torchlitho/model.py::TorchLithoLithography._hopkins_aerial`
  （前向）
- 迁移来源：TorchLitho-2.0 `pylitho/sim/hopkins/{tcc.py, func.py}`
  （Apache 2.0，Option 2 忠实迁移：数值链与原库逐位一致）
- 配置段：`[torchlitho]`（`method = "hopkins"`）

## 1. Hopkins 框架：把"对源点求和"打包成 TCC 二次型

部分相干成像的光强可以写成传输交叉系数（TCC）的二次型：

```
I(x) = ∫∫ TCC(f₁,f₂) · M(f₁) · M*(f₂) · e^{i2π(f₁−f₂)x} df₁ df₂

TCC(f₁,f₂) = ∫ S(ξ) · P(f₁+ξ) · P*(f₂+ξ) dξ
```

光源 S 的全部形状信息进入 TCC。直接算二重积分太贵，Hopkins 的经典做法是把
TCC 当 Hermitian 核做**本征分解（SVD）**：

```
TCC(f₁,f₂) = Σ_k  w_k · φ_k(f₁) · φ_k*(f₂)
   ⇒  I(x) = Σ_k  w_k · | F⁻¹{ M(f) · φ_k(f) } |²
```

每个本征核 φ_k 只需一次卷积。**部分相干（满秩源）下 TCC 是满秩的，需要几十
个核收敛**——这是 SVD 机器存在的意义。实测 disk σ=0.3（5 个离散源点，512nm
视场）的 TCC 恰有 5 个显著奇异值（4.23e-3, 6.90e-4, 6.82e-4, 3.87e-4,
1.96e-4），第 6 个起跌到 1e-10 量级被阈值 1e-6 滤掉：秩 = 离散源点数。

## 2. R1：点源极限下 TCC 恒为 rank-1（解析事实与测试资产）

点源 `S(ξ) = δ(ξ)` 代入 TCC 定义：

```
TCC(f₁,f₂) = P(f₁) · P*(f₂)
```

这恰是向量外积 u·u^H 的结构——矩阵第 (i,j) 元 = u_i·conj(u_j)，**秩为 1**：
唯一非零本征向量就是 pupil 自己，唯一非零奇异值 = ‖u‖。

代码层面同样成立：原库 `src/sum(src)` 对点源是单点 δ，`fft2(δ) = 全 1`，
之后 TCC 块公式 `w[i,j] = flip(roll(J,(i,j)))·h[i,j]·h*/N⁴` 中 J 恒为全 1，
整个 [N²,N²] 矩阵退化为 `(vec(h)/N²)·(vec(h)/N²)^H`。

实测印证（迁移后 `build_tcc_kernels`，64 网格/8nm/NA1.35/λ193）：恰 1 个核
通过阈值，权重 0.0090332，其余 63 个奇异值全部低于 1e-6。该事实的两个用途：

1. `TestGoldenRank1`：点源 SVD 输出必须满足"仅 1 核"且 Hopkins aerial 与
   Abbe DC 源 aerial 一致——**不依赖原库**的独立正确性校验；
2. 解释原库 README 的 "Abbe and Hopkins are well-aligned"：默认 σ=0.05 下
   Abbe 的源盘只含 DC 点，两方法物理上就是同一个完全相干模型。

## 3. genTCC 实现剖析（忠实迁移）

### 3.1 TCC 大矩阵构造（`compute_tcc`）

```
h  = fftshift(fft2(pupil))            # 光瞳谱（float64/complex128，对齐原库）
J  = fftshift(fft2(src / Σsrc))       # 归一源密度的谱
w[i,j] = flip(roll(J,(i,j))) · h[i,j] · h* / N⁴     # 每 [N,N] 块
W = w.reshape(N², N²)                 # 大矩阵按 complex64 存储
```

N=64 时 W 是 4096×4096 complex64 ≈ 128 MiB（构造后释放；每 defocus 值一份，
三命名条件最多两份）。双循环保持原库结构（约 0.1 秒，仅构造期执行）。
**实测与原库逐位一致**：disk 源权重 maxrel = 0.0、phis[0] maxdiff = 0.0。

### 3.2 randomized SVD（`randomized_svd`）

标准随机范围搜索算法，支持复数（scikit-learn 拒绝复数输入）：随机采样
(n_components+16 列) → 4 轮 QR 幂迭代 → 投影后精确 SVD。seed 固定 0，结果
确定。保留 64 分量、阈值 1e-6 过滤。

### 3.3 网格缩放与还原（`build_tcc_kernels`）

TCC 恒在 ≤64 的网格上构造（原库 MAX_TCC_SIZE，限制大矩阵内存）。缩放规则
与原库一致：仿真网格超过 64 时优先把 TCC 像素翻倍（padding 因子），视场超
过 2048nm 才把 TCC 画布减半（resize 因子）：

```
MyOPC 生产例：canvas 256 × 8nm（视场 2048nm）→ n=64、padding=4、resize=1
大视场例：  canvas 256 × 16nm（视场 4096nm）→ n=64、padding=2、resize=2
```

低分辨核还原到仿真网格的频域流程（与原库逐步同构）：

```
φ_low [n,n] → fftshift(fft2(fftshift(φ_low))) → 居中零嵌入 [n·padding]²
  → （padded ≠ size 时双线性插值到 size）→ fftshift(ifft2(fftshift(·)))
  → × padding²·resize²
```

**cv2 → torch 同构替换**：原库频域插值用 `cv2.resize(INTER_LINEAR)`，本实现
用 `F.interpolate(bilinear)`（MyOPC 环境不引入 opencv）。关键数值事实：
2048nm 视场的 padding 路径 padded == size，插值恒等跳过，与原库 cv2 恒等
resize **逐位一致**；只有 >2048nm 视场才发生真插值，torch 双线性与 cv2
双线性的核差异在测试报告中量化（实测 rel_L2 < 2%）。

## 4. 本征核前向（`_hopkins_aerial`）

```
M  = fft2(mask)                        # 标准谱（无 fftshift，与 Abbe 分支不同）
φ_kFFT = fft2(φ_k)
conved = ifft2(M · φ_kFFT)             # 循环卷积（ifft 内含 1/N²）
conved = fftshift(conved) / (H·W)      # 居中 + 归一
I = Σ_k  w_k · |conved_k|²
```

一处**有意修正**：原库归一用 `np.prod(conved.shape)`，批维 B>1 时会除以
B·H·W（幅度随批大小漂移，原库用例恒 B=1 故未暴露）；本实现按 H·W 归一，
B=1 时与原库逐位相同。记录于测试报告 §6。

phi 与 mask 恒同网格（TCC 输出已还原到仿真网格），原库前向的 phi 双线性
插值分支在此实现中为恒等、不迁移（死分支，AGENTS 禁投机保留）。

## 5. 非点源下的原库行为（忠实保留，如实记录）

原库 TCC 块公式里源以**谱 J(f₁+f₂)** 进入（flip+roll 实现的索引组合），而
非教科书"对离散源点求和"形式 `Σ_ξ S(ξ)P(f₁+ξ)P*(f₂+ξ)`。点源时 J≡1 两者
无差别；**非点源时两者不同**——实测 disk σ=0.3（5 源点）下原库 Hopkins 输出
比"逐源点平均"（修正版 Abbe）小约 1/S² 倍（幅度差）且形状有微差。

迁移裁决（用户知情，Option 2 忠实）：**保留原库行为**，与原库逐位一致优先；
差异的量化数据在测试报告 §5，物理正确的部分相干路径用 `method = "abbe"`
（R2 修正版逐源点求和）或后续增量（教科书式 TCC 离散化）。文档读者注意：
**非点源下不要把两方法输出互当回归基准**。

## 6. 两方法的对齐条件与 golden 证据

| 源形状 | Abbe ↔ Hopkins | 证据 |
|---|---|---|
| point（含 σ 小到盘只含 DC） | **严格一致**（rank-1 退化 = 相干成像） | golden 8 图案互证 + `TestGoldenRank1` |
| disk/dipole/quadrupole | 不对齐（原库 TCC 的 J(f₁+f₂) 行为，§5） | 测试报告 §5 差异量化 |

一致性总证（`tests/lithography/test_torchlitho_golden.py`，49 用例）：
point 源 8 图案 × 2 方法 × 2 离焦对原库 golden：abbe rtol 1e-6 逐位、
hopkins rtol 1e-5（float32 舍入包络，实测 ≤4.8e-7）；disk 源 hopkins 与原库
TCC 数值链逐位；resize 两分支如 §3.3。

## 7. 后续增量（当前不实现，均有真实场景再启）

- 教科书式 TCC 离散化（`Σ_ξ S(ξ)P(f₁+ξ)P*(f₂+ξ)`）修复非点源幅度语义；
- 任意旋转角的 dipole/quadrupole 与 annular 环形源；
- TCC 构造耗时优化（双循环向量化，数值不变的前提下）。
