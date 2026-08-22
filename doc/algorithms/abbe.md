# Abbe 光刻成像方法

- 代码入口：`lithography/torchlitho/source.py`（频率网格/源形状/光瞳）、
  `lithography/torchlitho/model.py::TorchLithoLithography._abbe_aerial`（前向）
- 迁移来源：TorchLitho-2.0 `pylitho/sim/abbe/`（Apache 2.0，归属见
  `lithography/TORCHLITHO_LICENSE.txt`）
- 配置段：`[torchlitho]`（`method = "abbe"`）

## 1. 物理背景：光源与光瞳是两个不同的部件

光刻成像链可以写成：

```
[有效光源 S(ξ)] →以倾角 ξ 照明→ [掩模 M] →衍射谱 M(f−ξ)→ [投影光瞳 P(f)] → [晶圆 强度 I(x)]
```

- **光瞳 P(f)**：投影物镜的通光孔。频率域里是实心圆盘，半径 NA/λ
  （数值孔径除以波长），盘内透过率 1、盘外 0。它决定掩模的哪些衍射级次
  能到达晶圆。**任何照明下光瞳都是这个圆盘，从不缩成点。**
- **有效光源 S(ξ)**：照明打到掩模上的角度分布（等价写成空间频率分布 ξ，
  单位 cycles/nm）。"点源/圆盘源/多极源"说的都是它。
- **部分相干因子 σ** = 光源盘半径 / 光瞳半径（NA 归一化）。σ→0 退化为
  完全相干照明；常规部分相干照明 σ 取 0.3～0.8。

**点源（σ=0）**：一束严格沿光轴的平行平面波照射掩模。掩模各衍射分量保持
固定相位关系，经光瞳在晶圆上**干涉**成一幅相干像：

```
I(x) = | F⁻¹{ M(f) · P(f) } |²
```

**圆盘源（σ>0）**：每个源点 ξₛ 是一束倾角不同的平行光，各自产生一幅独立
的相干像（掩模谱被平移：光瞳相对谱移动到 P(f+ξₛ)）。不同源点之间**互不相
干**——不能干涉，只能强度直接相加。这就是 Abbe 方法的物理图像。

## 2. Abbe 方法：逐源点相干成像叠加

对离散源点集合 {ξₛ}（每个等权）：

```
I(x) = (1/S) Σₛ  | F⁻¹{ M(f) · P(f + ξₛ) } |²
```

关键语义：**源点平移光瞳**。第 s 个源点把光瞳圆盘的中心从原点平移到 ξₛ，
掩模谱中只有平移后盘内的频率通过。频率域几何：

```
            频率域 (cycles/nm)
        ┌─────────────────────┐
        │      ╭─────╮        │
        │     ╱ ⬤    ╲_______╱│   ⬤ = 源点集 S（在 |ξ| ≤ σ·NA/λ 内离散采样）
        │     ╲  ξₛ  ╱       │   ○ = 光瞳 P，半径 NA/λ
        │      ╰──○──╯        │   每个源点把 ○ 平移到以 ξₛ 为中心
        └─────────────────────┘
```

本实现支持四种源形状（`source.py::source_points`，同一判定产出两种形态：
Abbe 消费格点坐标 [S,2]，Hopkins 的 TCC 消费 [N,N] 0/1 掩膜——两方法由此
描述同一物理光源）：

| source_shape | 几何 | 参数 |
|---|---|---|
| `point` | 仅 DC 单点（σ→0 极限，golden 一致性锚点） | — |
| `disk` | 圆心原点、半径 σ·NA/λ 的盘内全部频率格点 | `sigma` |
| `dipole` | 两极 (±c·NA/λ, 0)，每极一个 σ 盘 | `sigma`、`pole_center` |
| `quadrupole` | 四极 (±c,±c)/√2·NA/λ（±45° 方向） | `sigma`、`pole_center` |

dipole/quadrupole 的方向固定（X 轴 / 对角线）；任意旋转角属后续增量。

## 3. 离散化实现约定

- **频率网格**：`fftshift(fftfreq(canvas, d=pixel_nm))`——居中布局，DC 在
  索引 (canvas/2, canvas/2)，x 沿列、y 沿行；单位 cycles/nm，物理像素
  pixel_nm 由 `[lithography].pixel_nm` 提供（视场 = canvas × pixel_nm）。
- **掩模谱**：`fftshift(fft2(mask))`（居中谱，与光瞳同布局）；乘平移光瞳后
  `ifftshift` 回标准布局再 `ifft2`，取模平方。
- **平移光瞳**：`|f_grid − ξₛ| < NA/λ`（严格小于，与原库一致）。离焦相位
  同样按**平移后**频率计算（光瞳内传播角，物理正确）。
- **归一**：对源点取平均 = 单位剂量。出口按 ICCAD13 胶模型语义：
  `printed = sigmoid(steepness · (I · dose² − target_density))`。
- **分块**：源点堆 [块长,B,canvas,canvas] 广播乘受 `_MAX_ELEMENTS_PER_PASS`
  （64·256²）限制，源点更多时分块累加，数值不变（内存上界注释见代码）。

## 4. R2：原库缺陷与迁移修正（重要）

原库 `getSourcePoints` 返回的不是源点**二维坐标**，而是盘内每个格点的频率
**范数标量**（一维列表）；前向循环 `freq − freq_src` 因此成为"范数矩阵减
标量"——每个源点的光瞳不是**平移** NA/λ，而是**同心放大**成半径 NA/λ+rₛ
的圆瞳：

```
原库（错误）：pupil_s = 1{ |f| < NA/λ + |ξₛ| }      ← 半径随源点范数放大
本实现（正确）：pupil_s = 1{ |f − ξₛ| < NA/λ }      ← 圆心平移到源点
```

后果：σ 小到源盘只含 DC 点时两者无差别（原库自查不出的原因）；多源点时
原库平均的是"不同截止频率的相干像"，不是部分相干成像，且**标量范数无法区
分方向，dipole/quadrupole 在原实现里根本无法表达**。

迁移裁决（用户知情）：修正为标准向量源点。一致性分层——**点源参数下与原库
逐位一致**（缺陷不显形）；盘源下与原库的差异实测并写入测试报告
（`doc/changes/completed/CHG-20260823-torchlitho/test_report.md` §5）。
回归测试：`tests/lithography/test_torchlitho.py::TestPupilShiftRegression`
（单离轴源点手写平移瞳逐点对照 + 对称双源点镜像对称）。

## 5. 离焦相位

```
P_Δz(f) = 1{|f| < NA/λ} · exp( i · (2π/λ) · Δz · (n − sqrt(n² − λ²f²)) )
```

- n 为介质折射率（`refractive_index`，默认 1.44；原库 Abbe 侧硬编码 1.44、
  TCC 侧用 NA/0.9375，NA=1.35 时两者同为 1.44，本实现统一提参、默认值不变）。
- 根式在瞳内恒正（配置校验 NA < n），瞳外被 0/1 掩膜归零；`clamp(min=0)`
  仅防御瞳外负数开方的 NaN 进入未选中分支。

## 6. 离散特性数值表（粗网格是本征特性，不是 bug）

MyOPC 生产画布 256 × pixel_nm（如 8nm → 视场 2048nm）下：

| 量 | 公式 | 数值（pixel 8nm） |
|---|---|---|
| 频率格距 | 1/(256×8) | 4.88×10⁻⁴ cycles/nm |
| 光瞳半径 | NA/λ = 1.35/193 | 6.99×10⁻³（≈14.3 格） |
| σ=0.3 源盘半径 | 0.3·NA/λ | 2.10×10⁻³（≈4.3 格） |
| σ=0.3 盘内格点数 | π·4.3² | ≈61 |
| σ=0.05 盘 | 0.05·NA/λ < 格距 | 仅 DC（退化为点源） |

源点采样 = 频率网格采样：视场越小（或 pixel 越大）格距越大、源盘内格点越
少。测试 `TestDiscretization` 用数值断言锁定这些量，防止单位约定回归。

## 7. 与 ICCAD13 资产模型的对比

| 维度 | ICCAD13Lithography | TorchLithoLithography(method=abbe) |
|---|---|---|
| 光学来源 | 固定 Hopkins 核资产（35×35×24，OpenILT 生成） | 物理参数现算（NA/λ/σ/源形状/离焦） |
| 画布 | 256 冻结（资产契约） | 256 冻结（物理视场由 pixel_nm 表达） |
| 光源/离焦语义 | 固定 focus/defocus 两个 bank | 任意源形状 + 连续 defocus_nm |
| 工艺条件 | ProcessCondition(kernel, dose) | TorchLithoCondition(defocus_nm, dose) |
| 可微性 | 原生 autograd | 原生 autograd（无手写 backward，见下） |
| 运行时资产 | 4 份 .pt kernel bank | 无（全部构造期现算） |

原库 `AbbeFunc/AbbeGradient` 手写 vjp **不迁移**：前向全部由原生可微算子
组成（乘法/FFT/ifft/abs²），autograd 自动覆盖且与前向严格一致；手写 vjp
只有 forward+vjp 无标准 backward，且旧审计（00_PAST/findings.md §258）已
指出其"只以全一上游梯度展示"的缺陷。这是 Phase 5A「不迁手写 backward」
决策的延续。
