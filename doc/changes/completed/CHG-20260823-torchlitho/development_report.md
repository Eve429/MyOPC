# CHG-20260823-torchlitho 开发报告

## 1. 实施过程（四批，commit 链）

| 批 | commit | 内容 |
|---|---|---|
| A 模型核心 | 6e1f779 | `lithography/torchlitho/` 四文件 + LICENSE + 36 单元测试 |
| B golden 一致性 | 82f5789 | 8 图案 golden 三份 + 49 用例两级判定 |
| C 配置接线 | 9065555 | `[lithography].model` 分派 + 工厂 + 三入口 + 2 示例 config + 5 用例 |
| D 文档报告 | （本批） | abbe.md / hopkins.md / CHG 三件套 / 手册更新 / 记录三文件 |

## 2. 设计期关键发现与决策（R1–R5）

- **R1（数学事实）**：点源下 TCC 恒为外积 rank-1（源谱 J≡全 1 ⇒ w[i,j] =
  x_i·conj(x_j)）。曾据此提出 Hopkins 闭式实现（40 行替代 SVD 机器），用户
  裁决选 Option 2 忠实迁移（为多极源保留 SVD 的真实语义）；R1 转为文档证明
  与 `TestGoldenRank1` 独立校验资产。
- **R2（原库缺陷）**：`getSourcePoints` 返回频率范数标量 → 瞳同心放大而非
  平移。设计期会话内向用户报告并获修正批准（向量源点）；dipole/quadrupole
  需求使修正成为必要（标量无法表达方向）。
- **R3（源形状参数化）**：四种形状同一判定函数产出两形态（Abbe 格点坐标 /
  Hopkins 掩膜），两方法描述同一物理光源；dipole 沿 X 轴、quadrupole ±45°
  方向固定（旋转角后续增量）。
- **R4（配置组织）**：`LithographyConfig` 契约"算法无关"，物理参数放独立
  `[torchlitho]` 段（全默认，现有 config 零改动）。
- **R5（golden 位置）**：`tests/lithography/golden/`（纯测试对照数据），
  非运行时资产目录——TorchLitho 零运行时资产本身是卖点。

## 3. 实施中的偏差与修正（相对实施前计划）

1. **dtype 链对齐**：计划未预见原库 TCC 链是 numpy float64/complex128 而
   模型前向是 torch float32。实施改为：`frequency_grid/pupil_function` 支持
   dtype 跟随（f32→c64 前向、f64→c128 TCC 链），TCC 大矩阵按原库 complex64
   存储。此修正直接促成「权重与原库 maxrel=0.0」的逐位结果。
2. **hopkins 判定容差**：原计划 rtol 1e-6；实测原库前向多一步恒等
   interpolate 且求和顺序不同，6/32 用例落在 4.8e-7（float32 ulp 级）——
   判定调至 1e-5 并在 docstring/报告注明性质。这不是退让：rank-1 互证与
   TCC 逐位两组独立证据封闭了数值正确性。
3. **cv2→torch 差异远小于预期**：原计划给 4096nm 真插值路径留 5% 容差；
   实测 torch 与 cv2 双线性在半像素对齐下等价（差 ≤5.1e-7），容差断言保留
   0.05 作防回归上界。
4. **phi 前向插值分支不迁**：单网格设计下 phi 恒同网格，原库该分支为死代码
   （AGENTS 禁投机保留）；计划原文"保留插值机制"按此修正。

## 4. 简化审计（AGENTS 交付前清单）

- 未调用函数/重复实现：无（`_pole_union_mask` 仅 source_points 内部两调用
  方共用；padding 复制而非抽公共层是有意决策——避免动 iccad13.py）。
- 异常入口：构造期校验全集（config post_init / 模型构造 / condition 未知
  名 / forward_many 空条件、类型、重名 / TCC 缩放链对不齐 / 空源点），
  全部 ValueError/TypeError 带字段名，无吞错分支。
- 接口面：公共 3 名（Config/Condition/Lithography）+ 工厂 1 名；内部
  `_abbe_aerial/_hopkins_aerial/_prepare_mask/_tcc_for` 均有测试或生产调用方。
- 覆盖未命中分支：`TestSigmoidExit` monkeypatch 隔离出口；`TestForwardManyBatching`
  计数证明 TCC 每 defocus 恰一次；CUDA 路径由既有 workflow 测试覆盖（模型
  device 语义与 ICCAD13 同构）。

## 5. 性能记录（CPU，i7/64GB 便携级）

- TCC 构造（n=64，一次）：约 1.4 s（双循环 0.1 s + randomized SVD 与谱
  还原）；每 defocus 一份，惰性缓存。256 网格 hopkins 首次前向含构造
  共 1.44 s，之后每张 <0.1 s。
- Abbe 前向（256 网格点源）：向量化后 <0.1 s/张；σ=0.3（61 源点）分块
  累加仍 <0.3 s。
- ILT 冒烟全管线（含几何准备/求解/合并/PNG）：1.19 s（CUDA）。
- 内存：TCC 大矩阵峰值 128 MiB（构造后释放）；前向张量受
  `_MAX_ELEMENTS_PER_PASS` 上界约束。

## 6. 记录三文件与文档更新清单

task_plan.md（本节）、findings.md（§TorchLitho 迁移批次事实）、progress.md
（会话日志）；doc/algorithms/{abbe,hopkins}.md 新增 + index.md 登记；
doc/contracts/lithography.md 增补；doc/development_manual.md §5、
doc/test_manual.md 更新；requirements.txt 注释 opencv 仅 golden 再生成用。
