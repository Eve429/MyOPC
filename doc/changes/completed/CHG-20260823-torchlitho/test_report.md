# TorchLitho 迁移一致性测试报告（点名交付物）

- 日期：2026-08-23；执行：Claude（会话迁移实施）；对照基线：外部原库
  TorchLitho-2.0（`D:\00_WorkSpace\02_CodeStorage\01_OPC\TorchLitho-2.0`，
  工作树只读参照，未做任何修改）
- 目的：**证明迁移后的 `lithography/torchlitho/` 与迁移前（原库）的结果一致**，
  多图案覆盖，并如实标出每一处「不一致」的性质与量级。

## 1. 结论（先读这里）

| 对照项 | 覆盖 | 判定 | 实测差异 |
|---|---|---|---|
| Abbe、点源、focus/defocus | 8 图案 × 2 离焦 | **逐位一致**（rtol 1e-6） | 部分图案 maxdiff = 0，其余 ≤ 6e-8 |
| Hopkins、点源、focus/defocus | 8 图案 × 2 离焦 | **float32 舍入级一致**（rtol 1e-5） | 相对差 ≤ 4.8e-7 |
| Hopkins、disk σ=0.3（TCC 数值链） | 4 图案 | **逐位一致**（rtol 1e-5） | 权重 maxrel = 0.0、phis maxdiff = 0.0 |
| Hopkins resize 2048nm 视场（padding=4 纯零嵌入） | 3 图案 ×256 网格 | **float32 舍入级一致** | 相对差 ≤ 5.1e-7 |
| Hopkins resize 4096nm 视场（resize=2 真插值，torch 替代 cv2） | 3 图案 ×256 网格 | **float32 舍入级一致**（超出预期） | 相对差 ≤ 5.1e-7，rel_L2 = 0.0000 |
| Abbe、disk σ=0.3（R2 修正 vs 原库缺陷版） | 4 图案 | **有意的已知差异**（§5） | rel_L2 = 0.000 ~ 0.371 |

一句话：**除 R2 修正（用户知情批准）外，所有对照全部达到 float32 舍入级或
逐位一致；不存在任何未解释的数值偏差。**

## 2. 测试环境与命令

- 解释器：`D:/app/miniforge/envs/myopc/python.exe`（Windows 10，torch 2.5.1+cu124，
  numpy 2.5.1；golden 生成另需 opencv-python 5.0.0——仅原库 cv2 resize 分支
  对照用，**迁移后代码零 opencv 依赖**）
- 一致性测试：`python -m pytest tests/lithography/test_torchlitho_golden.py -q`
  → **49 passed**（含批 A 36 例的 `test_torchlitho.py` 与 iccad13 回归共
  170 passed）
- 全量：`python -m pytest -q tests` → 786 passed + 1 skipped + 10 failed
  （10 个 failed 全部位于 tests/main/，为并行工作现场既有失败，集合与批 A
  实施前完全相同，非本迁移引入）

## 3. golden 数据生成（可复现）

一次性脚本 `tests/lithography/golden/_generate_golden.py`（**不提交**，依赖
外部路径；本文附录 A 有全文）。生成命令：

```bash
D:/app/miniforge/envs/myopc/python.exe tests/lithography/golden/_generate_golden.py
```

脚本把外部原库 `sys.path` 前插后直接调其内部类（**输出取上采样 interpolate
之前的仿真网格 aerial，无 API 切入障碍**）：

- Abbe：`pylitho.sim.abbe.simulate.AbbeSim(pixel=8, sigma, NA=1.35,
  wavelength=193, defocus∈{0,40})(mask)`
- Hopkins：`pylitho.sim.hopkins.tcc.genTCC(...)` + `HopkinsFunc(tcc)(mask)`
  （point：genTCC(defocus=[0,40])；disk：`TCC(src_disk, funcPupil(...))` 原
  库数值链；resize：genTCC(8,2048) 与 genTCC(16,4096)，后者走原库 cv2 resize
  分支）
- 产物三份（提交于 `tests/lithography/golden/`，共 3.2 MiB，含 mask 本体与
  参数 meta，测试自含不依赖外部库）：`golden_point.pt`、`golden_disk.pt`、
  `golden_resize.pt`

## 4. 图案集（8 种，64×64，固定定义共享于脚本与测试）

| # | 图案 | 构造 | 考察点 |
|---|---|---|---|
| 1 | single_rect | 32×32 居中矩形（与原库 example/rect.py 同款） | 基础孤立图形 |
| 2 | dense_lines | 周期 8、线宽 4 的竖线阵 | 光栅/高频周期 |
| 3 | double_lines | 两条全宽横线（间隔 28） | 端头邻近效应 |
| 4 | l_shape | 竖条 + 底横条 | 内角几何 |
| 5 | isolated_hole | 16×16 单透光孔 | 暗背景孤孔 |
| 6 | bridge | 两竖块 + 顶横连 | 桥形/多端头 |
| 7 | random_blobs | 8×8 随机二值上采样 ×8（seed 20260823） | 非规则簇 |
| 8 | checkerboard | 8×8 棋盘 | 最高频混叠边界 |

resize 分支抽 3 种（single_rect / bridge / checkerboard）上采样到 256 网格。

## 5. R2 修正差异量化（唯一的"有意不一致"）

原库 Abbe 源点为范数标量 → 瞳同心放大（详见 `doc/algorithms/abbe.md` §4）。
迁移按用户批准修正为向量源点（瞳平移）。disk σ=0.3（512nm 视场，5 个离散
源点）下修正版与原库输出差异（rel_L2 = ‖ΔI‖/‖I‖）：

| 图案 | rel_L2 | 说明 |
|---|---|---|
| single_rect | 0.1341 | 孤立图形对瞳平移最敏感 |
| dense_lines | **0.0000** | 周期图案频谱落在对称采样点上，两种实现数值巧合相同 |
| l_shape | 0.2975 | |
| random_blobs | 0.3706 | 宽谱图案差异最大 |

上界断言 0.6（防回归恶化），实测最大 0.371。**方向性说明**：修正版
（瞳平移 + 对源点平均）是教科书部分相干成像；原库版是"不同截止半径相干像
的平均"。差异随 σ 增大而增大、σ→0（点源）时严格归零——点源 golden 逐位
一致正是该性质的直接证据。

## 6. 记录在案的两处原库行为差异（忠实迁移下的有意决策）

1. **Hopkins 前向 B>1 归一修正**：原库 `fftshift(conved)/np.prod(conved.shape)`
   在批维 B>1 时除以 B·H·W（幅度随批大小漂移；原库自家用例恒 B=1 未暴露）。
   迁移按 H·W 归一；B=1 时与原库逐位相同（golden 全部 B=1）。
2. **非点源 Hopkins 的谱进入形式**：原库 TCC 块公式里源以谱 J(f₁+f₂) 进入
   （`doc/algorithms/hopkins.md` §5），disk 源下幅度比逐源点平均小 ≈1/S² 且
   形状有微差。Option 2 忠实保留（与原库逐位一致优先），物理正确的部分
   相干路径用 `method="abbe"`；教科书式 TCC 离散化列为后续增量。

## 7. 独立正确性证据（不依赖原库，双保险）

- **rank-1 解析事实（R1）**：点源 TCC 恒为外积 rank-1（`hopkins.md` §2 有
  证明）。迁移 SVD 实测恰 1 核（权重 0.0090332）、其余 63 个奇异值 < 1e-6
  阈值；disk 5 源点恰 5 核（秩 = 离散源点数）——SVD 机器行为与理论吻合。
- **两方法互证**：点源下 8 图案 abbe ↔ hopkins aerial 一致（rtol 1e-4；
  梯度互证 rtol 1e-4）——两条完全独立的数值链（逐源点 FFT 卷积 vs
  TCC+SVD+本征核卷积）算出同一结果。
- **解析式对照**：点源无离焦 abbe 与测试内独立手写
  `|ifft2(ifftshift(fftshift(fft2(mask))·pupil))|²` 逐点一致（含 defocus 40nm
  变体）；autograd 梯度与手写链梯度一致。
- **padding 逐位**：与 `ICCAD13._prepare_mask` 在 4 组奇偶尺寸 × 3 种填充
  下逐位一致（同一几何同一 canvas 布局不变量）。
- **单位约定锁定**：`TestDiscretization` 断言 256/8nm 下格距 1/2048、瞳半径
  ≈14.3 格、σ=0.3 盘 ≈4.3 格（cycles/nm 约定回归即红）。

## 8. 管线级冒烟

`python main/run_ilt_simple.py config/torchlitho_abbe.toml`（CUDA）：三命名
条件管线贯通，1 macro / 16 tile / 16 张 PNG 留档 / 最终 GDS 合并，1.19s，
CUDA 峰值 54 MiB。iccad13 默认路径：现有 workflow/runner 全量测试为回归网，
批 C 前后行为不变（786 passed 集合一致）。

## 附录 A：golden 生成脚本全文

见 `tests/lithography/golden/_generate_golden.py`（工作树保留，未提交；如
被清理，参数与调用方式以本文 §3 为准重写即可，图案定义唯一来源是
`tests/lithography/test_torchlitho_golden.py::_patterns`）。
