# MyOPC 测试手册

## 1. 环境与直接运行

仓库不需要安装为 Python 包。当前已验证解释器：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
```

一次安装运行、测试和性能基准的直接依赖：

```powershell
& $python -m pip install -r requirements.txt
```

`requirements.txt` 与 `pyproject.toml` 使用相同版本范围；前者方便直接准备环境，
后者继续声明 Python 3.12 及以上和项目元数据。源码仍从仓库根目录直接运行，不需要
把 MyOPC 自身执行 `pip install`。运行依赖为 KLayout、NumPy、Pillow 和 PyTorch；
开发验收另外使用 pytest、pytest-cov、Ruff 和 psutil。

## 2. 前端输入验证

默认合成案例：

```powershell
& $python main\run_mbopc_frontend.py
```

真实版图：

```powershell
& $python main\run_mbopc_frontend.py TestReticle\gcd_45nm.gds `
  --layer 11/0 --grid 2 1 --probe-distance-nm 16 `
  --output-dir .benchmarks\gcd_frontend --skip-geometry-suite --json
```

该入口验证规范化、分段、唯一 owner、halo membership、固定参考位移、全局重建和 core 覆盖。输出为：

- `summary.json`：计数、耗时、内存和几何一致性；
- `segments.npz`：诊断格式版本 3、无稳定 key、按全局 segment 下标对齐；
- `overview.png`：core/owner/法向/inner/outer probe 标注；
- `reconstruction.gds`：参考与重建结果。

`--probe-distance-nm` 只控制展示探针，必须与 `--corner-nm` 相互独立。

## 3. 完整 MB-OPC

默认运行：

```powershell
& $python main\run_mbopc.py
```

整张 `gcd_45nm` 三轮复现：

```powershell
& $python main\run_mbopc.py TestReticle\gcd_45nm.gds `
  --layer 11/0 --iterations 3 --batch-size 8 --device cuda `
  --output-dir .benchmarks\gcd_mbopc --preview --json
```

完整入口只输出 `summary.json`、结果 GDS 和可选 PNG，不输出 NPZ。`--box` 可选择 DBU ROI；`--tile-size-nm` 和 `--halo-nm` 必须能由版图 DBU 精确表示且满足像素对齐。显存不足优先降低 `--batch-size`，不会改变 owner 和轮次屏障语义。

## 4. 自动测试与静态检查

```powershell
& $python -m pytest -q
& $python -m pytest tests\opc tests\lithography tests\evaluation `
  --cov=opc --cov=lithography --cov=evaluation --cov-branch `
  --cov-report=term-missing -q
& $python -m ruff check layout geometry opc lithography evaluation main tests benchmarks
& $python -m compileall -q layout geometry opc lithography evaluation main tests benchmarks
```

第一阶段 LevelSetILT 可单独执行：

```powershell
& $python -m pytest tests\opc\test_levelset_ilt.py -q
& $python main\run_ilt.py input.gds --method levelset --layer 1/0 `
  --box 0 0 2048 2048 --pixel-nm 8 --iterations 20 --device cuda `
  --output-dir output\levelset --json
```

第一阶段完成时全仓库 167 项通过（48.34 s），其中 8 项 LevelSet 专项覆盖精确 SDF、代理梯度、严格配置/输入、空工艺窗口、固定区域、曲率、零等值线、真实 Hopkins backward、GDS 入口及仓库外直接运行。入口保存 `ilt_result.npz`、`summary.json`、最终三工艺角 NPZ 和可选 PNG；时间分开记录输入、优化、评价和输出。

第二阶段 CurvMultiILT：

```powershell
& $python -m pytest tests\opc\test_curvmulti_ilt.py -q
& $python main\run_ilt.py TestReticle\simple.gds --method curvmulti `
  --box -2000 -1100 -200 948 --pixel-nm 8 --scales 4 2 1 `
  --iterations 20 --device cuda --output-dir output\curvmulti --json
```

`--iterations` 表示每个尺度的轮数，总记录数为 `len(scales) * iterations`。`--scales` 必须严格递减、以 1 结束且能整除 target 高宽；不指定 step/loss 权重时使用 CurvMulti 自身默认值。测试还必须核对 summary 的 stage 标记、完整物理光刻 shape、进程内存检查点和最终三工艺角产物。

第二阶段最终全仓回归为 180 项通过（55.06 s）；CurvMulti 专项 13 项、全部 ILT 定向 29 项通过。

第三阶段 MultilevelILT：

```powershell
& $python -m pytest tests\opc\test_multilevel_ilt.py -q
& $python main\run_ilt.py TestReticle\simple.gds --method multilevel --layer 1/0 `
  --box -2000 -1100 -200 948 --pixel-nm 8 --scales 2 1 `
  --stage-iterations 20 100 --stage-step-sizes 0.2 0.2 --device cuda `
  --output-dir output\multilevel --json
```

默认不写 `--scales/--stage-iterations/--stage-step-sizes` 时采用 `2 1`、`20 100`、`0.2 0.2`。`--iterations N` 会把所有级别设为 N 轮；自定义尺度必须同时给统一 `--iterations` 或逐级 `--stage-iterations`。测试必须确认低级监督公式、完整物理光刻 shape、不同级别记录边界、固定窗口、真实 backward、最终光刻产物及 CPU/CUDA 二值评价一致性。

第三阶段最终 Multilevel 专项 14 项、全仓回归 194 项通过。

第四阶段 DiffOPC：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest tests/opc/test_diffopc.py -q
D:\app\miniforge\envs\myopc\python.exe main/run_diffopc.py TestReticle/simple.gds --layer 1/0 --box -520 -20 -180 320 --tile-size-nm 256 --halo-nm 32 --corner-nm 8 --segment-nm 32 --max-displacement-nm 8 --pixel-nm 2 --epe-distance-nm 8 --iterations 2 --batch-size 2 --raster-chunk-size 4 --device cuda --no-preview --no-final-lithography-png --output-dir output/diffopc_phase4_cuda
```

专项 14 项必须覆盖：软栅格零位移严格一致与非饱和点中心有限差分；外轮廓/孔洞法向；左下原点像素中心 probe；batch 大小和 1-core/2-core 划分不变性；best 记录与位移快照一致；孔洞、斜边、跨 core membership；全局非法几何整轮回滚；资源配置拒绝；直接 GDS 的结果 NPZ、GDS、JSON 与最终光刻 manifest。正式入口的逐轮 `records` 同时保存连续训练损失和二值 L2/PVBand/EPE 诊断，二者不可混为同一单位。

当前 `simple.gds` 的 4-core、44-segment、60-membership 两轮 CPU/CUDA 验收中，L2 为 `773→687`、PVBand 为 `350→247`、EPE 为 `2→0`，两端最佳轮次均为 1。CUDA 峰值分配 133,264,384 bytes；首轮含 CUDA/NVRTC 初始化，不用该两轮总时长宣称加速比。

## 5. 严格性能基准

```powershell
& $python benchmarks\benchmark_mbopc_frontend.py --strict
```

严格门槛检查 5,000 个图形、110,000 个 segment：零位移 XOR、最大段长、owner 完整性、halo 稀疏性，以及紧凑常驻数组相对展开表示至少节省 60%。当前结果：

| 指标 | 本轮当前值 | 本轮修改前基线 |
|---|---:|---:|
| prepare | 122.358 ms | 117.32 ms |
| materialize | 13.097 ms | 13.48 ms |
| zero reconstruct（显式 API） | 398.755 ms | 394.95 ms |
| persistent arrays | 2.594 MiB | 2.594 MiB |
| 相对展开节省 | 67.46% | 67.46% |

基准不再测试 key lookup，因为生产迭代直接使用数组下标且已删除该路径。

补充热点等价基准：20,000 矩形 contour 峰值下降约 72%；CUDA batch=8 的 ICCAD13
三工艺角由 25.0241 ms 降到 16.4560 ms，GPU 峰值不增加，最大逐像素误差
5.82e-7；223,553 segment/870 core 的 owner 索引合成对照由 45.908 ms 降到
13.178 ms。`zero reconstruct` 仍保留为显式诊断/最终输出 API。

## 6. 图形与边界必测矩阵

每项图形逻辑至少检查：零位移 XOR 为 0、段长不超配置、法向为单位向量、每段恰有一个 owner、全局重建合法。

- 重叠矩形与角接触：重叠内边消失，角接触组件保持正确；
- 含孔图案：外环/内环法向与探针材料语义正确；
- 2 nm 中空壁、8 nm 探针：穿过窄壁的长边 probe 无效，不产生错误更新；
- 外轮廓越过内轮廓：拓扑保护拒绝 hole 逃逸并整轮回滚；
- 矩形左边越过右边：ring 绕向翻转被拒绝并整轮回滚；
- 斜边跨多 core：所有 tile 读取同一轮状态，最终不按 core 裁矢量，连接连续；
- 负坐标、多行多列 core、ROI 边界：内部半开、最大外边界闭合且无重复 owner；
- probe 冲突/越界/同像素：方向为 0，并分别记录 ambiguous 或 invalid。

图形标注图位于 `doc/images/mbopc/`。颜色说明：owner/core 使用不同颜色；黄色为外法向；青色为 inner probe；红色为 outer probe。

## 7. 同步迭代必测不变量

- 同一轮所有 batch 读取相同 `current`，`next_values` 只在轮次屏障后发布；
- 同一 segment 仅由唯一 owner 更新，halo 只读；
- target LRU 首次与命中结果一致，CPU 侧保持 `uint8 [0,255]`，设备 batch 统一归一化到 `[0,1]`；
- 二值 L2/PVBand 只累计 core ownership 像素，不重不漏，且不得改变输入张量；
- L2/PVBand 仅记录诊断，最佳 MB-OPC 状态只比较 EPE；
- batch 结束后不持有整张 reticle tensor；
- 非法候选整轮回滚，合法结果从固定全局参考边统一重建；
- 根脚本能从仓库外工作目录直接运行。

## 8. 真实版图验收基线

`gcd_45nm.gds` Layer 11/0 最新前端结果：1,776 polygons、21,590 edges、223,553 segments、870 cores、880,801 memberships；完整 problem 常驻 NumPy 数组 9,802,180 bytes，相比本次结构重构前同口径 10,688,650 bytes 减少 8.29%；prepare 233.34 ms；零位移重建、core coverage 和 overlap 校验均为 0。

历史三轮 CUDA 完整流程：870 cores、880,801 memberships；EPE `129645 -> 74592 -> 48348`；GPU 峰值分配 267,334,656 bytes；总耗时 79.834 s。历史报告中的浮点 L2/PVBand 使用旧连续平方和语义，不能与本阶段迁移后的二值像素计数直接比较；新入口会在 `records` 中明确保存整数指标，并在最终最佳几何上追加固定 512² shot 估计。

PVBand 在三轮中上升，必须在报告中原样保留，不能只报告改善指标。

## 9. 最终交付审计

交付前执行：完整 diff、`git diff --check`、受保护目录差异、删除符号调用点、AST 中文 docstring、未调用函数/重复实现/异常入口、覆盖率未命中分支和 Markdown 链接/代码围栏检查。用户 GDS、图片和无关工作树修改不得进入提交。

## 10. 离线光刻与迭代专项测试

准备一次输入：

```powershell
& $python main\offline_inputs.py raster TestReticle\simple.gds `
  output\workbench\raster_input.npz --box -2000 -1100 -200 948 `
  --pixel-nm 8 --canvas 256
& $python main\offline_inputs.py segments TestReticle\simple.gds `
  output\workbench\mbopc_input.npz --tile-size-nm 1024 --halo-nm 512
```

独立运行模型或迭代：

```powershell
& $python main\run_lithography.py `
  output\workbench\raster_input.npz --output-dir output\workbench\lithography --save-png
& $python main\run_mbopc_iteration.py `
  output\workbench\mbopc_input.npz --output-dir output\workbench\iteration `
  --iterations 3 --preview
& $python main\run_simpleilt.py `
  output\workbench\raster_input.npz --output-dir output\workbench\simpleilt `
  --iterations 20 --device cuda
```

光刻和 SimpleILT 也可以跳过输入 NPZ，直接读取版图 ROI：

```powershell
& $python main\run_lithography.py TestReticle\simple.gds `
  --box -2000 -1100 -200 948 --pixel-nm 8 --device cuda `
  --output-dir output\workbench\lithography_direct --save-png
& $python main\run_simpleilt.py TestReticle\simple.gds `
  --box -2000 -1100 -200 948 --pixel-nm 8 --iterations 20 --device cuda `
  --output-dir output\workbench\simpleilt_direct
```

若版图有多个 Layer，必须追加 `--layer LAYER/DATATYPE`；多顶层版图还要指定 `--top-cell`。直接模式仍在 Region 物化前执行相同 canvas、层级复杂度和预计内存保护，并且不会生成隐藏的输入 NPZ。

像素输入超过模型 canvas、源文件/图形/顶点/估计内存超限时，准备函数必须在公共 Region 物化前失败。读取测试还要覆盖缺字段、错误版本、越界 membership、错误参数区间和归档解压总量限制。

聚焦回归命令：

```powershell
& $python -m pytest tests\workbench\test_offline_workbench.py -q
& $python -m pytest tests\workbench\test_offline_workbench.py `
  --cov=tests.workbench --cov-branch --cov-report=term-missing -q
```

当前工作台回归包含光刻、MB-OPC 与 SimpleILT 三个真实模型成功路径、版图/NPZ 像素一致性、直接版图光刻结果一致性、直接版图 SimpleILT，以及四个 `main/` 工作台脚本的仓库外启动。详细矩阵与真实数据见 [离线工作台测试报告](offline_workbench_test_report.md)和[可微光刻/ILT 测试报告](lithography_ilt_evaluation_test_report.md)。

最终还对用户 `TestReticle/simple.gds` 执行 CPU 冒烟：独立光刻 256² 输出成功；SimpleILT 一轮 binary L2=1900；完整 MB-OPC 一轮得到 885 segments、8 cores、EPE/L2/PVBand=`338/3936/1607`、shot=325，GDS 可生成且 Region 合法。

本轮完整矩阵、性能和真实版图数据见[代码优化测试报告](code_optimization_test_report.md)。

## 11. 容量预检与资源统计测试

```powershell
& $python -m pytest tests\opc\test_preflight.py `
  tests\opc\test_artifacts_cli.py tests\opc\test_mbopc_cli.py -q
& $python benchmarks\benchmark_layout_geometry.py --runs 5 --raster-size 2048 --strict
```

必须覆盖：小版图完整扫描、低预算拒绝、百亿 segment 只估算不分配、正式求解预检不加载光刻模型、`skip-artifacts` 不生成大型文件、阶段时间非负和所有内存检查点字段完整。真实 `gcd_45nm` 还需比较预检/实际 segment，检查零位移 XOR、core 缺口和重叠均为 0。当前完整结果见[容量预检测试报告](frontend_preflight_test_report.md)。

最终光刻结果还需检查 MB-OPC 的 final_lithography/manifest.json、ownership-only tile NPZ 和 PNG，以及 SimpleILT 的 final_lithography.npz 和四张 final_*.png。使用 --no-final-lithography-png 验证 NPZ-only 路径。
