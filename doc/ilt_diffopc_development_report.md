# ILT 与 DiffOPC 迁移开发报告

本阶段新增 LevelSetILT、统一多尺度 ILT 调度和独立 DiffOPC 梯度边段求解器；现有 SimpleILT 与 Simple MB-OPC 保持不变。

- LevelSet 使用 `phi < 0` 硬二值前向和 `|grad(phi)|` 代理反向。
- CurvMulti 使用粗到细连续控制网格、平滑 sigmoid 参数化和统一 ICCAD13 光刻模型。
- DiffOPC 使用独立解析软边段栅格器；精确 KLayout raster 只保留在非梯度路径。
- 不复制 OpenILT/DiffOPC 的代码、资产、Hydra、数据集或日志框架。

当前 DiffOPC 首版完成 L2/PVBand 和固定 inner/outer probe 的连续 EPE hinge 梯度路径；MRC、SRAF 和多 GPU 仍属后续阶段。

## 第一阶段质量修正

LevelSetILT 默认初值已从二值正负常数改为一次性精确欧氏 SDF；配置、输入有限值、优化窗口和工艺条件执行严格校验，显式空工艺窗口不再回退默认条件。曲率项统一复用零和离散曲率核，公共 `soft_mask` 由最优 phi 生成连续诊断图，硬结果严格复用前向的 `phi < 0`。统一入口现已支持 GDS/OASIS 的 Layer、ROI、安全预检参数，并保存评价、分阶段时间、GPU 峰值和最终三工艺角 NPZ/PNG。

本阶段没有增加基类、注册器、contracts 或工具文件：`LevelSetConfig`、SDF、代理梯度和求解器同置一个实现文件，并复用 SimpleILT 的结果记录与曲率核。精确 SDF 使用 `O(HW)` 两遍距离变换，只在优化前运行一次；迭代主要常驻参数、固定初值、窗口和 Adam 状态，不引入边段或 KLayout 热循环。`layout/`、`geometry/` 均未修改。

当前光刻模型 canvas 为 256；同机纯 CPU 的一次性 SDF 实测 256² 为 1.459 s、结果 0.25 MiB，进程 RSS 增量约 3.38 MiB。512²/1024² 探查分别为 5.825/23.602 s，说明纯 Python 下包络适合当前固定 canvas，但未来若光刻 canvas 大幅提升，应在不改变接口的前提下换成编译型 EDT；本阶段不为尚未支持的大画布引入 SciPy/自定义扩展依赖。

## 第二阶段 CurvMultiILT

首版 `multiscale.py` 实际只是逐尺度调用 LevelSetILT，既没有 CurvMulti 的平均池化 sigmoid 参数化，也把粗图直接交给固定像素网格光刻模型；本阶段删除该错误原型和旧公共符号，不保留兼容包装，改为同目录单文件 `curvmulti.py`。该文件只包含一个配置、两个内部张量操作和一个求解函数；配置/记录/结果继续复用现有 ILT 契约，没有增加基类、注册器或 contracts。

算法以完整 raster 为固定监督，每个 scale 只构造较小的连续控制参数。参数经过奇数均值核和带 offset sigmoid 后，先近邻恢复到完整 target shape，再调用 `forward_many`；因此所有尺度共享相同 Hopkins 核物理坐标。每阶段使用 SGD，保存该阶段复合损失最优参数作为下一阶段 nearest warm-start，随即释放优化器和计算图。常驻设备内存上界由完整 mask/wafer 光刻图和当前尺度参数决定，不累计历史尺度。

相对 OpenILT 做了三项有依据的修正：nominal L2 使用具名 nominal 输出，不误用 `printedMax`；优化窗口外保持初始软 mask，不乘 filter 清零；曲率继续作用于 nominal wafer。`run_ilt.py` 增加尺度、平滑核、sigmoid 和阈值参数，并为每条公共记录附加可推导的 stage index/scale/iteration；没有另建重复记录结构。入口还记录输入、模型、优化、评价、输出的进程 RSS/USS/private/peak working set 和 CUDA 峰值。

真实 `TestReticle/simple.gds` CPU 三尺度一轮验证：完整 256² 网格，三阶段连续损失 `2131.974 -> 1691.847 -> 1602.002`；最终二值 L2/PVBand/shot=`2146/800/118`，优化 1.940 s、总计 2.176 s，峰值 working set 637,444,096 bytes。该数据用于功能与资源基线，不宣称一轮参数已达到收敛最优。

同一输入在当前 4 GiB GTX 1650 上完成真实 CUDA backward：三阶段损失与 CPU 仅有浮点尾差，最终二值指标完全一致；CUDA 峰值分配 69,222,400 bytes，优化 2.102 s、总计 2.580 s。首轮含 CUDA/NVRTC 初始化，因此此单轮时延不用于 CPU/GPU加速比结论；它只证明本阶段在显存远小于目标 24 GiB 的设备上可运行。

最终简化审计确认：旧 `MultiScaleILTConfig/optimize_multiscale`、`multilevel` CLI 别名和 `multiscale.py` 均无残留；新配置、两个私有张量操作和求解入口全部有当前调用方。函数体重复扫描无命中，阶段元数据不形成第二套结构，未增加仅为修 bug 存在的分支或包装层；`layout/`、`geometry/` 零差异。

## 第三阶段 MultilevelILT

OpenILT Multilevel 的算法身份是两个独立 CurvILT 求解：Low 级 256²/20 轮完成后，把该级最优参数近邻放大两倍，Mid 级 512²/100 轮以新的 Adam 精修。项目版保留“每级独立迭代数、实际 Adam 步长、最优参数 warm-start、逐级重建优化器”，默认 `scales=(2,1)`、`stage_iterations=(20,100)`、`stage_step_sizes=(0.2,0.2)`；也允许调用方配置更多级别。

为保持当前 ICCAD13 核的物理标定，每一级只降低参数和监督网格：级别 soft mask 先近邻恢复为完整 target shape，再执行 `forward_many`；完整 wafer 随后以 area 汇聚到本级监督网格计算损失。它与 CurvMulti 的差异是：Multilevel 使用每级独立 Adam/轮数并在级别网格监督，CurvMulti 使用统一 SGD/轮数且始终在完整网格监督。两者只共享 `simple.py` 中 `[B,H,W]` 缩放和“均值平滑+sigmoid”两个纯张量操作，不增加 utils、基类、注册器或阶段记录类。

相对参考代码继续修正三处历史问题：nominal L2 使用具名 nominal wafer，不误用 maximum；优化窗口外保持初始软 mask，不清零；曲率核只在权重大于零时作用于本级 nominal wafer。每级结束只保存最优参数供下一级 warm-start，旧 Adam、wafer 和 autograd 图不累计，设备内存上界仍由完整光刻图、当前级参数和 Adam 状态决定。

真实 `TestReticle/simple.gds` 当前 Layer 1/0、ROI `(-2000,-1100)-(-200,948)` 的 256² 两级各一轮验证通过。CPU 优化 1.984 s、总计 2.224 s、峰值 working set 638,906,368 bytes；4 GiB GTX 1650 CUDA 峰值分配 70,008,832 bytes。CPU/CUDA 最终二值 L2/PVBand/shot 均为 `1922/815/134`；首轮 CUDA 包含运行时初始化，不据此声明加速比。

最终简化审计删除了 CurvMulti/Multilevel 各自的一行平滑包装，两个现实算法直接复用 `simple.py` 的缩放与平滑纯张量实现；专项测试中的伪光刻模型和确定性 GDS 也收敛到已有测试包。生产与测试函数体重复扫描均为 0，全部新增配置/函数有调用方；未引入新结果结构、基类、注册器或仅为错误修复存在的分支，`layout/`、`geometry/` 零差异。
