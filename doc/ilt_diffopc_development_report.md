# ILT 与 DiffOPC 迁移开发报告

本阶段新增 LevelSetILT、统一多尺度 ILT 调度和独立 DiffOPC 梯度边段求解器；现有 SimpleILT 与 Simple MB-OPC 保持不变。

- LevelSet 使用 `phi < 0` 硬二值前向和 `|grad(phi)|` 代理反向。
- MultiScale 使用粗到细参数插值，统一复用当前 ICCAD13 光刻模型。
- DiffOPC 使用独立解析软边段栅格器；精确 KLayout raster 只保留在非梯度路径。
- 不复制 OpenILT/DiffOPC 的代码、资产、Hydra、数据集或日志框架。

当前 DiffOPC 首版完成 L2/PVBand 和固定 inner/outer probe 的连续 EPE hinge 梯度路径；MRC、SRAF 和多 GPU 仍属后续阶段。

## 第一阶段质量修正

LevelSetILT 默认初值已从二值正负常数改为一次性精确欧氏 SDF；配置、输入有限值、优化窗口和工艺条件执行严格校验，显式空工艺窗口不再回退默认条件。曲率项统一复用零和离散曲率核，公共 `soft_mask` 由最优 phi 生成连续诊断图，硬结果严格复用前向的 `phi < 0`。统一入口现已支持 GDS/OASIS 的 Layer、ROI、安全预检参数，并保存评价、分阶段时间、GPU 峰值和最终三工艺角 NPZ/PNG。

本阶段没有增加基类、注册器、contracts 或工具文件：`LevelSetConfig`、SDF、代理梯度和求解器同置一个实现文件，并复用 SimpleILT 的结果记录与曲率核。精确 SDF 使用 `O(HW)` 两遍距离变换，只在优化前运行一次；迭代主要常驻参数、固定初值、窗口和 Adam 状态，不引入边段或 KLayout 热循环。`layout/`、`geometry/` 均未修改。

当前光刻模型 canvas 为 256；同机纯 CPU 的一次性 SDF 实测 256² 为 1.459 s、结果 0.25 MiB，进程 RSS 增量约 3.38 MiB。512²/1024² 探查分别为 5.825/23.602 s，说明纯 Python 下包络适合当前固定 canvas，但未来若光刻 canvas 大幅提升，应在不改变接口的前提下换成编译型 EDT；本阶段不为尚未支持的大画布引入 SciPy/自定义扩展依赖。
