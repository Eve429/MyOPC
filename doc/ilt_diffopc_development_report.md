# ILT 与 DiffOPC 迁移开发报告

本阶段新增 LevelSetILT、统一多尺度 ILT 调度和独立 DiffOPC 梯度边段求解器；现有 SimpleILT 与 Simple MB-OPC 保持不变。

- LevelSet 使用 `phi < 0` 硬二值前向和 `|grad(phi)|` 代理反向。
- MultiScale 使用粗到细参数插值，统一复用当前 ICCAD13 光刻模型。
- DiffOPC 使用独立解析软边段栅格器；精确 KLayout raster 只保留在非梯度路径。
- 不复制 OpenILT/DiffOPC 的代码、资产、Hydra、数据集或日志框架。

当前 DiffOPC 首版完成 L2/PVBand 和固定 inner/outer probe 的连续 EPE hinge 梯度路径；MRC、SRAF 和多 GPU 仍属后续阶段。
