# Architecture — 数据流总览

每条核心工作流一个数据流文件，统一提供两种表示：**函数级流向**
（`module::function` 调用树）与**伪代码**（阶段化控制流与不变量）。
本文件只做导航与共享跨界标注。

## 工作流 → 文件映射

| 工作流 | 文件 | 入口 |
|---|---|---|
| 总管线（共享宏生命周期 + 验证管线） | `macro_pipeline.md` | `python main/run_macro_pipeline.py config/macro_pipeline.toml` |
| Simple MB-OPC | `simple_mbopc.md` | `python main/run_mbopc.py config/mbopc_single_macro.toml` |
| Gradient MB-OPC（含可微 EPE loss） | `gradient_mbopc.md` | `python main/run_gradient_mbopc.py config/gradient_mbopc.toml` |
| Simple ILT（像素型） | `simple_ilt.md` | `python main/run_simple_ilt.py config/simple_ilt.toml` |
| LevelSet ILT（SDF + STE + 宏 Adam） | `levelset_ilt.md` | `python main/run_levelset_ilt.py config/levelset_ilt.toml` |
| CurvMulti ILT（多尺度控制网格 + wafer 曲率） | `curvmulti_ilt.md` | `python main/run_curvmulti_ilt.py config/curvmulti_ilt.toml` |

四条工作流共享同一套宏生命周期（prepare → 逐 macro 独立求解 →
merge 恰一次），其完整定义见 `macro_pipeline.md`；其余文件在 workflow
层只描述与共享生命周期的差异。

## 共享跨界标注

- **Python/KLayout 边界**：Region 构造/布尔/写出/读取（layout、geometry、
  merge 裁剪）；栅格化在 `geometry.iter_region_coverage_tiles` 原生分块。
- **CPU/GPU 边界**：批组装与指标累计在 CPU（numpy）；光刻前向与张量评价
  在模型 device；方向/掩码/梯度张量整批一次搬运，批后释放。
- **文件边界**：NPZ（problem/result，`allow_pickle=False`）、GDS（KLayout
  原子写）、JSON（plan/metrics/summary/manifest，临时文件替换）。
