# 可微光刻、评价与 SimpleILT 测试报告

## 1. 环境与命令

解释器：`D:\app\miniforge\envs\myopc\python.exe`，Windows，Python 3.12。

```powershell
& $python -m ruff check layout geometry opc lithography evaluation main tests benchmarks
& $python -m compileall -q layout geometry opc lithography evaluation main tests benchmarks
& $python -m pytest -q
& $python -m pytest tests\lithography tests\evaluation `
  tests\opc\test_simple_ilt.py tests\opc\test_simple_mbopc.py `
  --cov=lithography --cov=evaluation --cov=opc.iteration.ilt `
  --cov=opc.iteration.mbopc --cov-branch --cov-report=term-missing -q
```

结果：Ruff 与 compileall 通过；全仓库最终复跑 `152 passed in 38.63s`；专项最终复跑 `39 passed in 13.54s`，四个目标模块综合 statement/branch coverage 为 92%。

## 2. 光刻测试

覆盖以下数值与接口不变量：

- OpenILT 四个 kernel/scale 资产 SHA-256 固定；
- batch 与逐图单条件结果一致，满 256² canvas 不错误返回输入；
- 共享频谱结果与每个条件独立 FFT 参考逐像素最大差不超过 `5e-6`；
- nominal/dose_max/defocus_min 的历史输出和基线一致；
- mask 梯度有限、非零；
- 使用正负混合的非均匀上游权重，对一个输入像素执行中心有限差分，autograd 与数值梯度在 `rtol=2e-2, atol=2e-2` 内一致；
- 自定义 kernel/dose 条件可单独运行，重复名称、未知默认条件、非法形状和超 canvas 明确拒绝；
- 可用 CUDA 时，由环境 Python 直接启动子进程并完成设备前向。

非均匀上游梯度测试专门排除了“只对全一上游梯度正确”的自定义反向实现。

## 3. 评价测试

- ownership 内外各放置错误像素，确认 halo 不重复累计；
- L2/PVBand 输出为二值不一致像素整数；
- 评价前后输入张量逐元素相同，防止 OpenILT 原位阈值化副作用；
- 矩形、L 形、中空图案和 batch 的 shot 估计分别为稳定的水平 run 合并结果；
- EPE 覆盖 inner 外移、outer 内移、同时冲突不移、越界、窄壁穿越和同像素无效；
- 形状、设备、ownership 与 shot 尺寸异常均在批量索引前拒绝。

## 4. SimpleILT 测试

- 低成本可微伪模型上，八轮总损失严格低于首轮；输出软 mask、bool 二值 mask 和 target 同形；
- 自定义 nominal/high/low 条件逐轮原样传入，没有被默认条件替换；
- optimization mask 外保持初始 soft 值；
- process conditions 可为空，process L2/PVBand 为零；
- 曲率项产生有限损失；
- 初始参数/窗口形状、窗口范围和重复条件名称错误明确拒绝；
- 真实 ICCAD13 Hopkins 模型完成一轮 forward、backward 和 SGD 路径，参数有限、soft mask 位于 `[0,1]`；
- `main/run_simpleilt.py` 从版本化 raster NPZ 完成真实模型运行，保存参数、soft/binary mask、L2/PVBand/shot、耗时、JSON/NPZ/PNG。

## 5. MB-OPC 与入口回归

- 两个 core 跨界图形仍在同一轮只读同一 `current`，owner 只写一次，屏障后才发布；
- 2 DBU 中空壁配 8 DBU probe、外轮廓越 hole、矩形对边穿越、斜边/孔洞/引用离线恢复均继续通过；
- 真实 ICCAD13 模型完成一轮流式 MB-OPC；
- 人工制造第二轮 L2 更优但 EPE 相同，最佳轮次仍为 0，证明诊断指标没有暗中控制几何；
- `main/run_layout_geometry.py`、`run_mbopc_frontend.py`、`run_mbopc.py` 的既有 CLI 回归通过；
- `main/offline_inputs.py`、`run_lithography.py`、`run_mbopc_iteration.py`、`run_simpleilt.py` 从仓库外工作目录执行 `--help` 均退出 0，无需安装项目。

## 6. 覆盖率与剩余风险

专项覆盖率：evaluation 96%、lithography 87%、SimpleILT 91%、MB-OPC solver 93%，合计 92%。未命中主要是 Windows/CUDA 环境分支、资产损坏异常、极少数空数组保护和 CLI 错误出口；核心数值路径、梯度路径、独立条件组合、跨 core 更新与真实模型集成均已执行。

本阶段没有重新运行 `gcd_45nm` 三轮完整 CUDA 基准，因此历史连续 L2/PVBand 不应与新二值计数直接比较。代码不改变版图输入、分段、owner、位移和重建，仅改变光刻返回接口、诊断指标语义并追加最终固定内存 shot 估计。下一次整图性能验收应建立新的整数 L2/PVBand 基线。

## 7. `simple.gds` 最终真实冒烟

在全部提交前，使用用户现有 `TestReticle/simple.gds` 执行四段真实 CPU 流程：指定 ROI 生成 256² raster NPZ；独立光刻保存三条件 NPZ/PNG；SimpleILT 一轮保存参数、软/二值 mask 和评价；完整 reticle MB-OPC 一轮保存 GDS/PNG/JSON。

结果：光刻输出 shape 为 256²；SimpleILT binary L2 为 1900；MB-OPC 恢复 10 polygons、107 edges、885 segments、8 cores、2658 memberships，单轮 EPE/L2/PVBand 为 `338/3936/1607`，shot estimate 为 325，结果 Region 合法，总耗时 0.749 s。全部命令退出码为 0，产物位于 `output/final_verification/`，该目录不进入 Git。
