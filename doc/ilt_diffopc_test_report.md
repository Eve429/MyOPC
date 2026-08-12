# ILT 与 DiffOPC 迁移测试报告

定向命令：

```powershell
& 'D:\app\miniforge\envs\myopc\python.exe' -m pytest tests/opc/test_new_methods.py -q
```

结果：3 passed。覆盖水平集/多尺度统一结果、软边段有限梯度和离线问题消费；既有光刻/工作台回归 26 passed，Ruff 与 compileall 通过。

后续补充孔洞/斜边跨 core 的数值梯度、MRC/SRAF 和大版图 CUDA 峰值测试。

第一阶段新增 8 项独立 LevelSet 回归：精确 SDF 的符号/距离、代理梯度局部性、配置/输入/窗口/条件拒绝、空工艺窗口与固定区域、曲率有限值、严格零等值线、真实 Hopkins backward、GDS 直接入口完整产物，以及仓库外直接运行。

最终验证：LevelSet 与既有 ILT 定向 16 项通过；全仓库 167 项通过（48.34 s）；Ruff、compileall、`git diff --check` 均通过。另以 1×1 至 8×8 随机二值图和暴力最近点算法逐像素比较，精确 EDT 全部一致。pytest-cov 在当前 Windows 环境触发 NumPy“同一进程重复加载”运行时错误，因此本阶段不虚报覆盖率百分比；功能分支以专项测试、全量回归和静态审计确认。

一次性 SDF 性能探查：256²/512²/1024² 分别为 1.459/5.825/23.602 s；当前 ICCAD13 canvas 上限为 256，因此该时间不随迭代次数重复。256² 结果占 262,144 bytes，进程 RSS 增量约 3.38 MiB；测试后已复用一维 scratch、预分配 batch 输出，避免逐行/逐图数组常驻。

## 第二阶段 CurvMultiILT 测试

专项测试覆盖：配置/有限值/尺度整除/窗口/重复条件拒绝；平均池化加 offset sigmoid 公式对照；恒等光刻下损失下降和参数实际更新；粗控制网格的光刻输入始终保持完整物理 shape；空工艺窗口和窗口外固定值；曲率确实施加于 nominal wafer；batch 和真实 Hopkins 多尺度 backward；孔洞、斜边、十字、多组件；GDS 函数入口完整产物/内存统计；仓库外直接 CLI。

真实 `simple.gds` 使用 `--scales 4 2 1 --iterations 1 --smoothing-kernel 3 --curvature-weight 0 --device cpu` 退出 0，保存 `ilt_result.npz`、`summary.json` 和 `final_lithography.npz`。三阶段损失单调下降，输出 shape 256²；CPU memory checkpoints 覆盖 start/input/model/optimization/evaluation/output。

最终结果：CurvMulti 专项 13 项、全部 ILT 定向 29 项、全仓 180 项通过；正式源码与 `tests/` 的 Ruff、compileall、中文模块/函数 docstring、Markdown 链接、旧 multiscale 符号、`git diff --check` 和保护目录差异审计均通过。根目录 Ruff 另命中用户 `Test/klayout.ipynb` 两处既有 SIM113，本阶段未修改该 notebook，也未把它误报为本阶段通过项。

CUDA 冒烟在 4 GiB GTX 1650 上完成 256²、scale 4/2/1 各一轮真实 backward；峰值分配 69,222,400 bytes，最终 L2/PVBand/shot 与 CPU 同为 `2146/800/118`。当前设备不是用户目标 24 GiB GPU，且首轮含 CUDA 初始化，因此只作为可运行和显存上界证据。
