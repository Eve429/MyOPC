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
