# 可微光刻、评价与 SimpleILT 开发报告

## 1. 交付范围

本阶段完成四项实际能力：ICCAD13 光刻模型改为独立工艺条件接口并保留完整 autograd；OpenILT 的二值 L2/PVBand/EPE/shot 评价语义迁入公共 `evaluation`；新增可运行的 SimpleILT；全部运行入口集中到 `main/`。本阶段未修改受保护的 `layout/` 和 `geometry/`。

参考实现包括 `OpenILT/pylitho/exact.py`、`simple.py`、`pyilt/evaluation.py`、`pyilt/simpleilt.py` 与 TorchLitho-2.0。保留了 Hopkins 核、连续工艺窗损失和像素参数优化的有效设计；没有复制 OpenILT 的原位阈值化、重复光刻仿真、逐像素 Python EPE、随机 adabox shot、导入期默认模型实例和自定义近似反向，也没有引入 TorchLitho 的运行时 TCC/SVD 或仅适配特定上游梯度的自定义 VJP。

## 2. 光刻模型

新增 `ProcessCondition(name, kernel, dose)`。`ICCAD13Lithography.forward(mask, condition)` 只运行一个条件；`forward_many(mask, conditions)` 返回名称到 wafer 张量的字典。默认 nominal、dose_max、defocus_min 只是便捷构造，不会强制一起运行。

一次 `forward_many` 只计算一份 mask FFT；相同 focus/defocus bank 的单位剂量 aerial 只传播一次，随后按 `dose²` 缩放。中间量只存在于当前调用的 autograd 图中，不跨 mask 缓存。实现完全由 PyTorch 标准复数 FFT、逐元素运算与 sigmoid 构成，MB-OPC 在 `no_grad` 下使用同一路径，ILT/梯度 OPC 可直接 backward。

删除了固定三字段 `LithographyResult`。该结构会把“模型能力”错误绑定为“三个特定工艺角”，也迫使只需一个条件的算法计算全部结果。字典只在一次 batch 内存活，未增加长期结构或注册器。

## 3. 评价

`evaluate_binary_l2` 统计 target/nominal 二值图不一致像素；`evaluate_pvband` 统计两个独立工艺条件二值图不一致像素。两者支持 ownership mask，不统计 halo，并且不原位修改输入。

现有 `evaluate_edge_probes` 已直接消费前端的稳定边段、法向和 inner/outer 探针，比 OpenILT 从像素轮廓重新找边更适合当前 MB-OPC，因此保留。EPE 是 simple MB-OPC 唯一的移动与最佳状态依据；L2/PVBand 仅记录。新增回归证明即使第二轮诊断 L2 更优，EPE 相同时仍保留更早几何。

`estimate_rectangular_shots` 在固定评价分辨率上提取每行前景 run，并把相邻行相同区间合并为一个矩形。它确定、无随机、无 OpenCV/adabox 依赖，适合版本间比较；它是 shot 估计，不宣称得到全局最小矩形分解。整图 MB-OPC 只对最终最佳 Region 栅格化一张 512² 诊断 mask 并计算一次，内存上界固定。

## 4. SimpleILT

`opc/iteration/ilt/simple.py` 同置 `SimpleILTConfig`、`ILTIterationRecord`、`SimpleILTResult` 与 `optimize`。当前只有一个真实 ILT 方法，没有为未来假设方法建立基类、注册器或单独 contracts 文件。

输入是 target、可选初始参数、可选 optimization mask、一个 nominal condition 与任意 process conditions。默认参数为 `target*2-1`；窗口外固定为初始 soft mask。每轮损失包括 nominal 连续 L2、各 process condition 相对 target 的连续 L2、process stack 逐像素范围的连续 PVBand，以及可选三乘三曲率项。SGD 直接更新参数，结果保存总损失最优轮的参数、软 mask、二值 mask 和标量记录。

该路径不构造 `MBOPCProblem`，不提边、不分 owner，也不依赖 `opc.input.edge`。共享的是像素输入、光刻模型和评价层，避免让 ILT 承担 MB-OPC 的边段常驻内存。

## 5. 入口迁移

全部可执行脚本位于 `main/`：

- `run_layout_geometry.py`、`run_mbopc_frontend.py`、`run_mbopc.py`；
- `offline_inputs.py`、`run_lithography.py`、`run_mbopc_iteration.py`；
- 新增 `run_simpleilt.py`。

旧根目录和 `tests/workbench` 中不保留包装脚本。每个入口按文件位置加入仓库根，可从任意工作目录由 Python 直接运行。像素/边段准备接口也移入 `main/offline_inputs.py`，生产入口不再反向导入测试包。JSON/NPZ/PNG 原子输出共用一份实现，清除了两个 runner 中重复的 PNG 转换函数。

## 6. Bug 与简化审计

开发中有三次补丁因上下文漂移未应用：光刻测试的大块补丁、计划锚点补丁、PNG 辅助函数收敛补丁。三次均为原子失败，无部分写入；随后改用小锚点补丁。没有为这些失败增加兼容分支或临时包装。

最终审计结论：

- `layout/`、`geometry/` 无差异；
- 删除固定 `LithographyResult` 和未再使用的 `QualityMetrics`，没有重复结果结构；
- ILT 两个源码文件（实现与包导出）是当前最小可直接导入布局；
- evaluation 继续单文件承载四个紧密相关指标，没有过度拆文件；
- `main/` 只按真实可执行工作流拆分，公共原子 I/O 和严格 nm→DBU 换算收敛到现有 `offline_inputs.py`，没有再创建 `utils.py`；
- 未新增空目录、注册器、抽象基类或无调用方生产函数；
- MB-OPC bug 修正没有残留旧评分逻辑，生产搜索无 `LithographyResult`、`QualityMetrics`、`evaluate_process_window` 调用。

详细验证见[可微光刻、评价与 SimpleILT 测试报告](lithography_ilt_evaluation_test_report.md)。
