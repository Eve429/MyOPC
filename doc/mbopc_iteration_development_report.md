# simple MB-OPC 迭代开发报告

## 1. 模块边界

`opc/iteration/mbopc` 只负责具体迭代；输入由 `opc.input.edge.prepare_problem` 构造，光刻由 `lithography.ICCAD13Lithography` 提供，评价位于 `evaluation`。`run_mbopc.py` 只做单位换算、生命周期编排和最终输出。

该边界支持分别更换输入构造、OPC 更新方法、光刻模型或评价方法。ILT 可复用物理 mask、Region 栅格、光刻和评价，不需要被迫使用 MB 边段。

## 2. 流式整图方案

CPU 常驻全局参考 problem 和一维 displacement。每个 batch 只生成 core+halo 的 current/target/ownership 图，在设备上运行模型后立即把 core 指标和 owner 方向累计到 CPU，并释放输出 tensor。target 使用有字节上限的 uint8 LRU，CPU batch 也保持 uint8，送设备时才一次性转为 float32；current mask 仍保留未量化覆盖率。

每轮所有 batch 只读 `current`，写入 `next_values`；完成全部 batch 和全局拓扑验证后才发布。该同步屏障保证 core 处理顺序不改变结果，也避免边被某个早完成 batch 提前更新。

最终从最佳全局 displacement 重建一次完整 Region。halo 从不写最终结果，core 也不裁剪最终 Polygon。

owner segment 索引由每 core 的 membership CSR 直接过滤，避免对全局 segment 数组重复扫描。ICCAD13 在一次 `forward` 内共享 mask FFT 和 focus 单位剂量强度；这些优化不改变轮次状态、segment 身份或模型输出接口。

## 3. 评价与更新

L2/PVBand 只在 core ownership 像素统计。EPE probe 使用当前 segment 中点、解析外法向和 `epe_distance_dbu`；inner/outer target 语义无效、越界、落同像素或相互冲突时不移动。有效方向乘本轮步长后，再由 `FragmentationConfig.max_displacement_dbu` 统一限幅。

模型的画布和 print threshold 只由 `ICCAD13Config` 提供；位移上限只由 `FragmentationConfig` 提供，删除了迭代配置中的重复权威字段。

## 4. 拓扑安全

候选轮次发布前检查每个 ring 的有向面积符号和 hull/hole 包含关系。矩形相对边交叉或外轮廓移动到孔内都会整轮回滚。当前没有增加未经验证的局部修补层；这是有明确测试的保守 v1 行为。

## 5. 产物

完整入口写 `summary.json`、结果 GDS 和可选预览 PNG，不再写 NPZ。迭代期间不保留整张 reticle tensor、逐 tile 模型输出或中间 GDS。

## 6. 简化审计

`QualityMetrics` 删除未使用的 `pixel_count`，避免额外 GPU 同步；`SimpleMBOPCConfig` 删除与模型/分段配置重复的 threshold 和位移字段；求解器删除仅验证自身构造结果的恒真检查。保留 canvas 是必要的，因为 tile 运算画布可小于模型支持的最大画布，求解器会验证兼容性。
