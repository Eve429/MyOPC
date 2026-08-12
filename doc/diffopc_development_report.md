# DiffOPC 第四阶段开发报告

## 1. 交付结果

本阶段把原有 DiffOPC 原型提升为可验收的固定参考边段梯度 OPC：输入可为 GDS/OASIS 或离线 segment NPZ，输出包含最佳位移 NPZ、参考/重建 GDS、summary JSON、可选边段标注图和 ownership-only 最终光刻 tile。生产代码未修改 `layout/` 与 `geometry/`。

参考 NVlabs DiffOPC 的边段参数化、连续工艺窗口损失和最佳轮次思想，但没有照搬 Hydra、数据集、日志框架、自定义二值直通算子或逐 edge Python backward。参考实现公开的二值 PVBand 只适合评价；项目版把 maximum/minimum 连续 wafer 的平方差用于梯度，二值 L2/PVBand/EPE 保留为诊断。

## 2. 数学与归属

对参考边的外法向有符号距离 `q` 和法向位移 `d`，局部占据变化定义为 `sigmoid((d-q)/T)-sigmoid(-q/T)`。因此 `d=0` 时两项解析抵消，输出逐像素等于精确 KLayout 参考覆盖率；孔洞的外法向指向孔内，同一公式无需 `is_hole` 分支。切向双 sigmoid 限制有限 segment 影响并平滑端帽。

一个 segment 可出现在多个 core context，但只有 `owner_indices[segment]` 指定的 core 计算它的 EPE。halo segment 只改变当前 tile 的软 mask/光刻上下文；L2 和连续 PV 损失只乘 `ownership_canvas`。全局损失以唯一 ownership 像素数和 segment probe 数归一化，所以 batch 大小与 halo membership 数不会改变同一问题的权重。

## 3. 流式资源路径

每个 batch 在同一只读 `current` 上完成 soft raster、三工艺角 forward、连续损失和 backward，随后立即释放光刻计算图；全部 batch 完成后才做一次 Adam step。这里的“立即 backward”只累积 `current.grad`，不会提前写位移，因此仍满足轮次屏障。

软栅格按 `raster_chunk_size` 分段，并对每个 H×W×chunk 中间量使用 PyTorch checkpoint：正向不保存 sigmoid/投影中间量，反向按 chunk 重算。CPU target 使用有字节上限的 uint8 LRU；simple MB-OPC 与 DiffOPC 共用一个 `ArrayTileCache`，删除了第二份私有缓存实现。

## 4. 几何约束与清理

候选位移先投影到 DiffOPC 上限，且该上限不得超过前端重建上限。轮次发布前调用公共 `reconstruct_region`；ring 方向翻转、孔洞越出所属 hull、自交、退化和原生无效 Polygon 都会拒绝整个候选。原先只存在 simple MB-OPC 内部的 ring/hole 拓扑检查已归入 `opc.input.edge.reconstruction`，两个迭代器不再维护重复逻辑。

最佳损失、最佳轮次和最佳位移现在绑定同一个已完整评价快照。最后一个配置轮次只评价不再产生无人评价的 step；不存在用 step 前损失保存 step 后位移的错配。直接 GDS 路径调用新增的 `materialize_segment_input` 内存层，不再写入并读回大型临时 NPZ；显式离线工作台仍由 `prepare_segment_input` 保存相同问题。

最终冗余审计确认：DiffOPC 只保留配置/记录、软栅格、求解三个职责文件；共享缓存只有两个现实调用方；没有注册器、基类、空目录、旧 `_TargetCache` 别名或旧拓扑包装。SRAF 会新增图形并改变 segment 身份，因此没有为“看似完整”而混入当前固定拓扑求解器。

## 5. 当前能力边界

GPU 只保存当前 batch，CPU 仍常驻完整 `MBOPCProblem`、owner/membership CSR 和全局位移。它适用于当前完整内存问题，不代表未来 macro shard/memmap 大 reticle 方案已经实现。工艺厂 MRC 规则尚无规则 deck；当前“几何 MRC”仅指显式最大位移和全局拓扑合法性，不能宣称完成未定义的最小线宽/间距规则。
