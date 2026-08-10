# MB-OPC 前端开发报告

## 1. 交付范围

前端负责把一个 Layer/ROI 的物理 Region 转为可供边段 OPC 使用的固定参考 problem，包含规范化、轮廓提取、边段切分、规则 core 归属、halo membership、按需物化、探针坐标和全局重建。它不包含具体光刻迭代；本次在用户明确授权后同步收敛了 `geometry` 轮廓契约。

## 2. 当前实现

- `PhysicalMask` 只保存 Layer、规范 Region 和 ROI，可供 ILT 等非边段方法直接复用；
- `ContourBatch` 用 `vertices/ring_offsets/polygon_ring_offsets` 表达两级 CSR；
- `SegmentBatch` 是轮廓的唯一数值持有者，并保存两个 `int32` edge cache、法向和参数区间；
- `MBOPCProblem` 直接保存规则 grid、唯一 owner 与 CSR halo membership；
- `edge_probe_points` 由当前端点、法向和显式距离生成 inner/outer probe；
- `reconstruct_region` 始终基于全局固定参考边和全局位移重建，不按 core 裁最终 Polygon；
- `opc.diagnostics` 集中处理显式 NPZ/GDS/PNG/图集输出。

前端验证器 `run_mbopc_frontend.py` 保留，方便不运行光刻时单独检查上述全部契约。其演示位移直接按全局 segment 下标写入，不模拟第二套更新协议。

## 3. 性能设计

输入阶段跨 KLayout/NumPy 边界使用批处理。迭代常驻表示不复制每个 segment 的参考端点和法向，也不保存逐 edge 起终点、ring ID 或 hole 标志。`edge_next_ids` 避免每轮重建闭环索引，`edge_polygon_ids` 支持 tile 快速选择 Polygon；其余元数据只在准备阶段临时推导。

本次减法删除稳定 key、排序查找表、外部更新批次、持久边长/边偏移、采样模板和可插拔 owner policy。这些对象在当前求解器中没有生产调用方，却占用内存并形成第二条更新路径。

阶段 28 最终严格 110,000-segment 基准中，prepare 由历史 168.41 ms 降为 125.64 ms，materialize 由 17.04 ms 降为 12.45 ms，重建由 477.95 ms 降为 427.83 ms；常驻数组相对展开表示节省 69.38%。

## 4. 边界归属和跨 core 图形

segment 中点决定唯一 owner；规则网格内部边界采用半开区间，版图最大边界归最后一列/行。一个 segment 可以作为 context 出现在多个 halo 中，但只能被一个 owner 写。

core 边界不用于最终矢量裁剪。斜边或长边跨多个 core 时，所有 tile 读取同一个全局位移状态，轮次结束后从同一固定参考边重建，因此不会因不同 Region 裁剪产生相邻点 33/34 DBU 的接缝差异。

## 5. 诊断格式

前端 NPZ 的 `format_version=3`，字段全部按当前全局 segment 下标对齐，不含稳定 key、展开 CoreSpec 或可推导 edge 数组。它是人工验证快照，不承诺跨 remesh 或跨版本恢复。显式 remesh 后必须重建 problem、owner 和优化状态。

## 6. 架构复核结论

当前只保留四个持久职责：`PhysicalMask` 表达原生物理覆盖，`ContourBatch` 表达通用数值拓扑，`SegmentBatch` 表达控制自由度，`MBOPCProblem` 聚合迭代输入和归属。删除 `EdgeBatch`、`OwnershipBatch` 以及重复领域入口后，调用方只有一处轮廓权威和一处 owner 权威。

没有保留无调用方的接口、注册器、兼容包装或 v1 转换分支；诊断集中为 1 个模块，输入包只剩计算职责。`SegmentGeometry` 仅为短生命周期对齐数组返回值，不进入常驻 problem。
