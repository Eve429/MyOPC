# MB-OPC 前端开发报告

## 1. 交付范围

前端负责把一个 Layer/ROI 的物理 Region 转为可供边段 OPC 使用的固定参考 problem，包含规范化、边段切分、规则 core 归属、halo membership、按需物化、探针坐标和全局重建。它不包含具体光刻迭代，也不修改 `layout/`、`geometry/`。

## 2. 当前实现

- `PhysicalMask` 一次保存规范 Region、轮廓和数学边；
- `SegmentBatch` 只保存共享参考几何、edge normals、ring offsets、edge IDs 和 `t0/t1`；
- `OwnershipBatch` 保存唯一 owner 与 CSR halo membership；
- `edge_probe_points` 由当前端点、法向和显式距离生成 inner/outer probe；
- `reconstruct_region` 始终基于全局固定参考边和全局位移重建，不按 core 裁最终 Polygon；
- `opc.diagnostics` 集中处理显式 NPZ/GDS/PNG/图集输出。

前端验证器 `run_mbopc_frontend.py` 保留，方便不运行光刻时单独检查上述全部契约。其演示位移直接按全局 segment 下标写入，不模拟第二套更新协议。

## 3. 性能设计

输入阶段跨 KLayout/NumPy 边界使用批处理。迭代常驻表示不复制每个 segment 的参考端点和法向，也不保存只供诊断的长度。`PhysicalMask` 和 `SegmentBatch` 的轮廓/边字段共享对象引用。

本次减法删除稳定 key、排序查找表、外部更新批次、持久边长/边偏移、采样模板和可插拔 owner policy。这些对象在当前求解器中没有生产调用方，却占用内存并形成第二条更新路径。

阶段 28 最终严格 110,000-segment 基准中，prepare 由历史 168.41 ms 降为 125.64 ms，materialize 由 17.04 ms 降为 12.45 ms，重建由 477.95 ms 降为 427.83 ms；常驻数组相对展开表示节省 69.38%。

## 4. 边界归属和跨 core 图形

segment 中点决定唯一 owner；规则网格内部边界采用半开区间，版图最大边界归最后一列/行。一个 segment 可以作为 context 出现在多个 halo 中，但只能被一个 owner 写。

core 边界不用于最终矢量裁剪。斜边或长边跨多个 core 时，所有 tile 读取同一个全局位移状态，轮次结束后从同一固定参考边重建，因此不会因不同 Region 裁剪产生相邻点 33/34 DBU 的接缝差异。

## 5. 诊断格式

前端 NPZ 的 `format_version=2`，字段全部按当前全局 segment 下标对齐，不含稳定 key。它是人工验证快照，不承诺跨 remesh 或跨版本恢复。显式 remesh 后必须重建 problem、owner 和优化状态。

## 6. 架构复核结论

当前对象各自承担不同职责：`PhysicalMask` 表达规范物理覆盖，`SegmentBatch` 表达控制自由度，`OwnershipBatch` 表达计算/写权限，`MBOPCProblem` 只聚合一次构造的参考状态。两处几何字段是共享浅引用而非重复存储，保留它们能避免调用方拆装数据且没有实际内存倍增。

没有保留无调用方的接口、注册器、兼容包装或异常分支；4 个旧模块删除后，诊断集中为 1 个模块，输入包只剩计算职责。
