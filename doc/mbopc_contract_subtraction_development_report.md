# MB-OPC 数据契约收敛开发报告

## 1. 目标与结论

本阶段针对 `PhysicalMask / ContourBatch / EdgeBatch / SegmentBatch / OwnershipBatch / MBOPCProblem` 的字段重叠重新划分数据权威。结论是保留四个持久职责，删除两个可合并结构：

- `PhysicalMask`：Layer、规范原生 Region、ROI；
- `ContourBatch`：Polygon/Ring/Vertex 两级 CSR；
- `SegmentBatch`：固定参考轮廓、最小 edge cache、法向和控制段参数；
- `MBOPCProblem`：完整迭代输入、规则 grid、owner/membership CSR；
- 删除 `EdgeBatch` 和 `OwnershipBatch`。

核心代码提交为 `48f7047 refactor(opc): 收敛边段问题数据契约`，同步手册与报告另形成一个本地文档里程碑；两者均未推送远端。用户 `TestReticle/JustPoly.gds` 和 `output/mbopc/*` 修改未进入提交。

## 2. 结构变化

`PhysicalMask` 不再为了 MB-OPC 提前提取轮廓，使 ILT 等非边段方法可以只消费规范 Region。`ContourBatch` 删除重复 Layer、逐 ring Polygon ID 和 hole 布尔列，改为：

```text
vertices                 int64[V,2]
ring_offsets             int64[R+1]  ring -> vertex
polygon_ring_offsets     int64[P+1]  polygon -> ring
```

每个 Polygon 范围首 ring 是 hull，其余 ring 是 hole。该不变量由构造函数统一校验，`contours_to_region` 可直接恢复原生 Polygon。

`EdgeBatch` 的 starts、ends、ring IDs、polygon IDs 和 hole 标志都能从轮廓推导，因此整体删除。`SegmentBatch` 只保留两个有当前热路径调用方的缓存：

- `edge_next_ids:int32[E]`：数学边终点顶点索引，避免每轮重建闭环索引；
- `edge_polygon_ids:int32[E]`：tile 从 member segment 快速得到完整 Polygon。

`OwnershipBatch` 删除。`MBOPCProblem` 直接持有 `RectilinearCoreGrid`、`owner_indices`、`core_offsets` 和 `member_segment_indices`，并提供 `core_count`、`segments_for_core()` 和完整数组内存统计。`CoreSpec` 仅在 solver/诊断入口按需展开一次。

## 3. 数据流与依赖

当前准备顺序为：

```text
RegionBatch
  -> normalize_physical_mask
  -> PhysicalMask(layer, region, query_box)
  -> extract_contour(region)
  -> fragment_edges(contours, config)
  -> SegmentBatch
  -> _build_ownership(segments, grid)
  -> MBOPCProblem
```

`_build_ownership` 是 builder 的私有实现步骤，不再作为公共类型工厂导出。依赖保持 `layout -> geometry -> opc.input -> opc.input.edge -> opc.iteration.mbopc`，基础层不依赖具体迭代方法。

## 4. 离线格式

可恢复 segment 归档直接升级为 version 2，保存 nested contour CSR、两个 edge cache、segment 数组、grid cuts/halo 和 owner/membership CSR。旧 version 1 即使缺少所有 v2 新字段，也先得到“请重新生成离线边段输入”的版本错误；没有转换分支。

前端人工诊断 NPZ 升级为 version 3。它不保存展开 CoreSpec、旧 edge 派生列或稳定 key，也不承诺跨 remesh 恢复。

## 5. 性能取舍

`gcd_45nm` 同口径实测中，完整 problem 常驻 NumPy 数组由 10,688,650 降至 9,802,180 bytes，减少 886,470 bytes（8.29%）。此外不再常驻由 488-byte grid 展开的约 200 KB CoreSpec Python 对象。

同一进程、同一 223,553-segment problem 的 30 次端点物化对照：旧 EdgeBatch 风格中位数 28.229 ms，新 nested-next 风格 28.205 ms，证明减法没有牺牲热路径速度。没有为了追求更低时间增加第三个 edge vector 缓存，因为现方案已经与旧路径等价且内存目标通过。

## 6. 最终简化审计

- 生产源码无 `EdgeBatch`、`OwnershipBatch`、旧公共 `extract_edges/build_ownership` 调用；
- Region 与 Contour 同时存在是必要双表示：前者服务原生布尔/栅格，后者服务数值拓扑/重建；
- `SegmentGeometry` 是短生命周期对齐返回值，不是重复常驻状态；
- 两个 edge cache 都有当前 solver/reconstruction/materialize 调用方；
- v1 没有兼容包装；诊断、PNG、GDS 仍只在明确请求时产生；
- 未发现为了本次回归错误增加的第二套校验函数或无调用变量；零生产引用的三个公开 API 均有直接测试调用，`ICCAD13.forward` 由 PyTorch 调度，因此没有为追求“零列表”误删有效入口。

测试结果见 [MB-OPC 数据契约收敛测试报告](mbopc_contract_subtraction_test_report.md)。
