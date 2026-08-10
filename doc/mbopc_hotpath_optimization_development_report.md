# MB-OPC 边段准备与零位移热路径开发报告

## 1. 开发范围

本阶段只实施两项已经确认的优化：内部 `_build_ownership` 不再为归属计算完整物化
`SegmentGeometry`；simple MB-OPC 的零位移初态和局部未变化 core 不再执行无意义的
轮廓/Region 重建。稀疏活跃 core、宏分块、梯度 OPC、remesh，以及 `layout/`、
`geometry/` 修改均明确排除。

## 2. ownership 准备路径

旧路径调用 `SegmentBatch.materialize()`，同时生成 starts、ends、normals 和
`SegmentGeometry`，但归属只读取端点。新路径直接用 `contours.vertices`、
`edge_next_ids/edge_ids`、`t0/t1` 批量生成参考端点和中点，之后继续使用原有
midpoint owner、bbox+halo 与 CSR
membership 算法。

端点构造完成后立即释放数学边起点/向量，四条 bbox 边界形成后再释放 starts、ends、
midpoints。这个生命周期控制是必要的：若只省掉 normals、却让其他 S×2 局部变量存活到
CSR 展开结束，实测峰值内存反而会上升。实现没有新增类、包装函数或第二套归属接口。

## 3. 零位移图形路径

求解器的 `current` 初值恒为全零，因此 `current_contours` 直接引用不可变参考
`problem.segments.contours`。只有轮次产生非零候选并通过拓扑检查后，才执行并发布一次
全局 `reconstruct_contours`。

`_current_tile` 新增读取全局 displacement 向量：若本 core 的全部 halo membership
位移均为精确零，则直接栅格化 `physical_mask.region`；否则仍执行原来的相关 polygon
子集、参考/当前 Region 差分和栅格化。判断基于 context membership，而非仅看 owner，
所以相邻 core 移动的 halo 边仍能进入慢路径。

target LRU 的常驻格式为 uint8，而历史 current mask 是未量化浮点覆盖率。为了保持
光刻和评价数值完全一致，快路径没有直接返回 target；它只省略几何变换，再从参考 Region
生成原精度 current raster。

## 4. 不变量与兼容性

- 全局 segment 下标、唯一 owner、halo 只读和轮次屏障不变；
- `next_values` 仍只在全部 batch 完成、全局候选合法后发布；
- 最终 GDS 仍从固定参考边和最佳全局位移统一重建；
- `SegmentBatch.materialize`、`SegmentGeometry` 仍服务探针、诊断和非零位移调用方，没有删除；
- 未新增面向假设方法的接口，ILT 和其他 OPC 的目录边界不变。

## 5. 代码审计结论

生产修改限于 `opc/input/edge/ownership.py` 与
`opc/iteration/mbopc/solver.py`。新增参数直接服务现有 `optimize` 调用；没有新增结构体、
生产函数、文件夹或兼容分支。调用搜索确认旧物化 API 仍有真实调用方，本阶段不应删除。
受保护的 `layout/`、`geometry/` 无差异。代码与回归已在本地提交 `56fd33d`，未推送远端。
