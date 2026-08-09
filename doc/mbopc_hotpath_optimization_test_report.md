# MB-OPC 边段准备与零位移热路径测试报告

## 1. 验收结论

两项优化均通过行为回归、结果等价微基准、严格合成基准、真实 `gcd_45nm` 前端、
三轮 CUDA 整图流程和全仓库质量门。质量指标与历史基线逐项一致，CPU 热路径耗时和
ownership 峰值内存下降，GPU 峰值未增加。

## 2. 自动测试与静态检查

| 检查 | 结果 |
|---|---|
| 归属/迭代定向回归 | 23 passed |
| 全仓库 pytest | 117 passed，最终复跑 19.49 s |
| OPC/光刻/评价覆盖测试 | 77 passed，16.13 s |
| 综合 statement/branch coverage | 92% |
| `opc.iteration.mbopc.solver` coverage | 92% |
| Ruff | passed |
| compileall | passed |
| `git diff --check` | passed |

新增回归直接把 `SegmentBatch.materialize` 替换为失败函数，证明 ownership 不再误用完整
几何；单轮求解同时把 `reconstruct_contours` 和 `contours_to_region` 替换为失败函数，
证明零位移全局/局部重建均被跳过。4 DBU 像素案例先证明 uint8 target 与浮点参考 raster
不同，再断言快路径结果逐像素等于浮点参考，防止量化语义回归。

## 3. 性能与内存等价基准

11 万 segment、64 cores、134,734 memberships 的 ownership 对照：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| 中位耗时 | 40.20 ms | 37.15 ms | -7.6% |
| tracemalloc 峰值 | 24.56 MiB | 17.85 MiB | -27.3% |

owner indices、core offsets、member segment indices 三组数组逐项相同。

500 polygons、11,000 segments 的单 core 零位移局部构造：旧路径 44.35 ms，新路径
11.06 ms，下降约 75.1%；输出 raster 逐像素相同。求解器初始化额外移除一次全局零位移
重建，本轮 11 万 segment 基线该调用为 579.56 ms。

严格 5,000 图形基准结果：prepare 115.55 ms、materialize 12.19 ms、显式零位移重建
576.01 ms、常驻数组 2.441 MiB、相对展开节省 69.38%，XOR=0、unowned=0、
strict failures 为空。

## 4. 真实版图验证

`gcd_45nm.gds` Layer 11/0 前端保持 1,776 polygons、21,590 edges、223,553
segments、4,830,716 persistent bytes；2-core 验证的 XOR、core coverage gap、overlap
均为 0，重建 Region 合法。

三轮 CUDA 整图仍为 870 cores、880,801 memberships：

| 轮次 | EPE | L2 | PVBand | valid | ambiguous | moved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129,645 | 1,038,629.522 | 115,626.751 | 223,298 | 51 | 129,594 |
| 1 | 74,592 | 563,335.522 | 134,540.869 | 223,298 | 5 | 74,587 |
| 2 | 48,348 | 440,251.431 | 147,186.806 | 223,298 | 2 | 48,346 |

上述值与阶段 28 基线逐项一致；PVBand 上升仍如实保留。GPU 峰值保持
271,544,320 bytes，总耗时由 85.892 s 降到 79.117 s，结果 GDS 合法。

## 5. 最终边界审计

`layout/`、`geometry/` 无修改；没有稀疏 core、`core_at`、core 数量或 JSON tiling
语义变化。新增生产抽象为零，所有修改函数都有现有调用方；未发现为修正本阶段问题引入
的包装层、重复字段、死函数或异常兼容分支。用户 GDS、历史设计评审和测试产物不纳入提交。
