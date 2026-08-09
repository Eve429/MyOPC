# OPC 架构精简测试报告

## 1. 验收结论

架构精简后，完整回归、专项覆盖、严格性能基准、真实 `gcd_45nm` 前端和三轮 CUDA 流程全部通过。几何与迭代数值保持一致，输入内存显著下降，未发现速度退化。

## 2. 自动测试

| 检查 | 结果 |
|---|---|
| 全仓库 pytest | 114 passed，16.03 s |
| OPC/光刻/评价专项 | 74 passed |
| 综合 statement/branch coverage | 92%（74 passed，15.94 s） |
| Ruff | passed |
| compileall | passed |
| Layout/Geometry 专项 | 38 passed，91% coverage |

相对最初 119 项基线，OPC 减法净减 1 项；Layout/Geometry 又删除 5 个仅服务旧门面/索引的用例并新增 1 个精确 ROI 回归，因此最终为 114 项。生产路径覆盖没有降低。

## 3. 严格性能基准

5,000 shapes / 110,000 segments：

| 指标 | 重构前 | 重构后 | 变化 |
|---|---:|---:|---:|
| prepare | 168.41 ms | 125.64 ms | -25.4% |
| materialize | 17.04 ms | 12.45 ms | -26.9% |
| zero reconstruct | 477.95 ms | 427.83 ms | -10.5% |
| 相对展开内存节省 | 43.43% | 69.38% | +25.95 pct |

重构后 persistent arrays 2.441 MiB，expanded representation 7.973 MiB；XOR=0、unowned=0、strict failures 为空。

## 4. 真实版图前端

`gcd_45nm.gds` Layer 11/0：223,553 segments；常驻数组由 12,675,300 降到 4,830,716 bytes，减少 61.89%；阶段 28 prepare 152.82 ms；总诊断流程 2.308 s；零位移 XOR/core gap/core overlap 均为 0。v2 NPZ 不含 segment key。

人工检查标注图确认 owner 分区、探针方向和跨 core 连续性正确。

## 5. 三轮 CUDA 完整流程

EPE `129645 -> 74592 -> 48348`，L2 `1038629.522 -> 563335.522 -> 440251.431`，与历史基线一致；PVBand `115626.751 -> 134540.869 -> 147186.806`，如实记录为上升。

GPU 峰值 271,544,320 bytes 与历史一致；授权精简后总耗时 85.892 s，对比 84.708 s 相差 1.40%。输出 Region/GDS/JSON/PNG 合法，完整 runner 没有 NPZ。

## 6. Layout/Geometry 精简验证

新增跨 ROI Polygon 回归要求 bbox 和面积精确等于 planner 查询框；属性回归同时裁断普通/带属性图形，并验证 tagged 属性不丢失。专项 38 项、综合覆盖率 91%。

百万逻辑实例 ROI 精确物化中位数 0.10435 ms，历史为 0.1058 ms；2048² raster 499.59 ms，覆盖一致；严格失败列表为空。真实 `gcd_45nm` 前端计数、XOR 和常驻内存完全不变。

## 7. 删除符号与过度设计审计

搜索确认生产 Python 不再引用：稳定 key、key lookup、external update batch/result、sample template/batch、选择性 materialize、持久 edge lengths/offsets 和旧 reconstruction 签名。

诊断代码只有根入口显式调用；输入包不再导出诊断函数。剩余结构体均有当前生产调用方，重复几何字段通过对象身份回归证明是共享引用。未发现为了旧错误保留的 wrapper、兼容分支或变量。

同时确认生产代码不再引用固定 backend、`GeometryEngine`、`UniformGridIndex`、`CoordinateSystemError`、edge bbox 或 `DbuBox.overlaps`。历史 `design_review.md` 保留为修改前的审计记录，不当作当前 API 手册。`hierarchy_summary` 是明确交付的只读层级检查能力，有外部 planner 使用价值且无热路径成本，因此保留。

## 8. 风险与后续

当前 segment 下标只在一个 prepared problem 生命周期内稳定；不支持跨 remesh/checkpoint/distributed update。这是有意缩小范围，不是遗漏。需要真实跨进程消费者时，应以实际持久化需求重新设计，而不是恢复已删除的通用 key 链。
