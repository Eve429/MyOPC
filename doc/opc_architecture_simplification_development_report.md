# OPC 架构精简开发报告

## 1. 背景与目标

本阶段不是增加新算法，而是审查 MB-OPC 前端与迭代实现是否因历史 bug 修复和未来假设形成过多对象、函数和文件。约束是结果与速度不能退化、未来 OPC/ILT 仍有清晰扩展边界，并且不得修改 `layout/`、`geometry/`。

## 2. 发现的问题

审计确认两类重复：

1. 求解器以全局数组下标同步更新，但前端还维护稳定 key、排序 token、外部更新批次和合并函数，形成没有生产调用方的第二套更新架构；
2. problem 常驻 edge lengths、edge offsets、采样模板和 materialized lengths/indices，实际只供诊断或根本没有调用方。

此外，输入包中混入 NPZ、GDS、PNG 和测试图集代码，4 个文件分别承担诊断生命周期，使输入层职责不纯。

## 3. 实施内容

- 删除稳定 edge/segment key、排序查找索引、外部 update/result 数据类和合并模块；
- 以当前 problem 内全局 segment 下标作为唯一进程内身份；
- `SegmentBatch` 只保留 `edge_normals`、`ring_segment_offsets`、`edge_ids`、`t0`、`t1` 及共享参考几何；
- 删除持久 edge lengths/offsets 和 materialized lengths/indices；
- 删除 boundary sample template/batch，增加求解器与诊断共用的 `edge_probe_points`；
- owner 构造只接受 `RectilinearCoreGrid`，删除没有第二个实现的 policy 抽象；
- `reconstruct_contours/region` 直接接收 `MBOPCProblem`，消除重复配置参数；
- 删除 3 个独立诊断文件，将显式输出合并到 `opc/diagnostics.py`；连同外部更新模块共删除 4 个旧模块；
- `SimpleMBOPCConfig` 删除重复的最大位移与打印阈值权威；
- `QualityMetrics` 删除未使用的 pixel count 及额外设备同步；
- 完整 runner 停止输出 NPZ；该阶段前端验证器为 key-free v2，后续数据契约收敛已升级为 v3。

核心提交：`09f898f refactor(opc): 精简边段输入与诊断架构`，净变化为新增 651 行、删除 1,034 行（其中包含测试、计划与新诊断模块）。

## 4. 后续数据契约收敛

本报告原阶段曾保留 `EdgeBatch`、`OwnershipBatch` 以及 `PhysicalMask/SegmentBatch` 的共享浅引用。后续调用和内存审计证明：浅引用虽不复制数组，仍形成重复的领域权威；`EdgeBatch` 自身五组数组也全部可推导。因此当前结构进一步收敛如下：

| 当前结构 | 保留理由 |
|---|---|
| `PhysicalMask` | 只保存 Layer/Region/ROI，是边段 OPC 与 ILT 的共享物理输入 |
| `ContourBatch` | 两级 CSR 数值拓扑，是 Region 与多轮数值边界之间的必要转换 |
| `SegmentBatch` | 唯一持有 contour，并保存控制段和两个有实测收益的 edge cache |
| `MBOPCProblem` | 直接持有 grid 与 owner/membership CSR，作为完整迭代输入 |
| `SegmentGeometry` | 明确标记短生命周期物化数组，避免混入常驻 batch |
| `SimpleMBOPCConfig/Result/Record` | 分别表达运行输入、最终输出和逐轮审计数据 |

`EdgeBatch`、`OwnershipBatch` 及其公共构造包装已经删除；离线 segment 归档直接升级 v2，不保留 v1 转换。详细数据与验证见 [MB-OPC 数据契约收敛开发报告](mbopc_contract_subtraction_development_report.md)。

## 5. 扩展性结论

架构没有通过空接口预支未来。新边段 OPC 可复用当前输入；更换迭代只需建立新的 `opc/iteration/<method>`。ILT 应复用物理 mask、Region 栅格、光刻和评价，而不强行依赖 segment。出现第二个真实 owner 策略、跨进程更新或 checkpoint 消费者时，再提取最小公共协议。

## 6. Layout/Geometry 授权精简

OPC 减法完成后，用户进一步明确授权修改原受保护目录。调用审计确认并删除 `GeometryEngine`、`UniformGridIndex`、固定 backend 字段/异常、只服务索引的 edge bbox、无调用方的 `DbuBox.overlaps` 糖衣，以及对应模块/测试/基准分支。

同时修复不同入口 ROI 语义不一致：`ShapeQuery.materialize` 现在每 Layer 一次原生精确裁剪。属性模式使用 KLayout `NoPropertyConstraint` 继承原图属性，普通模式使用直接 Region 相交；没有增加逐 polygon Python 循环。

用户 GDS、无关图片和其他工作树内容仍未修改，也没有推送远端。
