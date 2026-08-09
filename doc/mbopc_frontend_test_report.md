# MB-OPC 前端测试报告

## 1. 自动测试范围

前端测试覆盖重叠、角接触、孔洞、凹多边形、斜边、负坐标、跨 core 长边、规则网格边界、唯一 owner、halo membership、段长、法向、探针距离、合法/非法重建和全部显式产物。

新增架构回归包括：

- `PhysicalMask` 与 `SegmentBatch` 共享轮廓/边对象，不发生深复制；
- `SegmentBatch` 不再暴露 key、edge offset 或持久长度；
- probe 距离独立于 corner fragmentation 长度；
- 前端 NPZ 为格式 v2 且不含 `segment_keys`；
- 完整 runner 不生成 NPZ。

## 2. 严格合成基准

命令：

```powershell
& $python benchmarks\benchmark_mbopc_frontend.py --strict
```

阶段 28 最终结果：5,000 shapes、20,000 mathematical edges、110,000 segments；prepare 125.64 ms，materialize 12.45 ms，零位移重建 427.83 ms；persistent arrays 2.441 MiB，展开表示 7.973 MiB，节省 69.38%；XOR=0、unowned=0、严格失败列表为空。

## 3. 多图形专项

图集覆盖：`overlap_corner_touch`、`orthogonal_concave`、`negative_cross_core`、`hole_overlap`、`diagonal_angles`。生成文件位于 `doc/images/mbopc/`，每张图标出 segment、owner、core、外法向以及 inner/outer probe。

专项结论：重叠内边消失；角接触没有被错误合并；孔洞保留相反 ring 语义；负坐标 owner 正确；斜边跨 core 连续；零位移重建均与参考 Region 一致。

## 4. 真实 `gcd_45nm` 验证

Layer 11/0：1,776 polygons、21,590 edges、223,553 segments。常驻 segment 数组 4,830,716 bytes，较旧实现 12,675,300 bytes 减少 61.89%；prepare 152.82 ms，总诊断流程 2.308 s；零位移 XOR、core coverage gap、core overlap 均为 0。

JSON、key-free v2 NPZ、GDS 和 PNG 均成功生成。人工查看 `gcd_45nm_overview.png`，左右 owner 分区清楚，跨 core 图形连续，inner/outer 点与法向方向一致。

## 5. 结论

前端在删除稳定身份与外部更新链后，几何结果没有变化，准备、物化、重建和内存指标均改善；当前数组下标身份足以支持进程内同步求解。
