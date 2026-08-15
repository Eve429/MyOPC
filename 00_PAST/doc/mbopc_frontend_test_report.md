# MB-OPC 前端测试报告

## 1. 自动测试范围

前端测试覆盖重叠、角接触、孔洞、凹多边形、斜边、负坐标、跨 core 长边、规则网格边界、唯一 owner、halo membership、段长、法向、探针距离、合法/非法重建和全部显式产物。

新增架构回归包括：

- `PhysicalMask` 不再持有数值轮廓，`SegmentBatch` 是 `ContourBatch` 唯一所有者；
- `ContourBatch` 的两级 CSR 可精确恢复多 Polygon 和 hole；
- edge next/polygon 两个缓存必须与 contour 拓扑完全一致；
- `MBOPCProblem` 直接校验唯一 owner 与 membership CSR；
- `SegmentBatch` 不再暴露 key、edge offset 或持久长度；
- probe 距离独立于 corner fragmentation 长度；
- 前端诊断 NPZ 为格式 v3，离线 segment 归档为 v2 且明确拒绝 v1；
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

2026-08-10 使用 1024 nm tile、512 nm halo 的有界实测：Layer 11/0 为 1,776 polygons、21,590 edges、223,553 segments、870 cores、880,801 memberships。完整 problem 常驻 NumPy 数组 9,802,180 bytes，较本次重构前同口径 10,688,650 bytes 减少 886,470 bytes（8.29%）；`SegmentBatch` 自有数组为 5,003,436 bytes。prepare 233.34 ms，同机重构前只读基线为约 266 ms。

同一进程、同一 problem 的 30 次对照显示：模拟旧 `EdgeBatch` 数组访问中位数 28.229 ms，新 nested-next 访问中位数 28.205 ms，P95 31.082 ms，没有物化速度退化。零位移重建 5 次中位数 234.115 ms；XOR、core coverage gap、core overlap 均为 0。

JSON、key-free v3 诊断 NPZ、GDS 和 PNG 均成功生成。旧 v1 离线 segment 归档在检查新字段前即提示重新生成，不存在兼容转换分支。

## 5. 结论

前端完成第二次数据契约减法后，几何结果没有变化，完整问题数组内存下降 8.29%，同口径物化没有速度退化；当前数组下标身份足以支持进程内同步求解。
