# simple MB-OPC 迭代测试报告

## 1. 自动回归

OPC/光刻/评价专项 74 项通过，综合语句/分支覆盖率 92%；授权完成 Layout/Geometry 减法后，全仓库 114 项通过。覆盖内容包括 ICCAD13 CPU/CUDA、OpenILT patch 对齐、full-canvas 修复、process-window ownership、EPE 方向、target LRU、同步 batch、唯一 owner、非法拓扑回滚和 CLI 外部工作目录直接运行。

## 2. 关键场景

| 场景 | 预期与结果 |
|---|---|
| 同轮跨 core | 所有 batch 读取同一 `current`，仅 owner 写 `next_values`；通过 |
| 2 nm 中空壁、8 nm probe | 穿壁长边 probe 无效，局部角段按 target 语义判断；通过 |
| 外轮廓越入孔洞 | hole 关系失败，整轮回滚；通过 |
| 矩形左边越过右边 | ring 绕向改变，整轮回滚；通过 |
| 斜边跨多个 core | 最终全局重建连续，无 core 裁剪损失；通过 |
| inner/outer 冲突 | ambiguous 增加，方向为 0；通过 |
| target 缓存命中 | 首次/命中均为 `[0,1]` 且数值一致；通过 |

## 3. 完整 `gcd_45nm` CUDA 流程

配置处理 870 cores、223,553 segments、880,801 context memberships，运行 3 轮：

| 轮次 | EPE | L2 | PVBand | valid probes | ambiguous | moved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129,645 | 1,038,629.522 | 115,626.751 | 223,298 | 51 | 129,594 |
| 1 | 74,592 | 563,335.522 | 134,540.869 | 223,298 | 5 | 74,587 |
| 2 | 48,348 | 440,251.431 | 147,186.806 | 223,298 | 2 | 48,346 |

EPE 与 L2 下降；PVBand 上升，已原样记录。结果与重构前历史运行数值一致。

该轮次历史运行的常驻 segment 数组为 4,830,716 bytes；后续数据契约收敛加入两个必要 edge cache 后为 5,003,436 bytes，完整 problem 数组为 9,802,180 bytes。GPU 历史峰值分配 271,544,320 bytes；结果 Region 合法，GDS/JSON/PNG 成功生成，且没有全流程 NPZ。

## 4. 内存结论

当前实测设备只有 4 GiB 显存仍能完成该版图；算法不会把整张 reticle 曝光 tensor 常驻 GPU。24 GiB GPU、64 GiB RAM 的目标机器有更大 batch/缓存余量，但最大可处理 reticle 仍取决于 segment/轮廓 CPU 常驻数组和运行时间，不能由单个样例宣称无限规模。

## 5. 最终结论

架构减法没有改变 3 轮数值结果或 GPU 峰值；它显著降低输入常驻内存并改善前端耗时。同步轮次、跨 core 唯一 owner 和拓扑回滚均由回归测试覆盖。
