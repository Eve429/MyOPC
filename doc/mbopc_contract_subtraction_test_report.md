# MB-OPC 数据契约收敛测试报告

## 1. 测试结论

结构迁移后的 Ruff、compileall 和全仓库回归全部通过；`gcd_45nm` 有界整图前端计数、归属、重建和 core 覆盖保持正确。没有修改用户 GDS，没有把临时基准产物加入 Git。

## 2. 自动测试

```text
迁移聚焦测试：59 passed in 14.66 s
owner/v1 专项：23 passed in 12.60 s
全仓库 pytest：130 passed in 26.50 s（最终交付复跑）
Ruff：All checks passed
compileall：passed
CUDA 直接环境单测：1 passed in 9.56 s
```

覆盖场景包括：

- 空 Region、单 Polygon、多 Polygon、单/多 hole 和两级 CSR 往返；
- 重复点、零长度边、零面积 ring；
- 矩形、凹图形、斜边、负坐标、层级引用和旋转/镜像；
- 数学边分段长度、法向、独立 segment jog、miter/bevel；
- 一个边段跨一个或多个 core、唯一 owner、halo 多 membership；
- 2 DBU 中空壁配 8 DBU probe、外轮廓越入孔洞、矩形对边交叉；
- 同轮只读 `current`、owner-only 写入和轮次屏障；
- segment v2 往返、缺 cache、越界 membership、损坏 metadata 和真实 v1 拒绝；
- 前端 v3 诊断、根 CLI、离线光刻与离线 MB-OPC。

## 3. 回归问题及处理

首次聚焦运行 58/59 通过，唯一失败是 membership 已被新 `MBOPCProblem` 更早拒绝，但英文错误不匹配加载器的中文异常契约。处理方式是让 Problem 成为唯一范围校验权威并输出明确中文原因，没有在加载器增加重复分支。

首轮全仓库为 126/129 通过：两项测试仍引用旧诊断版本或旧 `PhysicalMask.contours`，已直接迁移；一个独立 CUDA 子进程临时报告设备 busy，进程退出后单独复跑和最终全量均通过，未修改光刻代码。

## 4. `gcd_45nm` 有界实测

配置：Layer 11/0、1024 nm tile、512 nm halo、16/32/24 nm 分段参数。输出写入系统临时目录，每个步骤只保存标量统计或显式最终产物，没有保留 870 份 tile 中间数组。

| 指标 | 结果 |
|---|---:|
| polygons / rings / edges | 1,776 / 1,776 / 21,590 |
| segments / cores / memberships | 223,553 / 870 / 880,801 |
| prepare | 233.339 ms |
| 完整 problem 常驻数组 | 9,802,180 bytes |
| 重构前同口径 | 10,688,650 bytes |
| 内存减少 | 886,470 bytes / 8.29% |
| SegmentBatch 自有数组 | 5,003,436 bytes |
| 零位移重建中位数（5 次） | 234.115 ms |
| XOR / core gap / core overlap | 0 / 0 / 0 |

30 次同进程物化对照：

| 访问方式 | 中位数 | P95 |
|---|---:|---:|
| 旧 EdgeBatch 起终点数组 | 28.229 ms | 35.407 ms |
| 新 contour + edge_next_ids | 28.205 ms | 31.082 ms |

新路径中位数没有退化，P95 更低。prepare 也低于本次修改前同机约 266 ms 的只读基线；不同历史阶段的 Python/系统负载不同，不混用 152.82 ms 旧报告作为同口径门槛。

## 5. 内存失败复盘

第一次探索性基准错误保留全部 870 个 core 的轮廓子集，多轮后触发 OpenBLAS 分配失败；紧接着启动 PyTorch 又遇到 WinError 1455 页面文件不足。正式测试改为逐次覆盖局部结果、只累计标量，并把 NumPy 几何基准与 CUDA 测试分进程执行。最终全量和 CUDA 均通过。

## 6. 交付审计

- `git diff --check`、旧符号搜索和公共调用点搜索通过；
- 76 个第一方 Python 文件的中文模块/函数 docstring 检查零缺失；20 份 `doc/*.md` 没有断链或不平衡代码围栏；
- 生产定义的零生产引用名称只有 `render_layout_region`、`hierarchy_summary`、`DbuBox.intersection` 和 `ICCAD13.forward`：前三者是有直接回归的公开接口，最后一个是 PyTorch 框架回调，不属于可删除死代码；
- 所有关键拓扑、内存和归属路径保留详细中文注释；
- `layout/` 未修改；`geometry/` 仅包含用户授权的轮廓契约收敛和 `edge.py` 删除；
- `doc/design_review.md` 保持历史原文；
- 用户 GDS 和 `output/mbopc/*` 未进入功能提交；
- 关键代码提交已本地创建，未推送远端。
