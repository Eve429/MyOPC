# MB-OPC 前端容量预检与结构收敛测试报告

## 1. 自动化结果

环境：Python 3.12.0、KLayout 0.30.10、NumPy 2.5.1、Windows 10。

| 检查 | 结果 |
|---|---:|
| Ruff（限定第一方源码、入口和 tests） | 通过 |
| compileall | 通过 |
| 全仓库 pytest | 141 passed，26.08 s |
| 阶段聚焦回归 | 28 passed，20.45 s |
| Geometry/OPC 聚焦 branch coverage | 108 passed，综合 92% |
| `git diff --check` | 通过 |

## 2. 功能矩阵

- Raster：孔洞、部分像素、空 Region、负坐标、上下方向、固定 canvas 超限均保持原结果；展示输出仍为顶部朝上 `uint8`，模型输出仍为左下原点 `float32`。
- 类型归位：包级导入、边段准备、ownership、重建、solver、诊断和离线 v2 归档全部通过；旧深层路径无第一方残留引用。
- Preflight：小版图完整扫描并接受；低预算在 Region 物化前拒绝；百亿切分估算不构造 SegmentBatch；`int32`、内存预算、扫描下界和推荐模式字段正确。
- 入口：frontend 的 `preflight_only/rejected/completed` 三状态、`skip_artifacts`、详细阶段统计均覆盖；正式 solver 的 preflight-only 通过 monkeypatch 证明未加载光刻模型。
- 离线工作台：像素/边段准备、读取、损坏归档、物化前保护和真实模型入口共 11 项通过。

## 3. 百亿容量测试

测试 GDS 只包含一个超长矩形，使用 0.2 DBU 最大段长使预检估算至少 10,000,000,000 个 segment。结果满足：

- `accepted=false`；
- `int32_capacity_ok=false`；
- `scan_complete=false` 且 `counts_are_lower_bounds=true`；
- `recommended_mode=sharded_required`；
- 测试过程中没有调用 `fragment_edges`，也没有按 segment 数分配 NumPy 数组。

该测试验证的是容量保护，不表示当前版本能求解百亿边段。

## 4. Raster 性能

严格 2048×2048 基准：416.94 ms、RSS 增量 7.62 MiB、覆盖率精确，`strict_failures=[]`。项目最近同机基线为 483–502 ms，本次公共底层复用未造成 10% 退化，反而更快。

## 5. `gcd_45nm` 真实验证

Layer 11/0、1024 nm core、512 nm halo：

| 指标 | 结果 |
|---|---:|
| occurrence / 原始顶点 | 1,776 / 21,590 |
| 预检 segment / membership 上界 | 223,553 / 1,167,992 |
| 实际 segment / membership | 223,553 / 880,801 |
| prepare 峰值估算 | 73,809,488 bytes |
| problem prepare | 212.72 ms |
| 完整前端总耗时（跳过产物） | 1.226 s |
| 进程 peak working set | 148,467,712 bytes |
| 零位移 XOR / core 缺口 / core 重叠 | 0 / 0 / 0 |

只预检模式总耗时约 0.33 s，未进入 Region 或 SegmentBatch 物化。完整前端计数、9,802,180-byte problem 常驻数组和几何结果均与重构前基线一致。

## 6. 已知边界

- 估算使用安全系数，不承诺等于操作系统峰值；实际运行同时报告进程内存用于校准。
- 预检基于源 occurrence，KLayout 布尔合并新增交点无法在物化前精确预测，因此采用保守内存系数。
- 当前超限输入只能安全拒绝；CPU macro、磁盘 shard 和跨 macro 更新属于阶段 2，未测试为当前能力。
