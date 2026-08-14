# 当前规则符合性修正开发报告

## 1. 目标与范围

本轮针对当前审查中已经有源码证据的问题做最小修正，目标是同时降低错误概率、GPU 同步开销和无效结构。实施范围包括 MB-OPC EPE 坐标、DiffOPC 软栅格热路径、边段参数单位换算、Macro 前端重复校验、无调用异常、项目元数据和当前文档事实源。

明确不处理：preflight 的既有入口行为、`main/` coverage、`MBOPCProblem`/归档改名、历史生成产物和未来 macro shard。`layout/`、`geometry/` 没有修改；用户已有 `.vscode/launch.json`、`config/mbopc.toml` 修改始终排除在提交之外。

## 2. 数值与接口修正

### 2.1 MB-OPC probe 坐标

左下原点 raster 的数组索引 0 表示第一个像素中心，而不是像素左下角。求解器原公式 `(probe-origin)/pixel_dbu` 整体偏移半格，现统一为：

```text
pixel_index = (probe_dbu - context_origin_dbu) / pixel_dbu - 0.5
```

该改动只发生在 EPE 评价输入边界，不改变 probe 的物理定义、边段坐标或 raster 方向。8 DBU 单像素矩形的端到端回归证明旧公式得到 0 个有效探针，修正后得到 10 个。

### 2.2 分段 nm→DBU

`main.configuration.fragmentation_dbu` 统一三个真实调用方：`run_mbopc.py`、`run_mbopc_frontend.py`、`offline_inputs.py`。corner 和最大 segment 长度决定实际分段端点，必须用 `exact_dbu` 严格落在整数格点；最大允许位移是连续优化上限，保留小数 DBU。没有把 helper 放入 OPC 基础层，也没有建立配置对象或注册器。

## 3. 性能修正

`rasterize_soft_edges` 保留正参数、浮点 Tensor、shape 和 device 等廉价结构检查，删除对已准备内部数组重复执行的有限性、正长度和单位法向设备 value 检查。原检查共包含 7 次 `.item()`，每个 tile 都会强制 CUDA 与 CPU 同步。

同一进程、GTX 1650、256² 空 tile、预生成像素中心、200 次调用的等效对照为：旧同步检查 1.844 ms/次，当前实现 0.288 ms/次，约 6.4 倍。该数字隔离的是热路径校验开销，不代表含光刻模型的整轮 OPC 加速比。

## 4. 精简审计

- 删除 `MacroPreparation.__post_init__`：对象只有 `prepare_macro` 一个构造点，全部数组刚由内部批量路径按正确 dtype/shape 生成，第二遍扫描没有外部输入可保护。
- 明确 Macro 的 segment 下标只在当前 `SegmentBatch` 内有效；仅 tile ID 是全局 ID。
- 删除没有抛出、捕获或测试调用方的 `OwnershipError`；现有 owner/membership 不变量继续直接抛 `ValueError`。
- 删除 `run_mbopc.py` 两组重复关键步骤注释，没有增加包装函数。
- 项目元数据名称更新为 `myopc`，描述覆盖当前 lithography、OPC 和 ILT。

本轮新增的唯一共享函数有三个当前调用方；两个新增回归直接覆盖生产错误/性能退化，没有为测试增加生产分支。核心代码提交为 `1aaf413`。

## 5. 文档治理

[`development_manual.md`](development_manual.md) 明确为当前架构事实源，[`module_interface_reference.md`](module_interface_reference.md) 明确为当前 API 事实源；[`项目开发手册.md`](项目开发手册.md) 保留为历史导航。阶段报告继续保留实施证据，但不覆盖当前源码事实。
