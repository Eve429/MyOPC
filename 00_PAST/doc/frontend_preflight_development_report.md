# MB-OPC 前端容量预检与结构收敛开发报告

## 1. 目标与边界

本阶段解决三个现实问题：重复 raster 底层实现、分散且命名模糊的 `types.py`、以及大版图在任何容量判断前就物化完整 Region/边段。实现范围只到“精确扫描、保守估算、安全拒绝”；CPU macro、磁盘 shard、跨 macro 身份和流式 epoch 属于后续阶段，当前没有伪装为已支持能力。

`layout/` 未修改。`geometry/` 只增加两个现实调用方共用的覆盖率分块生成器，并把三组只在单一几何操作中使用的结构体归回对应模块。

## 2. Raster 设计

两种上层 raster 保持独立：

- `geometry.render_region_batch`：可变图幅、左下原点、`uint8`；PNG/显示边界再翻转；
- `opc.input.rasterize_region_canvas`：固定方形、左下原点、`float32` 和零 padding，服务模型。

共同的 Region 裁剪、合并、原生面积栅格化、归一化和临时分块由 `geometry.iter_region_coverage_tiles` 完成。显示调用逐块翻转、量化，不产生整图 float64；OPC 调用逐块写入固定 float32 canvas。这样消除了重复逻辑，同时没有把模型坐标语义塞进通用 geometry。

## 3. 类型与文件收敛

- `ContourBatch`、`GeometryPatch`、`ValidationIssue/Report` 分别放回 contour、patch、validate；删除 `geometry/types.py`。
- `CoreSpec/RectilinearCoreGrid` 归入表达职责的 `opc/input/grid.py`。
- `FragmentationConfig/SegmentGeometry/SegmentBatch` 与切分操作同置于 fragmentation。
- `MBOPCProblem` 与唯一构造入口 `prepare_problem` 同置于 builder。
- 迭代配置/记录/结果改名为 `contracts.py`，避免与输入数据类型混淆。
- 两份重复 `_vector` 和相关矩阵/点校验合并为输入层私有 `_arrays.py`；没有建立全局 `utils.py`。

包级公开导入保持不变；已删除的深层内部路径不保留空兼容包装。文件调整后删除 4 个泛化 `types.py`，新增 3 个职责文件和一个有多个实际调用方的私有数组模块。

## 4. 生产预检

`opc.input.preflight_layout` 使用 KLayout 层级迭代器扫描指定 top/Layer/ROI，只在循环内短暂保存当前 polygon 的顶点。它按与生产切分一致的公式统计数学边对应 segment，并按 edge 扩展 bbox 估算 context membership 上界。

摘要包含：源文件字节、occurrence、顶点、segment、membership、prepare/solver 峰值、预算、`int32` 容量、扫描是否完整和推荐模式。默认预算为程序启动时系统可用内存的 70%。当统计下界已经超过预算或 `int32` 时立即停止，标记 `counts_are_lower_bounds=true`，避免只为得到更大的拒绝数字继续扫描海量 occurrence。

当前全局路径每个 segment 还需要边段索引/参数、owner、迭代三状态、写入检测和参考几何。百亿 segment 的状态量远高于 42.4 GiB，并超过局部 `int32` 身份容量，因此本阶段返回 `sharded_required` 和退出码 2，不尝试分配。

## 5. 运行入口与性能统计

`run_mbopc_frontend.py` 新增：

- `--preflight-only`：真实版图只扫描，不物化 Region/SegmentBatch；
- `--memory-budget-gib`：覆盖默认 CPU 预算；
- `--skip-artifacts`：完成几何验证但不生成 NPZ/GDS/PNG/图集；
- `--top-cell`：显式选择多顶层版图。

阶段统计覆盖 layout open、preflight、ROI materialize、problem prepare、演示更新、端点/probe、重建、验证及四类产物。每个检查点记录 RSS、USS、private、peak working set 和系统可用内存；NumPy/KLayout 的原生分配不使用 `tracemalloc` 代替进程统计。

演示 owner 选择同时由 `core × global segment` 扫描改为过滤当前 membership CSR。`run_mbopc.py` 复用同一预检，保证正式求解不能绕过保护；`--preflight-only` 在光刻模型加载前返回。

## 6. 简化审计

- 没有增加统计 dataclass、preflight problem 类型、注册器或阶段 2 空接口；摘要使用普通字典。
- 离线工作台删除原有的 KLayout 扫描、切分计数、membership 与内存估算重复实现，直接调用生产 preflight。
- 三个入口重复的固定物理 tile cuts 已收敛为 `grid.axis_cuts_by_size`；数量均分 cuts 语义不同，仍只留在 frontend。
- 全仓库搜索确认已删除 `geometry.types`、`opc.input.types`、`edge.types`、`mbopc.types` 深层引用。
- Stage 2 仍只保留设计文档，当前代码不声称百亿边段可以完成求解。
