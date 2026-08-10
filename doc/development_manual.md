# MyOPC 开发手册

函数级入口、调用顺序和数据生命周期见 [MyOPC 函数调用关系与数据流](function_call_architecture.md)。

## 1. 设计目标

MyOPC 面向整张 reticle 的流式 OPC：版图数据库只在输入阶段读取一次，物理边界只构造一次；迭代阶段按 core+halo 栅格化和批量执行光刻模型，不保存整张 reticle 的曝光张量；轮次结束后统一发布边段位移，最终只做一次全局矢量重建。

项目可直接运行根目录 Python 文件，不需要 `pip install` 当前仓库。当前依赖方向是：

```text
layout -> geometry -> opc.input -> opc.iteration.mbopc
                       |                    |
                       +-> diagnostics      +-> lithography + evaluation
```

`layout/` 与 `geometry/` 默认是受保护基础，新增 OPC 功能不得擅自修改；本次精简是在用户明确授权后执行，并留下 ROI/属性/性能回归。

## 2. 目录职责

| 路径 | 当前职责 |
|---|---|
| `layout/` | 层级版图加载、Layer/ROI 查询 |
| `geometry/` | Region、两级 CSR 轮廓、栅格化、输出 patch |
| `opc/input/` | 物理 mask、规则 core 网格等共享输入 |
| `opc/input/edge/` | 边段切分、唯一 owner、探针坐标、全局矢量重建 |
| `opc/iteration/mbopc/` | simple MB-OPC 的流式同步迭代 |
| `lithography/` | 可独立替换的光刻模型；当前为 ICCAD13 Hopkins 模型 |
| `evaluation/` | L2、PVBand、EPE 评价 |
| `opc/diagnostics.py` | 显式请求才执行的 NPZ/GDS/PNG 与几何图集 |
| `run_mbopc_frontend.py` | 不运行光刻的输入、分段、归属、重建验证器 |
| `run_mbopc.py` | 完整 MB-OPC 主程序 |

诊断代码不属于输入模型，求解器不反向依赖某个输出格式。未来 ILT 可复用版图、Region 栅格、光刻和评价层，但不必依赖边段重建。

## 3. 当前核心数据契约

### 3.1 `PhysicalMask`

只保存规范化后的原生 `Region`、查询框和 Layer，是 MB-OPC、ILT 等方法可以共享的最小物理输入。它不提前提取轮廓或数学边，避免不使用边段的方法承担无关时间和内存。

### 3.2 `ContourBatch`

由 `SegmentBatch` 唯一持有，使用 `vertices`、`ring_offsets` 和 `polygon_ring_offsets` 两级 CSR 表达 `Polygon -> ring -> vertex`。每个 Polygon 范围的首 ring 是 hull，后续 ring 是 hole，因此不再重复保存 Layer、逐 ring Polygon ID 和 hole 布尔列。

### 3.3 `SegmentBatch`

只常驻求解所需字段：

- `contours`：固定参考轮廓的唯一数值所有者；
- `edge_next_ids`、`edge_polygon_ids`：经实测保留的两个 `int32` 热路径缓存；
- `edge_normals`、`ring_segment_offsets`：边法向和每个 ring 的分段范围；
- `edge_ids`、`t0`、`t1`：每段在固定参考数学边上的索引和参数区间。

边段身份就是当前 problem 内的全局数组下标。已经删除无当前调用方的稳定 key、排序查找表、外部更新批次、持久化边长和边偏移表。`materialize(displacements)` 一次性用 NumPy 生成所有当前端点和法向；诊断需要长度时临时计算，不进入迭代常驻内存。

### 3.4 `MBOPCProblem`

聚合 `PhysicalMask`、`FragmentationConfig`、`SegmentBatch`、紧凑 `RectilinearCoreGrid` 以及 owner/membership CSR。`owner_indices[i]` 是 segment `i` 的唯一写入 core；`core_offsets/member_segment_indices` 表达 owner 与只读 halo context。展开的 `CoreSpec` 只在 solver 或诊断明确需要时生成一次，不再常驻 problem。

## 4. 输入构造和重建

`prepare_problem(batch, layer, config, grid)` 顺序执行：

1. `normalize_physical_mask` 合并重叠、清理属性并恢复合法孔洞；
2. `extract_contour` 一次生成两级 CSR 数值轮廓；
3. `fragment_edges` 生成最小边缓存、法向和参数化控制段；
4. 内部 `_build_ownership` 用规则网格批量计算唯一 owner 和 halo membership；
5. 返回只读参考 problem，不预生成图片、文件或探针缓存。

内部归属构造只需要参考端点和中点，不需要迭代阶段的逐段法向。实现直接由
`contours + edge_next_ids + edge_ids + t0/t1` 向量化生成端点，并在展开 CSR membership 前释放端点临时表；
不得为此调用 `SegmentBatch.materialize()`，否则会额外复制法向并抬高准备阶段峰值内存。

`edge_probe_points(starts, ends, normals, distance_dbu)` 是求解器和诊断共用的唯一探针坐标实现：以当前 segment 中点为基准，`inner = midpoint - normal * distance`，`outer = midpoint + normal * distance`。探针距离来自迭代配置，不与角段长度绑定。

`reconstruct_contours(problem, displacements)` 从固定参考边和位移生成 ring；相邻段位移不同时生成 jog，拐角优先解析 miter，超限时使用 bevel。`reconstruct_region` 再验证 ring 和孔洞关系并返回全局 Region。core 只分配计算和更新权，不裁剪最终矢量，因此跨多个 core 的斜边不会因多次整数裁剪出现 33/34 DBU 接缝差异。

## 5. simple MB-OPC 迭代

`optimize` 在 CPU 保存全局 `current`/`next_values` 位移；每个 batch 仅把当前 core+halo 的 mask、target 和 ownership 像素送到设备。处理流程为：

1. 本轮所有 batch 只读同一个 `current`；
2. 模型输出立即累计本 core 的 L2/PVBand/EPE 和 owner 更新，然后释放 tile tensor；
3. halo 只提供光学上下文，不累计指标、不写边；
4. 全部 batch 完成后验证全局候选轮廓；只有合法时才把 `next_values` 发布为下一轮 `current`；
5. 保存最佳一维位移，结束后只做一次全局重建。

因此“立即累计”只累计数值，不会提前移动参考边。跨 core 的同一边段只有一个 owner，其他 core 即使在 halo 中看到它也无写权限。

求解开始时的零位移状态直接共享 `SegmentBatch.contours`，不执行一次无意义的全局
`reconstruct_contours`。某个 core 的全部 context segment 仍为精确零位移时，
`_current_tile` 直接栅格化固定参考 Region，跳过局部 contour 子集和 Region 差分。
target 缓存为降低常驻内存使用 `uint8`，而 current mask 保留浮点面积覆盖率，因此快路
不能直接把量化 target 当作 current mask；这一数值边界有专门回归保护。

拓扑保护会拒绝 ring 绕向翻转和 hole 逃逸。当前 v1 采用整轮回滚，避免发布局部损坏图形；没有为假设中的局部恢复预建接口。

## 6. 光刻与评价替换

`ICCAD13Lithography` 独立位于 `lithography/`，迭代只依赖其批量 nominal/maximum/minimum 输出。新模型应保持当前张量坐标、画布和设备语义，不把模型细节写回输入层。

`evaluate_process_window` 只在 ownership 像素累计 L2/PVBand。`evaluate_edge_probes` 根据 target 的内外语义产生 `-1/0/+1` 法向移动方向；同一 probe 同时触发相反要求时记为 ambiguous 且不移动。更换迭代算法可以复用评价函数，也可以在独立方法目录实现新损失，但不能让公共输入依赖具体算法。

## 7. 输出约定

- `run_mbopc.py`：保存 `summary.json`、结果 GDS 和可选 PNG；不保存整轮 tensor，也不生成 NPZ。
- `run_mbopc_frontend.py`：用于人工检查输入契约，保存 key-free、按全局 segment 下标对齐的格式 v3 NPZ，并可保存 GDS/PNG/JSON。
- `opc.diagnostics`：只有调用者明确要求时才物化诊断长度、图片、GDS 或测试图集。

NPZ 是当前进程中 problem 的快照，不是跨 remesh、跨版本的持久身份协议。显式 remesh 必须重新分段、重新建立 owner，并由调用者重建优化状态。

## 8. 扩展原则

- 新抽象必须有当前调用方；不创建空接口、注册器或无实现目录。
- 替换输入构造和替换迭代是两个独立扩展点：前者产出实际方法需要的数据，后者消费数据并更新状态。
- ILT 优先复用 `layout`、`geometry`、`PhysicalMask`、栅格、`lithography` 和 `evaluation`；不要为了复用而套用 MB 边段结构。
- 保留层级与局部 ROI，跨 Python/KLayout 边界必须批处理；迭代数据要缓存，诊断默认关闭。
- 每个 bug 必须有回归测试，修复后搜索并删除仅服务于旧错误的包装、变量和分支。
- 关键节点只做本地 Git commit，未经明确授权不得 push。

## 9. 离线专项测试工作台

`tests/workbench/offline_inputs.py` 把“版图输入构造”和“模型/迭代优化”解耦。它提供四个稳定测试接口：

- `prepare_raster_input`：把一个明确 ROI 保存成模型左下原点的 `float32[canvas,canvas]`；
- `load_raster_input`：校验版本、方向、范围和数值后读取 mask；
- `prepare_segment_input`：一次性完成物化、规范化、切分和 owner/context 构造；
- `load_segment_input`：恢复现有 `MBOPCProblem`，不创建第二套 problem 类型。

像素归档只对应一个可直接送入模型的 canvas，超限 ROI 必须缩小，不隐式切 tile。边段归档是可恢复的 version 2 输入协议，保存两级 contour CSR、两个 edge cache、segment 数组、grid cuts 和 membership CSR；version 1 明确提示重新生成，不保留转换分支。它与 `opc.diagnostics.save_problem_npz` 的不可恢复 v3 诊断快照完全分离。

准备前先检查源文件、像素尺寸、层级展开图形/顶点和保守内存估计。严格预检有意额外读取一次版图：第一次只扫描并拒绝危险输入，第二次才通过现有 `LayoutDB` 公共接口物化。布尔合并可能产生的新交点无法在物化前完全预测，因此准备完成后还会按真实 segment/membership 数量复核内存估计。

`tests/workbench/run_lithography.py` 只消费像素归档，输出三工艺角连续数组和可选 PNG；`tests/workbench/run_mbopc_iteration.py` 只消费边段归档，输出最佳位移、迭代 JSON、GDS 和可选标注图。两个入口均可从任意工作目录直接运行，不需要安装本项目。

详细字段、内存边界和设计审计见 [离线工作台开发报告](offline_workbench_development_report.md)。
