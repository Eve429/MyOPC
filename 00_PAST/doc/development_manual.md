# MyOPC 开发手册

本文是项目**当前架构事实源**；架构职责、数据所有权、性能边界和扩展约束以本文及当前源码为准。逐函数 API 事实以[模块输入输出接口参考](module_interface_reference.md)为准，阶段报告只保存当时的实施证据。

当前全项目的结构、冗余与模块边界评审见[当前架构与精简性评审](current_architecture_review.md)。该文档区分确定问题、合理复杂度和需要用户授权的受保护目录改动；历史阶段报告不替代此当前结论。

Layout 层级接口的本轮精简、性能边界和删除项见[Layout 层级接口轻量化开发报告](layout_hierarchy_simplification_development_report.md)。

逐模块输入输出、数组 shape/dtype、坐标单位、异常和文件协议见[模块输入输出接口参考](module_interface_reference.md)；函数级入口、调用顺序和数据生命周期见[函数调用关系与数据流](function_call_architecture.md)。

## 1. 设计目标

MyOPC 面向整张 reticle 的流式 OPC：版图数据库只在输入阶段读取一次，物理边界只构造一次；迭代阶段按 core+halo 栅格化和批量执行光刻模型，不保存整张 reticle 的曝光张量；轮次结束后统一发布边段位移，最终只做一次全局矢量重建。

项目可直接运行 `main/` 中的 Python 文件，不需要 `pip install` 当前仓库。当前依赖方向是：

```text
layout -> geometry -> opc.input -> opc.input.edge -> opc.iteration.mbopc
                       |                              |
                       +-> raster --------------------+-> lithography + evaluation
                                                      |
                                                      +-> opc.iteration.ilt
```

`layout/` 与 `geometry/` 默认是受保护基础，新增 OPC 功能不得擅自修改；本次精简是在用户明确授权后执行，并留下 ROI/属性/性能回归。

## 2. 目录职责

| 路径 | 当前职责 |
|---|---|
| `layout/` | 层级版图加载、轻量 Cell 邻接查询、Layer/ROI 查询 |
| `geometry/` | Region、两级 CSR 轮廓、栅格化、输出 patch |
| `opc/input/` | 物理 mask、规则 core 网格等共享输入 |
| `opc/input/edge/` | 边段切分、唯一 owner、探针坐标、全局矢量重建 |
| `opc/iteration/mbopc/` | simple MB-OPC 的流式同步迭代 |
| `opc/iteration/ilt/` | 四种 ILT 求解器及包内共享张量操作；均不依赖边段输入 |
| `lithography/` | 最小模型 Protocol 与 ICCAD13 Hopkins 实现 |
| `evaluation/` | 二值 L2、PVBand、EPE 和确定性矩形 shot 估计 |
| `opc/diagnostics.py` | 显式请求才执行的 NPZ/GDS/PNG 与几何图集 |
| `main/artifacts.py` | 原子 JSON/NPZ/PNG 与完整/流式最终光刻产物 |
| `main/offline_inputs.py` | 可复用像素/边段物化、归档校验和准备 CLI |
| `main/run_mbopc_frontend.py` | 不运行光刻的输入、分段、归属、重建验证器 |
| `main/run_mbopc.py` | 完整 MB-OPC 主程序 |
| `main/run_lithography.py` | 独立光刻模型验证入口 |
| `main/run_mbopc_iteration.py` | 独立 MB-OPC 迭代验证入口 |
| `main/run_simpleilt.py` | 保留历史参数/返回值的 SimpleILT 适配入口 |
| `main/run_ilt.py` | 统一 ILT 入口；LevelSet、CurvMulti、Multilevel 均已验收 |
| `main/run_diffopc.py` | DiffOPC 入口；直接读取 GDS/OASIS 或 segment NPZ，保存最佳几何与流式最终光刻结果 |

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

`segments_for_core(i)` 返回该 core 的 owner+halo 只读 membership 视图；`owner_segments_for_core(i)` 只返回唯一可写 segment。两个边段求解器都复用该查询，不再各自复制全局 owner 过滤。真实版图入口把预检推导出的 `max_memberships` 传给 `prepare_problem`，归属构造会在 `np.repeat` 等大数组分配前拒绝超限；省略上限仍仅适用于调用方已确认可放入内存的小问题。

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

公共 raster 的像素 `[row=0,column=0]` 中心位于 `box.left/bottom + 0.5*pixel_dbu`。因此 MB-OPC 把全局 probe 转成数组连续索引时统一使用 `(probe-origin)/pixel_dbu-0.5`；该半像素偏移是坐标定义，不是插值补偿。

`reconstruct_contours(problem, displacements)` 从固定参考边和位移生成 ring；相邻段位移不同时生成 jog，拐角优先解析 miter，超限时使用 bevel。`reconstruct_region` 再验证 ring 和孔洞关系并返回全局 Region。core 只分配计算和更新权，不裁剪最终矢量，因此跨多个 core 的斜边不会因多次整数裁剪出现 33/34 DBU 接缝差异。

## 5. simple MB-OPC 迭代

`optimize` 在 CPU 保存全局 `current`/`next_values` 位移；每个 batch 仅把当前 core+halo 的 mask、target 和 ownership 像素送到设备。处理流程为：

1. 本轮所有 batch 只读同一个 `current`；
2. 模型输出立即累计本 core 的 L2/PVBand/EPE 和 owner 更新，然后释放 tile tensor；
3. halo 只提供光学上下文，不累计指标、不写边；
4. 全部 batch 完成后验证全局候选轮廓；只有合法时才把 `next_values` 发布为下一轮 `current`；
5. `iterations=N` 最多发布 N 次合法同步更新；初态和每次更新后的状态都会评价，最佳一维位移只来自已评价状态，结束后只做一次全局重建。

因此“立即累计”只累计数值，不会提前移动参考边。跨 core 的同一边段只有一个 owner，其他 core 即使在 halo 中看到它也无写权限。

Geometry 展示与 OPC/ILT 模型的公共 raster 返回数组都使用左下原点；PNG、查看器和诊断标注只在图片输出边界上下翻转。已物化 `RegionBatch` 也可在 `LayoutDB` 关闭后继续使用，只有尚未执行的惰性 `ShapeQuery` 依赖打开的数据库。

求解开始时的零位移状态直接共享 `SegmentBatch.contours`，不执行一次无意义的全局
`reconstruct_contours`。某个 core 的全部 context segment 仍为精确零位移时，
`_current_tile` 直接栅格化固定参考 Region，跳过局部 contour 子集和 Region 差分。
target 缓存为降低常驻内存使用 `uint8`，而 current mask 保留浮点面积覆盖率，因此快路
不能直接把量化 target 当作 current mask；这一数值边界有专门回归保护。target 在
CPU batch 中始终保持 `uint8`，只在一次性送到模型设备时转为 `float32` 并除以 255；
三组 CPU batch 数组都预分配后原位填充，不再逐 tile 建列表再 `stack`。

每个 core 的 owner segment 直接从现有 membership CSR 切片中过滤，不再对完整
`owner_indices` 做 `core_count` 次全局扫描。该变化只缩短索引准备，不改变 segment
全局下标、唯一 owner、halo 只读或轮次屏障。

拓扑保护会拒绝 ring 绕向翻转和 hole 逃逸。当前 v1 采用整轮回滚，避免发布局部损坏图形；没有为假设中的局部恢复预建接口。

## 6. 光刻与评价替换

`LithographyModel` 是零运行期开销的结构化 Protocol，只要求 `device`、`config.canvas/print_threshold`、`condition()` 和 `forward_many()`；求解器依赖该能力契约，runner 仍明确构造 `ICCAD13Lithography`，没有注册器或工厂。`ProcessCondition(name, kernel, dose)` 表示一次独立工艺条件；全部传播保留原生 autograd，可同时服务 MB-OPC、梯度 OPC 和 ILT。ICCAD13 的 35×35 数据是频域 Hopkins 核，不能把数组半宽误当成有限空间影响半径；`tile_halo_nm` 是用户按精度验证选取的有效光学截断范围。

`evaluate_binary_l2` 与 `evaluate_pvband` 采用 OpenILT 的二值语义，只在 ownership 像素累计不一致像素数；函数不会原位阈值化输入。`evaluate_edge_probes` 根据 target 的内外语义产生 `-1/0/+1` 法向移动方向；同一 probe 同时触发相反要求时记为 ambiguous 且不移动。simple MB-OPC 只用 EPE 决定更新和最佳轮次，L2/PVBand 仅记录诊断。`estimate_rectangular_shots` 在显式固定分辨率上逐行合并相同水平 run，提供确定、无随机和无 OpenCV/adabox 依赖的矩形 shot 估计；它不是最小 shot 数证明。

## 7. SimpleILT

`opc.iteration.ilt.optimize` 直接消费 `[H,W]` 或 `[B,H,W]` target，不构造 `MBOPCProblem` 或边段。默认参数由 target 映射到 `-1/+1`，每轮通过 sigmoid 得到软 mask；`optimization_mask` 可把窗口外区域固定为初始软值。损失由标称连续 L2、调用方传入的任意 process conditions 对 target 的连续 L2、这些条件逐像素范围的连续 PVBand，以及可选曲率项组成。

求解结果只保留历史总损失最优轮的参数、软 mask、二值 mask 和逐轮标量记录。默认条件是 nominal、dose_max、defocus_min，但调用方可传入完全不同的独立条件，也可传空元组只优化标称条件。配置、记录、结果和 Simple 算法集中在 `simple.py`；图像 batch、曲率、缩放和平滑 sigmoid 位于包内 `_common.py`，由现有四种算法复用。各 ILT 方法复用同一结果契约，没有建立基类或注册器。

LevelSetILT 以 `phi < 0` 作为硬开窗条件，并用 `-|∇phi|` 代理梯度把光刻损失传回边界。默认初值是一次性在 CPU 计算、再送到模型设备的精确像素中心欧氏 SDF，前景为负、背景为正；它不在迭代热路径重复计算。最终 `soft_mask=sigmoid(-phi)` 仅用于诊断，权威硬结果仍严格按 `phi < 0` 生成。算法常驻模型设备的主要新增状态为参数、固定初值、优化窗口和 Adam 两份状态，均为 `O(BHW)`；不会物化边段或整张 reticle 的矢量结构。

## 8. 输出约定

- `main/run_mbopc.py`：保存 `summary.json`、结果 GDS 和可选 PNG；最终最佳几何额外做一次固定 512² shot 估计，不保存整轮 tensor，也不生成 NPZ。
- `main/run_mbopc_frontend.py`：用于人工检查输入契约，保存 key-free、按全局 segment 下标对齐的格式 v3 NPZ，并可保存 GDS/PNG/JSON。
- `main/run_simpleilt.py`：只适配历史默认值并委托 `run_ilt(method="simple")`；统一保存 `ilt_result.npz`、summary、最终光刻 NPZ 和可选 PNG，同时保留 Python 返回 `(SimpleILTResult, summary)`。
- `opc.diagnostics`：只有调用者明确要求时才物化诊断长度、图片、GDS 或测试图集。

NPZ 是当前进程中 problem 的快照，不是跨 remesh、跨版本的持久身份协议。显式 remesh 必须重新分段、重新建立 owner，并由调用者重建优化状态。

## 9. 扩展原则

- 新抽象必须有当前调用方；不创建空接口、注册器或无实现目录。
- 替换输入构造和替换迭代是两个独立扩展点：前者产出实际方法需要的数据，后者消费数据并更新状态。
- ILT 优先复用 `layout`、`geometry`、`PhysicalMask`、栅格、`lithography` 和 `evaluation`；不要为了复用而套用 MB 边段结构。
- 保留层级与局部 ROI，跨 Python/KLayout 边界必须批处理；迭代数据要缓存，诊断默认关闭。
- 每个 bug 必须有回归测试，修复后搜索并删除仅服务于旧错误的包装、变量和分支。
- 关键节点只做本地 Git commit，未经明确授权不得 push。

## 10. 离线专项测试工作台

`main/offline_inputs.py` 把“版图输入构造”和“模型/迭代优化”解耦。它提供六个实际接口：

- `materialize_raster_input`：预检版图 ROI，并直接返回内存中的模型 mask 和 metadata；
- `prepare_raster_input`：把一个明确 ROI 保存成模型左下原点的 `float32[canvas,canvas]`；
- `load_raster_input`：校验版本、方向、范围和数值后读取 mask；
- `resolve_raster_input`：`.npz` 走加载路径，其他输入走版图内存物化路径；
- `prepare_segment_input`：一次性完成物化、规范化、切分和 owner/context 构造；
- `load_segment_input`：恢复现有 `MBOPCProblem`，不创建第二套 problem 类型。

像素归档只对应一个可直接送入模型的 canvas，超限 ROI 必须缩小，不隐式切 tile。当前 raster/segment 协议分别为 v2/v3并保存显式 polarity；loader 继续兼容缺少 polarity 的历史 v1/v2，按 clear 解释。它与 `opc.diagnostics.save_problem_npz` 的不可恢复 v3 诊断快照完全分离。

准备前先检查源文件、像素尺寸、层级展开图形/顶点和保守内存估计。严格预检通过 `LayoutDB.recursive_polygon_shapes()` 获取受数据库生命周期约束的只读迭代器，不再跨包读取 `_native_*`；源 GDS/OASIS/GLP 只解析一次，扫描通过才物化 Region。预检和真实切分共享纯 NumPy 边段计数公式，布尔合并仍可能产生无法预知的新交点，因此准备后还会按真实 segment/membership 数量复核内存估计。

`main/run_lithography.py` 和 `main/run_simpleilt.py` 都接受 GDS/OASIS 或 raster NPZ。直接版图模式用 `--layer/--top-cell/--box/--pixel-nm/--canvas` 选择目标并只在内存中生成 mask，不隐式保存 NPZ；归档模式保持原契约。`main/run_mbopc_iteration.py` 只消费边段归档，避免和完整 `run_mbopc.py` 重复版图前端。入口均可从任意工作目录直接运行，不需要安装本项目。

详细字段、内存边界和设计审计见 [离线工作台开发报告](offline_workbench_development_report.md)。

本轮函数与内存收敛见[代码优化开发报告](code_optimization_development_report.md)；macro 前端实现见[专项开发报告](macro_materialization_development_report.md)，仍未实现的多轮 shard 求解边界见[独立开发方案](large_reticle_streaming_plan.md)。

像素中心、DiffOPC 同步检查、分段单位与死抽象的本轮修正见[当前规则符合性修正开发报告](current_rule_compliance_fix_development_report.md)。

## 11. 物化前容量预检与资源统计

真实版图根入口必须先调用 `preflight_layout(database, layer, box, ...)`，通过后才能调用 `ShapeQuery.materialize()`。默认 CPU 预算是启动时系统可用内存的 70%；显式 `--memory-budget-gib` 只改变本次任务预算。超过预算或 `int32` 容量时返回 `sharded_required`，当前版本不会尝试继续分配。

```powershell
# 只做完整层级容量扫描，不物化 Region/边段
python main/run_mbopc_frontend.py TestReticle/gcd_45nm.gds --layer 11/0 `
  --tile-size-nm 1024 --tile-halo-nm 512 --preflight-only --json

# 完成前端几何验证，但跳过大 NPZ/GDS/PNG
python main/run_mbopc_frontend.py TestReticle/gcd_45nm.gds --layer 11/0 `
  --tile-size-nm 1024 --tile-halo-nm 512 --skip-artifacts --json
```

逐 CPU macro 验证未裁剪提边与栅格化时裁剪：

```powershell
python main/run_mbopc_frontend.py TestReticle/gcd_45nm.gds --layer 11/0 `
  --tile-size-nm 1024 --tile-halo-nm 512 --roi-halo-nm 536 `
  --macro-size-nm 8192 --pixel-nm 8 --macro-verify --json
```

`roi_halo_nm` 与 `tile_halo_nm` 是独立参数：前者控制 CPU macro 加载完整相交图形，后者控制光刻 tile context。默认可接近，但 ROI halo 还必须覆盖最大允许边位移。macro 模式只交付前端验证，不生成 shard、不运行多轮 OPC。

`MacroPreparation` 的 segment 下标只在当前 macro 自己的 `SegmentBatch` 内有效；只有 tile ID 来自全局网格。对象全部字段由 `prepare_macro` 一次构造，内部不重复执行面向外部任意输入的数组校验。

摘要 `memory_checkpoints` 使用进程 RSS/USS/private/peak working set，能覆盖 NumPy 与 KLayout 原生内存；`memory.problem_persistent_bytes` 只统计 problem 自有 NumPy 数组，两者不能混为同一指标。详细设计见[容量预检开发报告](frontend_preflight_development_report.md)。

## 12. 最终光刻结果

完整 MB-OPC 完成后会在 `final_lithography/` 写出 `manifest.json` 和按 core 的 ownership-only tile。每个 tile NPZ 含 `mask`、`nominal`、`dose_max`、`defocus_min` 四个二维 `float32` 数组；使用 `--no-final-lithography-png` 可关闭 PNG。SimpleILT 在输出目录直接保存同样字段的 `final_lithography.npz` 与 `final_*.png`。

## 13. 新增 ILT 与 DiffOPC

`main/run_ilt.py --method levelset|curvmulti|multilevel` 统一运行新增 ILT；`main/run_diffopc.py` 可直接读取 GDS/OASIS，也可复用 segment NPZ，使用独立软边段栅格器优化固定参考 segment 的全局绝对位移。LevelSetILT、CurvMultiILT、MultilevelILT 和 DiffOPC 已分别完成四个阶段专项验收。

DiffOPC 每个 tile 的 halo 只参与软 mask 与光刻传播，L2/PV 连续损失只在 `ownership_canvas` 内累计，EPE 只由 owner segment 贡献。所有 batch 读取同一 `current`；每个 batch 立即 backward 释放光刻图，只累积位移梯度，整轮结束后才执行 Adam step。候选位移同时受前端 `max_displacement_dbu` 与 `reconstruct_region` 全局环方向、孔洞归属和 Polygon 合法性约束。

软栅格以 `sigmoid((d-q)/T)-sigmoid(-q/T)` 表示边界平移产生的占据变化，`d=0` 严格返回参考覆盖率。有限 segment 使用平滑切向端帽；segment chunk 通过重计算式 checkpoint 限制反向中间量，而非只对前向循环分块。连续 PV 损失使用 maximum/minimum wafer 平方差；二值 PVBand、L2 和 EPE 只作为逐轮诊断，不能误接入 autograd。

当前 DiffOPC 与 simple MB-OPC 相同，CPU 仍常驻完整 `MBOPCProblem`、owner/membership 和全局位移；GPU 只常驻当前 batch。它不等同于 `large_reticle_streaming_plan.md` 中尚未实现的 macro shard/memmap 方案。SRAF 会改变图形和 segment 身份，属于输入构造/remesh 阶段，不在本求解器内隐式插入。

LevelSetILT 使用前景为负的精确欧氏 SDF 初始化和硬二值代理梯度；SDF 只在优化前计算一次。`run_ilt.py --method levelset` 支持 `--layer`、`--box`、`--top-cell`、像素/画布和容量上限，并保存 `ilt_result.npz`、`summary.json`、最终三工艺角 NPZ 及可选 PNG。

CurvMultiILT 使用 `[0,1]` 连续参数、奇数均值平滑核、带 offset 的 sigmoid 和 SGD。`scales` 严格递减且必须以 1 结束；粗尺度只减少控制参数自由度，每轮 soft mask 近邻恢复到完整物理网格后再执行统一 Hopkins 光刻，避免改变核的像素物理含义。曲率作用于 nominal wafer，不作用于 mask；窗口外在平滑前后均固定为初始参考值。入口按方法选择默认值：CurvMulti 的 step/PVBand/curvature 默认为 `0.5/1/200`，显式 CLI 参数才覆盖。

MultilevelILT 默认按 scale 2/1 运行两个独立 Adam 级别，各为 20/100 轮、实际步长 0.2；低级历史最优参数近邻放大给细级，但不传递 Adam 状态。每级参数和 target 位于本级网格，soft mask 恢复到完整物理网格执行光刻，wafer 再 area 汇聚到本级计算损失。`--iterations N` 表示所有级别同为 N 轮；需要不同轮数或步长时使用 `--stage-iterations`、`--stage-step-sizes`，其数量必须与 `--scales` 相同。

## 14. 版图极性、GLP 与 TOML 配置

版图极性必须显式选择 `clear` 或 `opaque`，内部 mask 永远以 1 表示透光。opaque 必须给出处理 `--box`；处理框只在 raster 边界反相，不进入边段轮廓。GLP 支持严格 ICCAD 子集，符号层可用 `--glp-layer NAME=LAYER/DATATYPE` 映射，结果仍统一写 GDS。

六份默认配置位于 `config/`。全部 `main/run_*.py` 和 `main/offline_inputs.py` 支持 `--config`；优先级为默认 common、默认 entry、自定义 common、自定义 entry、显式 CLI。配置中的相对路径相对配置文件目录解析，未知键或类型立即失败。动态 SRAF 仍是未来能力，实施约束见[动态 SRAF 设计](dynamic_sraf_design.md)，本轮没有加入空接口。
