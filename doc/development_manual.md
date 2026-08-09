# MyOPC 开发手册

配套的函数级调用图、数据生命周期和扩展接入点见 [MyOPC 函数调用关系与数据流](function_call_architecture.md)。

## 1. 目标与边界

MyOPC 将版图读取、物理几何和 OPC 方法前端分层，使 MB-OPC、ILT 及后续方法共用高成本的版图查询、mask 规范化、core 网格与边界采样，同时不把特定求解器逻辑塞入公共层。

项目当前使用 Python 3.12、KLayout 0.30.x、NumPy 与 PyTorch 2.5。项目本身不需要 `pip install`，直接运行根目录脚本即可使用；依赖只需存在于所选 Python 环境。

## 2. 目录职责

| 路径 | 职责 | 可复用范围 |
|---|---|---|
| `layout/` | 层级版图加载、Layer/ROI 查询、诊断 | 所有 OPC 方法 |
| `geometry/` | Region 运算、轮廓/边数组、Patch、栅格化 | 所有 OPC 方法 |
| `opc/input/` | 物理 mask、core/context 和方法无关输入契约 | MB-OPC、ILT 及后续方法 |
| `opc/input/edge/` | 边界采样、控制段、稳定 key、owner、位移载体和轮廓重建 | 边段型 OPC 方法 |
| `opc/iteration/mbopc/` | simple MB-OPC 的同步 owner-only 流式迭代 | 当前边段方法 |
| `lithography/` | 可独立替换的光刻模型；当前实现 ICCAD13 Hopkins 模型 | 各类 OPC/ILT 迭代 |
| `evaluation/` | L2、PVBand 与 EPE 探针评价 | 验证与迭代停止条件 |
| `run_mbopc_frontend.py` | 仅验证公共输入/重建前端 | 人工验证/调试 |
| `run_mbopc.py` | 从 GDS/OASIS 直接运行完整 simple MB-OPC | 整图/ROI 验证与产物输出 |
| `tests/opc/` | 功能、负向、集成、随机回归 | 自动验收 |
| `benchmarks/` | 可重复性能门槛 | 性能防退化 |

`layout/` 和 `geometry/` 是已稳定基础。当前开发不得修改这两个目录；如果新方法确实需要新能力，必须先停止开发并获得用户确认。

## 3. MB-OPC 数据流

1. `LayoutDB` 保留层级，仅对 planner 给定 ROI 和 Layer 物化局部 `RegionBatch`。
2. `normalize_physical_mask` 在副本上删除 Shape 属性、合并重叠、恢复孔洞，再一次提取轮廓和数学边。
3. `fragment_edges` 按拐角短段和最大段长切分数学边，只常驻 edge ID、`t0/t1`、稳定 key 和查找索引。
4. `MidpointOwnerPolicy` 使每段只有一个 owner，同时用 CSR 形式保存邻近 core 的 halo context membership。
5. 同进程迭代按已对齐 owner segment index 直接写 `next_values`；外部更新通过稳定 key 和 `merge_owner_updates` 拒绝未知、越权、重复和超限提交。
6. `SegmentBatch.materialize` 只在光学评估或输出前物化当前端点与法向；`sample_lines` 可复用输出缓冲区。
7. `reconstruct_region` 在同边位移差处生成 jog，在拐角处使用解析 miter，失控时使用 bevel，最后返回有效整数 DBU Region。
8. `rasterize_region_canvas` 只把当前 core+halo 栅格化；固定 target 以 uint8 LRU 缓存，当前 mask 由参考 tile 和邻近 Polygon 差分生成。
9. `optimize` 按 batch 在 GPU/CPU 上运行光刻和评价。所有 tile 只读 `current`，owner 方向写入 `next_values`，整轮完成且拓扑检查通过后才统一发布。
10. 每个 batch 结束后只保留标量指标和 segment 方向；完整曝光 tensor 释放。最终最佳位移只执行一次全局矢量重建，core 边界不裁最终 Polygon。

## 4. 主要公共 API

### 4.1 OPC 通用输入层

- `normalize_physical_mask(batch, layer) -> PhysicalMask`：生成方法无关的物理覆盖集合。
- `RectilinearCoreGrid(x_cuts, y_cuts, halo_dbu)`：定义半开内部边界和闭合外边界的规则网格。
- `build_sample_template(...)` / `sample_lines(...)`：为 MB-OPC 和 ILT 评估共用的切向/法向采样。
- `render_boundary_overlay(...)`：输出 mask、owner、core、法向和采样点标注 PNG。

### 4.2 边段输入层

- `FragmentationConfig`：角段长、最大段长、最大位移和 miter 限制。
- `prepare_problem(...) -> MBOPCProblem`：一次完成规范化、分段、owner 和采样模板。
- `SegmentBatch.lookup_keys(keys)`：批量稳定查找，未知 key 返回 `-1`。
- `merge_owner_updates(problem, updates, base_displacements=None)`：将各 core 更新合并到全局位移向量。
- `reconstruct_contours` / `reconstruct_region`：从固定参考边界重建当前 mask。
- `save_problem_npz` / `write_debug_gds`：输出纯数值调试数据和参考/重建 GDS。

### 4.3 光刻、评价与迭代层

- `ICCAD13Lithography(...)`：加载 OpenILT ICCAD13 的 24 个 Hopkins 核，批量返回 nominal/maximum/minimum 连续光刻胶图。
- `evaluate_process_window(...)`：只在 core ownership 像素累计 L2 与 PVBand，halo 只提供上下文。
- `evaluate_edge_probes(...)`：验证 target 的 inner/outer 语义，输出有效性、歧义和 `-1/0/+1` 法向方向。
- `SimpleMBOPCConfig`：定义轮次、步长衰减、位移/探针距离、像素、画布、batch 和 target 缓存上限。
- `optimize(problem, model, config)`：返回最佳全局位移、逐轮指标、最佳轮次和停止原因。

## 5. 扩展新 OPC 方法

ILT 或新方法应优先依赖 `opc.input`。只有当方法确实操作独立边段时才依赖 `opc.input.edge`。替换输入构造时保留 `MBOPCProblem` 侧契约，替换迭代时只组合新的 `opc.iteration.<method>` 与模型/指标；不要把求解器、像素模型或损失函数放进归属层。ILT 可直接复用 `PhysicalMask`、core/context、`rasterize_region_canvas` 和 ICCAD13 模型，不必依赖 MB 控制段重建。

扩展时遵守以下原则：

- 源版图只读，输出使用 Patch 或独立 GDS/OASIS。
- 不对整个层级版图 flatten，不在 Python 中长期保存 Polygon 对象列表。
- 重复迭代前缓存物理边界和索引，迭代中使用 NumPy 批处理。
- tile/halo 必须与像素晶格对齐，halo 必须覆盖模型有效半径和最大允许位移。
- normal iteration 固定参考分段；只有显式 remesh 才允许重提边，并必须重建 key、owner 和优化器状态。
- 不为未实现的求解器预先建立空抽象。

## 6. 代码与 Git 门槛

每个 Python 文件、类和函数都必须有中文注释或 docstring；函数内部对性能、所有权和拓扑正确性有影响的步骤必须紧凑但详细说明“为什么”。

每个 bug 必须先有可复现回归，修复后检查是否引入了临时 wrapper、重复分支或无使用字段。关键里程碑本地 commit，未授权时不 push。
