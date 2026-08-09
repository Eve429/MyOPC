# MyOPC 开发手册

## 1. 目标与边界

MyOPC 将版图读取、物理几何和 OPC 方法前端分层，使 MB-OPC、ILT 及后续方法共用高成本的版图查询、mask 规范化、core 网格与边界采样，同时不把特定求解器逻辑塞入公共层。

项目当前使用 Python 3.12、KLayout 0.30.x 与 NumPy。项目本身不需要 `pip install`，根目录主程序会把自身目录用作导入根。

## 2. 目录职责

| 路径 | 职责 | 可复用范围 |
|---|---|---|
| `layout/` | 层级版图加载、Layer/ROI 查询、诊断 | 所有 OPC 方法 |
| `geometry/` | Region 运算、轮廓/边数组、Patch、栅格化 | 所有 OPC 方法 |
| `opc/common/` | 物理 mask、core/context、边界采样、标注图 | MB-OPC、ILT 及后续方法 |
| `opc/mbopc/` | 控制段、稳定 key、owner、位移更新、轮廓重建 | MB-OPC |
| `run_mbopc_frontend.py` | 无安装主入口和产物输出 | 人工验证/调试 |
| `tests/opc/` | 功能、负向、集成、随机回归 | 自动验收 |
| `benchmarks/` | 可重复性能门槛 | 性能防退化 |

`layout/` 和 `geometry/` 是已稳定基础。当前开发不得修改这两个目录；如果新方法确实需要新能力，必须先停止开发并获得用户确认。

## 3. MB-OPC 数据流

1. `LayoutDB` 保留层级，仅对 planner 给定 ROI 和 Layer 物化局部 `RegionBatch`。
2. `normalize_physical_mask` 在副本上删除 Shape 属性、合并重叠、恢复孔洞，再一次提取轮廓和数学边。
3. `fragment_edges` 按拐角短段和最大段长切分数学边，只常驻 edge ID、`t0/t1`、稳定 key 和查找索引。
4. `MidpointOwnerPolicy` 使每段只有一个 owner，同时用 CSR 形式保存邻近 core 的 halo context membership。
5. 求解器用稳定 key 返回绝对法向位移，`merge_owner_updates` 拒绝未知、越权、重复和超限更新。
6. `SegmentBatch.materialize` 只在光学评估或输出前物化当前端点与法向；`sample_lines` 可复用输出缓冲区。
7. `reconstruct_region` 在同边位移差处生成 jog，在拐角处使用解析 miter，失控时使用 bevel，最后返回有效整数 DBU Region。

## 4. 主要公共 API

### 4.1 OPC 公共层

- `normalize_physical_mask(batch, layer) -> PhysicalMask`：生成方法无关的物理覆盖集合。
- `RectilinearCoreGrid(x_cuts, y_cuts, halo_dbu)`：定义半开内部边界和闭合外边界的规则网格。
- `build_sample_template(...)` / `sample_lines(...)`：为 MB-OPC 和 ILT 评估共用的切向/法向采样。
- `render_boundary_overlay(...)`：输出 mask、owner、core、法向和采样点标注 PNG。

### 4.2 MB-OPC 层

- `FragmentationConfig`：角段长、最大段长、最大位移和 miter 限制。
- `prepare_problem(...) -> MBOPCProblem`：一次完成规范化、分段、owner 和采样模板。
- `SegmentBatch.lookup_keys(keys)`：批量稳定查找，未知 key 返回 `-1`。
- `merge_owner_updates(problem, updates, base_displacements=None)`：将各 core 更新合并到全局位移向量。
- `reconstruct_contours` / `reconstruct_region`：从固定参考边界重建当前 mask。
- `save_problem_npz` / `write_debug_gds`：输出纯数值调试数据和参考/重建 GDS。

## 5. 扩展新 OPC 方法

ILT 或新方法应优先依赖 `opc.common`。只有当方法确实操作独立边段时才依赖 `opc.mbopc`。新 owner 规则只需实现 `OwnershipPolicy.assign`，不应把求解器、像素模型或损失函数放进归属层。

扩展时遵守以下原则：

- 源版图只读，输出使用 Patch 或独立 GDS/OASIS。
- 不对整个层级版图 flatten，不在 Python 中长期保存 Polygon 对象列表。
- 重复迭代前缓存物理边界和索引，迭代中使用 NumPy 批处理。
- 不为未实现的求解器预先建立空抽象。

## 6. 代码与 Git 门槛

每个 Python 文件、类和函数都必须有中文注释或 docstring；函数内部对性能、所有权和拓扑正确性有影响的步骤必须紧凑但详细说明“为什么”。

每个 bug 必须先有可复现回归，修复后检查是否引入了临时 wrapper、重复分支或无使用字段。关键里程碑本地 commit，未授权时不 push。
