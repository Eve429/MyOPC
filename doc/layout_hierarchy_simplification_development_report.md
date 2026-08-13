# Layout 层级接口轻量化开发报告

## 1. 目标与结论

本次改动把只读 Cell 层级查询从独立 `layout/hierarchy.py` 收回 `LayoutDB`。旧接口为每个 Cell 构造 `CellInfo`，同时保存 bbox、`CellRef`、直接子 Cell、实例记录数和 AREF 展开后的逻辑实例数，再包装为 `HierarchySummary`。当前没有 planner 或 OPC 路径消费这些统计，因此这些结构和字段增加了文件、类型与遍历成本。

当前接口只有：

```python
LayoutDB.cell_hierarchy() -> dict[str, tuple[str, ...]]
```

返回文件内全部 Cell 的直接引用邻接表。键是父 Cell 名称，值是按名称排序的直接子 Cell 名称元组；叶 Cell 显式返回空元组。共享 Cell 只定义一次，可同时出现在多个父节点下，所以语义是 DAG，不是 occurrence 展开树。

## 2. 精简内容

| 项目 | 修改前 | 修改后 |
|---|---|---|
| 实现位置 | `layout/hierarchy.py` + `LayoutDB` 包装方法 | `LayoutDB.cell_hierarchy()` 单一方法 |
| 返回对象 | `HierarchySummary`、`CellInfo` 两层结构体 | Python 普通字典与元组 |
| Cell 范围 | 全文件 | 全文件 |
| 子引用读取 | Python 遍历每条 `each_inst()` | KLayout 原生 `each_child_cell()` |
| SREF/AREF | 统计每条记录并计算逻辑实例数 | 只保留去重后的直接关系，不展开 occurrence |
| 附加信息 | top 列表、bbox、实例记录数、逻辑实例数 | 无 |
| 缓存 | 无 | 无 |

删除内容：

- `layout/hierarchy.py`；
- `CellInfo` 与 `HierarchySummary`；
- `build_hierarchy_summary()` 与 `LayoutDB.hierarchy_summary()`；
- `layout.__all__` 中两个旧类型导出。

没有保留兼容包装、旧属性或第二份缓存。仓库内没有生产调用方需要迁移，现有测试直接改为新契约。

## 3. 性能与内存设计

实现只遍历一次 `layout.each_cell()`。每个 Cell 调用一次 KLayout 原生 `each_child_cell()`，原生层已经把同一 child 的重复 SREF 和 AREF 合并为一条直接关系；Python 不进入每条实例记录，更不会按 AREF 的 `na × nb` 展开。

设 Cell 数为 `C`、去重后的直接引用关系数为 `E`：

- 时间复杂度为 `O(C + E)`，每个局部子列表另有确定性排序；
- 返回值内存为 `O(C + E)`；
- 不创建 bbox、`CellRef`、实例统计或 occurrence 路径；
- 不缓存结果，避免数据库关闭或版图状态变化后持有第二份快照。

本方法与已选择的 `top_cell` 无关，原因是用户要求返回版图文件的完整 Cell 层级。多 top 和未被当前 top 引用的 Cell 同样保留。调用前仍经过 `_native_layout` 生命周期守卫，数据库关闭后抛 `ClosedLayoutError`。

## 4. 架构影响

改动保持依赖方向不变，没有增加包、接口层或注册器。`layout` 仍只负责版图读取、元数据与 ROI 查询；`geometry`、OPC 输入、光刻和求解器没有修改，也没有新增对层级字典的隐式依赖。

该接口只回答“Cell A 直接引用哪些 Cell”，不回答 occurrence 的变换、实例路径、源 shape 身份或修正结果如何回写 master cell。因此它不替代未来层级 OPC 所需的 occurrence 追溯设计，也不把尚未实现的层级复用描述为当前能力。

## 5. 简化审计

- 两个只服务旧返回值的结构体和一个独立文件已经删除；
- 旧 API、旧构建函数和旧生产导入无残留；
- 没有为了兼容旧测试增加适配器；
- 新方法有直接测试和明确人工检查用途；
- `geometry/` 保持零改动；
- 用户已有 `.vscode/launch.json` 与 `config/mbopc.toml` 修改不属于本次交付。

测试证据见[Layout 层级接口轻量化测试报告](layout_hierarchy_simplification_test_report.md)。
