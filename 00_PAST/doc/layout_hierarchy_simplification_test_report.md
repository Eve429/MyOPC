# Layout 层级接口轻量化测试报告

## 1. 测试目标

验证 `LayoutDB.cell_hierarchy()` 的完整 DAG、去重、性能边界和生命周期语义，并确认删除旧类型后没有破坏版图、Geometry、OPC、光刻、ILT 或入口流程。

## 2. 专项用例

| 场景 | 输入结构 | 预期 |
|---|---|---|
| 基本层级 | `TOP -> LEAF` | 两个键；`LEAF: ()` |
| 三级层级 | `TOP -> MIDDLE_A -> LEAF` | 只保存直接关系，不跨级填入 `LEAF` |
| 共享 Cell | `MIDDLE_A -> LEAF`、`MIDDLE_B -> LEAF` | `LEAF` 只定义一次，可被两个父节点引用 |
| 重复 SREF | 同一父 Cell 两次引用 `LEAF` | 父节点值只有一个 `LEAF` |
| 大 AREF | 同一父 Cell 100×100 阵列引用 `LEAF` | 不生成 10,000 个 occurrence 条目 |
| 多顶层 | `TOP` 与独立 `INDEPENDENT` | 显式选择 `TOP` 后仍返回两个 top 对应的全部 Cell |
| 叶节点 | `LEAF`、`INDEPENDENT` | 两者均显式映射为 `()` |
| 返回类型 | 全部节点 | 外层严格为 `dict`，子列表严格为 `tuple` |
| 生命周期 | 调用 `close()` 后查询 | 抛 `ClosedLayoutError`，不返回缓存快照 |

测试版图由测试函数动态写入临时目录，不修改用户 GDS。

## 3. 执行命令与结果

使用项目环境直接运行，不安装 MyOPC 包：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
& $python -m pytest tests\layout\test_database.py -q
& $python -m ruff check layout tests\layout\test_database.py
```

最终结果：

- `tests/layout/test_database.py`：9 passed；
- 全仓库 pytest：250 passed，94.55 秒；
- Ruff：通过；
- compileall 与 `git diff --check`：通过；
- 106 个第一方 Python 文件：中文模块/函数 docstring 缺失 0，完全重复函数体 0；
- 47 份 `doc/*.md`：奇数代码围栏 0，本地断链 0；
- 旧符号生产调用点：0。

## 4. 正确性判定

邻接表比较使用精确字典等式，能够同时检查节点集合、每个直接子关系、叶节点和值类型。100×100 AREF 与重复 SREF 共用同一父节点，若实现误用逐 occurrence 展开，元组长度断言会立即失败。

测试特意加入当前选择 top 之外的独立 Cell，防止实现退化为“只从选中 top 向下遍历”。关闭生命周期用同一个 `LayoutDB` 实例回归，确保方法没有为省时间而引入可能过期的缓存。

## 5. 边界与非目标

- 本测试不验证 instance transform、instance path 或 source shape ID，因为当前轻量接口不承诺这些数据；
- 不用逻辑实例总数衡量性能，正确性能不变量是结果大小只随 Cell 和去重后的直接关系增长；
- 不修改或重新保存用户版图；
- 不把该邻接表当作层级 OPC occurrence 回写能力。

## 6. 清理与保护检查

`hierarchy_summary`、`HierarchySummary`、`CellInfo` 和 `build_hierarchy_summary` 在生产代码与测试中均为零匹配；没有兼容包装或旧错误分支。低引用函数审计只命中 dataclass/context manager/PyTorch 框架钩子，以及有直接回归的公共便利接口，没有发现本次新增死函数。

相对本阶段首个代码提交的父版本，`geometry/` 差异为空。用户的 `.vscode/launch.json` 和 `config/mbopc.toml` 修改保持原样，未纳入功能提交。
