# P1/P2 架构收敛测试报告

## 1. 环境

- 日期：2026-08-12
- 解释器：`D:\app\miniforge\envs\myopc\python.exe`
- 平台：Windows，CPU 全量回归；项目既有 CUDA 路径由真实模型专项覆盖
- 基线：阶段 88 的 226 项测试通过

## 2. 专项验证

P1 的 Layout 扫描、容量保护和模型 Protocol 共 56 项通过。P2 首轮修正后，边段切分/preflight、Simple MB-OPC、DiffOPC、四种 ILT 与离线工作台 96 项通过（53.77 秒）；新增随机计数等价与 SimpleILT 统一产物断言后，相关 21 项通过（24.87 秒）。

新增回归重点：

- 公共递归迭代器只扫描 Box/Path/Polygon，并受 LayoutDB 生命周期约束；
- membership 上限在 `np.repeat` 前失败；
- 结构兼容假模型满足 `LithographyModel`；
- 20 组固定随机矩形的真实 `edge_ids` 计数逐边等于共享公式；
- SimpleILT 返回的内存 binary mask 与 `ilt_result.npz` 完全一致，且不再生成 `simpleilt_result.npz`；
- owner 查询仍逐 core 等于全局参考过滤，跨 core 唯一写入不变。

## 3. 质量门命令

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
& $python -m ruff check layout geometry opc lithography evaluation main tests benchmarks
& $python -m compileall -q layout geometry opc lithography evaluation main tests benchmarks
& $python -m pytest -q
git diff --check
```

## 4. 最终结果

- 全仓 pytest：`230 passed in 74.60s`；基线 226 项，新增 4 项回归；
- CUDA：GTX 1650 真实 ICCAD13 CUDA 运行时 1 项通过；四种 ILT 定向 41 项通过；
- 严格前端基准：5,000 图形、110,000 segment、134,734 membership，`strict_failures=[]`；常驻数组 2.594 MiB，相对展开表示节省 67.46%，零位移 XOR=0、无未归属 segment；
- Ruff、compileall、`git diff --check` 全部通过；
- 104 个第一方 Python 文件的模块/函数/测试中文 docstring 缺失为 0，AST 完全重复函数体为 0；
- 正式代码跨模块导入旧 `_atomic_*`、`_exact_dbu`、SimpleILT 私有 helper、重复 owner helper 和重复 Layer parser 均为 0；
- `geometry/` 相对基线零差异；`layout/` 只有已授权的 14 行公共只读扫描接口。

测试过程中发现的命名遮蔽、漏导入和旧产物断言均已产生或更新回归；修复后没有保留别名、特殊分支或第二套产物文件。

尝试对三个新增模块追加 `pytest-cov --cov-branch` 时，Windows 进程在测试收集阶段触发 PyTorch docstring 重复注册及 NumPy 扩展“同一进程重复加载”，测试体没有执行。这与项目既有覆盖率插桩限制一致；本报告不提供虚假的覆盖率百分比，也没有为工具问题改变生产导入顺序。分支信心来自 230 项无插桩全量、96 项关键路径、真实 CUDA、严格基准及 AST/调用点审计。
