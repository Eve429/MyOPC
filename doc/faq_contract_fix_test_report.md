# FAQ 契约修复测试报告

## 1. 专项矩阵

| 场景 | 验收要求 |
|---|---|
| Geometry/OPC raster | 同一非对称 Region 的公共返回数组逐像素同向 |
| PNG 与查看器 | 文件/查看器只在输出边界上下翻转，返回数组不变 |
| OPC 标注图 | 左下原点 raster 翻为 Pillow 底图后，DBU 标注仍使用顶部向下坐标 |
| 单次 MB-OPC | `iterations=1` 评价初态、发布一次合法更新、评价更新态 |
| 多 core 屏障 | 同一状态所有 core 只读同一 `current`，更新态只在全局屏障后出现 |
| 最后候选拓扑 | 最后一次允许更新也调用全局重建校验，非法候选整次回滚 |
| 最佳状态 | 只按已评价状态 EPE 选择，诊断 L2/PVBand 不改变几何 |
| Region 生命周期 | 惰性查询关闭后拒绝；已物化批次关闭后计数和面积不变 |
| 用户 notebook | `ruff check .` 不扫描或修改 `Test/klayout.ipynb` |

## 2. 已完成回归

- raster、诊断、Geometry 图集与 simple MB-OPC 核心：56 项通过；
- 离线工作台、完整 MB-OPC 入口、诊断 CLI 和跨 core 重建：37 项通过；
- 修复前基线审查：40 项通过；
- 最终全仓库：210 项通过，耗时 72.93 s；
- Ruff 全仓库门禁：通过，用户 `Test/klayout.ipynb` 按配置排除；
- `compileall`：`layout/geometry/opc/lithography/evaluation/main/tests/benchmarks` 全部通过；
- 95 个第一方 Python 文件 AST 审计：中文模块/函数 docstring 缺失 0、非中文 0、重复函数体 0；
- 36 份 `doc/*.md`：断链 0、未闭合代码围栏 0；`git diff --check` 通过。

追加的 `pytest-cov` 插桩在 Windows 测试收集阶段触发 NumPy 扩展重复加载，测试体没有执行。该工具问题未通过修改运行时代码规避；本轮以专项分支回归、210 项无插桩全量测试和 AST 审计验收，不虚报新增覆盖率数值。

## 3. 性能检查原则

raster 严格基准继续要求覆盖率精确且不超过既有时间/内存门槛。MB-OPC 每个完整 N 次更新任务比旧错误实现多评价最终状态；报告应比较“每个已评价状态/每次真实更新”的成本，不能把修复后的 N 次更新与旧实现实际 N−1 次更新直接宣称为零开销。

Windows、Python 3.12、KLayout 0.30.10、NumPy 2.5.1 下，2048×2048 raster 严格基准耗时 471.94 ms、额外 RSS 7.90 MiB、覆盖率逐像素精确；百万逻辑实例的小 ROI 查询中位数 0.093 ms、额外 RSS 0.47 MiB，全部严格门槛通过。
