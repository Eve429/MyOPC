# Macro–Core 两级网格与双轮迭代重构 · 测试报告

日期：2026-08-15（首版）/ 2026-08-16（审查修复后更新） ｜ 解释器：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12）

## 1. 验收命令与结果（§17 全项）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests` | **135 passed**（基线 49 → 115 → 审查修复后 135） |
| `python -m ruff check opc main tests\opc tests\main geometry\patch.py tests\geometry\test_patch.py` | All checks passed |
| `python -m compileall -q layout geometry opc main tests` | 通过 |
| `python main/run_macro_pipeline.py config/macro_pipeline.toml` | 退出码 0，摘要见 §4 |

既存告警说明：`geometry/contour.py` 的一个导入空行告警在本任务范围外
（§17 已声明不借本任务修改），专项 Ruff 范围全部通过。

## 2. 测试套件构成

| 套件 | 用例数 | 覆盖 |
|---|---|---|
| tests/layout | 29 | 27 基线 + `layer_bbox` 2 例（多层/子树过滤 + AREF/R90/镜像实例） |
| tests/geometry | 25 | 22 基线 + `test_patch` 双模式写出 3 例 |
| tests/opc/input | 55 | `test_grid.py` 24（网格规划/校验/居中 canvas/极性/坐标映射）+ `test_macro_problem.py` 31（切线分裂/owner/CSR/复杂几何矩阵 11 例/NPZ 持久化与失败路径/空 CSR 回归 2 例） |
| tests/main | 26 | 配置校验 6（含位移冻结 2 + 阶段 0 零遍历守卫）+ 阶段产物 2 + 双轮状态机 8 + 最终合并 9 |

## 3. 设计文档测试矩阵对照（§15，审查后真实覆盖表）

- **15.1 网格与配置**：14 条全部覆盖；DBU 精确换算、位移冻结、context≥位移
  在 `tests/main/test_macro_pipeline.py::TestConfigValidation`，其余在
  `test_grid.py::TestMacroPlanningBySize/ByCount/Validation`。
- **15.2 Canvas 与极性**：1–8 全部自动化（`test_grid.py::TestCenteredCanvas`）；
  第 9 条（对照 00_PAST `_prepare_mask` 规则）为设计说明，不设自动用例。
- **15.3 跨 macro/core 几何**（图形类型 → 具体用例，全部为独立可定位失败的
  测试函数，共享 `_assert_problem_invariants` 全量不变量检查）：

| 图形类型 | 用例（test_macro_problem.py::TestGeometryMatrix） |
|---|---|
| 矩形/长条 | TestOwnershipSplit 系列（基线） |
| 凹多边形（跨横边） | `test_concave_polygon_crossing_horizontal_macro_boundary` |
| 孔洞 donut | `TestTopologyPreservation::test_donut_rings_survive_ownership_split` |
| 2 DBU 窄环（跨切线） | `test_two_dbu_narrow_ring_crossing_cuts` |
| 斜边跨竖边（斜率 1） | `test_slope_crossing_vertical_macro_boundary` |
| 陡斜边跨横边（8/3） | `test_steep_slope_crossing_horizontal_macro_boundary` |
| 斜边穿 macro 角点 | `test_slope_through_macro_corner_point` |
| 共享斜边分裂一致性 | `TestSharedDiagonal`（泛化助手 `_edge_split_params`） |
| 相接（共边跨切线） | `test_edge_touching_pair_across_core_cut` |
| 部分重叠 | `test_overlapping_pair_merges_before_extraction` |
| 完全包含 | `test_contained_pair_collapses_to_outer` |
| 单 SREF 跨边界 | `test_single_sref_crossing_macro_boundary` |
| 2×2 AREF 分落四 macro | `test_two_by_two_aref_spreads_across_macros` |
| 3×3 AREF（基线） | `TestTopologyPreservation::test_array_references_materialize_into_problem` |
| 连续跨 ≥3 core | `test_long_bar_crosses_at_least_three_cores` |

  每例验证：零位移 XOR==0（无查询框假边）、owned 开区间不跨切线、中点
  owner 一致、own⊆membership、ring 数与纯提取路径一致；斜边类额外验证
  两侧分裂参数逐位一致。
- **15.4 双轮迭代**：10 条全部覆盖；正逆序一致为双轮完整对照（两轮位移
  逐位一致 + 最终覆盖 XOR），阶段二零昂贵调用由 monkeypatch 计数证明，
  阶段 0 零逐 shape 遍历由 `recursive_polygon_shapes` 计数守卫证明。
- **15.5 最终合并**：8 条覆盖 7 条；「无效 polygon」为运行时守卫（见开发
  报告偏差表）。
- **15.6 持久化/性能/smoke**：roundtrip 逐项相等、allow_pickle=False、坏
  版本/截断/错位移长度、原子保存失败不替换旧文件；smoke 见 §4；
  coverage 数值见开发报告 §8。

## 4. gcd_45nm 2×2 完整 smoke

| 指标 | 数值 |
|---|---|
| macro / core 数 | 4 / 870 |
| problem 数 / 最大 problem | 4 / 3,528,928 B（mr0c0） |
| 段数总计 / membership 总计 | 343,018 / 722,161 |
| 准备耗时 | 0.45 s |
| 第一轮 / 第二轮 | 4.96 s / 4.88 s |
| 最终合并 | 0.17 s |
| 总耗时 | 10.64 s |
| RSS：准备 / 迭代 / 合并 | 74.8 MB / 80.6 MB / 75.2 MB（合并为完成后即时采样） |
| 产物 | 每轮 4 result NPZ + 4 macro GDS（两轮共 8 GDS）、1 个 final GDS、summary.json |
| **最终 XOR 面积** | **0**（第二轮回零后与原始目标层逐位一致） |

产物位于 `output/macro_pipeline/`（不提交）；smoke 最终版图按 TOML 相对
路径规则落在 `config/` 下，验证后已删除。原始 `TestReticle/gcd_45nm.gds`
未被修改。

## 5. 测试方法论要点

- 全部自动用例使用 `tmp_path` 动态生成 GDS/TOML/NPZ，不依赖 TestReticle
  用户数据；gcd_45nm 仅用于最终 smoke。
- 测试版图包含上下锚框 + 中条：锚框把层 bbox 撑到目标尺寸，中条完全在
  bbox 内部且跨 macro 切线——铺满 bbox 的图形外扩会全部落在 ownership
  之外被正确裁掉，无法证明 +2 生效（该几何病态在开发中确认是正确行为）。
- 阶段边界用调用计数证明（monkeypatch `prepare_macro_problem` 与
  `ShapeQuery.materialize_intersecting`，两轮全程零调用）。
