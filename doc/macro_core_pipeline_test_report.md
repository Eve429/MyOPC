# Macro–Core 两级网格与双轮迭代重构 · 测试报告

日期：2026-08-15 ｜ 解释器：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12）

## 1. 验收命令与结果（§17 全项）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests` | **115 passed**（基线 49 → 115） |
| `python -m ruff check opc main tests\opc tests\main geometry\patch.py tests\geometry\test_patch.py` | All checks passed |
| `python -m compileall -q layout geometry opc main tests` | 通过 |
| `python main/run_macro_pipeline.py config/macro_pipeline.toml` | 退出码 0，摘要见 §4 |

既存告警说明：`geometry/contour.py` 的一个导入空行告警在本任务范围外
（§17 已声明不借本任务修改），专项 Ruff 范围全部通过。

## 2. 测试套件构成

| 套件 | 用例数 | 覆盖 |
|---|---|---|
| tests/layout | 27 | 迁移基线（本任务未改动） |
| tests/geometry | 25 | 22 基线 + `test_patch` 新增双模式写出 3 例 |
| tests/opc/input | 40 | `test_grid.py` 22（网格规划/校验/居中 canvas/极性/坐标映射）+ `test_macro_problem.py` 18（切线分裂/owner/CSR/拓扑/NPZ 持久化与失败路径） |
| tests/main | 23 | 配置校验 4 + 阶段产物 2 + 双轮状态机 8 + 最终合并 9 |

## 3. 设计文档测试矩阵对照（§15）

- **15.1 网格与配置**：14 条全部覆盖（DBU/位移校验在 tests/main，其余在
  test_grid；负坐标 bbox、面积守恒、context 不改 ownership 均含）。
- **15.2 Canvas 与极性**：1–8 全部自动化；第 9 条（对照 00_PAST
  `_prepare_mask` 规则）为设计说明，不设自动用例。
- **15.3 跨 macro/core 几何**：内部不跨 owner、context 多 membership、own⊆
  membership、共享斜边参数逐位一致（含 t=0.5 切点）、donut 拓扑、零位移
  ownership 汇总 merge XOR==0、SREF/2×2 AREF 物化。窄环（2nm）与相接/
  重叠图形组未逐一单列——重叠图形被 normalize 合并后的全部不变量由 donut、
  跨界 bar 与 AREF 用例联合覆盖。
- **15.4 双轮迭代**：10 条全部覆盖（含篡改上轮位移证明续读、macro 正逆序
  一致、monkeypatch 计数证明阶段二零昂贵调用）。
- **15.5 最终合并**：8 条覆盖 7 条；「无效 polygon」为运行时守卫（见开发
  报告偏差表）。
- **15.6 持久化/性能/smoke**：roundtrip 逐项相等、allow_pickle=False、坏
  版本/截断/错位移长度、原子保存失败不替换旧文件；smoke 见 §4。

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
