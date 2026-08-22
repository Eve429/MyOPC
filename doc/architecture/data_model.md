# Architecture — 核心数据对象

跨模块数据对象的 ownership、生命周期与驻留位置（S=段数、C=core 数、M=membership 数）。

## 对象总表

| 对象 | 定义处 | Owner | 生命周期 | 驻留 | Mutability |
|---|---|---|---|---|---|
| `kdb.Region`（源/候选） | layout 查询 / `reconstruct_region` | 调用方 | 查询窗口内 / 单次重建 | CPU | 只读消费 |
| `MacroSpec`/`CoreSpec` | `opc/input/grid.py` | plan（无状态值对象） | 进程 | CPU | frozen |
| `MacroProblem` | `opc/input/edge/problem.py` | NPZ ↔ 内存 | 持久化（format v3，负板 prepare 前补铬，无 dark_box） | CPU | 构造后只读 |
| `SegmentGeometry` | `fragmentation.py::materialize` | 单次物化调用方 | 一次评价~整迭代（reference 复用） | CPU | 不可变（按值消费） |
| `displacements` | 求解器局部 | `optimize_simple_macro` | 单 macro 迭代 | CPU numpy | 唯一可写迭代态 |
| target uint8 画布 | `TargetCanvasCache` | 跨状态复用 | macro 内多状态 | CPU | 只读 |
| 批张量 | `evaluate_state` 局部 | 单批 | 批结束即释放 | GPU | 临时 |
| `SimpleMBOPCStep/IterationRecord/Result` | `mbopc/simple.py` | 求解器→workflow | 迭代结束落盘 | CPU | frozen |
| `PixelMacroProblem` | `opc/input/pixel/problem.py` | NPZ ↔ 内存 | 持久化（format v1） | CPU | 构造后只读 |
| macro 像素参数/梯度/best | `ilt/simple.py` 求解器局部 | 同步 SGD step / scatter-add | 单 macro 优化 | CPU float32 | 唯一可写迭代态 |
| `ILTStateRecord/ILTMacroResult` | `opc/iteration/ilt/_common.py` | 求解器→workflow | 迭代结束落盘 | CPU | frozen |
| plan/metrics/summary JSON | `main/_macro_pipeline.py` 等 | 文件 | 持久化 | 磁盘 | 原子写 |

## 关键规则

- **参考只读**：`MacroProblem` 的全部数组（contours/edge_*/owner/CSR）在
  迭代期只读；唯一写者是构造期的 `prepare_macro_problem` 与加载期的
  `MacroProblem.load` 归一化。
- **迭代态一维化**：一切可变状态收敛为 `float64[S]` 位移；context 段
  （owner==-1）恒 0；重建只在候选验证与最终输出发生。
- **owner 唯一写**：方向写 `next_values[owner 段]`，`written[S]` 标记恰写
  一次；同轮全部 batch 只读同一 current（Jacobi 屏障在批循环结束后的
  返回点）。
- **Region 生命周期**：窗口物化（`materialize_intersecting`）产生的 Region
  必须在 `with LayoutDB.open(...)` 内消费；已物化的独立 RegionBatch 在 DB
  关闭后存活（`tests/layout` 守卫）。
- **GPU 驻留上界**：每批 `B×3` 张 256² 图 + 光刻中间场（约 B×12MiB）；
  target 缓存显式字节上限（key 含 macro id）；不保存整张 reticle tensor。
- **数值域**：顶点 int32 DBU → 中间几何 float64（探针/参数插值可无理）→
  探针坐标 float64 换算 → round 成整数索引采样。像素值 float（覆盖率/
  sigmoid 输出），二值化只发生在评价内部（threshold 比较）。

## 持久化产物

| 产物 | 生产者 | 内容要点 |
|---|---|---|
| `problems/<macro_id>.npz` | `MacroProblem.save` | 全参考数组，format v3（负板补铬方案，无 dark_box），无 dbu_um |
| `macros/<id>/result.npz`、`gradient_result.npz` | `run_mbopc.py` / `run_mbopc_gradient.py` 入口（适配器已并入） | best_state_index、best_displacements、stop_reason |
| `macros/<id>/metrics.json`、`gradient_metrics.json` | 同上 | 逐轮/逐状态标量 + stop_detail |
| `round_*/results/*.npz` | `run_round` | 累计位移 + 每核 transmission（验证管线） |
| `final.gds` | `merge_macro_results` | ownership 权威覆盖，single_cell/macro_cells；空 macro 候选按零覆盖容忍 |
| `final_lithography/` | `save_final_lithography` | 逐 tile nominal/binary PNG + manifest |
| `plan.json`/`summary.json` | `atomic_write_json` | 网格契约 / 全流程摘要 |
