# 指标趋势图实现发现

- Simple 逐状态指标已经保存于各 macro 的 `metrics.json`，字段包括
  `state_index/epe/l2/pvband/moved_segments`，无需重新计算。
- `main/run_mbopc_simple.py::run_mbopc` 是 Simple 专用收尾入口；公共
  `run_mbopc_workflow` 负责保存各 macro 产物和初始 summary。
- `matplotlib` 已在 requirements.txt 中声明，项目已有 PNG 生成路径；趋势图
  应使用 Agg 后端并在保存后关闭 Figure。
- 当前工作树有用户未提交的 `config/gradient_mbopc.toml`，实现时不得覆盖。
- `run_mbopc_workflow` 的 macro summary 原本没有 `metrics_json`，趋势图实现补充
  该已有产物路径；没有改变 metrics.json 的 schema。
- 全量 `tests/main/test_mbopc_runners.py` 当前有 4 个旧的
  `main/run_mbopc.py` 直跑测试失败（仓库入口实际为 `run_mbopc_simple.py`），
  另有 1 个 field warning 文案失败；这些与趋势图逻辑无关，不能通过吞错掩盖。
