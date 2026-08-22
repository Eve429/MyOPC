# Simple MB-OPC 指标趋势图任务

## 目标

Simple MB-OPC 成功结束后，读取已保存的 metrics.json，自动生成每个 macro
和全局总览的 EPE/L2/PVBand/移动段数趋势图，并保存到 work_dir。

## 阶段

- [x] 增加输出配置和 workflow summary 元数据。
- [x] 增加 Simple MB-OPC 趋势图生成与 summary 回写。
- [x] 增加单元/端到端测试并更新手册。
- [x] 执行回归、静态检查和差异审查。

## 固定决策

- 默认保存；`[output].save_metric_trends = false` 可关闭。
- 每个 macro 一张四面板图；总览默认按 state_index 求平均。
- `overview_mode="lines"` 仅作为 Python 调用参数，不进入 TOML。
- 不重新运行光刻模型或评价，不修改 layout/geometry/00_PAST。

## 错误记录

暂无。
