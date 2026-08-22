# 指标趋势图实现进度

## 当前会话

- 已确认 Simple 指标来源、入口、配置解析和现有产物结构。
- 已建立本任务规划；尚未修改业务代码。
- 已完成：OutputConfig 增加 `save_metric_trends`，公共 MB-OPC summary 暴露
  `work_dir/save_metric_trends`，Simple summary 暴露 `metrics_json` 路径。
- 已完成：Simple 入口增加 macro 四面板图、mean/lines 总览图和 summary 回写。
- 已完成：配置、开发手册、测试手册及 smoke 配置同步更新。
- 当前：定向测试已通过；全量 runner 中存在与本任务无关的旧入口路径和 field
  warning 文案失败。
- 已完成：趋势图相关 4 项定向测试通过；Simple MB-OPC/配置相关回归共
  113 passed、5 deselected。
- 已完成：ruff check、ruff format --check 和 compileall 全部通过。
- 已完成：差异审查确认用户未提交的 `config/gradient_mbopc.toml` 与 gradient.py
  修改未被本任务改写。
