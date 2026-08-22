# 开发报告

## 实现内容

- 新增 `common/metric_trends.py`，抽出 JSON 记录读取、字段校验、动态面板布局、
  mean/lines 总览和 PNG manifest 输出。
- Simple/Gradient runner 共用该组件；Gradient 支持六项默认指标和
  `overview_mode` 参数。
- `MBOPCConfig`、`GradientConfig` 增加可配置 `metric_trend_fields`。
- 输出命名统一为 `series_pngs`、`series_<id>.png`。

## 简化审计

- 未新增 metrics 数据结构、注册器或算法抽象层。
- 未把趋势字段复制到 solver config；字段从用户配置进入公共 MBOPC summary。
- 未修改 layout/geometry/ILT runner。
- 未保留旧 macro 命名兼容分支，符合开发初期不要求后向兼容的决定。
