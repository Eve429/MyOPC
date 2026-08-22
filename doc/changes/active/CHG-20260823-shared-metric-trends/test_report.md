# 测试报告

## 已执行

- `tests/common/test_metric_trends.py`：5 passed
- `tests/main/test_configuration.py`：34 passed
- `tests/main/test_gradient_mbopc_runner.py`：30 passed
- Simple 趋势/留档专项：3 passed
- Ruff check/format：通过

## 公共接口覆盖

- Simple 四指标；Gradient 六指标；ILT 风格 loss 字段。
- 自定义字段、不同状态数量的 lines 总览。
- 空字段、重复字段、缺失字段错误传播。
- `series_pngs`、overview PNG 和字段 metadata。

## 已知存量失败

完整 `tests/main/test_mbopc_runners.py` 仍有 5 个既有失败：4 个测试引用已不存在的
`main/run_mbopc.py`，1 个测试匹配过期的 field warning 文本。本次未修改这些无关入口
和 warning 语义。
