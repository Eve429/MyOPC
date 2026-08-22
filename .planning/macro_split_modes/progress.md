# Macro 边界切分模式调整进度

## 当前会话

- 已读取当前计划、工作树状态和所有切分/ownership 调用点。
- 已确认本任务需要一次切分内核、两个语义入口，不应连续执行两次切分。
- 尚未修改生产代码。
- 已完成：将原切分内核改名为 `_split_segments_at_cuts`，新增
  `_split_segments_at_macro_cuts` 与 `_split_segments_at_macro_and_core_cuts`；
  `prepare_macro_problem` 当前只调用 macro-only 入口。
- 已完成：补充 macro-only 与 macro+core 两种路径的结构测试，并修复当前
  `SegmentBatch.segment_edge_ids` 重命名在切分返回值中的字段传递。
- 已完成：更新当前数据流、开发手册和边段契约中的切分语义。
- 已完成：定向回归 170 passed；ruff check、ruff format --check 和 compileall
  通过。首次回归发现并修正了持久化测试中的旧 `edge_ids` 字段名，以及
  跨 core 梯度计数仍按旧 core 切分数量断言的问题。
- 已完成：差异审查确认没有连续调用两种切分路径，也没有修改 layout、geometry
  或 `00_PAST`。
