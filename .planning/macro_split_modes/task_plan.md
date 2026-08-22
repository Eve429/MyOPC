# Macro 边界切分模式调整计划

## 目标

保留 macro+core ownership 切分能力，新增只切 macro ownership 的入口，并让当前 `prepare_macro_problem` 使用只切 macro 路径。

## 阶段

- [x] 增加两个语义清晰的切分封装并改 prepare 调用。
- [x] 核对调用点、测试契约和文档中的“core 必切”描述。
- [x] 执行静态检查与可用的定向测试，记录工作树既有问题。
- [x] 完成差异和过度设计审查。

## 约束

- 不连续执行两次切分；每条路径只调用一次切分内核。
- 不修改 `layout/`、`geometry/`、`00_PAST/`。
- 保留用户已有未提交改动，不覆盖其 `SegmentBatch` 重命名。

## 错误记录

暂无。
