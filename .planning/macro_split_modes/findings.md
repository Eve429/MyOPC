# Macro 边界切分模式调整发现

## 当前事实

- 当前切分函数同时接收 macro 内全部 core 切线，因此一次处理 macro 和 core 边界。
- 当前 Simple/Gradient 的 owner 是 segment 中点；membership 可跨 core，Gradient 可把同一参数的多 core 梯度累加。
- 切分实现是一次性分配新数组，不是 Python 原地插入；连续两次切分会重复扫描、排序和分配。

## 待核对

- 只切 macro 后哪些测试仍断言“owned segment 不跨 core”。
- 当前工作树已有 `SegmentBatch.edge_ids`→`segment_edge_ids` 的未完成重命名，会影响独立测试结果。

## 本次实现结论

- 两个公开给当前模块调用的语义入口共享一个向量化内核，避免先切 macro、再切
  core 的重复扫描和临时数组峰值。
- `prepare_macro_problem` 使用 macro-only；core 内跨界段仍按中点确定唯一 owner，
  并通过 membership 供相邻 core 读取。
- macro+core 入口保留为内部备用路径和结构回归测试入口，未改变当前主流程。
