# Development Report — CHG-20260815-macro-core-pipeline

原始完整报告：`doc_/archive/reports/macro_core_pipeline_development_report.md`。

## 实际交付（提交链）

```text
52fa90f  两级网格与居中光刻画布（plan_macros/_center_padding）
c1e26e0  持久化 MacroProblem（删除旧六文件）
36f2bf6  双轮 macro-core 迭代验证（run_macro_pipeline）
8e70a4a  macro 结果全局合并与双模式写出（PatchWriter.write_macro_results）
20c1d69  开发与测试报告
243a5fd  审查前置：layout 新增 layer_bbox，替换管线内的流式 bbox 扫描
fb80a4e  审查轮落实：契约冻结/空 membership/几何矩阵/正逆序/coverage 审计
         + 两个连带 bug 修复（切线交点重复分裂点、空 macro 崩溃）
d03e47b  文档与规划同步
```

## 关键实现事实

- 交付 `opc/input/grid.py`、`opc/input/raster.py`、`opc/input/edge/`
  （fragmentation/problem/reconstruction/sampling）、
  `main/run_macro_pipeline.py`、`geometry PatchWriter.write_macro_results`；
- gcd_45nm 2×2 smoke：343018 段 / 722161 membership / 870 core / 8 GDS /
  最终 XOR == 0 / 总 10.6s（审查轮后 10.67s 复验一致）；
- 测试：tests/opc/input 55 例 + tests/main 26 例（审查轮 +11 例）。

## 与规格偏差

按原始报告；主要项为 `_write_macro_gds` 增加 dbu_um 参数（NPZ 不含 DBU，
最小必要偏差）。
