# Development Report — CHG-20260815-single-pass-bias

> 本批无正式开发报告（历史记录缺失，如实标注）；以下从提交、源码与
> task_plan 会话记录整理。

## 实际交付

- 提交：`cac3930 feat(main): 单遍偏置扩张入口 run_single_pass`；
- 文件：`main/run_single_pass.py`（每行中文注释）、`config/single_pass.toml`、
  `tests/main/test_single_pass.py`；
- 复用面：验证管线全部核心符号，未新增领域抽象。

## 验证数字（从记录整理）

- gcd_45nm 单遍 +5nm 实测 0.80s（验证管线 10.6s——差异来自不做逐 core
  居中画布栅格化）；
- 环双向扩张正/负两方向用例、产物唯一性、未处理层、macro_cells 一致性
  均有用例（见 test_report）。

## 记录缺失项

- 无独立 coverage/性能报告；如需补齐以当前测试重跑为准。
