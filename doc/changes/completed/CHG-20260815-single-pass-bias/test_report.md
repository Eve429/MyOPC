# Test Report — CHG-20260815-single-pass-bias

> 本批无正式测试报告（历史记录缺失）；以下从测试源码与复验整理。

## 验证结论

- `tests/main/test_single_pass.py`：8 用例（2026-08-16 复验 8 passed /
  1.65s，myopc env）；
- 覆盖：环双向扩张正/负、产物唯一（每 macro 一 GDS + 最终合并）、
  未处理层不复制、macro_cells 与 single_cell 一致、配置校验 3 例。

## 环境

生成式 GDS/TOML（tmp_path）；解释器与依赖见 `requirements.txt`。
