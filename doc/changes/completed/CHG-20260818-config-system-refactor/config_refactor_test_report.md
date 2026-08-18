# 配置系统重构测试报告

日期：2026-08-18。环境：Windows、myopc conda env（Python 3.12、
torch 2.5.1+cu124、GTX 1650）。

## 实际执行命令与结果

```text
pytest -q tests/main/test_configuration.py      → 20 passed（loader 行为）
pytest -q tests/main                           → 119 passed（四流程全量）
pytest -q tests（全量）                          → 444 passed（86.3s）
ruff check layout opc lithography evaluation main tests → 全绿
compileall layout geometry opc lithography evaluation main tests → 全绿
python main/run_mbopc.py config/mbopc_multi_macro.toml        → 退出 0（47.6s CUDA）
python main/run_gradient_mbopc.py config/gradient_mbopc.toml  → 退出 0（205s CUDA）
python main/run_macro_pipeline.py config/macro_pipeline.toml  → 退出 0（4.1s，XOR=0）
python main/run_single_pass.py config/single_pass.toml        → 退出 0（0.12s）
```

全量 429 → **444 passed**（+15：configuration 20 新增 − 批 0 删 5 条约束用例；
无 skip）。失败项：0。

## §24 测试覆盖对照

| 规格 §24 | 实现（tests/main/test_configuration.py） | 结果 |
|---|---|---|
| 24.1 单 Config | test_single_config（含尾随解包、Decimal 字段） | 通过 |
| 24.2 多 Config | test_multi_config_order_and_types（顺序/类型/元组/Literal）+ 重复请求独立实例 | 通过 |
| 24.3 单次读取 | test_toml_read_exactly_once（mock 计数，请求 7 类仅 1 次读盘） | 通过 |
| 24.4 未请求 section 容忍 | test_unrequested_sections_allowed | 通过 |
| 24.5 未请求段内未知字段 | test_unknown_field_in_unrequested_section_still_fails | 通过 |
| 24.6 未知 section | test_unknown_section_fails_with_path（含配置路径） | 通过 |
| 24.7 Required | test_missing_required_fails | 通过 |
| 24.8 Default | test_default_applied_when_field_absent | 通过 |
| 24.9 Path 三态 | 相对 TOML 目录 / 绝对 / ~（假 HOME 隔离）三条 | 通过 |
| 24.10 MB-OPC 回归 | test_mbopc_runners 22 例全绿（产物/merge 一次/正逆序/batch 不变/差异上界） | 通过 |
| 24.11 ILT 回归 | 不适用（ILT 未迁移，规格调整项 2） | N/A |
| 24.12 派生值 | 全量 444（solver 单测直接构造 DBU 包不变）+ smoke 数值合理 | 通过 |
| 24.13 final_cell_mode | 保留（调整项 3）；双模式覆盖测试（macro_cells 子 Cell 数/双模式 XOR=0）照常通过 | 通过 |

附加：类型严格性（int 拒 float/bool/string ×6 参数化）、Decimal 拒字符串、
Partition 互斥在 post_init、跨段契约（步长/探针/lr warning）经流程入口
前置检查毫秒级触发（test_mbopc_runners/test_gradient_mbopc_runner）。

## smoke 结果（CUDA）

| 流程 | 配置 | 关键数字 |
|---|---|---|
| simple（bench_30um [2,2]，8 轮） | mbopc_multi_macro.toml | 4 macro/672 core，47.6s，best_epe 1596/1011/820/497（逐轮单调降） |
| gradient（gcd_30um [1,1]，10 轮，用户实验参数） | gradient_mbopc.toml | 870 core，205s，loss 0.137→**0.0691**（−50%），CUDA 峰 501MiB |
| 验证管线（bench_30um） | macro_pipeline.toml | 总 4.1s，**最终 XOR==0**（回零验证） |
| 单遍（bench_30um +5nm） | single_pass.toml | 0.12s，唯一产物 |

## 数值一致口径说明

旧基线数字（simple 7264/5893/5640/4892、gradient −10.1% 等）的对照版图
gcd_45nm.gds 已被用户删除，无法做同版图逐位对比；等价性证据改为：
①配置层重构零算法路径改动（diff 审查：仅 loader/装配/字段访问）；
②全量 444 passed（含全部 solver 单测的直接构造断言）；③三流程 smoke
端到端行为与指标形态与历史一致（simple EPE 单调降、gradient loss 降、
管线 XOR=0）。

## 已知口径

- gradient smoke 用的是用户实验中的 config（gcd_30um [1,1] context 512
  iterations 10）——非本重构默认参数，数字与历史 smoke 不可直接比；
- `geometry/contour.py` 既存 ruff 告警沿用排除口径（未纳入本任务）。
