---
id: CHG-20260818-config-system-refactor
title: 配置系统重构（统一 load_config + 业务 Config 划分）
type: implementation-spec
status: completed
---
# 配置系统重构（摘要版规格）

> 完整规格为用户本地文档 `MyOPC_config_system_refactor_spec.md`（1638 行，
> 未入库）；本文件为入库摘要。实施批准计划含 7 项按仓库现状调整。

## 需求摘要

统一 `load_config(path, *config_types)`（单次读 TOML、声明式
CONFIG_SECTIONS 映射、未请求 section 全量未知字段检查、required/
default 由 dataclass 表达、Path 三态、Decimal 精确链）；Config 按业务
划分（Layout/Partition/Lithography/Edge/MBOPC/Gradient/SinglePass/
Validation/Output）；删除 MacroCommon/MacroPipeline/MBOPCRun/
GradientMBOPCRun 四中转 Config 与五个旧 loader；runtime 派生值
（step_dbu 等）不进 Config；solver 层 Simple/Gradient MBOPCConfig
按 §25 豁免保留。

## 7 项规格偏差（均经用户批准）

GradientConfig=gradient 算法段解读；ILTConfig 不建（未迁移反投机）；
final_cell_mode 保留（macro_cells 真实第二模式）；新增 EdgeConfig 与
ValidationConfig/SinglePassConfig 段名（[iteration] 同名冲突）；
4 config 切 bench_30um（gcd_45nm 已删）；数值一致口径改全量+smoke。

## 提交链

94cd621（批 0 入口合并）→ 1db593d（批 1 configuration.py）→
8f71b5a（批 2+3 全流程迁移）→ bebddf2（批 4 报告）。

验证：全量 444 passed；四 smoke 全绿。详见同目录两报告。
