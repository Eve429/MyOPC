# 光刻模型与基础 MB-OPC 迁移审查计划

## 目标

只读审查已完成的光刻模型与基础 MB-OPC 迁移，核对设计规格、实现、测试、运行入口、性能/
内存契约和交付文档；输出按严重级别排列、带符号与证据的审查结论，不修改业务代码。

## 基线

- 当前分支与 HEAD：待核对。
- 光刻规格：`doc/lithography/lithography_migration_design.md`。
- MB-OPC 规格：`doc/opc/mbopc_migration_design.md`。
- 当前工作树非 clean；必须区分 HEAD 与未提交修改。

## 阶段

- [x] A. 核对 Git 基线、实际改动范围、规格与报告的一致性。
- [x] B. 审查 lithography 公共契约、数值路径、autograd、设备和资产处理。
- [x] C. 审查 evaluation 与 simple MB-OPC 状态/ownership/同步/异常/内存路径。
- [x] D. 审查单/多 macro main、merge、配置、产物和直接运行语义。
- [x] E. 审查测试矩阵并运行目标测试、静态检查和必要的最小复现。
- [x] F. 汇总 findings，区分 blocker/P1/P2/已知限制，并给出总体完成质量。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---|---|
| 组合读取 commit 范围与工作树 diff 的并行命令整体返回 1 | PowerShell 中首个 git 范围命令未返回结果，第二个 diff 正常输出 | 后续把 commit 范围查询拆成独立命令，不重复整组调用。 |
| 默认 `python -m pytest` 失败：base 环境没有 pytest | 1 | 改用报告记录的 `D:\\app\\miniforge\\envs\\myopc\\python.exe`，不在 base 安装依赖。 |
| myopc 目标测试在 1 秒工具超时内未完成 | 1 | 这是超时参数不足，不是测试失败；改为 5 分钟上限执行。 |
| 依赖/审计组合搜索返回 1 | `rg` 对无 TODO/宽泛 except 的正常零命中返回 1，导致组合调用标红 | 已取得依赖搜索输出；后续将文件统计与零命中搜索拆开，并显式输出结果。 |
| 文档旧符号搜索返回 1 | Windows `rg ... *.md` 不接受该 glob 写法；doc 内结果已正常输出 | 不重复该 glob；已有命中足以定位手册不一致。 |
