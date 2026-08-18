# ILT 多方法迁移规格制定计划

## Goal

基于当前 `2fa75ea` 源码架构与可核验参考实现，为若干 ILT 方法分别编写可由另一名 AI 独立实施的 Markdown 规格；首个方法同时冻结后续方法复用所需的最小兼容契约。

## Phases

### Phase 1：当前架构与既有契约核对 — Status: complete

- 读取当前配置、光刻、栅格、评价、MB-OPC workflow、文档契约和测试。
- 明确 ILT 可复用边界与不得反向依赖的模块。

### Phase 2：参考 ILT 方法盘点 — Status: complete

- 核对 OpenILT 中各 ILT 方法的算法、输入输出、状态、依赖和测试资产。
- 只选择源码事实足够、与当前项目目标相符的迁移候选。

### Phase 3：迁移顺序与兼容性决策 — Status: complete

- 确定首个方法及其最小公共契约。
- 明确每个后续方法复用什么、专有内容放在哪里、哪些内容不迁移。

### Phase 4：分别编写 implementation spec — Status: complete

- 每个方法单独一个 Markdown。
- 使用 `doc/implementation_spec_template.md` 的结构，写清具体文件、符号、数据契约、算法、测试与验收。

### Phase 5：交叉审查与交付 — Status: complete

- 检查规格间接口一致性、无过度抽象、无未披露假设。
- 检查只产生文档与规划记录，不修改业务代码。

## Result

- 已生成四份独立 draft implementation spec。
- 当前业务基线 446 passed；文档静态检查通过。
- `opc/input/grid.py` 的外部/并发注释改动已隔离，未由本任务修改或纳入计划提交。

## Constraints

- 本轮不实施 ILT，不修改 `layout/`、`geometry/`、`00_PAST/`。
- 无法从源码确认的产品选择写入 Blocking Open Questions，不自行决定。
- 计划必须以当前 HEAD 为基线，不能沿用旧目录或已删除接口。

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| PowerShell 下把 `00_PAST/tests/opc/test_*ilt.py` 作为 rg 路径导致 Windows 路径错误，命令退出 1 | 1 | 后续把目录作为搜索根并用 `-g 'test_*ilt.py'` 过滤，不重复该写法 |
| 全量 pytest 首次只给 1 秒 timeout，进程被工具终止且 stdout 收尾报 OSError | 1 | 这是调用参数错误；下一次使用足够 timeout 完整执行，不把该结果当测试失败 |
| PowerShell 再次把带 `*` 的多级路径直接交给 rg，产生 Windows 路径错误；其余审查命令继续完成 | 1 | 后续只把目录交给 rg 并使用 `-g` 过滤；不把该输出当占位符审查结果 |
