# 基于梯度的 MB-OPC 迁移计划编制

## 目标

依据当前源码、测试、既有基础 MB-OPC 迁移规格与梯度方法参考实现，编写一份可脱离聊天上下文执行的 implementation-spec；明确算法正确性、接口变化、状态归属、性能边界和验收测试。

## 阶段

- [complete] 1. 核对当前工作树、规格模板和现有模块接口
- [complete] 2. 核对梯度 MB-OPC 参考算法及其依赖
- [complete] 3. 确定最小架构变化与算法正确性条件
- [complete] 4. 编写迁移 implementation-spec
- [complete] 5. 交叉检查规格完整性、可执行性与过度抽象

## 约束

- 本任务只编写计划，不修改业务代码、测试代码、`layout/`、`geometry/` 或 `00_PAST/`。
- 以当前未提交工作树为 Current Behavior 依据，保留用户和其他开发者的修改。
- 无法从源码或参考实现确认的算法选择必须进入 Blocking Open Questions，不得猜测。
- 仅引入有当前调用方且实现梯度方法必需的抽象。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| `web.run click` 组合请求语法错误 | 1 | 改用直接 URL |
| GitHub API/DOI 直链被判 unsafe | 1 | 改用 GitHub HTML/raw 与检索结果 |
| 三个 GitHub raw 文件 cache miss | 1 | 核心文件已读取；其余改用 HTML/源码搜索 |
| `rg` 包含不存在的 `configs/` 路径 | 1 | 改用仓库实际 `config/` 路径 |
| 官方配置目录/raw YAML cache miss | 2 | 不依赖参考默认值；本项目配置全部显式必填 |
