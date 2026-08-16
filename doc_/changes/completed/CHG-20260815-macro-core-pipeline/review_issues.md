# Macro–Core 管线审查问题清单

日期：2026-08-15
审查范围：`doc/macro_core_pipeline_design.md` 对应的 Phase 4 实现与自动测试。
用途：本文只记录已经由代码、测试或只读诊断确认的问题，供项目所有者逐项实现；不包含实现代码。

## 1. 当前验证基线

- `python -m pytest -q tests`：`115 passed`。
- 专项 Ruff：通过。
- `compileall`：通过。
- Coverage：相关源代码整体 `84%`；本阶段核心文件约 `88%～95%`。
- 已独立回读 `gcd_45nm` smoke 产物：源版图与最终版图面积均为
  `28,594,652,500 DBU²`，XOR 面积为 `0`，polygon 数均为 `1776`。
- 两轮产物数量正确：每轮 4 个 macro GDS 和 4 个 result NPZ。

上述结果说明当前主链路可以运行，但不能代替下面尚未完成的专项测试和契约校验。

## 2. P1：补齐复杂几何自动测试矩阵

### 2.1 当前问题

设计文档 §15.3 要求动态覆盖：矩形、长条、凹多边形、孔洞、2 nm 窄环、多角度
斜边、相接/重叠图形、SREF、2×2 AREF，并分别跨 macro 横边、竖边、角点和多个
core。

当前 `tests/opc/input/test_macro_problem.py` 实际包含：

- 矩形和长条；
- 一个跨竖向 macro 边界的三角形斜边；
- donut；
- 一个 3×3 AREF。

当前没有独立覆盖：

- 凹多边形；
- 2 nm 窄环；
- 多角度斜边；
- 相接图形与重叠图形；
- 跨 macro 横边；
- 跨 macro 角点；
- 单独的 SREF；
- 设计指定的 2×2 AREF；
- 一个图形连续跨越多个 core 的组合情形。

测试报告目前把这一部分描述为已经覆盖，结论超过了实际断言范围。

### 2.2 需要增加的测试

建议全部使用生成式 KLayout 图形，不依赖用户 GDS：

1. 凹多边形分别跨横向、竖向 macro 边界；
2. 外框与内孔形成 2 DBU 或按测试 DBU 换算后的 2 nm 窄环；
3. 至少三种不同斜率的斜边跨竖边、横边和 macro 角点；
4. 两个 polygon 仅接触、部分重叠、完全包含三种组合；
5. 单个 SREF 跨 macro 边界；
6. 2×2 AREF 的多个 occurrence 分别落在不同 macro/core；
7. 长图形同时跨越至少三个 core。

每类图形至少验证：

- query/context 框没有成为物理边；
- owned segment 的开区间不跨 ownership 切线；
- owner 唯一且 `own ⊆ membership`；
- hole/ring 数量、绕向和 polygon 归属不变；
- 零位移逐 macro 裁 ownership、汇总并 merge 后与输入 XOR 为零；
- 共享斜边两侧使用相同参数分裂点。

### 2.3 完成标准

- 上述每种图形都有独立、可定位失败原因的测试函数；
- 测试报告不再使用其他图形“间接代替”2 nm 窄环、相接或重叠图形；
- 测试报告中的 SREF/AREF 尺寸与测试代码一致。

## 3. P1：落实冻结的配置契约

### 3.1 `macro_size_dbu > core_size_dbu` 未实现

设计文档 §5.3 要求名义 macro 大于 core，且为 core 的整数倍。当前
`opc/input/grid.py::plan_macros()` 只检查：

```text
macro_size_dbu > 0
macro_size_dbu % core_size_dbu == 0
```

因此 `macro_size_dbu == core_size_dbu` 当前会被接受。只读诊断已经确认这一行为。

需要明确实现选择：

- 若继续以批准的设计文档为准，应拒绝 `macro_size_dbu <= core_size_dbu`；
- 若项目所有者认为一个 macro 只含一个 core 合法，应先修改设计文档，再保留当前行为，
  不能让文档和代码维持冲突。

需要增加边界测试：

- `macro_size < core_size` 失败；
- `macro_size == core_size` 按最终决策成功或失败；
- `macro_size > core_size` 且整除成功；
- `macro_size > core_size` 但不整除失败。

### 3.2 双轮位移没有冻结为 `+2 nm/-2 nm`

设计文档要求本阶段固定执行 `+2 nm/-2 nm`。当前
`main/run_macro_pipeline.py::prepare_problems()` 只检查两项之和为零，所以
`[3, -3]` 等配置也会通过。

需要增加校验及测试：

- `[2, -2]` 成功；
- `[3, -3]` 失败；
- `[2, -1]` 失败；
- 项数不是 2 失败；
- `2 nm` 无法精确换算到当前 DBU 时失败并保留明确参数信息。

## 5. P1：避免阶段 0 逐 shape 跨越 Python/KLayout 边界

### 5.1 当前问题

`main/run_macro_pipeline.py::_layer_bounds()` 当前通过 Python `while` 循环逐 shape 获取
bbox 并累计极值。它不物化 Region，内存较低，但时间成本随 shape 数增长，而且每个 shape
都跨越一次 Python/KLayout 边界。

这与项目规则“批量跨越 Python/KLayout 边界，避免逐 polygon 热循环”冲突。当前
`gcd_45nm` smoke 只有 1776 个最终 polygon，不能代表完整 reticle 的形状规模。

### 5.2 实现边界

- 优先使用 KLayout 原生侧完成目标层层级 bbox 计算；
- 若现有 `LayoutDB` 公共接口无法提供目标层 bbox，按项目规则先说明是否必须最小修改
  `layout/`，不要从 main 访问 `_native_*` 私有字段；
- 不得用“先完整物化目标层 Region 再取 bbox”换取速度，因为这会破坏阶段 0 的内存边界。

### 5.3 测试要求

- flat、SREF、AREF、旋转/镜像实例的目标层 bbox 正确；
- 其他 layer 的远端图形不能扩大目标层 bbox；
- 测试应能证明 Python 侧没有逐 shape 遍历，例如替换/监控公开迭代入口；
- 不以耗时阈值代替调用路径断言。

## 6. P2：修正超出实际范围的测试断言和报告描述

### 6.1 正逆序测试只检查第一轮

`test_iteration_order_does_not_change_results` 的 docstring 和报告写的是“两轮状态和最终覆盖
一致”，但当前只运行第一轮并比较位移数组。

需要扩展为：

1. 正序完整运行第一轮、第二轮并合并；
2. 逆序在独立工作目录运行同样流程；
3. 比较两轮每个 macro 的 displacement；
4. 比较最终顶层覆盖 XOR；
5. 必要时比较 macro GDS 覆盖，不比较 GDS 二进制字节。

### 6.3 “不复制未处理层”测试为空验证

当前生成式输入 GDS 只有目标层 1/0，最终只含 1/0 不能证明其他层没有被复制。

需要在源 GDS 中增加至少一个非目标 layer，例如 2/0，并验证：

- 源版图确实同时包含 1/0 和 2/0；
- 最终版图只包含 1/0；
- 目标层覆盖仍正确。

## 7. P2：修复 `MacroProblem` 的空 membership 不变量漏洞

### 7.1 当前问题

`MacroProblem.__post_init__()` 仅在 `member_segment_indices` 非空时检查
`own ⊆ membership`。因此可以构造：

```text
owner_indices 中存在 owner >= 0
core_offsets 全为 0
member_segment_indices 为空
```

只读诊断已经确认：包含 16 个 owned segment、0 个 membership 的错误对象当前会被接受。

正常 `_build_macro_ownership()` 不会生成该状态，因此这是持久化加载和显式构造入口的
不变量漏洞，不是当前正常热路径错误。

### 7.2 完成标准

- 即使 membership 为空，也必须验证 `seen == (owners >= 0)`；
- 全部 segment 都是 context、owners 全为 `-1` 且 membership 为空时仍应合法；
- 增加上述两个最小回归测试；
- 不新增包装函数或第二套校验逻辑，直接修正现有交叉不变量检查。

## 8. P2：同步文档与实际状态

### 8.1 当前冲突

- `doc/macro_core_pipeline_design.md` 顶部仍写“等待用户审查，禁止开始实施”；
- `task_plan.md` 已把 Phase 4 标记为 complete，并声称 §21 逐项通过；
- 测试报告把尚未独立测试的复杂图形写成已经覆盖；
- 开发报告没有记录本次 coverage 未命中分支审计结果。

### 8.2 完成标准

所有 P1/P2 修复和测试完成后同步：

- 设计文档状态；
- `task_plan.md`、`findings.md`、`progress.md`；
- 开发报告；
- 测试报告；
- 开发手册与测试手册；
- Coverage 数值和关键未覆盖分支说明。

在问题修复前，文档应使用“主链路完成，审查问题待处理”，不要写“全部完成标准逐项通过”。

## 9. P2：注释规则的少量遗漏

AST 审查发现以下嵌套测试辅助函数没有 docstring：

- `tests/opc/input/test_macro_problem.py::_boom`；
- `tests/main/test_macro_pipeline.py::_counting_prepare`；
- `tests/main/test_macro_pipeline.py::_counting_materialize`。

此外，开发报告声称 `main/run_macro_pipeline.py` 每一行都有注释，但多行导入、函数签名、
字面量结束行等并不全部带注释。实现时需要在“逐行注释”与项目最高准则“注释解释原因，
不逐行复述语法”之间采用一致口径，并同步修正文档中的绝对化描述。

## 10. 修复顺序建议

1. 决定 `macro_size == core_size` 是否允许，并同步设计契约；
2. 修正固定 `+2/-2 nm` 校验；
3. 修复空 membership 不变量；
4. 补齐复杂几何测试矩阵；
5. 执行完整回归、Coverage、gcd smoke；
6. 最后统一更新报告和规划状态。

## 11. 最终验收清单

- [ ] P1/P2 条目均已处理或由项目所有者明确决定不处理；
- [ ] 全量 pytest 通过；
- [ ] 专项 Ruff 与 compileall 通过；
- [ ] 新增几何矩阵全部自动化；
- [ ] 固定双轮和 macro 尺寸契约与设计文档一致；
- [ ] 性能字段名称、取值和报告口径一致；
- [ ] `gcd_45nm` 2×2 smoke 仍为 8 个 macro GDS，最终 XOR 为零；
- [ ] `single_cell` 无 seam，`macro_cells` 物理覆盖与其一致；
- [ ] Coverage 未命中分支完成审计并记录；
- [ ] 文档不再声称未被测试覆盖的能力；
- [ ] 未修改 `00_PAST/` 和用户 GDS；
- [ ] 只提交本次修复相关文件，不包含无关工作树修改。
