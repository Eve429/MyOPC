# 审查发现

## 已确认事实

- 项目主计划记录 Phase 5A lithography、Phase 5B evaluation、Phase 6A simple MB-OPC 已完成。
- 主计划声称全量 330 passed；本次必须重新验证当前工作树，不能直接复用该结论。

## Findings

### 基线与范围

- 当前分支 `migration`，HEAD=`0b21e54666f604a36c25e06edc6dbccf77eac394`；HEAD 是
  `test(mbopc): 完成端到端验证与迁移报告`。
- 迁移提交链与主计划一致：lithography `6338710/8773e37/5f0747a`，随后 contract、
  evaluation、坐标、共享生命周期、solver、runner 和报告至 `0b21e54`。
- 当前工作树非 clean：`AGENTS.md`、`CLAUDE.md`、MB-OPC 设计文档和
  `main/main_test_lithography.py` 有修改，另有 `.planning/`、模板目录。审查结论必须注明
  是针对当前工作树还是 HEAD。
- 当前未提交的 MB-OPC 设计文档已改为独立 macro 完整迭代/最终一次 merge；实际 HEAD
  代码需要核对是否已是该语义，不能仅凭修改后文档认定。
- 当前未提交的 `main/main_test_lithography.py` 扩充了演示几何和英文图题；它不是已提交
  迁移基线的一部分，应单独核对，不归因于迁移提交。

### Lithography 初读

- `lithography/iccad13.py::ICCAD13Lithography.forward_many` 共享一次 mask FFT，并按
  focus/defocus bank 各传播一次；没有逐 kernel Python 循环，符合设计性能路径。
- `ICCAD13Lithography._prepare_mask` 的 `.to(device, float32)` 仍在 autograd 图内；没有
  `detach/no_grad`，backward 结构上可贯通输入 mask。
- 输出在 sigmoid 后裁回原输入尺寸，单图 `[H,W]` 与 batch `[B,H,W]` 分支明确；需要由测试
  继续核对奇偶 padding、数值基线、有限差分和 CUDA parity。
- `lithography/contracts.py::LithographyModel` 只暴露 device/config/condition/forward_many，
  当前 simple MB-OPC 是真实调用方；未看到空注册器或统一 optimizer 抽象。
- Windows DLL 路径修复发生在 `import torch` 之前并保留目录句柄；这是报告记录的实测兼容
  路径，不属于静默 fallback。

### Evaluation / simple MB-OPC 初读

- `evaluation/metrics.py` 的 L2/PVBand 只统计 ownership；EPE 对越界、重合探针和 target
  内外语义无效化，inner/outer 同时违规显式记 ambiguous 且方向为 0，逻辑与规格一致。
- `evaluate_and_propose` 的 current 在整次 macro 评价中只读，方向写独立 next；owner 写集在
  出口核对，context 始终为零，batch size 不改变同步语义。
- 性能疑点：`problem.segments.materialize()` 在每次 `evaluate_and_propose` 调用都执行；
  `optimize_macro` 每轮都会重新进入该函数，因此固定参考端点/法向可能被反复物化。对大 S
  macro 这是 O(iterations×S) 的重复 CPU/内存工作，需结合 `SegmentBatch.materialize` 核实。
- 性能疑点：baseline 在 `epe>0` 但 `proposal.moved_segments==0` 时没有立即停止，而是重建同一
  状态并再做完整一轮 raster/lithography/evaluation，之后才报告 `no_update`。
- 最后允许的 round 仍以 `can_update=True` 生成一个不会被消费的下一轮位移提案；指标本身属于
  已移动候选，语义没有混写，但存在不必要的方向计算/写入并可能影响停止原因优先级。
- `SegmentBatch.materialize` 确认会重新分配 starts/ends/normals 等全长浮点数组；上面的重复
  物化不是常数级调用，确属随 segment 数线性增长的重复热路径。
- `optimize_macro` 捕获 `(ValueError, ReconstructionError)`；当前内部候选 shape 固定正确，
  `reconstruct_region` 的 ValueError 代表调用契约/程序错误，不应被转换成
  `invalid_geometry`。按项目“禁止吞错误、信任内部契约”规则，疑似只应捕获领域
  `ReconstructionError`。
- GPU 热路径问题：`evaluate_and_propose` 在每个 core 的回切循环内分别执行
  `valid.sum().item()`、`ambiguous.sum().item()` 和 `directions[piece].cpu().numpy()`；这些会
  形成每 core 多次 GPU→CPU 同步/小拷贝。应至少按整个 batch 一次汇总标量、一次取回方向，
  再在 CPU 上按 piece 分配，保留 batch 的意义。

### Main / workflow 初读

- 单/多入口均为薄 main，共享 `_run_mbopc`；多 macro 按稳定顺序各自跑完全部轮次，最后只调用
  一次 `merge_macro_results`，与用户最新选择一致。
- `save_final_lithography` 虽然 GPU tensor 按 batch 释放，但先用固定
  `[-2^30,2^30]` 查询框把最终目标层完整 `.materialize()` 成一个 Region，再逐 tile 栅格。
  这既可能漏掉合法范围外几何，也不是真正的大版图几何流式处理；配置默认样例启用该功能时，
  full-reticle 可能在进入 batch 前耗尽 CPU 内存。
- `_run_mbopc` 在 `prepare_problems()` 完整准备之后才检查 single/multi macro 数量和每 macro
  tile 数；错误入口可能先付出全部边段物化与 NPZ 写出成本再失败。
- `load_config` 用 `int(value)` 读取整数项，TOML 浮点 `1.5` 会被静默截断，布尔值也会被当成
  `1/0`；这不符合配置字段的整数 contract，测试列表尚未覆盖该输入。
- `solve_macro` 的 tqdm 关闭不在 `finally` 中；求解异常会保留未关闭进度条。属于终端资源清理
  问题，不改变算法状态。
- `merge_macro_results` 对每个 macro GDS 使用固定 ±2^30 查询框，并把所有 clipped Region
  同时保存在 `patches`，写出后又完整物化最终层做面积验证。它提供了正确性复核，但不是
  O(单 macro) 峰值内存的 merge；需对照报告判断这是已披露限制还是能力误报。

### 测试覆盖初读

- 光刻测试覆盖配置、资产哈希、shape/padding/Y 方向、OpenILT 数值和、batch/单图一致、
  FFT/bank 调用计数、有限差分、CUDA parity、直接环境和 main，核心合同覆盖较扎实。
- MB-OPC 的图形矩阵大多数只断言“以四种允许原因之一停止、位移有限/context=0”，没有断言
  对应图形的具体边段归属、移动方向、重建覆盖或预期非法原因。场景名称覆盖多，但对图形逻辑
  回归的约束偏弱。
- `TestSingleVersusMulti.test_difference_area_is_quantified` 只断言 `difference >= 0`，对任何
  Region XOR 都恒真；它既不要求预期非零，也不检查差异集中在 macro 边界带，无法自动拦截
  “意外变成同结果”或“差异扩散到内部”的回归。
- runner 配置测试未覆盖 fractional/bool integer 字段，因此 `int()` 静默截断没有被发现。

### 实际验证

- 当前工作树目标套件：
  `D:\\app\\miniforge\\envs\\myopc\\python.exe -m pytest -q tests/lithography tests/evaluation tests/opc/iteration tests/main/test_mbopc_runners.py`
  → **178 passed in 60.85s**。
- 通过结论仅证明当前断言成立；弱断言和未构造的明确边界场景仍属于覆盖缺口。
- 当前工作树全量：`...python.exe -m pytest -q tests` → **330 passed in 70.62s**。
- 目标范围 ruff：All checks passed；compileall 通过；`git diff --check` 通过（仅有 Windows
  LF→CRLF 提示）。
- 配置最小复现：把 TOML 解析结果中的 `iterations=1.5`、`batch_size=true` 传给
  `main._mbopc_workflow.load_config`，实际得到 `iterations=1, batch_size=1`，确认存在静默
  截断/布尔接受，不只是静态推测。
- no-update 最小复现：4 core、batch=2 的全 ambiguous baseline 得到 2 条完全未移动记录，
  `forward_many` 共调用 4 次；正确早停只需要 baseline 的 2 次 batch 调用。
- 物化计数最小复现：baseline+2 round 时 `SegmentBatch.materialize(None)` 被调用 3 次，固定
  参考几何确实每次评价重复生成；另有 4 次带位移物化（包含 baseline/target/candidate 重建）。
- 实际几何保护复现：4 DBU 宽矩形左边越过右边时，`reconstruct_region` 正确抛
  `reconstructed ring reversed orientation`；2 DBU 壁厚环的外 hull 缩入 hole 时，正确抛
  `reconstructed hole escaped its hull`。因此这两项当前是 MB-OPC 回归测试缺失，不是重建
  守卫失效。
- **高优先级正确性问题**：按用户明确场景构造 2 DBU 壁厚 hollow + 8 DBU EPE 探针，真实
  ICCAD13 baseline 返回 `valid_probes=0, epe=0`，`optimize_macro` 随即以 `zero_epe` 停止。
  由于 inner 探针跨过窄壁落入 hole，target inner/outer 语义均不成立；“无法评价”被错误解释
  为“零违规”。如何处理薄于探针距离的图形需要产品/算法裁决（拒绝 problem、记录
  insufficient probes、或自适应距离），但 MUST NOT 报 zero_epe。

### 依赖方向

- 搜索未发现 `layout/geometry/opc.input` 反向导入 main、具体 iteration、lithography 或
  evaluation；simple MB-OPC 单向消费 evaluation/lithography，main 消费 iteration，符合
  项目依赖方向。

### 结构与可扩展性

- 当前主要文件虽为 350 行左右，但职责边界清楚：iccad13 数值模型、simple solver、MB-OPC
  workflow、macro 生命周期各自独立；Config/Cache/Step/Record/Result 均有真实调用方，没有
  发现因 bug 修复遗留的空结构或未调用包装函数。
- `LithographyModel` 的签名直接引用 `iccad13.ProcessCondition`，而该类型把 kernel bank 限定为
  `focus/defocus`；因此它能解耦 simple solver 与 ICCAD13 类名，却不能保证未来不同工艺条件
  模型在不改接口的情况下替换。当前 ICCAD13/MB-OPC 功能不受影响，属于后续模型兼容性风险。

### 文档一致性

- `doc/development_manual.md:68` 仍指向已被共享生命周期重构移走的
  `main.run_macro_pipeline.PipelineConfig/load_config`；当前真实符号是
  `main/_macro_pipeline.py::MacroPipelineConfig/load_macro_config`。
- 同一手册 `doc/development_manual.md:72-73` 仍声称 lithography“不建 Protocol”，但当前已经有
  `LithographyModel/LithographyConfigView`，且 simple MB-OPC 是真实调用方。历史 lithography
  开发报告保留当时结论可以接受，但当前开发手册必须反映迁移后的最终状态。

## 最终分级

### P1 — 完成声明前应修复

1. **无有效 EPE 探针被误判为 zero_epe**：2nm 壁/8nm 探针已真实复现；属于算法正确性。
2. **大版图最终路径仍全局物化**：final PNG 前完整物化 reticle，merge 同时持有全部 patch
   并回读完整覆盖；未证明 64GiB 主机可运行普通 full reticle，且默认配置开启 PNG。
3. **整数配置静默转换**：fraction/bool 被 `int()` 接受，已真实复现，违反显式配置契约。

### P2 — 建议紧随修复

1. 固定参考 SegmentGeometry 每轮重复物化，且 EPE 回切每 core 多次 GPU 同步/小拷贝。
2. baseline 无移动提案仍完整重算一轮；最终轮生成未消费 proposal。
3. 实际越界守卫有效，但设计指定的 2nm/8nm、outer→hole、left→right 缺真实回归；图形矩阵
   大多只断言允许停止集合，single-vs-multi 差异断言 `>=0` 无回归约束。
4. 内部 `ValueError` 被转换成 invalid_geometry；single/multi 数量校验发生在完整准备之后；
   tqdm 异常路径未 finally 关闭。
5. LithographyModel 条件类型仍绑定 ICCAD13 focus/defocus；开发手册有两个过时接口描述。

### 通过项

- ICCAD13 数值前向、资产身份、batch 复用、autograd/有限差分、CPU/CUDA parity 和直接入口证据完整。
- owner/context、Jacobi 同步、round 后指标、best 选择、独立 macro 最终一次 merge 的主逻辑正确。
- 依赖方向、主要结构必要性、异常不静默、直接 Python 运行和产物合同总体清晰。
- 当前树目标 178 passed、全量 330 passed、ruff/compileall/diff-check 通过。
