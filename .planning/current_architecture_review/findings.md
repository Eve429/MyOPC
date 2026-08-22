# 当前项目架构复核发现

## 基线

- 当前 HEAD：`8244f59 docs: 风格迁移三文件记录`。
- 工作树存在用户修改：`config/gradient_mbopc.toml`、`config/mbopc_single_macro.toml`、`TestReticle/p50_1024/p50_1024_loose_clear.gds`，以及未跟踪 `p50_1024_dense_line_clear.gds`。
- 当前阶段记录：Simple/LevelSet/CurvMulti ILT 已完成；Multilevel ILT 仍为 active draft；Phase 7 收尾审计未完成。

## 初步模块事实

- 顶层模块：`layout`、`geometry`、`opc.input`、`opc.iteration`、`lithography`、`evaluation`、`main`、`common`。
- 入口：Macro 管线、Simple/Gradient MB-OPC、Simple/LevelSet/CurvMulti ILT、单遍 bias、GDS 光刻验证。
- 文档体系已拆成 architecture/contracts/changes/review/manuals，旧材料进入 archive。

## 待核对

- 文档所述依赖方向是否与真实 import 一致。
- MacroProblem、PixelMacroProblem、各 solver state/result 的真实 owner、生命周期和写出边界。
- 独立 macro、同 macro 内同步 core、ownership-only 写回等不变量是否落在源码和测试中。
- 当前测试基线与文档记录的 695 passed + 1 skipped 是否一致。

## 文档宣称的当前架构

- 合法依赖主线：`layout -> geometry -> opc.input -> edge/pixel`；具体迭代方法向下依赖输入、光刻和评价，`main` 只编排。
- Edge 路径持久对象是 `MacroProblem`（NPZ v3），参考数组只读，可变状态是每段一维位移；Pixel 路径持久对象是 `PixelMacroProblem`（NPZ v1），每 macro 只保存 query transmission，不保存重复 core canvas。
- 全部求解工作流按“prepare 一次 → macro 独立求解 → 最终 merge 一次”执行；GPU 仅保存当前 batch，不物化整张 reticle tensor。
- MB-OPC 有离散 EPE 驱动和梯度法；ILT 有 Simple、LevelSet、CurvMulti，并已抽出三者确实共用的 `_skeleton` state×batch 循环。
- 光刻层只有 ICCAD13 Hopkins 具体实现和一个薄 `LithographyModel` Protocol；evaluation 是独立纯消费者。

## 源码规模与结构

- 生产 Python 约 12k 行；较大的核心文件是 `gradient.py` 682 行、`_macro_pipeline.py` 536 行、`problem.py` 435 行、`configuration.py` 435 行、`simple.py` 390 行。
- `main` 的共享编排与方法入口已分层，但方法注入对象 `MBOPCMethod`/`ILTMethod` 仍需源码核对是否是实际多调用方而非未来抽象。
- 文档明确记录当前已知限制：macro 间不交换优化后状态，ILT 输出为像素阶梯几何，且目标 bbox 需整像素。

## 输入对象源码核对

- `MacroSpec` 不保存 CoreSpec 列表，按索引即时构造；macro/core ownership 都是不重叠半开网格，context 仅扩张读取范围。
- `MacroProblem` 的字段与文档一致：`macro/layer/polarity/fragmentation/segments/owner_indices/core_offsets/member_segment_indices`。owner 是每段唯一写者；CSR membership 是每 core 可见段集合，两者没有合并成一个概念。
- `SegmentBatch` 以 ContourBatch 为几何唯一持有者，段只保存 edge_id+t0/t1；边级 next/polygon/normal 各存一次，没有按段复制 polygon/ring/hole 元数据。
- `PixelMacroProblem` 只持有 query transmission uint8；target/ownership/trainable/valid canvas 都按 core 即时构造。trainable 索引是 int64，且只构造当前窗口的索引块，没有 O(macro pixels) 的临时全局 arange。
- 两条 prepare 路径对 opaque 都在栅格化/提边前补画 `query - data_bounds` 铬区并 merge；clear 不补。极性进入求解器前统一为 transmission。

## 工作流源码核对

- MB-OPC 与 ILT 都逐 macro 加载 problem、完成该 macro 全部迭代、写 NPZ/JSON/GDS，再释放并处理下一 macro；全部 macro 完成后仅调用一次 `merge_macro_results`。
- ILT 公共骨架确有三个当前调用方；`BatchPack` 每 macro 构建一次并跨 state 复用，GPU 每批仅保留当前画布与 autograd 图。
- ILT state 内全部 core/batch 读同一参数快照，批间只累积梯度，更新器在 state 屏障后执行；最终 binary 指标另做一次只读前向。
- `ILTMethod` 与 `MBOPCMethod` 是 main 层真实方法注入点，分别已有 3 个和 2 个入口调用方，不属于无调用的一次性抽象。

## prepare、合并与依赖源码核对

- Edge prepare 按 macro 逐个 `materialize_intersecting()`、构造并立即保存 NPZ，记录统计后删除内存对象；不会把全 reticle 边段同时放入内存。Pixel prepare 采用同一 macro 生命周期。
- 最终 merge 先逐 macro 回读完整候选，精确裁到该 macro ownership，构成 `GeometryPatch`；`PatchWriter.write_macro_results` 负责全局表示合并/写出，随后再按 macro 窗口回读核对覆盖面积守恒。
- 真实顶层 import 依赖为：layout 无第一方依赖；geometry→layout；lithography/evaluation 独立；opc→common/evaluation/geometry/layout/lithography；main→全部下层。未发现基础层反向 import main/iteration 的拓扑违例。
- Simple MB-OPC 是 Jacobi 式“全 batch 评价同一 current，再发布 next”；Gradient MB-OPC 和三个 ILT 均在完整 macro 的 state 屏障后更新参数，不会因批完成回调提前发布边或像素状态。
- 当前 macro 策略明确是独立求解、固定初始 context；合并只解决最终几何覆盖与表示 seam，不解决相邻 macro 优化状态耦合。

## 测试清单

- 当前收集到 696 个测试；compileall、ruff check、ruff format --check 均通过。
- 测试不仅覆盖字段校验，还明确覆盖跨 3 core 图形、跨 macro 斜边、中空/窄壁/凹多边形、边反穿导致的非法几何、双极性、batch 不变性、state 屏障、缓存命中、真实 ICCAD13 数值与 autograd、三种 ILT、最终 merge 恰一次和入口产物 schema。
- 第一次完整回归因外部 244 秒工具超时被终止，尚不能据此确认 695 pass + 1 skip；需延长工具超时复跑。

## 最终验证基线

- WSL `myopc312`：`695 passed, 1 skipped, 3 warnings in 216.82s`。
- 唯一 skip 是 LevelSet 单类退化的一个参数化项，测试明确说明由常量场用例覆盖；3 个 warning 都是 field 大于 layer bbox 时的预期正板环带提示。
- 因此当前文档记录的 695 passed + 1 skipped 与实际工作树一致。

## 当前能力边界

- 已实现：GDS/OASIS/GLP 读取；Macro–Core 规划；edge/pixel problem 持久化；Simple/Gradient MB-OPC；Simple/LevelSet/CurvMulti ILT；ICCAD13 可微光刻；L2/PVBand/EPE；单/多 macro 编排、进度、产物和最终光刻 PNG。
- 未实现或不应误报：Multilevel ILT（仅 active draft）；macro 间迭代状态交换；整 reticle GPU 常驻；ILT EPE/shot/MRC；消除独立 macro 优化导致的物理 seam。
- 当前“最终全局 merge”只保证 ownership 后物理覆盖合并与面积守恒；它不会重新协调各 macro 的优化决策。
