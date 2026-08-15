# 大 Reticle 流式处理独立开发方案

> 当前已完成阶段 2 的前端基础：未裁剪相交物化、tile 对齐 macro planner、逐 macro 局部边段/owner/membership 准备、逐 tile 栅格对照和局部容量守卫。磁盘 shard、全局状态文件、多轮流式求解、恢复和流式最终输出仍未实现，不得描述为当前能力。

## 1. 当前能力、瓶颈与目标

当前两个求解器仍让 CPU 保存完整 ROI 的物理 `Region`、参考边段、membership、`d_current/d_next` 和拓扑；GPU 只保存当前 batch 的 core+tile halo 张量。新增 macro 前端可在不构造完整 ROI Problem 的情况下逐 macro 验证几何和 raster，但尚未把这些局部问题接入多轮求解状态。

因此 24 GiB GPU 不是整版运行的首要限制，64 GiB CPU 内存中的完整 `Region`、数十亿至百亿边段、membership 和全局重建才是阶段 2 要解决的问题。阶段 2 的目标不是让全部数据进入内存，而是只展开当前 macro 和当前 GPU batch；即使运行很慢，也必须在 24 GiB GPU、64 GiB CPU 和可配置本地磁盘空间内有界运行。

## 2. 不可破坏的全局语义

- 一次 prepare 内固定参考数学边、分段相位、法向、拓扑顺序和 owner；普通迭代只更新相对参考边界的绝对位移，只有显式 remesh 才能重新提边和迁移优化器状态。
- 所有 macro/tile 在第 `k` 轮只读同一份 `d_current[k]`；owner 是 segment 唯一写入者，所有 macro 完成后才原子发布 `d_next[k+1]`，禁止边计算边修改本轮输入。
- core 和 macro 都只是调度、加载与写入归属边界，不能成为新的物理边；跨边界数学边不能按 core 或 macro 重新起算分段。
- halo 至少覆盖光刻模型有效影响半径、最大允许边位移和栅格化安全余量；halo 只读，自身 ownership 区域才允许累计指标和 owner 更新。
- GPU 只保留当前若干 core+halo 的 mask、光刻中间量和局部评估结果；CPU 不常驻整版像素图，也不保存全部 tile 输出。
- 源层级 GDS/OASIS 始终只读；PNG、完整诊断和中间 GDS 仅在明确请求时生成，不能进入每轮热路径。

## 3. CPU macro 与 GPU core 的两级切分

两层切分解决不同问题，不能合并成一个网格：

1. **CPU macro 层**控制完整候选图形、参考边段、局部拓扑和状态页的工作集，使进程常驻内存不随整版面积线性增长；
2. **GPU core+halo 层**控制单次光刻张量和 batch 大小，使显存峰值稳定低于配置预算。

每轮按以下顺序执行：读取当前 macro 的参考 shard 和 `d_current` 页，按其 active core 生成 GPU batch，立即累计自身 core 的 L2/PVBand/EPE 与 owner 更新，写入对应 `d_next` 页后释放所有 GPU 输出和 macro 临时对象。所有 macro 完成并通过重复写入/缺失写入检查后，交换 current/next 文件并进入下一轮。

不能让 macro 0 独立完成全部迭代后再处理 macro 1，因为后续轮次的光学上下文会读取邻近 macro 的更新；这种做法会让结果依赖 macro 顺序。允许的是“一轮内展开 macro 0、计算并释放，再展开 macro 1”，轮末统一发布。

## 4. macro 边界与精确 ROI 裁剪

### 4.1 两类边界

每个 macro 同时具有：

- `ownership_box`：决定哪些参考区间、segment、指标和更新由该 macro 唯一发布；采用全局半开边界规则消除边界点双 owner；
- `context_box`：`ownership_box` 向外扩展准备 halo，只负责提供邻域完整图形和只读光学上下文。

`ShapeQuery.materialize()` 继续返回精确裁到查询框的 `Region`，适合显示、ILT 像素 ROI 和普通查询。`materialize_intersecting()` 已新增：它使用同一原生层级空间索引，只扁平化与查询框相交的完整 occurrence，不与框求交，因此查询框不会成为物理边。

### 4.2 真实边提取路径

当前 `run_mbopc_frontend.py --macro-verify` 已实际消费该入口；每个 macro 在物化前独立预检，完成准备和 tile raster 对照后立即释放。这个入口不生成 shard，也不会调用 simple MB-OPC/DiffOPC solver。

单个 macro 的准备顺序固定为：

1. 批量取得与 `context_box` 相交的完整候选 occurrence，不逐图形跨越 Python/KLayout；
2. 在 macro 临时工作集中按物理覆盖合并，恢复孔洞并提取真实数学边；
3. 只发布数学边落入 `ownership_box` 的参数区间，丢弃 context 外结果，但不把参数区间端点之间的 macro 框线当成边；
4. 数学边的整数直线、方向和全局分段相位来自完整真实边，segment 以全局参考起点切分，再筛选 owner 区间；
5. 写入 shard 后立即释放候选 `Region`、临时轮廓和 Python 对象。

斜边即使跨多个 macro，也必须由同一全局直线和分段相位计算交点。禁止相邻 macro 分别对裁剪后的短边重新均分，否则同一理论交点会因独立整数量化得到 33/34 DBU，进而产生断点、重复 segment 或微小缝隙。

### 4.3 去重和拓扑

- macro 准备记录完整数学边描述及其 owned 参数区间；相邻 macro 的重复 context 记录通过精确整数端点、方向和局部拓扑做本次 prepare 内去重，签名不承诺跨 remesh 或跨输入文件稳定。
- ring/polygon 跨 macro 时不能由各 macro 独立闭合。参考记录采用 shard-local `int32` 索引和全局 `int64` shard offset，轮次只消费局部索引；最终按端点邻接和确定性顺序恢复全局 ring、hole 与 polygon owner。
- 如果一个物理 polygon 横跨多个 macro，默认由固定 polygon owner 写最终结果；非 owner macro 只提交其拥有的边段更新。只有在单 polygon 本身超过工作集预算时，才允许在最终输出阶段做确定性精确切片，并验证相邻片的 gap、正面积 overlap 和 XOR 都为零。

## 5. 全局状态的两种存储路径

阶段 2 只保留一个逻辑问题和同一套求解语义，根据预检结果选择内部存储后端，不建立两个公开 `MBOPCProblem` 类型。

### 5.1 RAM 紧凑状态

当参考 shard 保存在磁盘、但紧凑索引和 `d_current/d_next` 能进入内存时，位移、owner 写入 bitset、少量优化器状态常驻 RAM；macro 几何、membership 和像素按需加载。这是速度优先路径。

### 5.2 完全 out-of-core 状态

当仅位移和 owner 状态也超过内存预算时，使用预分配二进制 shard/memmap 保存 `d_current/d_next`、写入 epoch/bitset 和必要优化器状态。每次只映射当前 macro 的连续页，轮末通过文件级 generation 元数据原子发布；中断时保留上一代完整 current，不能留下 current/next 混合状态。

百亿 segment 不能继续使用单一全局 `int32` 下标。每个 shard 内使用 `int32` 以减小内存，全局定位使用 `int64 shard_offset + local_index`。默认位移为 `float32`；如果磁盘容量仍不足且迭代步长是离散 DBU，可在经过数值等价测试后启用有界 `int16/int32` 位移格点，不能无验证地降低精度。

预检同时报告 RAM 峰值、磁盘持久状态、临时双缓冲和 checkpoint 空间。磁盘不足与内存不足一样在物化前拒绝，不能依赖操作系统分页文件兜底。

## 6. active core、membership 与迭代

- 规则 `x_cuts/y_cuts` 保留全局坐标和 owner 语义，只为有 mask 或可能受 halo 影响的 active core 保存 ID；近满铺版图可走密集内部快路，但公开接口和数值结果不变。
- membership 按 macro 生成并随 shard 保存或按成本重建，不形成整版 Python 对象表；GPU batch 只读取当前 active core 的连续 membership 范围。
- `d_next` 在每轮开始时按状态策略初始化，owner segment 直接 scatter 到固定位置；epoch/bitset 检测重复写入和漏写，非 owner contribution 只能参与只读指标归约。
- L2/PVBand/EPE 使用可结合的标量/计数器流式归约，不保存整版 tensor。归约发生在当前 `d_current` 上，不会提前改变边；只有轮末状态文件发布才改变下一轮可见位移。
- 第一版每轮遍历全部 active core。只有确认光学有效半径和 dirty 传播规则后，才允许跳过 clean tile；“当前 core 没有 owner 更新”不足以证明其光学上下文未变化。

## 7. ILT、PB-OPC 与层级版图复用

- macro planner、active tile、halo、栅格调度、GPU batch、标量归约和 out-of-core generation 可由 MB-OPC、ILT 与像素型 OPC 共用。
- segment、owner、ring 拓扑和边重建只属于边段型 OPC，ILT 不依赖这些结构；ILT/PB-OPC 的主要 CPU 压力是整版像素、梯度和优化器状态，应使用同一 macro 调度但保存像素 shard。
- 当前物化会把 SREF/AREF occurrence 转为 top 全局坐标。阶段 2 第一版仍按物理 occurrence 独立优化，不修改 master cell；层级 cell variant 复用属于后续独立阶段，不能阻塞正确的整版流式路径。

## 8. 实施阶段与提交边界

1. **测量与契约基线**：选择真实大版图，记录 active core 密度、完整 Region、segment、membership、状态和磁盘估算；冻结一次性 prepare/迭代对照结果。
2. **active core 与 shard 格式**：实现 shard-local 索引、全局 offset、原子 generation 和读取校验；先用现有完整问题生成 shard，验证存储层等价。
3. **macro 参考准备（前端基础已完成）**：未裁剪候选查询、真实边局部分段、tile owner/membership 和单 macro/多 macro owned 集合对照已经实现；跨进程稳定身份与 shard 格式未实现。
4. **RAM 状态流式迭代**：每轮逐 macro 加载、GPU core batch、owner scatter 和全局屏障；与一次性求解逐轮对照。
5. **out-of-core 状态与恢复**：增加 memmap 双代状态、容量预检、checkpoint 和异常恢复；模拟中断验证上一代不损坏。
6. **流式拓扑输出**：按 polygon owner 重建并写 GDS/OASIS，最后再评估超大单 polygon 精确切片。
7. **性能优化**：仅在正确性门槛全部通过后评估 dirty tile、状态量化、层级 variant 和预取；每项单独基准、提交并可回退。

## 9. 验收与测试矩阵

- 单个矩形、斜边、孔洞、窄环和相接/重叠图形分别跨两个 macro 及 `2×3` core；和一次性 prepare 比较真实边、segment 分段、法向、owner、ring/hole、零位移 XOR。
- 专门验证 macro/context 裁剪框上没有新增可移动边；斜边跨三个 macro 时所有共享端点一致，不出现 33/34 DBU 分歧。
- macro 正序、逆序和随机顺序运行同一轮，`d_next`、L2、PVBand、EPE、歧义计数和最终 GDS 必须一致。
- SREF/AREF 多 occurrence 在不同光学上下文中保持独立，不能因修改一个 occurrence 传播到 master 或其他引用。
- RAM 和 memmap 两条内部路径逐轮对照；模拟进程在 macro 中途退出，恢复后结果与无中断运行一致。
- 空白超过 90% 的合成整版验证 active core 内存收益；接近满铺版图验证内部稀疏管理不会造成超过 10% 的准备时间退化。
- 在 24 GiB GPU、64 GiB CPU 目标机逐级放大图形数量，记录 Python RSS、KLayout Region、NumPy/memmap、磁盘双缓冲和 CUDA 峰值；超过预算必须在对应物化前明确拒绝。
- 最终输出重新由 KLayout 打开，检查 polygon/hole 合法性、边界 gap、正面积 overlap 和与流式重建参考的 XOR。

## 10. 明确不在第一版阶段 2 中实施

- 不让某个 macro 独立完成全部 OPC 轮次；
- 不把整版 mask 或所有 tile 输出常驻 GPU/CPU；
- 不把 `ShapeQuery.materialize()` 改成默认返回未裁剪图形；
- 不在普通轮次重新提边、重新分段或改变 segment 身份；
- 不提前实现 dirty tile 跳过、跨进程永久 segment key、分布式执行或自动修改 master cell；
- 不为了阶段 2 建立没有当前调用方的注册器、空方法目录或第二套 problem 类型。

每一阶段完成后同步开发手册、测试手册、专项开发/测试报告和项目规划记录，并执行差异、重复实现、未调用函数、异常入口与 bug 修复遗留审计。关键阶段只做本地 Git commit，不推送远端。
