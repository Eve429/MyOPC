# 调研发现

## 当前工作树

- 调研开始时存在未提交的 MB-OPC 修复；调研期间它们已由外部工作流提交到 `e289f2c`。
- 当前与本 change 相关的业务代码和测试 clean；用户修改的 `doc/opc/mbopc_migration_design.md` 仍未提交，本任务不得覆盖。
- `00_PAST/` 中存在只读的 `opc/iteration/diffopc/` 参考迁移代码和对应测试、main。

## 已完成核对

- 已核对 `MacroProblem`、光刻模型协议、求解器及 macro 工作流的精确接口。
- 已确认官方 DiffOPC hard-raster midpoint surrogate 与 `00_PAST` sigmoid soft-raster 不同。
- 已完成 wafer loss → mask gradient → segment midpoint → scalar displacement 的链式推导和真实模型符号实验。
- 已确认现有光刻/Problem 接口无需改动；唯一必要共享抽取是 target cache。

## 当前接口事实

- `lithography/contracts.py::LithographyModel` 已提供有当前调用方的可微批量光刻协议：`device`、`config.canvas/print_threshold`、`condition()`、`forward_many()`；新增梯度求解器不需要再建光刻模型抽象。
- `opc/input/edge/problem.py::MacroProblem` 已持有固定参考 `SegmentBatch`、唯一 owner、core membership CSR、macro 网格、极性和位移上限，且 NPZ 格式版本为 1。
- `opc/input/edge/fragmentation.py::SegmentBatch` 以数学边索引和参数区间存段，`materialize(displacements)` 是 NumPy 硬几何路径；它不提供可微栅格化。
- `opc/input/edge/reconstruction.py::reconstruct_region` 把 segment 位移重建为整数 KLayout Region，并检查环翻转、hole 越界、最大位移和轮廓有效性；只能用于候选发布/输出验证，不能位于 autograd 热路径。
- 当前 `opc/iteration/mbopc/simple.py::optimize_macro` 是同步离散 EPE 方法：固定参考段、NumPy `float64[S]` 位移、KLayout 硬栅格、EPE 提案、每轮候选重建验证。
- 当前 simple 求解器已经修复无有效探针、重复末轮评价、逐 core GPU 同步和过宽异常捕获；这些修复已包含在 `e289f2c` 基线。
- `evaluation/metrics.py` 的指标函数返回 Python 整数并对 wafer 二值化，适合作为无梯度报告指标，不适合作为训练 loss；梯度方法必须直接用连续 aerial/resist tensor 定义可导目标。
- 调研期间基础 MB-OPC 修复已由外部工作流提交；最终规格基线更新为 `e289f2c60c3db6302c687bdf30c6977f108c47f0`。相关业务代码和测试已 clean，只有依赖文档 `doc/opc/mbopc_migration_design.md` 仍 dirty，必须在 front matter 记录。
- `main/_macro_pipeline.py` 已提供方法无关的严格配置、逐 macro problem 准备、局部 GDS 写出和最终 ownership merge；梯度方法必须复用，不重复 layout/geometry 前端。
- `main/_mbopc_workflow.py` 当前把 simple 配置、simple 求解、progress、result NPZ、macro 循环、merge 与最终光刻留档放在同一文件。它的 `save_final_lithography()` 实际方法无关，但只有当前 simple 调用方。
- 当前 `TargetCanvasCache` 位于 `opc/iteration/mbopc/simple.py`，其语义（固定 target 的 CPU uint8 有界 LRU）也适用于梯度方法。迁移到同包 `_cache.py` 并保持包级 re-export，可产生两个当前调用方且不改变 public import，是合理且最小的共享抽取。
- `MacroPipelineConfig.canvas_pixels` 当前明确固定为 256；梯度配置不应再重复声明 pixel/canvas，必须从 `MacroProblem.macro` 和光刻模型 config 读取并交叉校验。

## 工作流设计倾向

- 不建立 ILT/MB-OPC 统一 Problem 或通用 Solver 注册器：ILT 优化像素/水平集，不消费 `MacroProblem`。
- 不为 simple 与 gradient 强制共享结果结构：二者分别有 EPE step 记录和连续 loss 记录，共同字段不足以抵消适配复杂度。
- 梯度工作流可加入现有 `main/_mbopc_workflow.py`，复用同一 macro pipeline 和最终光刻函数；只增加一个公开 `run_gradient_mbopc()` 入口。这样不复制整份 orchestration，也不增加一次性 workflow 文件。
- 唯一计划中的共享抽取是 `TargetCanvasCache`；自定义 edge-gradient 只被梯度求解器使用，应保留在 `gradient.py`，不单独建立 rasterizer 接口/目录。

## 初步架构约束

- 梯度求解器可直接消费 `MacroProblem`，但必须新增“segment 位移 -> 可微 mask”的具体实现；不能复用 KLayout raster 作为反向路径。
- `reconstruct_region()` 应只在优化状态准备发布时验证离散 DBU 几何，不能每个梯度 step 调用，否则会切断梯度且跨 Python/KLayout 热循环。
- simple 与 gradient 方法的优化变量/损失/停止条件不同，不应为了统一而抽象成通用 Solver 基类或注册器。

## 参考实现初查

- `00_PAST/opc/iteration/diffopc/` 不是当前能力，只能作为只读参考。它包含配置/结果结构、解析 sigmoid 软边栅格器、逐 batch backward 和 Adam 更新。
- 归档软栅格器采用“参考 hard mask + 每条有限 segment 的 occupancy delta 求和后 clamp”。它的单段零位移与有限差分测试成立，但该公式没有通过完整 polygon/hole/相邻边联合覆盖的数值等价证明；不能仅凭旧测试就认定为正确迁移算法。
- 归档求解器依赖已经删除的 `MBOPCProblem.physical_mask/grid/config` 与旧 raster API，不能复制即用；它还把 `ValueError` 与 `ReconstructionError` 一起转为 `invalid_geometry`，违反当前“禁止吞错误”的规则。
- NVIDIA 官方 DiffOPC 仓库声明其为 ICCAD 2024 “Differentiable Edge-based OPC”参考实现，当前仓库结构包含 OPC、数据、光刻、SRAF 和 Hydra 运行框架；README 给出单 case 和 10 case 参考结果，但 README 本身不足以确认边段梯度的具体公式。
- 官方仓库的工程框架、Hydra、日志、数据集和 SRAF 不应整体迁移；需要继续核对产生 edge gradient 的精确源码和论文语义。
- 官方仓库许可证包含“仅非商业研究或评估使用”限制；计划必须采用算法思想并由本项目重新实现，不能在未确认项目用途与许可证兼容性时逐段复制官方代码。

## 官方 DiffOPC 核心算法证据

- `src/opc/edgeilt.py::Binarize.forward` 从量化、角点协调后的 edge endpoints 生成 hard binary mask；`Binarize.backward` 不对 rasterizer 做普通 autograd，而是把 `dLoss/dMask` 在每个 segment 的代表点采样，再乘该边段允许移动的 velocity，形成 edge endpoint 梯度。
- `src/opc/edgeilt.py::StraightThroughEstimator` 前向对 edge 参数取整，反向直接传梯度；所以官方方法优化的是离散像素边位置的 STE surrogate，不是归档代码的 sigmoid occupancy-delta 软栅格。
- `src/opc/edgeilt.py::EdgeMerger` 前向协调相邻 corner edge 的端点、反向恒等传递；这对应本项目中“控制段位移 -> 连续闭合轮廓”的连接约束，但官方实现只展示 Manhattan H/V 分支，不能直接声称支持本项目斜边。
- 官方 loss 主体是 nominal L2、max/min 对 target 的 PVBL2、max-min PVBand surrogate，并可选沿 H/V EPE 点邻域的 loss；PVBand 离散指标只用于报告。
- 官方每轮执行 hard mask forward、光刻、loss.backward、optimizer.step；best 参数与 best mask 来自同一次已评价 forward。其实现未展示本项目需要的 macro/core ownership 与流式 batch 语义，这部分必须由本项目契约补齐。
- 因此，若目标是“迁移 DiffOPC 方法”，归档的 sigmoid 软边栅格只能作为实验替代，不能作为本次最终算法。计划应采用本项目重新实现的 hard raster + 自定义 edge-gradient/STE，并用有限差分方向性、平移对称和端到端 loss 下降测试验证 surrogate。
- 官方 `src/opc/utils.py::edge_params_merge2mask` 的 forward 从 edge endpoint/polygon id 创建二值 mask，并显式令 mask 可求导；它不是 sigmoid mask。官方边段预处理把 H/V 边切成接近固定长度的片段并记录 midpoint、corner、direction。
- 官方代码的边界检测、mask 构造和 EPE 分支大量假定 Manhattan H/V；本项目必须明确首版梯度方法是否支持斜边。基于当前 `SegmentBatch` 已有任意单位法向和重建能力，计划采用“任意直线段中点双线性采样”的 surrogate，并用斜边方向性测试，不复制 H/V 类型分支。
- 官方 hard mask forward 在每个 polygon bbox 内做射线奇偶填充，并按 polygon Python 循环；这不符合本项目跨 Python/KLayout 边界批量化和大版图性能要求。当前 KLayout Region + `rasterize_mask_canvas()` 可作为更准确的 hard forward，同时自定义 backward 保留官方 edge-gradient 思想。
- 论文 Algorithm 4 明确规定：在每条 segment 中点插值 `dL/dMask` 得到 `g_mid`，两个 endpoint 的梯度都等于 `g_mid * velocity`。本项目用一个标量 `d_s` 同时驱动两个 endpoint 沿单位公共法向移动，因此链式法则给出 `dL/dd_s = 2 * g_mid`；不得另乘 segment length 或自创 soft-temperature。
- 论文 MRC-aware velocity、SRAF 和可选 EPE loss 属于更完整 DiffOPC；本次“最小梯度 MB-OPC”只迁移 STE hard-forward、midpoint edge gradient、连续 L2/PV loss 和同步 Adam，必须明确不能称为完整 MRC-clean DiffOPC。

## 光刻与配置事实

- `lithography/iccad13.py::ICCAD13Lithography.forward_many` 对输入 mask 保留 autograd，多个工艺条件共享一次 FFT，并返回与输入 H×W 相同的连续 resist；梯度 loss 可直接基于该连续输出。
- `ICCAD13Config` 同时提供 `target_density`（resist sigmoid 的物理阈值）与 `print_threshold`（离散评价阈值）。训练 loss 不应再次用 `target_density` 阈值化；离散 L2/PV/EPE 报告继续使用 `print_threshold`。
- 当前 `config/mbopc_*.toml` 已把 pixel/canvas/grid/edge 放在公共段、simple 参数放在 `[mbopc]`。梯度方法应新增独立 `[gradient_mbopc]` 段，避免让 simple 配置出现不消费的 loss/optimizer 字段。
- 项目依赖已经包含 torch/tqdm 等所需包；hard-mask surrogate 不需要新增 OpenCV、Hydra 或自定义 CUDA 依赖。
- `opc/input/raster.py::points_to_canvas` 已给出唯一 DBU→居中 canvas 连续坐标公式；自定义 backward 必须复用它，不能在梯度模块重写 pixel/padding 变换。
- `rasterize_mask_canvas` 返回面积覆盖率（边缘像素可为 0..1），而不是严格 0/1。计划中的“hard forward”应定义为“精确 KLayout 面积覆盖率 forward、对几何离散构造不求导”；不得把它描述成逐像素二值填充。
- 对 gradient loss，context mask 参与光刻但 loss 只乘 `ownership_canvas`；同一 segment 在相邻 tile membership 的梯度可通过 Torch gather 的反向 scatter-add 累计到同一全局参数，optimizer 仍在 macro 全部 tile 完成后才 step。

## 数值方向验证

- 使用当前真实 CPU `ICCAD13Lithography` 对 256² 矩形做只读实验：当前 mask 小于 target 时，四条边界处 `dLoss/dMask` 均为负（约 `-1.4e-5`）；当前 mask 大于 target 时均为正（约 `+1.5e-5`）。
- 本项目正 segment displacement 的公共语义是“扩大透光区域”（clear hull 外扩；opaque 法向翻转后同样扩大透光）。因此 SGD/Adam 的 `d -= lr*grad` 在 target 较大时增加位移、target 较小时减小位移，符号闭合，没有反号。
- hard forward 不可要求普通有限差分“数值值相等”：它在 DBU 格点间是分段常量。正确测试应比较 surrogate gradient 与整数 DBU 精确重建有限差分的方向/余弦，而不是幅值逐值相等。

## 梯度状态与内存决策

- 只把 `owner_indices>=0` 的 O 个位移作为 Adam 参数；context segment 始终是常数 0。避免为只读 context 分配 parameter/gradient/Adam 一阶与二阶矩。
- 维护 CPU `segment_to_parameter[int32,S]` 映射；每 batch 根据 membership 构造局部 parameter 索引，Torch gather 的反向自动把多 tile 梯度累加到同一 owner 参数。
- 不跨迭代保留 `SegmentGeometry` 的 starts/ends/normals 三个 float64[S,2] 数组。只缓存梯度采样所需的 reference midpoint；normal 从 edge-level `edge_normals[edge_ids]` 按 batch gather。论文 surrogate 对标量位移的梯度为 `2*g_mid`，不需要 segment length 常驻数组。
- 每个已评价状态只构造一次完整 float64[S] 位移并调用 `reconstruct_region()`；候选通过后得到的 Region 直接作为下一状态 batch raster 输入，禁止在同一状态重复重建。
- 最佳状态内部只保存 owner 参数快照；返回 `GradientMBOPCResult.best_displacements` 时才展开为与 `MacroProblem.segments` 对齐的 float64[S]，保持现有写出接口可复用。

## 现有回归可复用点

- simple 测试已经覆盖 clear/opaque、hole、斜边、跨 core、跨 macro、窄壁、边越界和无效重建；gradient 测试必须重用同类几何矩阵，但不能只断言“某个允许停止原因”，而要断言梯度方向、context 归零、best 状态和最终几何。
- 基线相关测试命令通过：lithography、evaluation、opc.input、simple MB-OPC、MB-OPC runners 共 `249 passed in 60.40s`。

## 工具错误

- 一次 `web.run click` 组合请求产生语法错误；已改用直接 URL 打开，不重复该调用。
- GitHub API 目录 URL 与 DOI 直链被浏览工具判为 unsafe；后续改用 GitHub HTML/raw 文件和 ACM 可检索页面。
- 三个 raw 文件发生缓存 miss；核心 `edgeilt.py` 已成功读取，其他文件改用 GitHub HTML 或源码搜索定位。
- 一次 `rg` 同时搜索不存在的 `configs/` 路径返回错误；有效的 `config/` 搜索结果已取得，后续只使用存在路径。
- 官方 `configs/opc` HTML 与 raw YAML 连续 cache miss，无法从浏览工具确认参考默认超参数。该信息不是算法 contract；本计划将所有梯度超参数设为 TOML 必填项，不冒充官方默认值。
