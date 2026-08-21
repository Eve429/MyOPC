# MB-OPC（simple + gradient）全链路三角度审查

> 2026-08-21 · 基线 `13cf41e` · 审查方式：全链路源码逐文件通读（约 3700 行）
> + 单项微基准 + 端到端实测。发现按 C（算法正确性）/A（架构）/P（性能）
> 编号与严重度标注，**全部仅立案未实施**，优化方向由用户决策。

## 1. 审查范围（完整调用链）

| 层 | 文件 |
|---|---|
| 入口 | `main/run_mbopc.py`、`main/run_mbopc_gradient.py`（适配器内联） |
| 编排 | `main/_mbopc_workflow.py`（MBOPCMethod 七钩子）、`main/_macro_pipeline.py`（prepare/merge/留档/field） |
| 配置 | `main/configuration.py`（Partition/Lithography/Edge/MBOPC/Gradient + resolve_*） |
| 输入 | `opc/input/mask.py`、`grid.py`、`raster.py`、`_fragmentation.py`、`edge/{fragmentation,problem,reconstruction,sampling}.py` |
| 求解 | `opc/iteration/mbopc/{simple,gradient,_cache}.py` |
| 评价 | `evaluation/metrics.py`（L2/PVBand/EPE 探针） |
| 光刻 | `lithography/iccad13.py` 消费面（forward_many 三条件共享 FFT、autograd 保持） |

## 2. 总体结论

链路成熟度高：核心不变量逐条核对全部通过（§3.1），六文件 250 例测试成体系
（§6），**未发现行为级正确性缺陷**。值得立项的发现集中在四处：一条代码
注释与实现矛盾（C1）、gradient 停止判据语义与注释偏差（C2）、双求解器
批量评价段重复且已漂移（A1）、候选栅格化存在 ~1.8× 结构性冗余（P1）。

## 3. 算法正确性

### 3.1 核对通过的不变量（证据锚点）

| 不变量 | 证据 |
|---|---|
| 探针/计分/mask 三画布同布局 | 三函数共 `_center_padding`（raster.py:37-50）；`points_to_canvas` 与 ownership 正向公式互逆 |
| 极性统一"透光→不透光"、+位移扩大透光 | fragmentation.py:238-242 法向翻转；metrics.py:106-118 方向约定逐极性核对一致 |
| owner 唯一写 | own⊆membership 构造校验（problem.py:85-98）；written 恰一次守卫（simple.py:250-252） |
| Jacobi 屏障 | simple：方向只写提案缓冲、current 全程只读（simple.py:234-244）；gradient：批间梯度累积同一参数快照、每态恰一步（gradient.py:511,666,704） |
| best 严格更小、平局保早 | simple.py:369（EPE）；gradient.py:690（total_loss） |
| "无法评价"≠"零违规" | insufficient_probes 先于 best 比较（simple.py:362-368，含注释依据） |
| records 语义 | records[0]=baseline；Round N 指标属第 N 次位移后状态、step_dbu 记产生步长（simple.py:353-361） |
| 位移界 | 双路径 clip ±max_displacement、context 段恒 0（simple.py:253-255；gradient.py:596 + candidate_full 仅写 owner_ids） |
| 候选守卫后发布 | reconstruct 的绕向翻转/hole 逃逸/有效性三重拒绝（reconstruction.py:30-65）；失败留 stop_detail 不吞错 |
| nm→DBU 精确换算 | 全部落格参数走 exact_dbu；lr 为连续量例外且有注释（configuration.py:440-442） |
| 暗界贯通 | dark_box 像素中心判据与 ownership 同式（raster.py:96-109）；双求解器四调用点 + 最终留档同源；MacroProblem v2 持久化拒绝旧版 |
| STE 单位契约 | 2·g_mid/pixel_dbu = 两端点链式和 × DBU 换算（gradient.py:168-210）；跨 core membership 梯度经 gather 自动 scatter_add 累加 |
| merge 正确性 | 显式映射不猜路径、ownership 权威裁剪、面积守恒回读验证、空候选容忍（_macro_pipeline.py:242-325） |
| loss 归一 | 各批除以全宏 total_pixels 后求和 = 全宏平均（gradient.py:497-499,531-533）；L_epe 分母全宏常量 |

### 3.2 发现

**C1（低 · 注释与实现矛盾）** `main/_macro_pipeline.py:59-63`
`resolve_field_bounds` docstring 仍写"环带光学语义 = 极性背景外推
（clear→不透光、opaque→透光）"，与同函数 warning（L106"环带恒不透光"）
及 2026-08-21 语义修订直接矛盾。行为正确、注释误导维护者。建议改写注释
（属一行级修复）。

**C2（低 · 停止判词语义）** `gradient.py:600`
`torch.equal(parameters, before)` 的 no_update 注释称"梯度全零时步长为
零"，不精确：Adam 动量使梯度归零后参数仍以 lr·0.9^k 量级移动多个状态，
equality 迟迟不触发——典型 iterations≤20 内 no_update 几乎不会出现，
尾部若干状态属无效评价。行为无害（best 只认已评价合法状态）。建议：修正
注释；可选增强"连续 k 态梯度全零即停"。

**C3（信息 · 无需修改）** membership 是 bbox 判定的超集：斜边段的扩张
bbox 角落会多包 core。梯度采样 midpoints 越界整体置零（gradient.py:184,
207）已消解多包影响；EPE 探针仅建在 owner core 不受影响。

**C4（信息 · 既定设计）** `_macro_cuts_by_count` 末端 macro 吸收余量
（grid.py:150-157），边缘缩短 core 可为非 pixel 整数倍：edge 路径按
coverage 语义容忍半像素，pixel ILT 路径在 prepare 显式拒绝
（pixel/problem.py:253-260）。两路径不对称是已知设计；配 field_box/
field_size 可控对齐。

**C5（信息）** `count_edge_fragments` 在 L=2·max 处段数 2→4 跳变
（角部约束启动），覆盖率连续、无正确性影响。

**C6（低 · 契约补一句）** `evaluate_edge_probes.violation_count` 包含
ambiguous 段（inner|outer 违规并集，metrics.py:20-24）；contracts/mbopc.md
未写明该口径，建议契约注明"ambiguous 计入 EPE 违规且方向为 0"。

## 4. 架构合理性简洁性

正面：分层干净——求解器不知 config/layout，编排不知算法数学；MBOPCMethod
鸭子契约（三属性 + best_displacements）文档化；适配器并入入口与 ILT 系
一致；plan/problem/result 三级产物全部版本化并在 load 显式拒绝旧版。

**A1（中 · 重复且已漂移）** simple 与 gradient 的批量评价段 ~80 行同构
重复：target 缓存 miss 回填、mask 栅格、ownership 画布、forward_many、
离散诊断、进度回调（simple.py:143-248 vs gradient.py:385-563）。且两份
**已经漂移**：gradient 以 `_GradientMacroContext` 预计算探针坐标/参考
几何/ownership 计数，simple 每态重算探针坐标与 ownership。建议镜像 ILT
`_skeleton.py`（P3 批次先例）抽 `mbopc/_batching.py`：静态画布 per macro
打包一次（BatchPack 模式）+ 批组装 + 离散诊断复用。两个真实调用方，
不属投机抽象；须以 golden A/B（位移与记录逐位一致）保护迁移。

**A2（低）** `evaluate_and_propose`（~160 行）与 `_evaluate_state`
（~210 行）均多功能混合；A1 落地时自然拆分，不建议单独动。

**A3（低 · CLI 一致性）** gradient 入口打印峰值 RSS/CUDA，simple 不打印
（run_mbopc.py:103-115 vs run_mbopc_gradient.py:123-129）；summary 公共键
两者都有，simple 补两行打印即齐。

**A4（低）** `opc/input/_fragmentation.py`（17 行共享计数公式）与
`edge/fragmentation.py` 同名易混；可降为后者的私有函数再 re-export。

**A5（信息）** `_solve_macro` 对 best 再重构一次（solver 不返回几何是
接口简洁的代价；每 macro 一次，量级可忽略）。

## 5. 性能

微基准（WSL myopc312、core 1024/context 400/pixel 8）：

| 项 | 实测 |
|---|---|
| mask 栅格化（50 矩形 region、1824² 窗口） | **5.8 ms/core** |
| ownership_canvas | 36 µs（栅格化的 0.6%） |
| 探针坐标（500 段/core，simple 逐态重算量） | 40 µs |
| sparse_6um 端到端（4 macro/36 core、2 轮、CUDA） | 2.1 s（本次实测） |

**P1（中高收益 · 中风险）候选 mask 栅格化是 CPU 侧绝对主导，且存在
结构性冗余**：同一候选 Region 按每个 core 的 context 窗口各自栅格化，
context 带被邻窗重复计算。例：1024 macro、2×2 个 512 core、ctx 256——
窗口面积和 4×1024² px vs macro query 一次 1536² px，冗余 **1.78×**。
优化方向：每态在 macro query box 栅格化一次，按 core 窗口切片。前置
条件：全部窗口像素网格对齐（要求 bounds 与 core 切线均为 pixel 整数倍；
现仅校验 core/context 本身），不满足时回退逐窗路径。必须以"切片 vs
逐窗逐位一致"golden 测试保护。预计整链 CPU 侧 ~1.5-1.7×。

**P2（低中）gradient 每态 reconstruct 的固定开销**：
`_validate_reference_topology` 对每个 hole 做一次 klayout 布尔
（Python 循环，reconstruction.py:51-65）+ `contours_to_region`。孔洞
密集版图（opaque 极性孔阵：每宏数百孔）下每态 O(holes) 次 klayout 调用
（估 ~60µs/孔）；可做面积/包络预筛或仅在位移跨越拓扑阈值时复检。

**P3（引用既有结论）** GPU 侧无 MB-OPC 特有新发现：forward_many 三条件
共享一次 mask FFT（iccad13.py:320-340）已是最优形态；ILT 审查结论适用
（backward 占 60-65%、batch_size 扫描未做）。simple 为 no_grad 前向。

**P4（微 · 随 A1 顺带）** simple 逐态重算探针/ownership 实测全程 ~0.2s
（bench_30um 量级），不单独立项，A1 的静态打包顺带消除。

内存上界复查通过：LRU 字节上限与驱逐语义正确（_cache.py）、批张量 del
后报进度、参数 O 级、problem 逐 macro 加载即弃。

## 6. 测试覆盖（250 例盘点）

| 文件 | 例数 | 代表矩阵 |
|---|---|---|
| test_simple_mbopc.py | 56 | 缓存 LRU/配置校验/入口契约/stub 方向与步长/批不变性/几何矩阵（含 opaque、空宏、窄壁 insufficient、invalid_geometry）/步长衰减 |
| test_gradient_mbopc.py | ~95 | STE 三单位测试（2×、1/pixel、无 Python 循环）/loss 独立复算/批与屏障/EPE 公式与切段不变性/调用计数/真实模型 CPU-CUDA 方向一致 |
| test_macro_problem.py | 38 | ownership 分裂/共享对角线逐位一致/几何矩阵/中点语义/NPZ 往返与拒旧 |
| test_grid.py | 38 | 两种规划模式/切线契约/居中画布/探针映射/dark_box 画布三例 |
| test_mbopc_runners.py | 23 | 单/多宏 e2e、merge 恰一次、跨宏顺序不变、field 扩网格 |
| test_gradient_mbopc_runner.py | 20+ | 配置校验（含 lr 超限警告）/摘要契约/多宏/进度收尾/免安装直跑 |

**缺口**（均为低价值或随 A1 补）：切线近切微碎段（1e-9 量级）无专门用例；
membership 超集语义无断言（设计上不可见）；Adam 动量下 no_update 时序
未钉；simple 与 gradient 探针坐标一致性（A1 漂移的回归锚）；
final_lithography tile 网格在 clear 输出（包络=图形区）与 opaque 输出
（包络=整版）间的覆盖差异未断言。

## 7. 决策清单（按建议优先级）

| 编号 | 项 | 收益 | 风险/前置 | 工作量 |
|---|---|---|---|---|
| A1(+P4) | 批量段骨架化（BatchPack 模式） | 消除漂移 + 微性能 + 为 P1 铺路 | golden A/B 逐位保护 | 中 |
| C1/C2/A3/C6 | 注释矛盾、停止判词、CLI、契约口径 | 一致性 | 零行为变化 | 小 |
| P1 | macro 级一次栅格化 + 对齐切片 | CPU 侧 ~1.5-1.7× | pixel 对齐校验 + 逐位一致 golden | 中大 |
| P2 | hole 拓扑校验降频/预筛 | 孔阵版图可感知 | 低 | 小 |
| — | C3/C4/C5/A4/A5 | 立案不改 | — | — |

## 8. 顺带实证（输入 doc 清扫）

1. **sparse_6um 端到端成功**（本次实测：空宏 mr1c1 zero_epe、merge 正常、
   final GDS 产出）——P1-2（merge 抛 LayerNotFoundError）已被现行空候选
   容忍逻辑（_macro_pipeline.py:270-279）修复；
   `TestReticle/reticle_build_plan.md` §5.7"现状警示"已过期。
2. `doc/contracts/edge.md`、`doc/contracts/opc_input.md` 均无 dark_box
   记录（MacroProblem v2 与 rasterize_mask_canvas 新参数未同步契约）。
3. `doc/INDEX.md` 目录一览"dataflow/（index + 四工作流文件）"实为
   index + 六工作流文件（含 macro_pipeline 与三个 ILT）。
