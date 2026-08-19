# MyOPC 迁移研究发现

## 2026-08-19 Gradient MB-OPC EPE loss 更新设计（进行中）

- 当前生产 `gradient.py` 已采用 DiffOPC Algorithm 4 midpoint STE：硬几何栅格作为
  forward，`grad_output` 在当前已发布重构段中点双线性采样，单个 DBU 法向位移参数的
  梯度为 `2*g_mid/pixel_dbu`。同一 owner 参数会从所有包含该段的 core membership
  累加贡献；loss 仍只在各 core ownership 像素计分。
- 当前训练目标只有按全 macro ownership 像素数 P 归一的 nominal L2、两 process-corner
  L2 与 PVBand；`epe_distance_dbu`、参考 inner/outer probes 和离散 `epe` 仅用于同状态
  diagnostics，best 由连续 `total_loss` 严格更小决定，末状态纯评价。
- 当前已有关键保护：参数是唯一 owner 段连续 DBU 位移；跨 core 梯度 SUM 而非平均；采样
  中点来自与 forward Region 同一次合法重构；所有 core batch 同参数快照、屏障后一次 Adam；
  macro 之间仍独立。因此 EPE loss 设计必须接入现有 `_evaluate_state` 的同一 backward，
  不能另建 per-core 参数、单独 step 或回退到参考中点刚体近似。
- 项目内存在只读归档 `00_PAST/opc/iteration/diffopc/`、旧 DiffOPC 报告和配置，可用于
  核对公式；当前完成规格的 DEC-003 明确首版不加入 EPE training loss，理由是官方可选
  分支偏 H/V 且尚无任意斜边契约。本次设计必须正面解决斜边、符号、归一化和无效探针，
  不能简单把现有离散 EPE 计数当可导 loss。
- 归档 DiffOPC 的连续 EPE 公式是在固定参考段的 inner/outer probe 上采 nominal wafer：仅当
  target 最近邻采样满足 `target_inner>=threshold` 且 `target_outer<threshold` 且两点均在画布内
  时有效；每段 penalty 为
  `ReLU(threshold-wafer_inner)^2 + ReLU(wafer_outer-threshold)^2`。它只惩罚
  “inner 未印上”和“outer 错印上”，恰对应现有离散 EPE 两种违规，ambiguous 不单独入 loss。
- 归档实现用 owner core 唯一计算每个物理 segment 的 EPE，halo membership 只贡献光学/L2/PV
  上下文；分母固定为 `2*segment_count`。该分母把无 owner/context 段也计入，且使用全段数而非
  有效 owner probe 数，不能未经复核直接搬到当前 MacroProblem（存在 context 段、无效 probe、
  多 macro owner 语义）。更合理的候选分母应从“唯一有效 owner probe slot”定义并固定于 macro。
- 当前生产 loss 梯度最终都通过同一个 hard-mask STE 的 `grad_output` 在所有 core membership 中点
  回到 owner 位移。若 EPE loss 在 owner probe 所在 core 的 nominal wafer 上构造，它对 mask 的
  空间梯度仍会经光刻反向传播，并由现有 membership midpoint SUM 汇入唯一参数；EPE 自身不能按
  membership 重复计分，否则一个物理段会因 halo 数量改变权重。
- 官方一手来源已定位：NVlabs/DiffOPC 仓库与 ICCAD 2024 论文
  “Differentiable Edge-based OPC”（DOI 10.1145/3676536.3676764，arXiv 2408.08969）。
  论文问题定义把 EPE、L2、PVB、shot count 组成加权目标，并明确 DiffOPC 集成 EPE loss；但公开
  摘要/论文当前已读片段尚不足以确定代码中的具体 EPE surrogate，必须继续精读官方源码。
- 官方仓库已浅克隆到 `/tmp/myopc-diffopc-reference`，固定核对 commit
  `bdc6e72ce6d7f8b1092e4177fadc670a5207bf42`。官方 `edgeilt.py::cal_epe_loss`
  与旧归档公式不同：先令 `D=(target-printedNom)^2`；对每个 H 段中点取竖直
  `[y-15:y+15,x]`，对 V 段取水平 `[y,x-15:x+15]`，求和后计算
  `sigmoid(SigmoidSteepness*sum(D_line))`，再跨段求和。配置 debug 使用
  `EPELoss=True, WeightEPE=100, SigmoidSteepness=4`，default 则关闭 EPELoss。
- 官方 EPE 实现仅接受精确 `seg_type == "H"/"V"`；`CH/CV` corner 段和其他方向直接
  continue。法向半径 15 是硬编码像素数，没有使用评价侧 `EPE_CONSTRAINT=15` 的公共
  显式接口，也没有边界裁剪/有效点归一化。它是“沿参考边法向带聚焦 target-wafer 误差”的
  surrogate，不等同于 inner/outer 阈值违规，也不能直接覆盖当前项目已支持的斜边。
- 官方公式每个零误差段仍贡献 `sigmoid(0)=0.5` 常数；常数不改变局部梯度或同一段集合下的
  best 排序，但让 loss 数值不以 0 为收敛点，且总量随参与段数变化。官方 notes 原文还记录
  “EPE loss is not good”及后续消融待办，说明公开实现本身不足以成为无需裁决的生产契约。
- 因此本次不应原样复制官方 H/V Python 循环。可保留其核心思想：以每个唯一 owner 段的参考
  中点为中心，沿单位法向采样一条固定物理宽度的连续 target/nominal squared-error profile；
  对斜边用向量化双线性采样，把固定像素 15 改为由现有 `epe_distance_dbu`/pixel_dbu 定义的
  对称物理采样范围，并让 corner/斜边遵守同一法向公式。
- 官方论文 §3 的 eq. (6)-(8) 与源码一致：对 target boundary 的 H/V measure point，沿法向
  `[-th_epe,+th_epe]` 累加 `D=(Z_nom-T)^2` 得到 `D_sum`，再定义
  `L_epe=sum(sigmoid(gamma*D_sum))`，总目标为 `w1*L2+w2*L_pvb+w3*L_epe`。
  eq. (9)-(16) 明确 EPE 对 mask 的梯度与 L2/PVB 一起经 lithography 和 midpoint edge
  Jacobian 回传；这支持把 EPE 作为现有同次 `batch_loss.backward()` 的第四分量。
- 论文和官方代码的适用域是 Manhattan H/V target pattern；当前 MyOPC 测试与分段契约包含
  45° 斜边。设计若宣称“参考 DiffOPC”而非“逐值复现”，应把 H/V 法向行/列推广成任意单位
  法向的双线性 profile，同时用 H/V 用例证明推广在轴对齐时退化到相同采样几何。
- pypdf 提取确认论文没有给跨 core ownership、无效画布点、batch 归一化或 partial profile
  的工程规则；这些必须由 MyOPC 自己冻结，不能冒充论文事实。
- MyOPC 兼容语义候选已收敛：EPE measurement 仍固定在 reference target segment，而不是随
  current mask 移动；每个 owner segment 只在 owner core 生成一条法向 profile，因此 loss
  唯一计分。profile 可穿过 core ownership 进入 simulation context，这是 EPE 测量所需，不是
  重复 pixel loss；其光学梯度再由现有所有 core membership midpoint STE 汇入唯一 owner 参数。
- 为兼顾官方 2R 点 profile 与当前面积像素坐标，候选采样采用 target boundary 两侧各 R 个
  pixel-center slot：要求 `epe_distance_dbu = R*pixel_dbu`，offset 为
  `(-R+0.5,...,-0.5,0.5,...,R-0.5)*pixel_dbu`。这在 H/V 边上形成对称 2R 像素法向线，
  对斜边用同一单位法向和双线性采样；避免采到 context 几何边界本身。
- 候选连续项：在 nominal error map `D=(Z_nom-T)^2` 上采 `[E,2R]` profile，逐段取
  `D_mean`，再用 `penalty=2*(sigmoid(gamma*D_mean)-0.5)`，最后对 macro 的 E 个唯一
  owner segment 求平均。相对论文的必要工程修正是：减去 0.5 使完美匹配为 0、乘 2 归一到
  `[0,1)`、profile mean 与 owner mean 消除 pixel 分辨率/segment 数/batch 划分的权重漂移。
- EPE loss 必须成为现有 `batch_loss.backward()` 的第四分量，记录独立 `epe_loss`，并参与
  `total_loss`/macro best；离散 EPE、valid/ambiguous diagnostics 保持原接口与数值，用于判断
  surrogate 是否真正改善最终阈值轮廓，但不单独打破 total-loss 平局。
- 实现不应调用 evaluation 的离散 `evaluate_edge_probes` 来构图：该 API 面向二值诊断且当前在
  `no_grad` 中消费。连续 profile 采样应留在 `opc.iteration.mbopc.gradient`，对批内变长 owner
  profile 做一次向量化四邻域 gather，CPU 常驻静态坐标，GPU 只转移当前 batch。
- `main/configuration.py` 的 dataclass 解析会使用字段默认值，因此在
  `GradientMBOPCConfig`/`GradientConfig` 尾部增加默认关闭的 `weight_epe=0.0` 与
  `epe_steepness=4.0` 可让旧 TOML 继续加载；关闭时必须跳过 profile 构造与采样，才能保证旧
  数值路径逐值不变。项目示例 `config/gradient_mbopc.toml` 则可显式启用该功能。
- 当前 `resolve_gradient_config()` 不接收 lithography 配置；EPE profile 的
  `epe_distance_dbu % pixel_dbu == 0` 应在 `optimize_gradient_macro()` 入口、任何 GPU 分配前
  校验，避免为这一个约束扩大公共配置调用链。
- metrics record 当前只含三项连续 loss，summary/控制台只展示三项权重。更新必须把
  `epe_loss` 作为 additive record 字段，并同步 summary 的 `loss_weights.epe` 与 runner 输出；
  gradient result NPZ 的位移/状态契约不需要改版。
- 外部 DiffOPC 仓库许可证限制非商业研究/评估使用。本设计只采用论文公开公式及行为证据，
  不复制外部源码，也不增加外部依赖。
- 最终设计已写入
  `doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md`：默认
  `weight_epe=0` 保持旧数值路径，示例配置显式用 1.0；采用 fixed reference、任意方向
  2R pixel-center profile、zero-based normalized sigmoid、全 macro owner 分母和同次 backward。
- 设计审计补齐了两个容易遗漏的契约：四个 loss 权重至少一项为正（允许 EPE-only 回归），以及
  canvas 最外侧整数 pixel center 的邻居可退化为同一 pixel，真正越界才失败。公开 iteration
  record 的新增字段放在末尾且默认 0，保留旧构造方式；summary 额外公开 best EPE loss。
- 当前仓库不存在 `doc/opc/`；本 change 完成时应随目录从 active 移到 completed，并把专项
  `development_report.md`/`test_report.md` 放在同一 completed change 内，不改旧 archive reports。
- 规格没有 blocking open question，但状态保持 draft 等待用户批准；实现尚未开始。


## 2026-08-19 Simple ILT 规格语义修订（完成）

- 用户指出的主问题与草案原文完全对应：REQ-005/008/009、INV-002/004、
  §7.3/7.4/7.5、`ILTBatchResult`、IF-004、§10.2/10.3、§11、PERF-002/003、
  TEST-007/008/010、DEC-001/004、§20/21/24 都把 core 定义成独立优化问题。
- 该语义会让一个 core ownership loss 无法把梯度传到其 simulation context 内
  属于同 macro 的可训练像素，并允许拼接不同 core 的不同 best state；用户提出
  的 macro 唯一 parameter snapshot、跨 core gradient sum、全 core 屏障后统一
  SGD step 和 macro 级 best，在算法与状态一致性上成立。
- coverage 初始化问题也成立：草案同时要求 `target_u8` 保存 0..255 面积覆盖率，
  又用 `p0=2*T-1` 后套 sigmoid，只有少数 T 能碰巧保持原值。应改为
  `p0=logit(clamp(T, eps, 1-eps))/beta`，并以 state0 soft mask 对 transmission
  raster 的容差一致作为测试契约；严格 0/1 仅受 epsilon 近似影响。
- 规格必须继续保留两级唯一性：core ownership 只决定 loss 统计；macro
  trainable domain 决定参数/梯度/最终写回；simulation context 决定 forward
  读取范围。不得使用 overlap loss averaging，也不得为同一物理 pixel 建多个参数。
- core/pixel 整除限制是否已由当前 `plan_macros` 强制，尚待源码与测试核对；
  在确认前不把用户建议写成当前能力事实。
- 网格源码核对结果：`plan_macros` 已拒绝名义 `core_size_dbu` 或
  `context_dbu` 不是 `pixel_dbu` 整数倍，且有
  `test_core_or_context_not_pixel_multiple_fails` 回归；但 `_core_cuts` 会把
  最外侧实际 core 强制截到版图 bbox 终点，因此当 bounds 宽/高不是 pixel
  整数倍时，末端实际 `CoreSpec.ownership_box` 仍可能含 partial pixel。
- 用户的第 4 点因此“部分已满足、边缘情形仍成立”。Simple ILT 规格应在其
  pixel problem/workflow 边界要求每个实际 macro/core ownership box 的宽高
  都是 `pixel_dbu` 整数倍，并在 KLayout/raster 分配前失败；无需改共享 raster
  的通用 partial-coverage 能力，也不应继续定义“末端 partial pixel 归属/回写”。
- 为避免无关地改变现有 edge MB-OPC 对任意 bbox 的兼容性，最小规格改法是让
  ILT 的 `prepare_pixel_problems` 对 `plan_macros` 结果做实际 cuts 对齐校验；
  不要求修改 `opc/input/grid.py`，除非实施核对发现多个当前调用方共同需要该
  新限制并经用户另行扩大范围。
- 当前 architecture/contracts 不定义 ILT 实现，因而不阻碍 macro 同步语义；
  现有 MB-OPC 已提供可类比的“所有 core 读取同一状态、跨 batch 累积、屏障后
  发布下一状态”模式。Simple ILT 可沿用该状态纪律而无需改变基础层依赖方向。
- LevelSet、CurvMulti、Multilevel 三份后续草案大量继承了 `optimize_*_batch`、
  per-sample best、fixed context 与 tile-independent seam 语义。用户本次只授权
  更新 Simple 规格，因此不联动修改它们；Simple 修订完成后，这三份草案都将
  成为已知不一致的依赖文档，必须在各自批准/实施前单独修订。
- 规格接口需要随语义最小调整：optimizer 单位从 core batch 提升为 macro，
  `ILTBatchResult` 改为 macro 结果，best state 从 `[B]` 变为单个 macro state；
  core batch 仍是 solver 内部 GPU 分块，不新增 registry/base class 或三套概念对象。
- 第一轮残留搜索未发现仍具约束力的 per-core best、`ILTBatchResult`、
  `optimize_simple_batch`、`place_owned_canvas` 或 partial-pixel 处理；命中项仅有
  OpenILT/Revision 0.1 历史事实和对被拒方案的说明，均应保留。
- 审读修改后正文发现两处小型清理项：Scope 仍写“batch optimizer”，应改为
  macro optimizer；PERF-004 有空格排版问题。其余数据流、状态、接口和产物
  已围绕一个 `ILTMacroResult` 收敛。
- 第二轮残留扫描仅命中 Revision 0.1 的历史 per-sample 记录；当前约束区零旧
  语义命中，`git diff --check` 通过。
- 草案 front matter/§2.1 仍指向旧 `2fa75ea` 和已不存在的 grid.py dirty
  状态。当前 HEAD 是 `540a0121eb06904bdc44ae7fe3bd491aeff22fb5`，生产
  代码相对 HEAD 干净；本轮仅规格、规划与 `.learnings` 记录未提交。为了让
  实现 AI 能按 Document Contract 核对，应同步刷新 baseline，并保留 draft
  与未勾选 approval gate。
- 终审确认 state N 应只评价而不构建无效 backward；§10.2/10.5 与
  TEST-006/010 已同步收紧。core batch 的固定外部 context 明确按
  `target_u8/255` 进入光刻，避免把 uint8 0..255 误传模型。
- 修订完成后的已知问题：跨 macro 光学耦合仍未纳入梯度和 best 评价；非整像素
  layer bbox 现在会在 ILT prepare 前置拒绝；三份后续 ILT 草案仍引用旧
  `ILTBatchResult/optimize_*_batch/per-sample best/fixed context`，必须在各自
  批准前修订；规格仍为 draft 且 approval gate 未确认，当前不得实施。

## 2026-08-19 项目现状复核（完成）

- 当前分支为 `migration`，复核开始时与 `origin/migration` 同步且工作树干净；
  本轮仅因持续规划要求修改根目录三份工作记录。
- 当前生产模块为 `layout`、`geometry`、`opc.input`、`opc.input.edge`、
  `lithography`、`evaluation`、`opc.iteration.mbopc` 与应用编排 `main`。
  `opc.iteration.ilt` 尚不存在；四份 ILT 文档位于 `doc/changes/active/`，属于
  待评审目标，不能当作已实现能力。
- 当前合法依赖拓扑由 `doc/architecture/system.md` 明确为
  `layout -> geometry -> opc.input -> opc.input.edge`，方法层可依赖输入层、
  `lithography`、`evaluation`，`main` 只负责编排。
- 当前主数据流是：版图读取与局部物化 → Macro/Core 两级网格 →
  `MacroProblem` NPZ → 逐 macro 求解 → ownership 裁剪合并 → final GDS 与
  JSON 摘要。参考数组只读，唯一迭代态是一维 `float64[S]` 位移。
- 已实现运行路径包括验证管线、单遍偏置、simple MB-OPC 和 gradient
  MB-OPC；文档总入口要求以源码/测试确认当前事实，以 active spec 描述目标。
- 外部依赖基线：KLayout、NumPy、PyTorch、Pillow、psutil、tqdm；测试门禁为
  pytest/ruff/compileall；本轮已在 Linux `myopc312` 环境完成 CPU 门禁实测。
- 配置已集中在 `main/configuration.py`：版图、分区、光刻、边、simple、
  gradient、单遍、验证与输出均使用 dataclass 声明式解析；五份 TOML 分别覆盖
  simple 单/多 macro、gradient、±2nm 验证管线和单遍偏置。
- 对外执行入口为 `main/run_mbopc.py`、`main/run_gradient_mbopc.py`、
  `main/run_macro_pipeline.py`、`main/run_single_pass.py`；`main_test_*` 是教学演示。
- 当前源码同时存在 simple 与 gradient 两个 MB-OPC 算法；gradient 使用自定义
  autograd 边缘掩膜并在最新提交中拆成 prepare/evaluate/step 三段。架构文档的
  模块表把该包简称为“最简 MB-OPC 求解器”，阅读时须结合源码和 dataflow，
  不能据此漏掉 gradient 能力。
- `doc/test_manual.md` 宣称全量 458 用例；本轮 pytest 实际收集数与之吻合。
- ILT 有四份 `draft` 规格，依赖顺序为 Simple（同时建立 pixel problem 与公共
  workflow）→ LevelSet；Simple → CurvMulti → Multilevel。四份规格均无 blocking
  open question，但 approval gate 全未勾选，且 baseline 仍停在旧提交
  `2fa75ea`，明显早于当前 `540a012`；因此任何实施前都必须先做基线漂移复核并
  获得用户批准，不能直接按草案编码。
- 四条候选算法路线边界：Simple=像素 sigmoid+SGD；LevelSet=硬二值水平集+
  SDF+Adam；CurvMulti=平滑 sigmoid 控制场+粗到细 SGD；Multilevel=逐级不同
  Adam 参数和 stage-grid 监督。共同约束是完整 256×256 物理光刻、ownership
  唯一回写、context 固定、逐样本 best，以及不依赖 edge/SegmentBatch。
- 本轮 Linux 环境全量实测：`450 passed, 8 skipped in 87.86s`；总收集数确为
  458。8 项全部因当前 Linux 环境无 CUDA 而按测试声明跳过，CPU 覆盖范围零
  失败。可用解释器是 `/home/wzh/miniconda3/envs/myopc312/bin/python`。
- 同一 Linux 环境的显式范围门禁通过：ruff `All checks passed!`；compileall
  零错误。当前依赖足以执行全部 CPU 测试，但 CUDA 路径仍只能引用既有测试/
  报告证据，不能声称本轮 GPU 复验。
- 文档有一处表述需谨慎：`doc/development_manual.md` 的依赖清单同时写了
  `layout -> geometry` 与 `geometry -> layout`，而实际源码是 geometry import
  layout、layout 不 import geometry；架构文档中的 `layout -> geometry` 更像
  数据流方向。后续做依赖设计时应以源码 import 和禁止反向依赖规则为准，
  必要时单独修正文档，不能把该两行解释为允许循环依赖。
- 源码交叉核对确认共享生命周期：`prepare_problems` 在一个 LayoutDB 会话中
  逐 macro 局部物化并原子保存 NPZ，全部成功后才发布 plan；workflow 稳定顺序
  逐 macro 加载/求解/写 best GDS，任何异常不进入最终 merge；全部成功后恰一次
  ownership 裁剪合并，回读按窗口验证覆盖面积守恒。最终光刻 PNG 也是独立规则
  tile 网格流式生成，不保留整 reticle tensor。
- simple 算法是同步 Jacobi 式离散更新：每个已评价状态在三工艺角上计算
  EPE/L2/PVBand，只有 owner 段按参考边探针方向生成下一状态；全批完成后发布
  位移，context 恒零并裁到上限。best 只按严格更低 EPE 更新；停止原因明确区分
  `zero_epe`、`no_update`、`insufficient_probes`、`invalid_geometry` 与
  `iteration_limit`，末轮只评价、不产生无用提案。
- gradient 算法的唯一参数是 owner 段的连续 DBU 法向位移；每个 state 在同一
  参数快照上跨 core batch 累积三项连续 loss 梯度，屏障后恰做一次 Adam step。
  硬几何前向通过 midpoint STE 回传 `2*g_mid/pixel_dbu`；梯度采样使用
  membership（让跨 core 可见贡献聚合到同一 owner 参数），EPE 诊断仍用唯一
  owner 段。每次候选重建同时发布 Region 与实际重构中点，非法几何保留最近
  已评价 best；best 按严格更低 total loss 选择，末状态纯评价。
- 两种 MB-OPC 均是“独立 macro 求解 + 最后合并”，macro 之间不交换优化态；
  这是当前架构事实与精度/并行边界，后续方案若要求跨 macro 全局耦合，属于
  明确的新设计而非参数调整。
- AST import 图复核：`common/evaluation/layout/lithography` 无第一方跨包依赖，
  geometry 仅依赖 layout，opc 依赖 common/evaluation/geometry/layout/
  lithography，main 依赖全部应用所需层；未发现基础层反向 import main 或
  iteration。`doc/development_manual.md` 的双向箭头确属文字歧义，不是代码循环。
- 当前 HEAD 最近三批聚焦 gradient 正确性与结构：重构函数拆分（声明无数值
  变化）、STE 补 DBU 链式换算、重构中点按需计算；本轮全量回归覆盖这些提交。
- 历史 `findings.md` 不能整体当作当前事实：其中仍记录
  `mbopc_single_macro.toml context_nm=1024` 导致配置损坏，但当前文件已是
  512，恰好满足 1024nm core + 两侧 context = 2048nm canvas；这是已过期的
  历史发现，当前配置不再有该损坏。
- 当前仍需保留的设计局限：现有和候选 ILT 都按 core/macro 独立优化，邻块不
  交换优化后 context；像素 ILT 还会产生 stair-step GDS，且草案明确不覆盖
  MRC、shot、checkpoint/distributed。方案评估时这些必须作为精度/工程取舍，
  不能只比较 loss 公式。
- 两项历史待办从当前源码看仍未形成闭环测试：① 空 macro 的 solver 可合法
  返回空 best，但 `merge_macro_results` 要求每个候选 GDS 的目标层非空，历史
  sparse smoke 已在此复现 LayerNotFoundError；② 通用配置解析和 Config
  `__post_init__` 没有统一的 Decimal/float 非有限值守卫。用户此前明确裁决
  暂不修，本轮只记录，不擅自改动。
- 非有限配置的当前精确表现：`MBOPCConfig(initial_step_nm=NaN)` 在
  `__post_init__` 泄漏 `decimal.InvalidOperation`；gradient 全 NaN/零权重会以
  “至少一个为正”的 ValueError 拒绝，但非有限值没有在统一解析边界用明确原因
  处理；`exact_dbu(NaN)` 自身会以不可精确换算 ValueError 失败。问题是异常
  边界/原因不统一，并非静默接受所有 NaN。
- 发现一处当前 contract 漂移：源码 `evaluate_edge_probes` 默认阈值已由提交
  `61b6a63` 改为 0.5，simple/gradient 还显式传模型 PrintThresh；但
  `doc/contracts/evaluation.md` 仍写默认 0.499 和“保留默认”。当前行为应以
  源码与测试为准，后续触及 evaluation 文档时需修正；本轮不顺带改文档。

## API 变更记录（旧 → 新，评审后续迁移代码的对照基准）

| 旧（00_PAST） | 新（migration） | 备注 |
|---|---|---|
| `CellRef(name, index)` 凭证类型 | 已删除，全链路 `str` 名称 | index 直查+name 交叉验证随之删除；名称查找即校验 |
| `db.top_cell` → CellRef | `db.top_cell_name` → str | 用户更名 |
| `db.cell(name) -> CellRef` | 已删除 | str→str 往返无意义 |
| `query(cell=CellRef\|str\|None)` 三分支 | `query(cell: str \| None)` 两分支 | 存在性校验由 `_native_cell` 统一完成 |
| `RegionBatch(regions, box, cell)` | `RegionBatch(regions, box, stats=None)` | cell 字段零消费者，已删 |
| `read_layout(path, glp_layer_map)` 门面 | `read_layout(path)` + `read_glp(path, map)` | 格式分派在 `LayoutDB.open`；GLP 误用拒绝消息逐字保留 |
| `layout/hierarchy.py` HierarchySummary 全家 | `LayoutDB.cell_hierarchy() -> dict[str, tuple[str, ...]]` | 直接邻接 DAG；each_child_cell 原生去重，不按 occurrence 展开 |

## 测试与验证纪律（已验证有效）

- 全生成式数据（`tests/fixtures/layout_factory.write_advanced_layout`、tmp_path 内 klayout 构造），
  不迁 TestReticle 用户 GDS 依赖；旧库中依赖 reticle 的用例已改生成式等价
  （双顶层 GDS、SREF R90+AREF 2×2 展开断言 `bbox==(0,0,1000,60)`、count==5）。
- 每批次交付三件套：包迁移 + `main/main_test_<模块>.py`（无断言教学演示，逐调用注释
  作用/输入/输出）+ `tests/<模块>/` pytest（参照旧库组织：helpers + 按模块分文件）。
- 门禁命令：`pytest -q tests` / `ruff check layout geometry tests main` / `compileall`；
  解释器 `D:/app/miniforge/envs/myopc/python.exe`；绝不 `ruff check .`。
- 提交纪律：只提交当批模块 + 其测试演示；排除 AGENTS.md/CLAUDE.md/TestReticle/*.glp。

## 架构事实（从旧库蒸馏，迁移评审对照）

- 分层单向：`layout → geometry → opc.input → opc.input.edge`；`opc.iteration.<method>`
  可依赖输入层 + `lithography` + `evaluation`，基础层不得反向依赖。
- `prepare_problem()` 是架构中心：产出四个固定参考对象（PhysicalMask / SegmentBatch /
  OwnershipBatch / BoundarySampleTemplate），迭代态只有一维 displacements 数组。
- `geometry.iter_region_coverage_tiles` 是栅格化共享原语：显示层（uint8 PNG）与
  `opc.input.raster`（float32 光刻 canvas）共用；左下原点，PNG 翻转仅在 I/O 边界。
- Region 生命周期：materialize()/prepare_problem() 必须在 `with LayoutDB.open(...)` 内；
  已物化 RegionBatch 独立存活（test_materialized_region_batch_survives_database_close 守卫）。
- 光刻画布：canvas 256²、Hopkins 核 35×35×24、tile 1024nm + halo 512nm + pixel 8nm 恰满画布；
  FFT 循环卷积污染由 halo 吸收，ownership_canvas 仅 core 像素计分。
- 已知过滤决策点现状：hierarchy.py 已删（Phase 2）；`render_layout_region` 保留
  （零生产引用但有直接回归 + 演示使用）。
- AGENTS.md「未来优化内容」新增：全局同层几何合并/规范化步骤（tile seam 碎片治理），
  属未来功能，迁移时不实现。

## opc 批次进行中的发现

- `opc/input/edge/ownership.py` 输出契约（注释已加厚，示例已验证）：
  `owners[S]` 每段唯一 owner（中点定归属，边界归右/上）；`core_offsets/members`
  是 core 视角 CSR = 段 bbox±halo 接触窗口；own ⊆ membership 恒成立。
  验证示例：2×2 网格 + halo 30 + 横跨切线横条 → 10 段，跨界段同时出现在
  相邻两 core 的 membership 中但 owner 唯一。
- `reconstruction.py` 拐角块逐行注释已加（miter 解析交点 + bevel 退化）；
  关键隐蔽约定：方向向量取原始顶点而非位移后端点（位移沿法向不改变边方向）。
- **已修 bug**：用户为 SegmentBatch 字段加注释时把 `edge_ids` 挪到第二位
  （按段级/边级分组），而 `fragment_edges` 尾部仍是旧字段顺序的位置传参，
  导致 normals(E×2) 落进 edge_polygon_ids 槽位（报"非一维"）。修复为关键字
  传参，此后字段顺序调整不再错位。教训：**给 frozen dataclass 字段重新排序后，
  必须检查所有位置构造点**。验证：零位移 XOR==0；全段 +3 DBU 重建面积
  2400→3276（=126×26，四角 miter 精确）。
- 「owner 唯一」的正确断言是每段恰有一个有效 owner（0≤o<C），
  不是 owners 值互不相同——写测试时别用集合去重误判。
- opc 首次过 ruff（5 个复制时带入的导入排序已 `ruff check opc --fix` 修复）。
- opc 核心链与新 layout/geometry 兼容（RegionBatch 三参 OK）；
  `opc/diagnostics.py:15,125,233` 残留 CellRef + 四参构造，Phase 4 适配点。

## Macro–Core 管线事实（Phase 4 重构产出，2026-08-15）

- **两级网格**（`opc/input/grid.py`）：`plan_macros` 先切不重叠 macro（size 模式
  名义整数倍 / count 模式按 core 单元均衡分配，较前 macro 多一单元），macro 内
  再切 core；半开区间归右/上、最外沿归末行/列；`MacroSpec.locate_owned_points`
  对 macro 外的点返回 **-1**（与全局网格的 clip 语义不同）。
- **ownership 切线分裂**：斜边交点参数 t 必须由「原始整数端点 + 全局整数切线」
  计算，共享边界两侧逐位一致；把边裁成整数短边再均分会产生 33/34 DBU 分歧。
  分裂碎片沿用原段数学边号（edge_ids），否则 SegmentBatch 校验失败。
- **单 macro membership**：context 是均匀扩张，候选 core 范围可由 searchsorted
  精确求出；越出 macro 的远端段必须得到空范围而非 clip 到边界 core。
- **居中 canvas**（`opc/input/raster.py`）：差值平均分配、奇数余量归高坐标侧；
  全局 DBU→canvas 映射 `x_canvas = (x_dbu-context.left)/pixel - 0.5 + low_x`
  固定在 ownership_canvas 注释中，后续 EPE/probe 必须复用。
- **NPZ 契约**：problem format_version=1 不含 dbu_um（GDS 写出由调用方传入）；
  result NPZ 记录 round_index 供合并期一致性校验。
- **测试几何病态**：铺满层 bbox 的图形外扩位移全部落在 macro ownership 之外被
  正确裁掉（第一轮 XOR==0 是正确行为）；证明 +2 生效需要「锚框撑 bbox + 完全
  内部的动图形」布局。
- **性能参考**（gcd_45nm 2×2）：准备 0.45s、每轮 ~4.9s、合并 0.17s、总 10.6s；
  RSS 峰值 ~80MB；343018 段 / 722161 membership / 870 core。
- 已知保留的零消费符号：`edge_probe_points`（sampling.py 文档保护）、
  `reconstruct_contours`（公共中间入口）、`rasterize_region_window`（底层，测试直用）。

## 审查轮新事实（2026-08-16，commit fb80a4e）

- **切线交点重复分裂点**：斜边精确穿过 x/y 切线交点时（同一参数 t 同时满足
  两条切线），_split_segments_at_ownership_cuts 会产生两个等值穿越点拼接出
  零长碎段；修复为段内 isclose 去重。构造此类几何的最小例子：边
  (90,50)→(60,20) 在 t=1/3 处同时穿过 x=80 与 y=40。
- **空 macro 是合法状态**：查询框不接触任何图形的 macro（如远端 SREF 场景）
  产出空 SegmentBatch，切线分裂必须对空批次原样返回。
- **契约冻结点**：macro_size 严格大于 core（等于即拒绝）；双轮位移必须是
  [+2nm,-2nm] 的精确 DBU（和为零不够）。
- **own⊆membership 检查不得被空 membership 短路**：空 CSR 下 seen 全 False，
  恰好给出「全 -1 合法 / 有 owner 拒绝」的正确语义。
- 测试对照层技巧：验证「未处理层不复制」时源 GDS 必须含非目标层，否则断言
  是同义反复；验证位移生效时图形必须完全在层 bbox 内部（锚框撑 bbox）。

## lithography 批次事实（2026-08-16，Phase 5A）

- **Hopkins 前向公式链**（`lithography/iccad13.py`）：pad → fft2(norm="forward")
  → 四象限 kernel 相乘 → ifft2(norm="forward") → scale 加权 |field|² →
  dose² 缩放 → sigmoid(steepness×(I−target)) → crop。全原生可微算子，
  无手写 backward。
- **四象限映射的关键事实**：象限块尺寸由 **kernel 自身**（35→18/17）决定，
  不是频谱尺寸——频谱只有四角低频块（±17 频率）与 kernel 相乘，其余频率
  恒零；赋值顺序固定（左上→右上→左下→右下），DC/Nyquist 重叠行列由后写
  覆盖。探索转述易把象限索引误读为 256 频谱块，实施以旧代码原文为准。
- **数值身份**：新实现三工艺角 sums 与 OpenILT 同资产基线**逐位相等**
  （差 0.0）；确定性 mask 构造（[2,200,150] 固定公式）与期望值已移植进
  `tests/lithography/test_iccad13.py`。资产 SHA-256 是模型身份，硬断言。
- **居中 padding 双实现共享同一公式**：`_prepare_mask` 与
  `opc.input.raster._center_padding` 都是差值均分 + 奇数余量归高侧；
  模型对满 256 输入 padding 全零、不二次移动，raster canvas 可直传。
- **Windows DLL 事实**：环境 python.exe 直跑（非 conda run）时
  `torch.cuda.is_available()` 为 True 但首次 CUDA FFT 抛
  `nvrtc-builtins64_124.dll` 缺失——`<env>/bin` 不在搜索路径。
  最小修复 = 模块级 `os.add_dll_directory` + PATH 前置，必须在
  `import torch` 之前执行（lithography/iccad13.py 模块头）。
- **依赖纪律**：lithography 只 import torch + 标准库；main_test_lithography
  才桥接 opc.input.raster。测试导入 opc.input 无碍（tests 无此限制）。
- **性能**：GTX 1650 上三条件 256 canvas 前向 172.4ms / peak 32MiB；
  一次 forward_many = 1 次 mask fft2 + 每 bank 1 次传播（monkeypatch
  计数测试固化）。
- coverage 100%（204/204 语句），无豁免分支。

## run_single_pass 批次事实（2026-08-16）

- **边压切线退化**：图形边恰好与内部 macro 切线重合时，边整条归一侧 macro
  （中点归右/上），另一侧以 context 原位参与该侧拐角重建；两侧拼合处出现
  一位移宽度的台阶（XOR = 2×d²）。切线分裂只保证段不**跨越**切线；core 级
  切线无此问题（同 macro 内所有 owner 段统一位移），仅 macro 边界受影响；
  bbox 外沿例外（邻侧副本被裁剪成零宽）。测试几何须避开切线重合。
- **孔闭合算术**：+d 双向收缩孔，孔必须在两个维度都 > 2d 才不闭合
  （10 宽孔 +5/边 = 闭合；正向用例孔取 16×16 → 余 6×6）。
- 单遍入口复用验证管线全部核心（exact_dbu/plan_macros/prepare_macro_problem/
  reconstruct_region/write_macro_results），`[lithography]` 段仅为网格契约
  校验保留（两套网格合法性标准不可分叉）。
- gcd_45nm 单遍 +5nm 实测 0.80s（验证管线 10.6s——差异主要来自每 core 的
  居中画布栅格化，单遍入口不栅格化）。

## 旧库规模（迁移批次预估基准）

| 模块 | 行数 | 状态 |
|---|---|---|
| layout | 616 | ✅ 已迁移 |
| geometry | 495 | ✅ 已迁移 |
| lithography | 318 | ✅ 已迁移（Phase 5A，重写为 ~370 行 + main 入口 + 81 测试） |
| evaluation | 153 | ✅ 已迁移（Phase 6A 最小子集：metrics 100% coverage） |
| opc/input | 1315 | ✅ 已迁移（Phase 4 重构为 Macro–Core） |
| opc/input/edge | 758 | ✅ 已迁移（Phase 4） |
| opc/iteration | 1670 | mbopc ✅ 已迁移（Phase 6A，simple.py ~430 行）；diffopc/ilt 待独立设计 |
| main | 3357 | 验证管线 + MB-OPC 两入口 ✅；旧入口剩余待评审 |
| tests | 4177（旧） | 按批次对照移植（新树 330 用例） |

## MB-OPC 审查修复轮事实（2026-08-16）

- **「无法评价 ≠ 零违规」**：valid_probes==0 时 epe 恒 0（violation 只在
  有效探针上累计），旧逻辑把探针全无效（2nm 壁 + 8nm 探针穿壁）判成
  zero_epe。修复为 insufficient_probes 停止状态；循环内检查必须放在 best
  比较之前（valid==0 的 epe=0 会被 epe<best 误当改善）。空 macro（零段）
  的 zero_epe 语义正确（无违规对象），两者必须区分。
- **几何退化的异常形态**：reconstruct_region 的越界守卫不止抛
  ReconstructionError——四边共线退化（位移 −20）以 ValueError
  （"every ring must contain at least three vertices"）从 KLayout 数组
  校验冒出；且更大幅度（−25/−30）的边交叉会被 miter 解析成**反向合法
  ring**（正面积、不触发守卫）。测试构造越界场景时用 −20 的共线退化
  （最先触发的守卫形态），不要用 −30 翻转（守卫不炸）。
- **窗口物化防重复计数**：merge 回读验证逐 macro 窗口累加时，
  materialize_intersecting 不裁剪——跨界 polygon 伸出窗口的部分会被
  相邻窗口各算一次，必须显式 `& kdb.Region(ownership)` 裁回（与主路径
  clipped 同款）。
- **±2^30 魔法框的真实风险**：GDS int32 域 ±2^31，固定 ±2^30 只盖一半，
  域外图形静默不进 Region（无报错的数据损坏）；正确写法是
  `db.layer_bbox(layer)`（原生逐层包络，图形必然全含）。
- **TOML int() 静默截断**：`int(1.5)→1`、`int(True)→1`；配置层整数必须
  `isinstance(v, int) and not isinstance(v, bool)`（_as_int）。
- **stub 方向构造的三个变换**：_zero（全暗→全 +1 外移）、_ones（全亮→
  全 -1 内移）、_invert（反相→全 ambiguous 方向 0）；大幅移动后参考探针
  的判定基于 printed（模型输出）而非 mask——_ones 在任意位移下都给 -1，
  是构造越界场景的可靠变换。
- **无变化提案跳过**：directions 全 0 时 next==current，同一状态再评一轮
  无新信息（指标几何全同）；跳过后 no_update 的 records 只含 baseline
  （行为变化，metrics.json 消费方须知）。
- gcd_45nm smoke 三版本（迁移/P1/P2）四 macro best_epe 逐位一致——
  几何流式、窗口化验证、性能修复均零算法漂移。

## 最简 MB-OPC 批次事实（2026-08-16，Phase 6A）

- **评价层默认阈值分叉**：`evaluate_edge_probes` 旧默认 threshold=**0.499**、
  L2/PVBand=0.5（设计文档 §8.2 误写 0.5，已裁决保留 0.499——0.4995 这类
  边界灰度在两阈值下打印判定相反，测试固化该差异）。
- **探针坐标必须过 `points_to_canvas`**：旧 solver 公式
  `(x-left)/pixel-0.5` 与旧 raster 自洽（旧契约 tile+2×halo 恰满 256 画布、
  无 padding）；新 Macro–Core 的 228px 居中 + 14px padding 下必须补
  `+low_x/+low_y`，否则探针整体向左下偏 14 像素。ownership 全部 True 像素
  中心整数回映是批量一致性锚点。
- **方向写入漏乘步长是首版真实 bug**：`next[idx]=directions` 会把步长丢成
  ±1 DBU；正确为 `next[idx] += directions.astype(f64)*step`（测试
  `values==2.0` 一步拦截）。
- **stub 直通模型的量化陷阱**：`nominal==mask` 时零位移无违规成立（同图同
  采样），但**移动后**的直通输出因边界半像素灰度（step 非像素整数倍）会残留
  少量 outer 违规——「移动后归零」测试必须用像素整数倍步长（step=4×pixel=4）
  构造，否则断言脆弱。
- **invalid_geometry 测试的重建计数陷阱**：evaluate_and_propose 内 cache miss
  会重建零位移参考 Region（cache 预算 0 时每次评价都重建），monkeypatch
  reconstruct 计数会混入参考重建；按「首个非零位移候选」判别而非纯计数。
- **独立 macro 边界代价实测**（gcd_45nm CUDA，870 tile）：single（全 ROI 一
  macro）总 EPE 23440 vs multi（2×2）之和 23676——差 236 段（~1.0%）；
  最终覆盖 XOR 34650860 DBU²。EPE 逐轮单调下降但 8 轮未归零（启发式已知
  行为）。两入口各 ~126s。
- **merge 显式映射重构**：`merge_macro_results(plan, {macro_id: Path}, out,
  cell_mode)` 不读 result/不猜路径/键集必须与 plan 一致；轮次一致性校验
  （防旧轮 GDS 冒充最新）归验证 runner 的 `collect_round_macro_gds`。
  重构后 +2/-2 与 gcd XOR 零变化（TestTwoRounds/TestFinalMerge 全绿）。
- **load_macro_config 的段白名单机制**：共享六段键校验 + `extra_sections`
  放行流程专属段（iteration/mbopc），拼错段名进不了任何白名单；段内键由
  各流程 loader 自校验。
- **plan.json 不存 macro 切线**：save_final_lithography 用独立规整 tile 网格
  （单 macro 全 ROI 按 core 切分）并写入 manifest 对账；MacroProblem 不含
  dbu_um，GDS 写出函数必须由调用方传 dbu（solve_macro 同款补参）。
- simple.py coverage 99%：缺两行防御 RuntimeError（需破坏构造期不变量，
  不可构造）；evaluation metrics 100%。


## 梯度 MB-OPC 批次事实（2026-08-17，Phase 6A-G）

- **Adam 方向与梯度符号**：dL/dMask<0（印刷不足）→ 最小化器沿负梯度走 →
  位移为正（外移）；构造"内移退化"测试需印刷过量（dL/dMask>0）。Adam 单步
  幅值与梯度大小无关（首步 ≈ lr×0.316…实际 m̂/√v̂=sign(g)，首步 |Δ|≈lr），
  大 lr + clamp 可精确控制候选到恰 ±max_displacement（共线退化真构造）。
- **autograd.Function 直通返回**：forward 返回输入 tensor 时 apply 输出与
  输入 torch.equal 但不是同一 Python 对象（requires_grad 输入下被包装）；
  测试断言用逐位相等而非 `is`。
- **`2·g_mid` 手算基准**：半像素点 = 四角均值；跨批采样注意第二张图的扁平
  基址偏移（[B,H,W] 扁平索引 base=b·H·W，自检时两次算错都错在这里）。
- **rasterize_mask_canvas 边界对齐**：Box 边界恰在像素边界（20/4=5.0）时
  中点采样落在整数格点，mask 值取像素 5（完全覆盖）——线性模型梯度非零
  的取值来源。
- **gradient 产物三件套命名**：gradient_result.npz/gradient_metrics.json/
  best.gds（独立于 simple 的 result.npz/metrics.json，同目录共存互不覆盖）。
- **_resolve_device/_as_number** 为 simple+gradient 共享的最小抽取（DEC-004
  边界内：两个真实调用方才抽）。
- **P=0 防御分支不可达**：core ownership box 恒含至少一个 canvas 像素，
  total_pixels==0 仅在数据损坏时出现；保留 ValueError 防御。

## 全项目审查与 P1-1 修复事实（2026-08-17）

- **P1-1 复现口径**：`segments_for_core(c)` 过滤 `segment_to_parameter>=0`
  vs `owner_segments_for_core(c)`——2×2 跨界矩形 40 条 vs 24 条采样；丢失
  的 16 条全部是跨 core 边界段在邻 core 的 membership 条目（前向含其几何、
  反向不采）。
- **聚合机制**：autograd 的 `parameters[owned]` advanced-indexing gather 梯度
  回传天然 SUM（重复索引求和），无需手写 index_add_；修复只改"采哪些条目"。
- **EPE slots 必须独立**：修复前 EPE batch_index 复用梯度条目的
  member_slots（两者条目数恰同）；membership 采样后条目数不同，共用会错乱
  探针批号映射。
- **frozen slots 实例不可 monkeypatch.setattr**（pytest MonkeyPatch 内部走
  super 失败）；测试需打类级补丁（测试独占实例时无交叉）。
- **Adam 首步幅值 ≈ lr**（m̂/√v̂ 比率≈1，与梯度大小无关）；梯度幅值均匀
  缩放被自适应 largely 掩盖——这是 owner-only 采样未被 smoke 发现的原因。
- 其余 P1/P2（空 macro merge 崩溃、run_single_pass 校验漂移、simple loader
  NaN、cuda:N 峰值、EPE 阈值解耦等）已记录待用户裁决处置。

## P1-3 修复事实（2026-08-17，single-pass 配置收敛）

- **共享配置层**：`_macro_pipeline.MacroCommonConfig`（17 公共字段，frozen
  slots 基类）+ `load_macro_common_config(path, extra_sections, output_keys,
  output_required)`；`MacroPipelineConfig`/`SinglePassConfig` 继承基类各自加
  work_dir / displacement_nm——字段访问与消费方零改动。
- **dataclass 组装陷阱两枚**：`dataclasses.replace(基类实例, 扩展字段=…)`
  按基类构造直接 TypeError；`asdict` 会把嵌套 LayerSpec 递归转 dict
  （unhashable）。正确做法 `子类(**{f.name: getattr(common, f.name) for f
  in fields(基类)}, 扩展字段=…)` 浅拷贝。
- **output 段参数化**：公共层默认只必填 final_layout/final_cell_mode；
  work_dir 由 load_macro_config 经 output_keys 放行 + output_required 保序
  追加必填（错误文本与旧版逐字一致）；single-pass 不放行 work_dir（配置里
  出现即未知键拒绝，不静默忽略）。
- **入口→入口依赖消除**：run_single_pass 改从 `main._macro_pipeline`
  import exact_dbu/load_macro_common_config（原 `from
  main.run_macro_pipeline import exact_dbu`）。
- 用户裁决：P1-2（空 macro merge 崩溃）与 P1-4（Decimal 击穿）暂不修，
  立案待办。

## 审查问题 1/2/3/5 修复事实（2026-08-17）

- **CUDA 峰值显式设备**：`cuda_stats_device = torch.device(device)` 传给
  reset/max（不传时 PyTorch 统计 current device=cuda:0，多卡量错）；不调
  set_device 改全局。测试用真 CUDA 小跑 + spy 透传断言收到 torch.device
  对象（cuda:1 的映射是同一表达式，无需真卡）。
- **macro 前置校验成真**：_run_mbopc 的 macro_grid 检查上移到 prepare 之前、
  模型构造挪到全部校验后；plan 后兜底保留（macro_size_nm 模式只有 plan
  知道数量）。两条 monkeypatch"被调用即 AssertionError"证明零执行。
- **EPE 阈值统一**：evaluate_edge_probes 默认 0.499→0.5（standalone 三指标
  一致）；simple/gradient 显式传 threshold=model PrintThresh。数值影响实测：
  target 侧 uint8 量化在 [0.499,0.5) 无格点，但 nominal 侧是连续 sigmoid——
  带内确有探针采样点，判定翻转改变 simple 的方向序列。gradient smoke best
  loss 逐位不变（EPE 仅诊断、不驱动梯度）；simple multi best_epe 漂移
  ±1~15 段（7263→7264/5904→5893/5625→5640/4884→4892，<0.3%，方向不恒定）
  ——指标一致化的预期行为变化，非回归。计划原预期"simple 零漂移"只考虑
  了 target 量化、漏了 nominal 连续带，如实记录。
- **lr 超限 UserWarning**：load_gradient_config 在 lr>max_displacement 时
  warnings.warn（stacklevel=2），合法集合不变、参数原样；不自动截断。

## TestReticle 版图集事实（2026-08-17）

- **纯空白不贡献 layer bbox**：稀疏版图必须有角标记图形撑开包围盒，
  否则"空 macro"根本进不了网格域——sparse_6um 用右下/左上两个 200²
  标记（刻意避开右上象限）撑到 5.7×5.7µm，实测右上宏 S=0。
- **正负板成对产出**：GDS 不携带极性，_clear/_opaque 文件字节相同，
  文件名即预期 config 极性值（防呆）；SHA-256 抽查一致。
- **bench 尺寸实测**：母题（六族两列自然摆位）≈9.6×10.5µm；30µm 版 =
  母题 3×2 = 32×21µm/672 core；100µm 版 = 母题 10×7 = 109×76µm/
  8025 core。原"格框"设计在图形自然尺寸下撑不满格，实施改为平铺。
- **100µm 压力实测**：16 macro/8025 core 梯度一轮 CUDA 176s（≈46 core/s，
  稀疏图形快于 gcd 的 21 core/s）；CUDA 峰值 495MiB 与 30µm 版完全相同
  ——批内张量尺寸不随 core 数变化，显存与规模解耦，瓶颈在吞吐。
- **P1-2 素材有效**：sparse_6um [2,2] 3.1s 后精确失败于 merge
  （LayerNotFoundError 11/0，调用链确认）。
- 生成器 TestReticle/build_reticles.py（仅依赖 klayout，--list/--only，
  幂等）；20 份 GDS 已入库；单测仍用自建生成式 GDS（纪律不变）。

## 配置系统重构事实（2026-08-18）

- **两轮都漏了 EdgeConfig**：方案 §4 与批准计划的首批清单都没有 [edge]
  段的归属——边段化（corner/segment/max_disp/miter）是 simple/gradient/
  验证/单遍四方共用的算法无关配置，实施时补第八个 Config。
- **[iteration] 同名冲突**：单遍（displacement_nm）与验证管线
  （round_deltas_nm）共用段名但字段不同——全量未知字段检查会互相误伤；
  单遍改 [single_pass]、验证建 ValidationConfig（冻结 ±2nm 迁入
  __post_init__，load_validation_deltas 删除）。
- **f-string 模板的正则清理陷阱**：`device = "{values["device"]}"` 行被
  两次不同正则删（一次删错一次补回），模板键挪移用"段尾锚点插入"比
  "先删后补"稳。
- **_prepare 元组切片错位**：configs[:5] 把 Validation 当 output 传入
  （Validation 无 work_dir 属性报 AttributeError）——元组多态装配必须
  解包命名，不能位置切片。
- **plan dict 的值是 str**：run_macro_pipeline 里 plan["work_dir"] / x
  报 str/str TypeError——plan JSON 序列化产物一律 Path() 包裹再拼接。
- **workflow.load_config 幽灵调用**：_mbopc_workflow 从 configuration
  import 了 load_config → 旧测试 workflow.load_config(path) 不 AttributeError
  而是静默返回空元组（无 config_types）→ "DID NOT RAISE"——跨模块同名
  导入会制造静默成功路径，测试调用点要显式传类型。
- smoke：simple（bench_30um 8 轮）47.6s best_epe 至 497；gradient
  （用户实验 config gcd_30um [1,1] 10 轮）205s loss −50%；管线 XOR=0。
  gcd_45nm 已删，旧基线数字不可比（报告如实记录口径）。

## common 包集中事实（2026-08-18）

- **相对导入漏检陷阱**：grep `from opc.input._arrays import` 漏掉
  `from ._arrays import`（grid.py）——as_points 实际有真实调用方
  （grid.locate_owned_points），"零调用可删"结论错误，删文件后
  ModuleNotFoundError 才暴露。教训：删模块前必须同时搜绝对与相对导入。
- **用户清单三处与现状不符**：_as_int 已被配置重构删除（casting 因此
  取消——唯一调用方 iccad13 按用户裁决不动）；ensure_2d_float32 不存在
  （实为 as_vector/as_matrix/as_points）；iccad13 的 as_integer/
  as_finite_float 是 from_file 嵌套闭包非模块级。
- **切片删除的锚点顺序**：t[:start]+t[end:] 在 end<start 时变复制——
  _macro_pipeline 的 _PLAN_FORMAT_VERSION 在函数定义前，删除区间反转
  导致定义重复，git checkout 恢复重做。
- **_center_padding 双实现维持**（共用需 lithography import common，
  随"litho 不动"裁决一并搁置）。
- main 内三份 NPZ 原子写归一（workflow 2 + run_macro_pipeline 内联 1）；
  四组旧符号残留 grep 零命中；全量 444 passed；smoke 基线逐位复现。
- **NPZ 原子写收口补遗**（用户指出）：MacroProblem.save() 内联的第 4 份
  npz 原子写改用 common.io.atomic_write_npz（problem.py 的 os/tempfile
  import 随之清零）；收口后 mkstemp 模式全仓仅存 common/io.py 两处唯一
  实现 + _macro_pipeline.write_macro_gds 的 GDS 载荷版（不同载荷、单
  调用点，不收）。

## _mbopc_workflow 拆分事实（2026-08-18）

- 按算法拆分（用户方案全采纳）：_simple/_gradient 两个 workflow 各自
  独立（配置/结果版本/求解器 import 全分家）；save_final_lithography
  迁 _macro_pipeline（公共后处理，该文件因此新增 torch/PIL/numpy 依赖
  ——main 层可接受）；_mbopc_workflow.py 删除，不建 shared 中间层。
- **测试 import 巧劲**：两 runner 测试都是 `import X as workflow` 单别名
  ——只改 import 行，全部 monkeypatch 目标（prepare_problems/
  ICCAD13Lithography/merge_macro_results/optimize_*）随别名自动跟随新
  模块，测试体零改动。
- save_final 直测用直通 stub（forward_many=mask 恒等）：四成员
  device/config/condition/forward_many 即满足留档消费面，无需真模型。
- 手术陷阱延续：_macro_pipeline 追加函数后补依赖要连带 numpy（np.rint/
  where 在 PNG 变换里）；第三方 import 排序 klayout<numpy<psutil。

## doc_ 切换为 doc（2026-08-18）

- 12 个 doc/ 旧文件在 doc_/archive 有同源副本直接删；8 个增量迁移
  （两手册至根活跃位、gradient design Rev0.2 归位 completed CHG、两报告
  原件入 archive/reports、mbopc design 用户新版覆盖副本、config_refactor
  新 CHG 含摘要版 spec——1638 行规格原件在用户本地不入库）。
- **git mv 与文件系统移动混用陷阱**：shutil.move（active→completed）绕过
  git 索引后 git mv doc_ doc 报 bad source——统一走文件系统 mv + git
  add -A，让 rename 由相似度推断（18 R 记录）。
- CLAUDE.md 仅做路径字符串级更新（用户领地纪律）；瘦身/重写仍留待
  用户另行指派。changes/active 清空（下一个 CHG 自建）。

## resolve_*_config 集中事实（2026-08-18）

- 职责三分：load_config=TOML→Config；resolve_prepare/mbopc/gradient_config
  =组合校验+nm→DBU+运行时配置构造（PrepareRuntime 打包返回，4 个真实
  调用方）；prepare_problems/run_*=流程调度。
- **跨段校验时机后移**（行为变化）：step≤max/epe≤context/lr warning 从
  prepare 前移到 prepare 后（resolve 需要 dbu_nm）——非法配置先跑一次
  prepare（bench_30um 0.07s/gcd ~1s）再失败；"非整除 dbu"类本就在
  prepare 后，两级时序归一。
- **用户清单外补第 4 消费方**：run_single_pass 与 prepare 有完全相同的
  6 项换算+FragmentationConfig 构造——只改 3 个 workflow 会留第四份
  副本；displacement 换算与 |d|≤max 留在入口（单遍专属）。
- 类型注解的 Config import 不能随构造职责一起删（solve_macro/
  solve_gradient_macro 签名仍用）——"构造走 resolve"≠"类型不引用"。
- 残留检查口径：main/ 内 Simple/Gradient MBOPCConfig( 与
  FragmentationConfig( 的构造点应仅存 configuration.py；exact_dbu
  仅存 run_single_pass 的 displacement 一处。

## MB-OPC 公共 workflow 上提事实（2026-08-18）

- **显式 supersede**：拆分轮记录的"不建 shared 中间层"由本轮推翻——
  用户主动提出 adapter 方案（callback 注入 + 生命周期唯一化；防的是机械
  合并与巨型 if 分支，两者都不发生）。新增方法自此只写一个 adapter 文件。
- MBOPCMethod 七字段（method_name/algo_config_type/build_solver_config/
  solve_macro/save_macro_result/macro_summary/summary_extras）；**不建
  MacroSolveOutput**——序列化与摘要全在 adapter 侧后公共层对 result 零
  字段消费（仅透传 best_gds），无真实调用方（用户方案评估后的裁剪）。
- **晚绑定纪律**：adapter 的 solve 必须以模块全局名调用 optimize_*，
  测试 monkeypatch(workflow, "optimize_gradient_macro") 才能拦截——禁止
  把 optimizer 作为 MBOPCMethod 字段在 import 期捕获（捕获即冻结原函数）。
- **幽灵调用重现**：merge 计数测试的 patch 宿主必须随循环迁到
  _mbopc_workflow（两处已改）；patch 打在不再被消费的名字上会以
  calls==[] 失败暴露，不会静默通过。
- **资源统计上提**（行为变化，加法式）：simple summary 新增 method/
  rss_*/cuda_peak 五键（test_summary_and_artifacts 补断言）；逐 macro
  RSS 采样物理上住在循环体内，不可能留在 adapter 侧。
- 行为零变化验证：gradient loss 0.069138/CUDA 峰 501MiB 逐位；simple
  bench_30um 多 macro best_epe 1596/1011/820/497（45.5s）逐位；445 passed。
- **既存损坏（非本轮引入，未修）**：config/mbopc_single_macro.toml 的
  context_nm 在用户 7b3ca1e tmp 提交改为 1024，core 1024+2×1024=3072
  超 2048（256×8nm）画布上限，prepare 即失败——"单 macro 497"口径有误，
  1596/1011/820/497 是 multi 配置四 macro 的 best_epe；单 macro 配置
  待用户裁决（改回 400 或缩 core）。
- 行数：simple 123/gradient 125（原 178/203），公共层 150；run_* 入口
  零改动；文档 5 处同步（development_manual/architecture×3/contracts）。

## solve 上提与外层条收尾事实（2026-08-18 P2 两项）

- **solve 包装去重（用户 P2-1）**：MBOPCMethod 字段 solve_macro 改
  optimize_macro（裸 optimizer 本体），tqdm/(iterations+1)×core_count/
  finally/reconstruct best/write best.gds 迁为 _mbopc_workflow._solve_macro；
  两 adapter 降至 optimizer + 序列化/摘要钩子 + METHOD + 薄代理（各 ~85 行）。
- **可测性模式改写（supersede 本日上午的晚绑定纪律）**：frozen 字段在
  import 期捕获函数本体后，模块属性 monkeypatch 失效；测试注入改
  dataclasses.replace(METHOD, optimize_macro=替身) + 重绑定 adapter 模块
  全局 METHOD（run_* 代理按模块全局名晚绑定读取）。生产代码不留
  仅为测试服务的转发壳。
- **鸭子契约扩一项**：result 必须暴露 best_displacements（best GDS 重建
  消费），与 solver_config 三属性同记 MBOPCMethod docstring。
- **外层条 finally（用户 P2-2，bug 修复）**：macro 循环包 try/finally；
  回归 test_outer_bar_closes_on_midway_error 双向验证——无修复
  closed(1) != created(2) 失败、有修复通过。全量 445 → 446 passed。
- 数值零变化复验：gradient loss 0.069138/CUDA 501MiB、simple 多 macro
  1596/1011/820/497 逐位。

## 注释整改事实（2026-08-19，用户三规则 + AGENTS.md 授权改写）

- **三条新规**：去变更管理 ID/设计文档章节引用（REQ/ERR/INV/DEC/§N/
  阶段 N/"本 change 清单"类）；难懂变量必须注释（segment_to_parameter
  正式说明替换 scratch 示例）；跨行语句注释前置、不逐行加行尾注释，
  单行语句行尾注释保留。用户裁决：tests 不纳入；lithography/geometry
  纳入；AGENTS.md 授权改写。
- **AST 口径陷阱**：ast.Try/With 的 end_lineno 覆盖整个块，行尾注释
  计数虚增约一倍；正确口径是 tokenize 逻辑行（独立注释仅在括号延续内
  并入跨度）。首版脚本把语句前独立注释折进跨度，导致单行语句被误处理
  且注释插到既有块注释上方（三明治）——修正为独立注释仅在 cur 活跃
  （括号内）时并入，git checkout 回滚重做。
- **脚本机械 + 人工合并**：单片段自动上移；≥2 片段（main 69 处）逐处
  复核——per-key 纯标签（macro 总数等键名自释）直接丢弃，携带独立信息
  的并入首行注释（三工艺角条件、Round N 记录、两处 mkstemp）。
- **I001 空行修复安全**：import 注释上移触发 ruff I001，--diff 确认
  期望仅为注释前补空行（无重排）后 --fix；此前"ruff --fix 拉乱 import"
  前科的场景是重排，本次不适用。
- **设计文档引用清理清单**：§16/§5.3/§7.1/§7.3/§5.1/§11.7、阶段 0/1/
  3、阶段 0 步骤 7、"消 main 内第三副本"、"本轮不修改清单"；demo 自身
  流程节标（main_test_lithography 阶段 1–6）为自含结构保留。
- AST 等价校验全绿（22 文件 vs HEAD）；残留复查归零（范围内跨行行尾
  注释 0、ID 引用 0）；446 passed ×4 批；双 smoke 逐位复现。
- 一次性脚本存 D:/temp_hoist/（hoist_comments.py + ast_check.py，
  collect 可复用作残留复查），不入库。

## TestReticle 负板重制事实（2026-08-19）

- **旧正负板规则作废**：原"两份内容完全相同、文件名即极性"使用户无法
  区分正负——改为真互补板：_opaque.gds = 图形包围盒补区，配 polarity=
  "opaque" 与 _clear 表达同一透光目标。
- **Region(RecursiveShapeIterator) 惰性挂接陷阱**：Region 借迭代器构造
  后不立即物化，layout 被 GC 即变空Region——验证脚本两轮全 0 就是它
  （bbox/并集全 0 的"OK"是空洞真）。回读 Region 必须保 layout 存活或
  在其作用域内消费（管线 LayoutDB.open with 块内物化的既有纪律同源）。
- **GDS 头时间戳**：同参数重跑字节必变（BGNLIB 时间戳），plan 文档
  "再生成幂等逐字节一致"表述过强——幂等的是几何，不是字节；clear 十份
  按几何等价从 git 恢复避免无谓 churn。
- **贴边图形收缩负板 bbox**：图形贴住包围盒边的方向补区够不到框边
  （lines_dense/dense_iso），两份 layer bbox 不同→网格划分不同，对照
  实验须知情（文档已记）。
- **巨型负板多边形**：bench_100um 补区单多边形含 6860 孔，GDS 记录超
  0x8000（klayout 可写读、读时告警，标准严格读端不兼容）。

## gradient 采样中点一致性修复事实（2026-08-19，用户 P1）

- **问题机理**：reconstruct_contours 在 corner 按相邻 offset 线交点重接
  （junctions[corners]=intersections），相邻段位移不同时候选段端点含
  切向调整；旧 backward 采样点 = 参考中点 + 法向×位移（刚体假设），
  与 forward 几何脱钩。即使全边同位移，corner 邻段中点也偏移
  邻边法向分量（矩形 +8 时偏差 8 DBU = 2 像素）。
- **实现**：_reconstruct_geometry 在 two_points 后向量化产出
  segment_midpoints（与拼接规则一一对应：two_points 边界前段终于
  previous_end/后段始于 current_start、普通边界共享 junction、
  same_position 内部取共线中点；float64 连续域不随 np.rint）；
  gradient 以 reconstruct_region_with_midpoints 一次重构绑定发布
  Region+中点，批内 gather 已发布中点（删两条常驻数组）。重构计数
  契约不变（iterations=2 恰 3 次）。
- **判别证据链**：几何单测（解析期望：corner 邻段刚体 [56,17] vs
  实际 [55,17]、45° 角偏差 1.66 DBU）+ spy 成员关系测试（apply 实收
  中点 ∈ 已发布重构换算集合，旧刚体值对不上任何发布行）。**如实
  记录**：非均匀状态 FD 方向测试在 4 DBU 像素下不判别旧新（切向偏差
  1~1.7 DBU 亚像素、STE 梯度带平滑，矩形与 45° 两种几何实测旧/新
  surrogate 均与真实差分同号，仅幅值差 ~3%）——它是固定后语义的
  回归守卫，不是旧代码捕捉器。
- **数值行为变化（修复生效的证明）**：gradient smoke（gcd_30um [1,1]
  iterations=10）旧 0.069138/iteration_limit/215s → 新 state1
  0.134467（baseline 0.1498 的 −10.3%）后 state2 候选 zero_length_edge
  被守卫拒绝、invalid_geometry 终止（两次复跑逐位一致，58s/CUDA
  501MiB）——修正后 corner 梯度走不同微观轨迹，撞上密集小特征的
  整数化退化；守卫按设计保留 best、留 stop_detail。这暴露一个后续
  观察项：~1nm 级位移即可触发 zero_length_edge 拒绝（密集特征的
  rint 脆弱性），若 invalid_geometry 早停频发需评估候选回退/步长
  衰减策略（本次不做）。
- 测试 4 → 452 passed；simple/单遍/验证管线零影响（simple smoke
  逐位不变）。
- **立案待办**：optimize_gradient_macro 结构拆分（prepare/evaluate/
  step/orchestrate）——用户要求 midpoint 修复先行落地、数值变化
  归因清晰后再拆，结构重构另开任务。

- **中点按需计算（用户 P2）**：_reconstruct_geometry 增 with_midpoints
  旗标（默认 False），simple/验证/单遍等 reconstruct_region 热路径不再
  付中点四个数组（约 56S 字节临时内存）的成本；仅
  reconstruct_region_with_midpoints 传 True。ReconstructionResult.
  segment_midpoints 类型放宽为 | None；新增旗标契约测试（默认 None、
  请求才产出、两种请求轮廓逐位一致）。双 smoke 基线逐位不变，
  453 passed。

- **梯度单位契约（用户审查）**：sampling midpoint 在 canvas/pixel 域而
  位移参数是 DBU——∂x_canvas/∂d_dbu=1/pixel_dbu，旧 backward 返回
  2·g_mid 等价把参数当 pixel 位移。修复：apply 增末位 pixel_dbu（ctx
  普通属性），backward 返回 2·g_mid/pixel_dbu（2 与单位换算两件独立
  事）；无 lr 补偿、参数/几何层保持 DBU。新增 pixel-size invariance
  测试（pixel_dbu 1/2/4 → 方向一致、幅值 g、g/2、g/4）；公式测试改
  pixel_dbu=4 非平凡值锁定 ÷4。两个子类化 forward 的测试代理同步签名。
- **三个 gradient 基线**（gcd_30um [1,1]×10，同 config）：刚体中点
  0.069138/iteration_limit（采样位置错误）→ 真实中点 0.134467/
  invalid_geometry（state2 候选撞 zero_length_edge）→ 真实中点+单位
  修正 0.106994/iteration_limit（两次复跑逐位；÷4 缩放经 Adam eps/
  偏差校正瞬态改变早期微观轨迹，避开了退化候选）。Adam 对统一缩放
  仅近似不变（eps=1e-8 非零、偏差校正期敏感）——印证用户"实际优化
  影响往往不大但非零"的判断。454 passed；simple 逐位不变。

## gradient 结构重构事实（2026-08-19，收口上一节立案待办）

- **边界**（用户三段方案，两私有 dataclass）：`_GradientMacroContext`
  只存整个优化不变的静态量（owner 映射、参考 Region+零位移中点、逐 core
  sampling/owner membership、探针坐标、total_pixels、device/threshold/
  conditions）；`_GradientStateEvaluation` 只描述一次评价的多指标输出。
  parameters/optimizer/current 几何/best 刻意不入 ctx——静态上下文与
  迭代态显式分离。
- **三段函数**：`_prepare_macro_context`（原 L187–242，含 total_pixels==0
  数据损坏 raise；`del reference` 释放随迁）；`_evaluate_state`（原 batch
  循环 L262–405 逐字搬入；只 backward，绝不 zero_grad/step）；
  `_take_optimizer_step`（原 L433–447；None 即 no_update，非法重构的
  ValueError/ReconstructionError 原样上抛由主函数定停止——不引入
  valid:bool 复制异常体系）。主函数只留编排（入口校验、no_owner 快速
  返回、循环控制、record 构造、best 严格更小、停止判断、成对发布
  Region+midpoint），308 → ~110 行。
- **裁决记录**：用户曾提议把 reconstruction.py 整理成共享
  materialize_reconstructed_geometry——该 drift 已由上节 P1 修复消除
  （reconstruct_region_with_midpoints 即同源接口），本轮裁定不改
  reconstruction.py。`_take_optimizer_step` 增 macro_id/state_index 两个
  仅服务错误消息的关键字参数（保留 FloatingPointError 原消息文本）。
- **唯一接受的非契约微差**：records 的 elapsed_seconds 语义收窄为纯
  评价耗时（原值额外含一次 parameters.detach().cpu() 同步）；无测试
  断言、无消费方比较。
- **行为不变验证方式**：既有 45 例期望零改动（只经公共入口与模块级
  _EdgeGradientMask monkeypatch）；新增 TestStructuralSplit 4 例结构单测
  （ctx 映射与 problem 一致、build_gradient=False 不建 grad、=True 只
  累积不改参数、step 返回 None/二元组/异常上抛三态）；数值等价以
  CPU A/B 逐项对比验收（重构前后各跑一次 gradient_mbopc，逐 state
  loss/诊断指标/best/stop 排除计时字段后全等，best_displacements
  npz 逐位一致）。


## Simple ILT 迁移事实（2026-08-19，CHG-20260818-simple-ilt）

- **像素输入层**（`opc/input/pixel`）：每宏一次 `rasterize_region_window`
  存 uint8 transmission（NPZ v1，不存每 core 重复画布）；core 画布按需
  切片 + `_center_padding` 居中（包内私有复用保证与模型 padding 逐位一致）；
  trainable 索引定义在 macro 网格（跨 core 恒同值）。**实际 box 整像素契约**：
  等价于 bbox 宽高 pixel 整数倍——像素管线比 edge 严，smoke GDS 选型必须
  核对（corners_unit 1900² 是 4 的倍数而非 8）。
- **宏同步语义**（Rev 0.2 演进）：同一 state 全批读同一 CPU 宏参数快照
  （numpy 取值即快照、无 autograd 直通）；梯度经每批 leaf 张量 backward
  后 `np.add.at` scatter-add 求和（avgpool 耦合 stub + batch1/2 对照锁定，
  逐点 stub 无空间耦合、测不出跨 core 和）；屏障后单次 SGD；macro best
  严格更低（4-core 常数数值表使 macro 最优 ≠ 材料核局部最优）。
- **logit 饱和性质**：严格 0/1 像素 sigmoid 斜率 ~β·eps（≈5e-7），有效梯度
  经分数覆盖格进入；纯对齐几何一轮更新低于 float64 记录精度——测试几何
  必须含非整像素边界。OpenILT ±1 初始化不满足 1e-6 恢复契约，规格有意
  取 coverage-preserving；全域活跃梯度需求另立 change。
- **共享层必要修复**：空 macro 候选（无材料区域）GDS 无目标层 →
  `merge_macro_results` 两处回读（候选 + 面积守恒验证）改为层缺失按零覆盖；
  稀疏版图正常形态，端到端 mr1c1 全空宏为回归。
- **配置直注册**（DEC-007）：`_parse_config` 经 `get_type_hints` 后
  SimpleILTConfig（postponed annotations）直接挂 `[simple_ilt]`，无第二份
  用户配置；runner 内函数级 dataclass 探针锁定 int/float/Path/tuple 解析。
- 全量 458 → 525 passed；阶段 A 单遍 smoke XOR==0（GridRuntime/writer 迁移
  零回归）；smoke：corners_unit 16 core / 225,625 像素 / CUDA 1.90s /
  best_state=1。


## Simple ILT P1-1 修复事实（2026-08-19，CHG-20260819-simple-ilt-openilt-init）

- **问题定性（用户裁决）**：logit+float32-eps 初始化把 0/1 像素参数推到
  ±logit(1.2e-7)/β，sigmoid 斜率仅 β·eps≈4.8e-7（β=4）——内部像素几乎
  不可优化，而 ILT 的拓扑变化/开孔/SRAF 恰需这些区域可激活。这不是
  "观察性质"而是缺陷；纯对齐几何一轮更新低于记录精度即为实证。
- **修复**：`params = 2·T − 1`（00_PAST/OpenILT 同式），斜率
  β·σ(β)σ(−β)≈0.0707（约 1.5×10⁵ 倍）。保持性质
  σ(β(2T−1)) ≥ 0.5 ⟺ T ≥ 0.5（state0 二值掩膜与 T 二值化一致）；
  废除 1e-6 恢复契约（新 change supersedes REQ-006 初始化子句，不回改
  历史 CHG 文档）。
- **回归证据**：新增纯对齐几何真模型用例——loss₁≠loss₀ 且 max|Δp|≈
  3.9e-4（阈值 1e-5；饱和方案 <1e-6）。阈值依据：真实光刻链路的
  dprinted/dmask 低于线性 stub 估计，位移量级 1e-4 而非 1e-2。
- **测试改写要点**：常数/逐调用模型的损失监督是 T 而非 mask——其期望值
  与初始化无关（曾误改为 soft₀ 监督，已纠正）；曲率参考的卷积输入是
  state0 soft（随初始化变化）；float64 镜像 init 同步。
- **smoke 重调**：梯度尺度随初始化增大 ~10⁵，step_size 10 严重过冲
  （state1 损失反升至 43400）；定格 1.0：records 7952→6233（−21.6%）、
  binaryL2 2893（优于旧基线 2896）、0.99s/CUDA 503MiB。教训：初始
  化方案与步长尺度耦合，后续方法换参数化时 smoke 步长须同步评估。
- 全量 525 → 526 passed。


## Simple ILT P1-4/P2-1 修复事实（2026-08-20，用户算法审查）

- **P1-4**：trainable 全局扁平索引原 int32，宏总像素 >2^31（4nm pixel 约
  185µm² 见方）在 CPU 构造期溢出——负值经 `>=0` 判据误判为 macro 外
  context、正值错位；GPU 前的 int64 转换无效。全链改 int64（返回/canvas/
  两个 arange），删 solver 冗余转换；256² 画布 +0.5MB 可忽略。教训：索引
  dtype 是值域契约，与 numpy 弱提升（python int 不升位数）叠加时溢出点在
  构造侧而非消费侧。
- **P2-1**：context=0 合法 + curvature valid 卷积 → ownership 边缘一圈
  曲率按 core 切分丢失（同宏 2×2 vs 4×4 core 损失不同，网格切分不应改变
  损失语义）。裁决：不改编卷积，入口联合约束 curvature_weight>0 ⟹
  context ≥ 1 像素；关曲率时 context=0 仍合法。
- **契约措辞**：REQ-B 的 σ(β(2T−1))≥0.5 ⟺ T≥0.5 仅在 mask_threshold=0.5
  成立；原表述把它写成任意阈值的通用不变量。修正为限定 0.5；threshold
  可调性保留。OBS：监督目标是 u8 量化 T（≤1/510）为规格契约。
- 全量 528 → 529 passed。


## 2026-08-20 Gradient EPE loss 规格修正（用户 P1/P2，Rev 0.2）

- **P1（profile 聚合 mean→sum）**：固定边缘偏移下 D 的非零槽位数 ≈ 过渡区宽±偏移，
  与 R 无关——sum 使 d_s 近似"偏移的 pixel 数"，Q/epe_distance_nm 改变不再漂移
  loss 尺度；mean 随 Q 线性稀释。原 DEC-002"不使用 sum"的裁决反转（zero-based
  ×(sigmoid−0.5) 与 [0,1) 值域保持）。连带：sum 使 gamma·d_s 更易饱和，
  epe_steepness=4.0 列为实施 smoke 待验证项。
- **P2（segment 归约等权→参考长度加权）**：等权下 16nm corner 段与 32nm 长段贡献
  相同（短段单位边长权重 ×2），仅改 fragmentation 即改目标；长度加权
  Σlen·pen/Σlen = 沿 target 边界的均匀离散积分，对切段基本不敏感。len_s 由参考段
  两端点计算、随 EPE 元数据缓存（§8.3 新数据行）。
- 规格同步面：REQ-003/INV-005/§7.3–7.4/§8.3/§10.1–10.2/DEC-002 重写 + DEC-007
  新增；新测试 TEST-013（Q 不变性，mean 版在该用例必然线性衰减构成判别）与
  TEST-014（切段不变性，含非等长混合段）+ 矩阵行 + AC-011；基线刷新至 08f4866。
  两个修正相互独立：P1 治 profile 宽度漂移，P2 治 fragmentation 漂移。
- 规格仍为 draft，待用户批准后实施。
