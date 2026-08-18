# ILT 多方法迁移规格研究记录

## Baseline

- Git commit：`2fa75ea89ea6cd64122214f1e2e0ed14cae518c3`
- 当前工作树在首次检查时无未提交修改；交叉审查时出现 `opc/input/grid.py` 一处注释删改，
  删除“设计文档 §5.3”字样、行为不变、来源无法确认。本任务不覆盖、不提交该差异，规格基线改为 dirty 并逐项列出。
- 当前已实现：ICCAD13 可微光刻、evaluation、simple MB-OPC、gradient MB-OPC、Macro–Core 输入与共享 MB-OPC workflow。

## 待核对

- 当前 ILT 契约是否只有文档、是否存在生产实现。
- OpenILT 内可迁移方法集合与方法之间真实共享点。
- 首个 ILT 方法要冻结的最小公共接口。

## 初步源码事实

- 当前 `doc/contracts/ilt.md` 明确声明 ILT 尚无生产实现，且 ILT 优化变量是像素或水平集，不经过 `SegmentBatch`、segment owner 或边段重建。
- 当前光刻公共契约已经为 ILT 预留原生 autograd：`lithography/contracts.py::LithographyModel.forward_many`；不需要再抽象一套模型接口。
- 当前新树没有 `opc/iteration/ilt/`，不能把旧 ILT 目录结构误写成当前行为。
- `00_PAST/opc/iteration/ilt/` 确认有四个候选：`simple.py`、`multilevel.py`、`levelset.py`、`curvmulti.py`，并有对应四组测试及旧入口。
- OpenILT 原项目同样有 `pyilt` 方法族；后续需逐文件核对旧适配与上游算法的差异，不能只依据文件名决定迁移范围。
- 当前配置已集中到 `main/configuration.py`；当前 MB-OPC workflow 是方法描述对象 + 公共宏循环，但该抽象是否适合像素型 ILT 必须以数据流核对，不能直接复用。

## 方法候选与初步顺序

- OpenILT `pyilt/` 的四个优化器是 `simpleilt.py::SimpleILT`、`levelset.py::LevelSetILT`、`multilevel.py::CurvILT`、`curvmulti.py::CurvILT`；后两个类同名但算法文件和配置不同，计划中必须使用不混淆的本项目名称。
- `00_PAST` 已把四者适配为函数式入口：`optimize`、`optimize_levelset`、`optimize_multilevel`、`optimize_curvmulti`。该形态比上游“solver 类持有 lithography”更贴近当前项目，但仍需审查是否存在旧版重复结构和全图内存假设。
- 首个方法最合理的候选是 simple ILT：它是最低复杂度的像素参数优化，可先冻结目标画布、current/best mask、评价状态、结果与 workflow 的兼容边界；level-set 与多层方法随后只复用已经有真实调用方的部分。
- 当前规格模板比用户最初模板多了 baseline/evidence、阶段边界、traceability、approval gate 等交接必要章节，应直接复制使用，不另造格式。

## 当前架构边界

- `lithography/contracts.py::LithographyModel` 已是模型公共协议；首个 ILT 不应新增第二个模型抽象。
- 当前 `main/_macro_pipeline.py` 承担 GDS→MacroProblem（边段化）、GDS merge、最终光刻 PNG；ILT 不经过 MacroProblem，因此不能直接把 MB-OPC workflow 或 MacroProblem 当作 ILT 公共输入。
- ILT 若以现有 macro/core 全掩膜流程运行，需要新的像素型 problem/workflow；是否在首版支持多 macro，必须由参考算法边界与 seam 语义进一步决定。

## `00_PAST` 算法适配事实

- `simple.py::optimize`：像素参数经 sigmoid 生成软 mask，SGD；默认 nominal+dose_max+defocus_min，loss 为 nominal L2 + process L2 + 连续 PV 差 + 可选 mask 曲率；支持 `[H,W]`/`[B,H,W]`、初值和 optimization mask；按 total loss 保存 best。
- `levelset.py::optimize_levelset`：精确 CPU SDF 初始化，前向 `phi<0` 硬二值，反向用 `-|grad(phi)| * upstream` 代理梯度，Adam；复用同一 loss 与旧版 `SimpleILTResult`。这说明通用结果确有多个真实算法消费者，但旧命名造成 method-to-method 耦合，应在首迁时使用中性类型名。
- `multilevel.py::optimize_multilevel`：严格递减且以 1 结束的 scale 序列；每级独立 Adam 和 best，控制网格逐级放大，但光刻始终在完整物理网格计算；曲率作用在当前级 wafer（需继续核对完整源码）。
- `curvmulti.py::optimize_curvmulti`：同为多尺度控制网格；与 multilevel 的关键差异是平滑 sigmoid 参数化、监督/曲率位置（旧注释指出曲率在 nominal wafer），需从未截断源码和 OpenILT 原文再确认。
- 旧四组测试覆盖算法公式、非法输入、batch、真实 ICCAD13 backward、GDS runner、CLI 和产物；不能整套照搬，因为旧 runner/config/offline input 已不存在，但这些场景可作为新规格测试矩阵依据。

## 当前配置与栅格约束

- `main/configuration.py::load_config` 是 section→dataclass 唯一解析入口；新增 ILT 配置必须注册 `CONFIG_SECTIONS`，不能恢复旧 argparse/configured parser。
- 当前 `_parse_scalar` 对 `tuple[int, int]` 有效，但按 `len(get_args(annotation))` 处理元组；`tuple[int, ...]` 会被误判为恰好两项且还会遇到 `Ellipsis`。多尺度计划若使用可变长度 tuple，必须把这一接口改动明确列出并做现有 `macro_grid` 回归。
- `opc/input/raster.py::rasterize_mask_canvas` 的数组约定是 `[y,x]`、行 0 为最低 Y、值 1=透光；局部 context 居中，canvas 外 padding 恒 0。ILT target 与 mask 不能混淆：target 是期望晶圆图形，初始 mask 是源版图透光率；两者即便初始数值相同也应在 data contract 中分开。
- 当前 raster 只能从 KLayout Region 生成 numpy canvas，没有 pixel mask→Region/GDS 的反向接口。ILT 首版若要求最终 GDS，必须新增明确的矢量化/坐标映射；若只要求 NPZ/PNG，则必须把不输出 GDS 写为限制，不能假装复用 MB-OPC merge。

## 上游与旧迁移差异

- OpenILT Simple 直接对完整 tile 参数用 sigmoid+SGD；中央 filter 可动、外围固定；best 在 step 前按连续训练 loss 选择，最后一次 step 未评价。`00_PAST` 保留核心参数化/损失，但把条件改成具名 `ProcessCondition`，支持任意独立工艺条件，并返回结构化记录。
- OpenILT LevelSet 使用 `phi<0` 硬 mask + `-|∇phi|` 自定义 backward + Adam。`00_PAST` 额外用精确 SDF 初始化并修正输出 zero-boundary 一致性，是可借鉴的工程修正，不应退回上游二值 ±1 初值。
- OpenILT CurvMulti 源码的 nominal L2 实际误用 `printedMax`，且曲率权重硬编码 `2e2`；`00_PAST` 改为具名 nominal 并显式 `curvature_weight`。计划必须明确这是有意修正而非逐行复刻。
- 进一步核对 OpenILT runner：`PixelInit` 虽返回 `2*target-1`，但 CurvMulti/Multilevel 最低级都显式调用 `solve(target, target)`，因此 `00_PAST` 的 `[0,1] target` 初值是忠实于实际入口，而不是误迁移。
- `00_PAST` Multilevel 和 CurvMulti 都确保 Hopkins 始终在完整物理像素网格运行；粗尺度只减少控制参数/监督自由度。直接把粗图补零进 256 canvas 会改变像素物理尺寸，是计划中必须禁止的错误迁移。
- 旧 `main/run_ilt.py` 是单张 target 的统一 method 字符串分派器，依赖已删除的 offline_inputs/artifacts/configured argparse，并只保存 NPZ/PNG/光刻数组，不输出 GDS、不支持当前 Macro–Core 管线。其方法分派与超长参数表不适合恢复。
- 旧 runner 对所有方法统一做二值 L2、PVBand 和 rectangular shot；当前 `evaluation/metrics.py` 已无 shot API，因此新计划不能声称 shot 仍可用，除非单独迁移（本任务不应顺带迁移）。

## 首个方法兼容性方向（待最终核对）

- 应建立一个**像素 ILT 专用**共享 problem/workflow，而不是复用边段 `MacroProblem` 或 `MBOPCMethod`。
- 可共享内容仅限已有四个候选都会实际消费的事实：规范化 `[B,H,W]` 输入、连续三项损失、通用 iteration record/result、逐 core/macro 生命周期、结果持久化；参数化和 optimizer 必须留在各方法模块。
- 当前最小可行大版图语义倾向于：每个 core 的 context 只读，core ownership 像素可优化；同一 batch 的 core 独立完成全部迭代；只回写 ownership 像素；macro 完成后写一个结果，全部 macro 完成后合并。该语义需要继续核对当前 `MacroSpec/CoreSpec` 和输出几何路径。

## 当前网格与可复用输出事实

- `opc/input/grid.py::plan_macros` 已保证 core/context 是 pixel 整数倍、`core+2*context<=256*pixel`，末端 core 允许缩短；可直接作为像素 ILT 的空间切分，不应新建第二套网格。
- `MacroSpec.query_box` 恰为所有 core context 的包围框。像素 problem 可在该框只栅格一次为 `uint8[Hq,Wq]`，再按 core 切片和居中 padding，避免每轮/每方法重复跨 Python/KLayout。
- 当前 `ownership_canvas` 使用像素中心落入半开区间判定；版图 bbox 末端不足一个 pixel 时可能排除最后部分像素。ILT 若要完整写回，规格必须显式决定是复用中心规则还是按像素格与 ownership 的正面积相交归属，不能含混。
- `main/_macro_pipeline.py::merge_macro_results` 与 `save_final_lithography` 只依赖 plan/GDS/grid 语义，可供 ILT 复用；`write_macro_gds` 当前参数类型绑定 `MacroProblem`，但行为只需 layer+Region，首计划应明确最小接口调整而非复制写出函数。
- `evaluation.evaluate_binary_l2/evaluate_pvband` 已支持 ownership mask，适合 ILT 最终二值评价；EPE 是边段探针指标，不应硬塞进像素 ILT 首版。
- OpenILT 仓库根许可证为 MIT；迁移算法代码时须保留许可证归属，当前仓库已有 `lithography/OPENILT_LICENSE.txt`，ILT 文档和源码头仍应注明算法来源。
- `main/_macro_pipeline.py::write_macro_gds` 当前仅有两个生产调用点：`main/_mbopc_workflow.py::_solve_macro` 与 `main/run_macro_pipeline.py::run_round`。若改为显式 `LayerSpec` 参数，迁移范围小且可完整回归。

## 推荐的首版执行模型

- 一个 macro problem 在磁盘只保存一张 query-box target transmission `uint8`，不保存 `[core,256,256]` 重复 canvas。
- 求解按 core batch 进行；每个 batch 内每个 core 独立完成全部迭代，context 固定为初始 target，只有 ownership 像素参与参数更新和 loss；batch 完成立即把最佳 ownership 像素写回 macro 聚合数组并释放 GPU 张量。
- 选择 tile 独立而非 macro 全场参数的理由：这与 OpenILT 固定 canvas 算法一致；后续 level-set/multiscale 都可在同一 256 canvas 工作；GPU/CPU 峰值与 reticle 总 core 数解耦。代价是相邻 core 不交换更新后的 context，可能出现 seam；必须列为已知限制，不能宣称全局最优或无缝。
- batch best 必须逐样本选择，不能像 OpenILT parallel 按 batch 总 loss 共享 best iteration；否则改变 `batch_size` 会改变输出。总 loss 仍可一次 backward，逐样本 best 用无梯度向量比较维护。

## 基线验证

- 命令：`D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests`
- 结果：`446 passed in 98.28s`，当前 `2fa75ea` 业务基线全绿。

## 最终设计结论

- 计划共四份，实施顺序：Simple → LevelSet → CurvMulti → Multilevel。
- Simple 是唯一会建立公共像素输入/workflow 的 change；LevelSet 必须零修改共享层；CurvMulti 只在出现真实调用方时为 `_common` 增加 resize/smooth；Multilevel 只把共享 loss 的 selection 从 bool 向后兼容扩为 `[0,1]` 权重。
- 第一份新增的结构仅四类且都有明确当前职责：GridRuntime（edge+pixel 两调用方）、PixelMacroProblem（持久输入）、ILTStateRecord/ILTBatchResult（算法与 workflow 边界）、ILTMethod（公共 workflow 注入）。没有 solver 基类、registry、PixelCoreWindow 或重复 config dataclass。
- Config 直接使用算法 dataclass；为避免 main/runtime 两份同构配置，Simple 计划明确让 `_parse_config` 通过 `get_type_hints` 支持外部 postponed annotations。CurvMulti 再按真实需求增加 variadic tuple。
- 四份规格均明确：tile-independent context 不交换是首版已知限制；输出 pixel-grid GDS；不迁 EPE/shot/MRC；不修改 layout/geometry/00_PAST。
- 文档静态审计：四份均含 0–26 全部章节、无模板占位符、无行尾空白、Blocking 均为 None，File-Level Change Plan 已使用精确路径。
