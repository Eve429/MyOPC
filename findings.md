# MyOPC 迁移研究发现

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
