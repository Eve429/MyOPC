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
| lithography | 318 | 待迁移 |
| evaluation | 153 | 待迁移 |
| opc/input | 1315 | 复制到新树，未适配 |
| opc/input/edge | 758 | 复制到新树，未适配 |
| opc/iteration | 1670 | 待迁移 |
| main | 3357 | 待迁移 |
| tests | 4177（旧） | 按批次对照移植 |
