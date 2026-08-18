# 单遍偏置扩张管线 run_single_pass 设计文档

> 状态：**已批准实施，2026-08-16 完成**（用户审查本文件后放行；实测数据见 §9）
> 本文件是 `main/run_single_pass.py` 及其配置、测试、文档的唯一决策依据。
> 若实际代码与本文记录不一致，必须先报告差异，不得自行调整本文接口或
> 扩大任务范围。

## 1. 本次目标

`main/run_macro_pipeline.py` 是双轮 ±2 nm 回零的**验证管线**：它的 plan.json、
problem/result NPZ、每 macro GDS、transmission sums、RSS 采样与回读 XOR 都
服务于"证明管线正确"，不服务于产出。本次新建一个贴近实际使用的单遍入口：

1. 读取一份精简 TOML 配置；
2. 阶段 0 完成与验证管线完全相同的网格契约校验与两级规划；
3. 逐 macro **全内存**执行一次 prepare → 单遍位移 → 重建 → 裁 ownership；
4. 全部 macro 完成后按 ownership 权威覆盖 + 全局 merge 写出**唯一产物**——
   最终目标层 GDS/OASIS；
5. 仅保留时间统计（准备/执行/写出/总计）；
6. 位移为配置项 `displacement_nm`，默认示例 5 nm，方向沿外法向。

不实现：真实求解器、光刻前向、多轮迭代、断点恢复、中间产物持久化。

## 2. 对本轮设计问题的明确结论

### 2.1 core_transmission_sums 为什么删除

该数组在验证管线中的唯一作用是**活性证据**（原设计文档 §12 阶段 2 步骤 8：
"保存每 core transmission sum，证明所有 core 均执行"）。它不参与任何决策，
没有评价消费者；真实求解器接入后会被真正的每 core 评价量（光刻输出/EPE）
取代。单遍入口无验证诉求，删除。

### 2.2 "一个环，外面向外 5nm，里面向内 5nm" 的实现方式

本项目法向约定恒为**从材料指向空区**（`fragmentation.py::_outward_normals`，
opaque 极性翻转后仍保持该不变量）。对带孔图形：

```text
外环（hull）法向：指离材料 → +d 使外边界向外扩张 d
孔壁（hole）法向：指向孔内（孔是空区）→ +d 使孔壁向孔内推进 d，孔缩小
```

因此**全部 owner 段统一 +5 nm** 即实现"外面向外 5nm、里面向内 5nm"，
环带总宽增加 10 nm，无需按环角色区分位移符号。数值例（直角图形，miter
角点精确）：

```text
输入 donut：外 (20,20)-(100,80)，孔 (40,40)-(80,60)，环带宽 20
+5 之后：  外 (15,15)-(105,85)，孔 (45,45)-(75,55)，环带宽 30
```

`displacement_nm` 允许负值：`-5` 使外环内收、孔壁外扩（环带变窄），语义
为沿法向反向；绝对值仍受 `max_displacement_nm` 约束。

### 2.3 `[lithography]` 段为什么保留

单遍入口不构造光刻画布（无 transmission sums、无模型前向），但
`plan_macros()` 的网格契约校验需要 `pixel_dbu` 与 `canvas_pixels`：

- `core_size_dbu % pixel_dbu == 0`、`context_dbu % pixel_dbu == 0`；
- `ceil((core + 2×context)/pixel) ≤ canvas(=256)`。

这两个契约保证网格与 ICCAD13 画布兼容，是几何规划的一部分，不是栅格化
的一部分。删掉它们会让同一套网格在单遍与验证管线之间出现两套合法性标准。
**结论：保留 `[lithography]` 段，仅供契约校验。**

### 2.4 回读验证为什么可以安全删除

验证管线的回读（XOR、面积不变）证明的是"双轮协议 + 持久化往返"没有引入
偏差。单遍入口的几何在内存中直通（prepare → reconstruct → 裁剪 → 写出，
无序列化往返），其正确性已由 `tests/opc/input` 的 55 个 problem 级用例与
`PatchWriter.write_macro_results` 的既有测试覆盖；删除回读不降低几何正确性
保障，只去掉重复取证。

## 3. 配置文件

新增 `config/single_pass.toml`。所有生产参数显式填写，不在 Python 中维护
另一套默认值；路径相对 TOML 文件目录解析；未知段/键一律拒绝。

```toml
[input]
layout = "../TestReticle/gcd_45nm.gds"
top_cell = "TOP"                  # 可省略；省略时使用唯一顶层规则
layer = 11
datatype = 0
polarity = "clear"                # clear 或 opaque

[grid]
macro_grid = [2, 2]               # 与 macro_size_nm 恰好填写一个
# macro_size_nm = 4096
core_size_nm = 1024
context_nm = 400                  # core 四边各自扩展的通用只读上下文

[lithography]
pixel_nm = 8                      # 仅供网格契约校验，本入口不栅格化
canvas_pixels = 256               # 冻结为 ICCAD13 画布

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24          # |displacement_nm| 不得超过此值
miter_limit = 4.0

[iteration]
displacement_nm = 5               # 单遍位移；正=沿外法向，负=反向

[output]
final_layout = "../output/single_pass/gcd_45nm_result.gds"
final_cell_mode = "single_cell"   # single_cell 或 macro_cells
```

与验证管线配置的差异：`[iteration]` 由 `round_deltas_nm`（两项）改为
`displacement_nm`（单项）；`[output]` 删除 `work_dir`（无中间产物目录）；
其余段与键完全一致。

## 4. 校验清单（阶段 0，全部失败即终止）

沿用 `exact_dbu`（自 `main.run_macro_pipeline` 导入，不复制第二份换算）：

```text
全部 nm 值（core/context/pixel/corner/segment/max_displacement/displacement）
    经 Decimal 精确换算为整数 DBU，失败报错含参数名、nm 值与 dbu_nm
|displacement_dbu| <= max_displacement_dbu
max_displacement_dbu <= context_dbu
macro_grid 与 macro_size_nm 恰好一个；size 模式要求
    macro_size_dbu > core_size_dbu 且为整数倍
canvas_pixels == 256；像素整除与画布容量由 plan_macros 校验
macro ownership 面积和 == 目标层 bbox 面积（O(macro 数) 复核，保留）
polarity、final_cell_mode 枚举合法
```

## 5. 执行流程与资源边界

```text
阶段 0：LayoutDB.open → layer_bbox → 换算与校验 → plan_macros
执行：对每个 macro（行优先）：
    db.query([layer], macro.query_box).materialize_intersecting()
    prepare_macro_problem(...)                 # 全内存，不调用 save
    displacements = np.where(owner>=0, displacement_dbu, 0)
    region = reconstruct_region(problem, displacements)
    clipped = region & ownership_box           # 权威覆盖选择
    patches.append(GeometryPatch(macro_id, layer, clipped, box))
    del batch, problem, region                 # 释放后进入下一个 macro
写出：PatchWriter.write_macro_results(patches, final_layout, dbu_um,
                                       cell_mode=final_cell_mode)
```

资源边界：任一时刻常驻的几何 = 当前 macro 的 problem + 已收集的权威覆盖
patch（裁剪后）+ 最终 merge（single_cell 模式的一次全局 Region）。与验证
管线相同的"逐 macro 释放"纪律。

产物边界：**唯一产物是 `final_layout`**。不创建 work_dir、plan.json、
problems/、round 目录、每 macro GDS、result NPZ、summary.json。

时间统计（perf_counter，打印到 stdout）：准备（阶段 0）、执行（全部 macro
合计）、写出（merge + 落盘）、总计；附 macro 数与段数总计。

## 6. 函数与结构定义

`main/run_single_pass.py`（每行中文短注释）：

```python
@dataclass(frozen=True, slots=True)
class SinglePassConfig:
    """保存一次单遍偏置扩张所需的全部显式配置。"""
    layout_path: Path
    top_cell: str | None
    layer: LayerSpec
    polarity: MaskPolarity
    macro_size_nm: Decimal | None
    macro_grid: tuple[int, int] | None
    core_size_nm: Decimal
    context_nm: Decimal
    pixel_nm: Decimal
    canvas_pixels: int
    corner_nm: Decimal
    segment_nm: Decimal
    max_displacement_nm: Decimal
    miter_limit: float
    displacement_nm: Decimal
    final_layout: Path
    final_cell_mode: str

def load_config(path: str | Path) -> SinglePassConfig:
    """严格读取单遍 TOML、解析相对路径并拒绝未知或互斥字段。"""
    ...

def run_single_pass(config: SinglePassConfig) -> Path:
    """单遍执行两级网格偏置扩张并写出最终目标层版图。"""
    ...

def main() -> int:
    """读取唯一位置参数 config，执行并打印中文摘要与耗时。"""
    ...
```

复用清单（零新算法代码）：`exact_dbu`、`LayoutDB.layer_bbox`、
`plan_macros`、`prepare_macro_problem`、`reconstruct_region`、
`PatchWriter.write_macro_results`。导入方式：仓库根 sys.path 引导后
`from main.run_macro_pipeline import exact_dbu`（脚本直跑与 pytest 两种
加载路径下均为同一模块实例）。

## 7. 与 run_macro_pipeline 的差异表

| 能力 | 验证管线 | 单遍入口 | 单遍去除理由 |
|---|---|---|---|
| plan.json / problem NPZ | 有 | 无 | 单遍无跨阶段复用与断点诉求 |
| 双轮 ±2 nm 回零 | 有 | 单遍 `displacement_nm` | 实际使用形态 |
| 每 macro GDS / result NPZ | 有 | 无 | 中间产物只服务双轮协议与合并输入 |
| core_transmission_sums | 有 | 无 | 活性证据，无消费者（§2.1） |
| RSS / summary.json | 有 | 无 | 验证性设施 |
| 回读 XOR / 面积校验 | 有 | 无 | §2.4 |
| 时间统计 | 有 | **保留** | 用户要求 |
| 网格/problem/重建/合并算法 | — | 共享 | 零重复实现 |

## 8. 测试矩阵（tests/main/test_single_pass.py，全生成式）

1. **donut 双向扩张**：输入外 (20,20)-(100,80)、孔 (40,40)-(80,60)，
   `displacement_nm = 5` → 最终覆盖与期望 Region（外 (15,15)-(105,85)、
   孔 (45,45)-(75,55)）XOR 面积为零，polygon 数 2（hull+hole）；
2. **负位移反向**：同输入 `displacement_nm = -5` → 外 (25,25)-(95,75)、
   孔 (35,35)-(85,65)，环带宽 20→10；
3. **产物唯一**：执行后输出目录树中仅存在 `final_layout` 一个文件；
4. **未处理层不复制**：源含 1/0 与 2/0，最终仅 1/0，覆盖正确；
5. **配置校验**：`displacement_nm` 无法精确落格点失败（报错含参数名）、
   `|displacement| > max_displacement` 失败、macro 入口同现/同缺失败；
6. **macro_cells 可用**：与 single_cell 顶层物理覆盖 XOR 为零。

## 9. 验收命令

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check main tests\main
D:\app\miniforge\envs\myopc\python.exe -m compileall -q main tests
D:\app\miniforge\envs\myopc\python.exe main\run_single_pass.py config\single_pass.toml
```

基线：全量 pytest 135 passed；gcd smoke 仅产出
`output/single_pass/gcd_45nm_result.gds` 一个文件。

**实测（2026-08-16）**：全量 pytest **143 passed**（新增 8 用例）；gcd_45nm
2×2 单遍 +5 nm：4 macro / 870 core / 343018 段，准备 0.00s、执行 0.76s、
写出 0.03s、总计 0.80s；产物仅 `output/single_pass/gcd_45nm_result.gds`。

**实施中确认的几何退化（记入 findings）**：图形边恰好与**内部 macro 切线**
重合时（如孔壁压在 x=80 切线上），该边整条归一侧 macro，另一侧 macro 以
context 原位参与拐角重建，两侧拼合处出现一位移宽度的台阶（XOR = 2×d²）。
ownership 切线分裂只保证段不**跨越**切线，不处理边**落在**切线上的退化；
测试几何须避开（bbox 外沿例外——邻侧副本被裁剪成零宽，无影响）。该限制
对验证管线同样成立，属已知边界情形，不是单遍入口的回归。

## 10. 实施顺序与提交

1. `main/run_single_pass.py` + `config/single_pass.toml`；
2. `tests/main/test_single_pass.py`；
3. 本文档状态行改为"已批准实施"，补 smoke 实测数据；
4. 单次本地提交：`feat(main): 单遍偏置扩张入口 run_single_pass`；不推送。

## 11. 已知限制

- 巨大单 polygon 的单 problem 内存上限与验证管线相同（§20.1 遗留）；
- `single_cell` 全局 merge 是内存峰值；
- 单遍位移不做任何光学评价，偏置量完全由配置决定；
- 斜边图形的位移后角点由 miter/bevel 规则重建，极尖角在
  `miter_limit` 约束下退化为 bevel，属重建契约而非本入口行为。
