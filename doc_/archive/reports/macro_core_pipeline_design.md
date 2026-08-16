# Macro–Core 两级任务与双轮迭代重构计划

> 状态：**已批准实施，2026-08-16 完成并通过审查修复**（实施记录见
> `macro_core_pipeline_development_report.md` 与 `..._test_report.md`；
> 审查问题清单 `macro_core_pipeline_review_issues.md` 已逐项处理）
> 本文件是后续实现的唯一决策依据。重开上下文后，只需读取本文件、仓库根目录
> `AGENTS.md` 和实际代码即可实施。若实际代码与本文记录不一致，必须先报告差异，
> 不得自行调整本文接口或扩大任务范围。

## 1. 本次目标

当前新树正在从只读归档 `00_PAST/` 逐模块重新梳理。现有 OPC 输入代码采用
“先建立全局 core 网格，再把 core 分组为 macro”的路线，并同时保留
`MBOPCProblem`、`MacroPreparation` 等重复结构。本次不兼容旧路线，直接重构为：

1. 对完整目标层 bbox 先切不重叠 macro；
2. 再在每个 macro 内切不重叠 core；
3. 当前主机逐 macro 生成并持久化一个 `MacroProblem`；
4. 不接真实 OPC 求解器，先按真实迭代状态机运行两轮：第一轮所有 owner 边段累计
   向外 `2 nm`，第二轮累计向内 `2 nm`，最终回到零位移；
5. 每个 macro 的全部 core 完成一轮后，保存一个状态文件和一个 macro GDS；
6. 若版图按 `2×2` macro 切分，则第一轮产生 4 个 macro GDS，第二轮再产生 4 个；
7. 第二轮结束后，合并全部 macro 的权威区域，执行全局同层 merge/normalize，写出
   一个完整目标层 GDS/OASIS；
8. 整个流程由一个 Python 文件直接运行，不需要安装本项目包。

本次只实施完成上述闭环所需的最小代码，不实现真实 MB-OPC、ILT、Worker、任务队列、
失败重试、断点恢复、方法注册器或未来分布式接口。`00_PAST/` 只读，不得修改。

## 2. 对本轮设计问题的明确结论

### 2.1 最终合并是否会截断跨 macro 图形

最终合并不能把“按 ownership 裁剪后的碎片”直接作为最终版图而不再处理，否则跨 macro
polygon 会在表示层留下 seam。新方案分三层：

1. **每轮 macro GDS**：保存当前 macro 重建出的完整候选 polygon，不先裁成最终碎片；
2. **权威覆盖选择**：最终合并时，每个 macro 的完整结果只贡献自身 ownership 内的覆盖，
   以消除相邻 macro context 的重复写入；
3. **最终全局 merge/normalize**：把所有 ownership 覆盖汇总到同层 `Region` 后执行一次
   `merged()`，重新连接因为 ownership 选择而被表示性切开的连续 polygon，并清理重复边、
   零面积碎片和不必要切割边。

所以，内部仍然必须按 ownership 选择权威覆盖，否则相邻 macro 的完整 context polygon
会正面积重复；但裁剪不是最终结果，默认的单 Cell 输出会在写盘前完成全局 merge。
零位移或第二轮回零后，最终目标层与输入目标层的 XOR 面积必须为零。

### 2.2 为什么不直接让一个 macro 独占整个跨界 polygon

“给完整 polygon 指定唯一 macro owner，再整 polygon 写一次”只适用于 polygon 永远作为
整体更新的算法。边段 OPC 中，同一个跨 macro polygon 的不同边段可能分别由多个 macro
内的 core 更新。如果只让 polygon owner 写出完整 polygon，它必须收集其他 macro 的边段
位移；这要求跨 macro 稳定 segment ID、全局位移归并和完整 polygon 拓扑路由。本阶段没有
这些已验证接口，提前加入会形成一次性抽象。

因此本阶段采用：macro 保存完整候选 polygon，边段仍按空间 owner 更新，最终按 ownership
选覆盖并全局 merge。以后若实现跨 macro 全局 segment generation，可以再评审 polygon
owner 输出，不在本次假装已经解决。

### 2.3 “不实现全局 polygon merge”是否就是同一个问题

是。旧计划中“允许最终保留 macro seam”就是上述表示碎片问题。本版计划已改为：

- 默认 `single_cell` 最终输出必须全局 merge/normalize，不保留 seam；
- 同时提供 `macro_cells` 输出模式，供调试或内存受限时保留一个 macro 一个 Cell；该模式
  物理覆盖仍正确，但跨 Cell 无法合并成一个 polygon，文档和摘要必须明确这一差异。

## 3. 阶段边界总览

| 阶段 | 要解决的问题 | 输入 | 输出 | 明确禁止重复的工作 |
|---|---|---|---|---|
| 阶段 0：配置与网格 | 确定 DBU、macro/core/context/canvas 和任务顺序 | TOML、版图元数据 | 内存网格计划；阶段 1 全部成功后才写入 `plan.json` | 不物化图形、不提边 |
| 阶段 1：Problem 准备 | 把每个 macro 的参考几何变成可重复迭代的持久数组 | 打开的 `LayoutDB`、macro 计划 | 每 macro 一个 problem NPZ | 每个 macro 只物化、合并、提边、分段、算 owner/membership 一次 |
| 阶段 2：双轮迭代 | 验证逐 core 更新、macro 汇总、轮间状态传递和 canvas 构造 | problem NPZ、上一轮状态 | 每轮每 macro 一个 result NPZ 和一个完整候选 GDS | 不打开原始 GDS，不重新物化、提边、分段、算 owner/membership |
| 阶段 3：最终合并 | 选取各 macro 权威覆盖并消除 seam | 第二轮 macro GDS、plan ownership | 一个完整目标层 GDS/OASIS | 不重新运行迭代，不重新提边 |
| 阶段 4：验证与报告 | 证明回零、覆盖、文件数量、性能和架构清理正确 | 全部产物 | 测试/开发报告 | 不修改算法结果来“修测试” |

任何阶段只允许消费上一阶段公开产物。阶段二发现 problem 缺少必要数据时必须返回阶段一
修订计划，不得在迭代热路径临时重新扫描原始版图。

## 4. 配置文件

新增 `config/macro_pipeline.toml`。不再使用 `macro_mode`。`macro_size_nm` 与
`macro_grid` 写哪个就使用哪种方式；两者同时出现或同时缺失直接报错。

`gcd_45nm.gds` 的实际元数据已经只读确认：

```text
dbu_um = 0.0001（0.1 nm/DBU）
top = TOP
bbox = (11400, 13150, 317300, 308850) DBU
layer = 11/0
```

默认 smoke 配置使用 `2×2` macro：

```toml
[input]
layout = "../TestReticle/gcd_45nm.gds"
top_cell = "TOP"                  # 可省略；省略时使用 LayoutDB 的唯一顶层规则
layer = 11
datatype = 0
polarity = "clear"               # clear 或 opaque

[grid]
macro_grid = [2, 2]               # 与 macro_size_nm 恰好填写一个
# macro_size_nm = 4096
core_size_nm = 1024
context_nm = 400                  # core 四边各自扩展的通用只读上下文

[lithography]
pixel_nm = 8
canvas_pixels = 256               # 当前 ICCAD13 模型固定 Canvas

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24
miter_limit = 4.0

[iteration]
round_deltas_nm = [2, -2]         # 恰好两轮；第二轮结束累计位移回到零

[output]
work_dir = "../output/macro_pipeline"
final_layout = "../output/macro_pipeline/gcd_45nm_result.gds"
final_cell_mode = "single_cell"  # single_cell 或 macro_cells
```

所有生产参数必须显式填写，不在 Python 中维护另一套默认值。路径相对于 TOML 文件目录解析。

## 5. 单位、整除与画布校验

### 5.1 nm 到 DBU

打开版图取得 `dbu_um` 后：

```text
dbu_nm = dbu_um × 1000
value_dbu = value_nm / dbu_nm
```

TOML 数值通过 `Decimal(str(value))` 进入换算。不得用 `round()` 吸收误差；不能精确得到
整数 DBU 时直接报错，并写明参数名、nm 值和当前 `dbu_nm`。

### 5.2 共有校验

```text
core_size_dbu > 0
context_dbu >= 0
pixel_dbu > 0
canvas_pixels == 256
core_size_dbu % pixel_dbu == 0
context_dbu % pixel_dbu == 0
ceil((core_size_dbu + 2 × context_dbu) / pixel_dbu) <= canvas_pixels
max_displacement_dbu <= context_dbu
round_deltas_dbu == [+2 nm 对应 DBU, -2 nm 对应 DBU]
sum(round_deltas_dbu) == 0
```

边缘缩短 core 小于名义 core，因此只要名义 core 通过画布校验，缩短 core 也不会超过
canvas。若版图外边界使缩短 core 宽度不是 pixel 的整数倍，局部栅格使用向上取整并保留
边缘像素的面积覆盖率，不移动几何边界来迎合 pixel。

### 5.3 `macro_size_nm` 模式

```text
macro_size_dbu > core_size_dbu
macro_size_dbu % core_size_dbu == 0
```

要求的是配置中的**名义 macro 大小**为 core 整数倍，版图轴长不需要整除 macro。
例如版图宽 21 µm、macro 11 µm、core 1 µm，切分结果为 11 µm 和 10 µm；若版图宽
21.5 µm，则为 11 µm 和 10.5 µm，只有最外侧 macro 内允许出现 0.5 µm 缩短 core。

### 5.4 `macro_grid` 模式

`macro_grid=[columns, rows]` 两项均为正整数，且不得超过对应轴的 core 单元数，保证每个
macro 至少包含一个 core。按 core 单元数尽量均分，较前的 macro 多分一个 core：

```text
unit_count = ceil(axis_length / core_size_dbu)
base, remainder = divmod(unit_count, macro_count)
前 remainder 个 macro 获得 base+1 个 core 单元
其余 macro 获得 base 个 core 单元
最后一条 macro 切线强制等于版图轴终点
```

所以 21 µm、core 1 µm、横向 2 块得到 11 µm + 10 µm，不做两个 10.5 µm 的几何硬等分。

## 6. ICCAD13 kernel 契约与 context/canvas 语义

### 6.1 已从 `00_PAST` 核对的事实

本计划只读核对了 `00_PAST/lithography`，不修改也不让新生产代码依赖归档路径：

```text
KernelNum = 24
Canvas = 256
Resolution = 256
focus.pt / defocus.pt = complex64[35, 35, 24]
focus_scale.pt / defocus_scale.pt = float32[24]
FFT = fft2/ifft2，norm="forward"
```

旧模型 `_prepare_mask()` 接受 `[H,W]` 或 `[B,H,W]`，要求 `H,W <= 256`，把不足 256 的
mask 居中补零到 256 后执行 FFT；输出再裁回输入局部尺寸。35×35 是频域 Hopkins kernel
尺寸，不能被解释成 17 pixel 的有限空间光学半径，因此不能从 kernel shape 自动推导
`context_nm`。context 是否足够仍需以后做收敛测试。

### 6.2 `context_dbu` 取代 `optical_range_dbu`

使用 `context_dbu` 更准确。它表示 core 四边的通用只读上下文，可以同时容纳：

- 光学影响范围；
- 当前方法允许的最大几何位移；
- EPE/probe 或其他评价所需范围；
- 以后方法明确证明需要的额外邻域。

本阶段只强制 `context_dbu >= max_displacement_dbu` 和画布可容纳，不谎称已经从 kernel
推导出充分的光学 context。

### 6.3 Core、context 和 canvas 的关系

不再使用含义模糊的 `input_box`。每个 core 只有：

- `ownership_box`：该 core 唯一可以更新和计分的非重叠区域；
- `context_box`：`ownership_box` 四边各扩 `context_dbu`，可以跨 core、macro 和版图 bbox；
- `canvas`：固定 256×256 的光刻张量，不是版图坐标框。

处理顺序：

```text
完整候选 Region
    -> 只栅格化 core.context_box，得到实际 local_mask[H,W]
    -> 把 local_mask 居中放入 canvas[256,256]
    -> low/high padding 按差值平均分配，奇数余量放到高坐标侧
    -> canvas 其余位置全部补 0
    -> 送入 ICCAD13；因为已经是 256×256，模型自身不再增加 padding
```

数组值在所有极性下都遵守同一个光学定义：

```text
1.0 = 透光
0.0 = 不透光
```

- clear：源 polygon coverage 是 1，其他 local context 是 0；
- opaque：local context 内先填 1，再减去源 polygon coverage；
- 两种极性在 local context 之外的 canvas padding 都是 0；
- 极性只影响“源 polygon 如何转换为透光率”，不改变 0/1 的物理含义。

以默认参数为例：

```text
core = 1024 nm
context = 400 nm/side
pixel = 8 nm
local = (1024 + 2×400) / 8 = 228 pixels
canvas = 256 pixels
每侧补 14 pixels 的 0
```

## 7. 网格结构与术语

### 7.1 `CoreSpec`

`CoreSpec` 按需构造，不为整版常驻对象列表：

```python
@dataclass(frozen=True, slots=True)
class CoreSpec:
    """描述一个 core 的唯一写入范围和只读上下文范围。"""

    core_id: str             # macro 内稳定行优先 ID，例如 c_r1c2
    ownership_box: DbuBox    # 唯一可更新、可计分、最终可回写的非重叠区域
    context_box: DbuBox      # ownership 四边扩 context_dbu 后的只读计算范围
```

### 7.2 `MacroSpec`

```python
@dataclass(frozen=True, slots=True)
class MacroSpec:
    """保存一个 macro 的唯一写入框和局部 core 切线，不展开 core 对象列表。"""

    macro_id: str                    # 全版稳定行优先 ID，例如 mr0c1
    ownership_box: DbuBox            # 当前 macro 对最终版图负责的非重叠范围
    x_cuts: NDArray[np.int64]        # macro 内 core 的全局 x 切线，严格递增
    y_cuts: NDArray[np.int64]        # macro 内 core 的全局 y 切线，严格递增
    context_dbu: int                 # 每个 core 四边扩展的只读上下文 DBU
    pixel_dbu: int                   # 一个光刻像素对应的整数 DBU
    canvas_pixels: int               # ICCAD13 固定方形 canvas，当前必须为 256

    @property
    def column_count(self) -> int:
        """返回当前 macro 横向 core 数。"""
        ...

    @property
    def row_count(self) -> int:
        """返回当前 macro 纵向 core 数。"""
        ...

    @property
    def core_count(self) -> int:
        """返回当前 macro 的 core 总数。"""
        ...

    @property
    def query_box(self) -> DbuBox:
        """返回所有 core context_box 的最小包围框，供完整相交物化使用。"""
        ...

    def core(self, core_index: int) -> CoreSpec:
        """按局部行优先索引即时构造 CoreSpec，不缓存 CoreSpec 列表。"""
        ...

    def locate_owned_points(self, points: object) -> NDArray[np.int32]:
        """返回点的局部 core owner；macro ownership 外返回 -1。"""
        ...
```

内部共享边按半开区间稳定归右/上；macro 整体最大边界归最后一列/行。

### 7.3 网格规划函数

`opc/input/grid.py` 公开：

```python
def plan_macros(
    bounds: DbuBox,
    *,
    core_size_dbu: int,
    context_dbu: int,
    pixel_dbu: int,
    canvas_pixels: int,
    macro_size_dbu: int | None = None,
    macro_grid: tuple[int, int] | None = None,
) -> tuple[MacroSpec, ...]:
    """按 size 或 count 二选一先规划 macro，再在每个 macro 内规划 core。"""
    ...
```

只保留三个同时被 x/y 轴使用的私有函数：

```python
def _cuts_by_size(start: int, end: int, size: int) -> NDArray[np.int64]:
    """从轴起点按固定尺寸切分，最后一个区间允许缩短。"""
    ...

def _macro_cuts_by_count(
    start: int, end: int, core_size: int, count: int
) -> NDArray[np.int64]:
    """按 core 单元数平衡分配指定数量的 macro。"""
    ...

def _core_cuts(start: int, end: int, core_size: int) -> NDArray[np.int64]:
    """在一个已确定 macro 内切 core，末端 core 允许缩短。"""
    ...
```

## 8. `MacroProblem` 与三个索引数组

### 8.1 结构定义

`opc/input/edge/problem.py` 同时保存结构、准备和 ownership 构造，不再单独保留
`ownership.py`：

```python
@dataclass(frozen=True, slots=True)
class MacroProblem:
    """一个 macro 可独立保存、加载和重复迭代的全部参考输入。"""

    macro: MacroSpec
    # 当前任务的 macro/core 网格、context、pixel 和 canvas 契约。

    layer: LayerSpec
    # 当前 problem 处理和最终输出的唯一 GDS layer/datatype。

    polarity: MaskPolarity
    # 源 polygon 的 mask 极性；栅格输出仍统一使用 1=透光、0=不透光。

    fragmentation: FragmentationConfig
    # 参考边段长度、最大允许位移和 miter 限制；阶段二不得重新计算。

    segments: SegmentBatch
    # 完整候选 polygon 的轮廓拓扑、数学边和控制边段，是参考几何唯一数组真源。

    owner_indices: NDArray[np.int32]
    # 长度 S。owner_indices[s] 是 segment s 唯一可写的 macro 局部 core 编号；
    # -1 表示该 segment 只因 context 被当前 macro 看见，当前 macro 不得修改它。

    core_offsets: NDArray[np.int64]
    # 长度 C+1 的 CSR 偏移。core c 的可见 segment 位于
    # member_segment_indices[core_offsets[c]:core_offsets[c+1]]。
    # 使用 int64 是因为 membership 总量 M 可能超过 int32 累计范围。

    member_segment_indices: NDArray[np.int32]
    # 长度 M。按 core 连续存储 context 内所有 segment 的局部 segment 编号；
    # 同一 segment 可以因 halo/context 同时出现在多个 core 的 membership 中，
    # 但 owner_indices 仍只允许一个 core 写入。segment 局部编号使用 int32 节省内存。

    def segments_for_core(self, core_index: int) -> NDArray[np.int32]:
        """返回一个 core 可读取的 owned + context segment 索引视图。"""
        ...

    def owner_segments_for_core(self, core_index: int) -> NDArray[np.int32]:
        """从该 core 的 membership 中筛出唯一允许当前 core 更新的 segment。"""
        ...

    def save(self, path: str | Path) -> Path:
        """把 problem 以不压缩 NPZ 原子保存，不写重复几何数组。"""
        ...

    @classmethod
    def load(cls, path: str | Path) -> MacroProblem:
        """使用 allow_pickle=False 读取并通过现有结构不变量校验 problem。"""
        ...
```

### 8.2 CSR 直观示例

假设 macro 内有两个 core、四个 segment：

```text
owner_indices = [0, 0, 1, -1]
```

含义：segment 0/1 由 core 0 写，segment 2 由 core 1 写，segment 3 是当前 macro 的只读
context。若两个 core 分别能看到 `[0,1,2,3]` 和 `[1,2,3]`：

```text
member_segment_indices = [0,1,2,3, 1,2,3]
core_offsets = [0,4,7]

core 0 -> members[0:4] = [0,1,2,3]
core 1 -> members[4:7] = [1,2,3]
```

`owner_indices`回答“谁能写”；CSR 回答“每个 core 为计算 context 需要读谁”。二者不是
重复字段。owned segment 列表可由二者按需过滤，不额外常驻第四份数组。

### 8.3 Problem 准备函数

```python
def prepare_macro_problem(
    batch: RegionBatch,
    layer: LayerSpec,
    polarity: MaskPolarity | str,
    fragmentation: FragmentationConfig,
    macro: MacroSpec,
) -> MacroProblem:
    """从完整相交图形一次生成可供多轮迭代复用的 macro 参考问题。"""
    ...
```

`problem.py` 内只保留两个有明确职责的私有函数：

```python
def _split_segments_at_ownership_cuts(
    segments: SegmentBatch,
    x_cuts: NDArray[np.int64],
    y_cuts: NDArray[np.int64],
) -> SegmentBatch:
    """在 macro/core ownership 切线交点处分裂控制段，保证一段不跨两个 owner。"""
    ...

def _build_macro_ownership(
    segments: SegmentBatch,
    macro: MacroSpec,
) -> tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int32]]:
    """生成每段唯一 owner 和每个 core 的 context membership CSR。"""
    ...
```

不再建立单独 `ownership.py`。ownership 是 `MacroProblem` 构造的一部分，当前只有一个调用方。

### 8.4 为什么必须在 ownership 切线处分段

仅用 segment 中点选 owner 不够：若一个长 segment 横跨 macro/core 边界，整个 segment 会
被某一侧 owner 更新，而另一侧 macro 的副本仍保持旧位置，最终会在边界产生不一致。

阶段一先按现有 `fragment_edges()` 做物理边段切分，再把每段参数区间与当前 macro 的
`x_cuts/y_cuts` 求交，在共享切线的参数位置继续分裂。分裂点仍属于同一条真实数学边，
不是新物理边；只增加控制段边界。斜边交点以原始整数端点和全局整数切线计算参数 `t`，
共享 macro 边界两侧使用同一公式，禁止分别裁成整数短边后重新均分，从而避免 33/34 DBU
分歧。分裂后每个可写 segment 的内部不得跨越任何 ownership 切线。

### 8.5 准备顺序

```text
materialize_intersecting(macro.query_box)
    -> normalize_mask() 合并物理覆盖
    -> extract_contour() 提取完整真实轮廓
    -> fragment_edges() 按边段长度切分
    -> _split_segments_at_ownership_cuts() 按 macro/core 切线再分裂
    -> _build_macro_ownership() 计算 owner 和 membership
    -> MacroProblem.save()
```

`batch.query_box` 必须等于 `macro.query_box`。context 框、macro 框和 core 框不得进入
`ContourBatch`，只作为候选筛选、owner 和 membership 的空间条件。

## 9. 栅格化接口

保留一套 raster，不在 geometry/opc 两处复制。`opc/input/raster.py` 负责把局部 context
放到固定模型 canvas 的中心：

```python
def rasterize_region_window(
    region: kdb.Region,
    box: DbuBox,
    pixel_dbu: int,
) -> NDArray[np.float32]:
    """把物理 box 栅格为最小 H×W 覆盖率数组，不添加模型 canvas padding。"""
    ...

def rasterize_mask_canvas(
    region: kdb.Region,
    context_box: DbuBox,
    pixel_dbu: int,
    canvas_pixels: int,
    *,
    polarity: MaskPolarity | str,
) -> NDArray[np.float32]:
    """把 context 透光率居中放入固定 canvas，所有外围 padding 填 0。"""
    ...

def ownership_canvas(
    ownership_box: DbuBox,
    context_box: DbuBox,
    pixel_dbu: int,
    canvas_pixels: int,
) -> NDArray[np.bool_]:
    """返回与居中 mask canvas 对齐的唯一计分像素，context/padding 为 False。"""
    ...
```

私有函数：

```python
def _center_padding(
    local_height: int,
    local_width: int,
    canvas_pixels: int,
) -> tuple[int, int, int, int]:
    """返回低/高 y 和低/高 x 的居中零填充宽度。"""
    ...
```

全局 DBU 坐标映射到 canvas 像素时必须把低坐标侧 padding 加入索引：

```text
x_canvas = (x_dbu - context.left) / pixel_dbu - 0.5 + low_x_padding
y_canvas = (y_dbu - context.bottom) / pixel_dbu - 0.5 + low_y_padding
```

后续 EPE/probe 必须复用这一映射，不能继续假设 context 位于 canvas 左下角。

## 10. Problem 与轮次文件格式

### 10.1 Problem NPZ

使用不压缩 `np.savez`，同目录临时文件成功关闭后 `os.replace()`。读取固定
`allow_pickle=False`。格式版本为 1：

```text
format_version                  int32[1]
macro_id                        unicode[1]
macro_ownership_box             int64[4]
macro_x_cuts                    int64[X+1]
macro_y_cuts                    int64[Y+1]
context_dbu                     int64[1]
pixel_dbu                       int64[1]
canvas_pixels                   int64[1]
layer                           int32[1]
datatype                        int32[1]
polarity                        unicode[1]
corner_length_dbu               float64[1]
max_segment_length_dbu          float64[1]
max_displacement_dbu            float64[1]
miter_limit                     float64[1]
contour_vertices                int64[V,2]
contour_ring_offsets            int64[R+1]
contour_polygon_ring_offsets    int64[P+1]
edge_next_ids                   int32[E]
edge_polygon_ids                int32[E]
edge_normals                    float64[E,2]
ring_segment_offsets            int64[R+1]
segment_edge_ids                int32[S]
segment_t0                      float64[S]
segment_t1                      float64[S]
owner_indices                   int32[S]
core_offsets                    int64[C+1]
member_segment_indices          int32[M]
```

不保存原生 `Region`、重复 starts/ends、owned segment 副本、全局 segment ID 或每 core
Python 对象。完整候选 polygon 拓扑必须保存，否则阶段二无法在不重开原 GDS 的情况下重建。

### 10.2 每轮 MacroResult NPZ

Result 不定义 dataclass，避免只服务于主流程的一次性结构：

```text
format_version                  int32[1]
macro_id                        unicode[1]
round_index                     int32[1]
round_delta_dbu                 float64[1]
segment_displacements           float64[S]
written_owner_count             int64[1]
core_transmission_sums          float64[C]
```

第一轮从全零位移开始，owner segment 加 `+2 nm`；第二轮读取第一轮位移，owner segment
加 `-2 nm`。每轮维护 `written[S]`：

- 每个 `owner>=0` 的 segment 必须且只能写一次；
- `owner==-1` 的 context segment 不得写；
- 第二轮结束后全部 owner segment 位移必须精确回到 0 DBU；
- 位移范围不得超过 `FragmentationConfig.max_displacement_dbu`。

## 11. 主流程和函数定义

新增 `main/run_macro_pipeline.py`，按现有 main 演示脚本把仓库根加入 `sys.path`：

```powershell
D:\app\miniforge\envs\myopc\python.exe main\run_macro_pipeline.py config\macro_pipeline.toml
```

### 11.1 配置结构

只定义一个配置结构，不再拆 input/grid/lithography/edge/output 四套 dataclass：

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """保存一次两级网格、双轮迭代和最终合并所需的全部显式配置。"""

    layout_path: Path                     # 输入 GDS/OASIS/GLP 的绝对路径
    top_cell: str | None                  # 显式顶层；None 表示要求版图只有一个顶层
    layer: LayerSpec                      # 本次处理的唯一目标 layer/datatype
    polarity: MaskPolarity                # 源 polygon 的 clear/opaque 极性
    macro_size_nm: Decimal | None         # 按 nm 切 macro；与 macro_grid 恰好一个非空
    macro_grid: tuple[int, int] | None    # 按 [列,行] 数量切 macro
    core_size_nm: Decimal                 # 名义 core 边长
    context_nm: Decimal                   # core 每侧通用上下文宽度
    pixel_nm: Decimal                     # 光刻采样像素尺寸
    canvas_pixels: int                    # 当前 ICCAD13 固定为 256
    corner_nm: Decimal                    # 拐角控制段长度
    segment_nm: Decimal                   # 普通控制段最大长度
    max_displacement_nm: Decimal          # 允许的绝对位移上限
    miter_limit: float                    # 拐角重建 miter 上限
    round_deltas_nm: tuple[Decimal, Decimal]  # 固定两轮 [+2,-2]
    work_dir: Path                        # problem/result/macro GDS/summary 根目录
    final_layout: Path                    # 最终完整目标层版图路径
    final_cell_mode: str                  # single_cell 或 macro_cells
```

### 11.2 公开和阶段函数

```python
def load_config(path: str | Path) -> PipelineConfig:
    """严格读取 TOML、解析相对路径并拒绝未知或互斥字段。"""
    ...

def exact_dbu(value_nm: Decimal, dbu_nm: Decimal, name: str) -> int:
    """把必须落在版图格点上的 nm 参数精确转换为整数 DBU。"""
    ...

def prepare_problems(config: PipelineConfig) -> dict[str, object]:
    """执行阶段 0/1，逐 macro 生成 problem，并写出 plan.json。"""
    ...

def run_round(
    plan: dict[str, object],
    round_index: int,
    delta_dbu: int,
) -> dict[str, object]:
    """执行一个全局轮次：逐 macro、逐 core 更新并保存状态与完整候选 GDS。"""
    ...

def merge_final(
    plan: dict[str, object],
    round_index: int,
    output_path: Path,
) -> Path:
    """合并指定轮次全部 macro 权威覆盖，并按 cell mode 写出最终版图。"""
    ...

def run(config_path: str | Path) -> dict[str, object]:
    """按准备、两轮迭代、最终合并、验证顺序执行完整流程并返回摘要。"""
    ...

def main() -> int:
    """读取唯一位置参数 config，运行流程并打印中文摘要。"""
    ...
```

辅助逻辑只在确有两个调用方或独立阶段含义时抽函数，不创建 runner 类、stage 类、Worker、
Protocol 或注册器。

## 12. 各阶段详细执行流程

### 阶段 0：配置、模型契约与网格计划

#### 本阶段要做什么

在任何大几何物化前，确定一次运行的全部数值契约、任务数量、macro/core ownership、
context 和固定 canvas 可容纳性。

#### 输入

- TOML；
- `LayoutDB` 元数据：DBU、顶层、目标层、bbox；
- 本计划冻结的 ICCAD13 Canvas=256 契约。

#### 输出

- 内存中的 `tuple[MacroSpec,...]`；
- `plan.json` 所需的小型网格元数据。

#### 执行流程

1. 读取 TOML，拒绝未知字段；
2. 打开版图，解析 top/layer/bbox/DBU；
3. 精确换算全部 nm 参数；
4. 校验 `macro_size_nm`/`macro_grid` 恰好一个；
5. 校验 core/context/pixel/canvas/kernel 契约；
6. `plan_macros()` 先生成 macro，再为每个 macro 生成局部 core；
7. 验证 macro/core ownership 面积和、无正面积重叠、稳定顺序；
8. 创建空的工作目录结构，但还不生成 result/GDS。

#### 阶段边界

本阶段不调用 `materialize*()`、`extract_contour()`、`fragment_edges()` 或 raster，不读取
kernel tensor，不进入 GPU。

### 阶段 1：逐 macro 准备并持久化 Problem

#### 本阶段要做什么

把原始层级版图中每个 macro 后续所有轮次重复使用的昂贵参考信息计算一次并落盘。

#### 输入

- 同一个已打开的 `LayoutDB`；
- 阶段 0 的 MacroSpec；
- Layer、极性、FragmentationConfig。

#### 输出

- `problems/{macro_id}.npz`；
- 完整 `plan.json`；
- 每 macro 的 segment/core/membership 数量、problem 字节和准备耗时。

#### 执行流程

对每个 macro 按行优先顺序：

1. `db.query([layer], macro.query_box).materialize_intersecting()`；
2. `normalize_mask()` 删除属性并合并物理覆盖；
3. `extract_contour()` 提取完整 polygon/ring/hole；
4. `fragment_edges()` 做一次参考分段；
5. `_split_segments_at_ownership_cuts()` 确保一段不跨 owner 边界；
6. `_build_macro_ownership()` 生成 owner 和 context CSR；
7. 原子保存 `MacroProblem`；
8. 记录时间/RSS/数组字节后释放当前 Region、Contour、Segment 和 membership；
9. 继续下一个 macro。

所有 problem 成功后关闭 LayoutDB，再原子写 `plan.json`。任一 macro 失败时错误直接传播，
不写表示“准备完成”的 plan。

#### 阶段边界

以下昂贵操作在整个流程中每 macro 只允许出现一次：

```text
层级 ROI 查询
完整相交图形物化
物理 Region merge
轮廓提取
边段切分
ownership 切线再分段
owner 计算
membership CSR 构造
```

阶段二不得导入 `LayoutDB` 来补数据。

### 阶段 2：两轮逐 core 迭代与逐 macro 汇总

#### 本阶段要做什么

用确定性的 `+2 nm/-2 nm` 更新验证真实迭代骨架：上一轮状态读取、core 唯一写入、macro
轮末汇总、每轮持久化、完整候选 GDS 输出和回零。

#### 输入

- 阶段 1 problem NPZ；
- 第一轮初始全零位移或第二轮上一轮 result NPZ；
- 当前轮 delta。

#### 每轮每 macro 输出

```text
round_001/results/{macro_id}.npz
round_001/gds/{macro_id}.gds
round_002/results/{macro_id}.npz
round_002/gds/{macro_id}.gds
```

`2×2` macro 时，每轮恰好 4 个 GDS，两轮共 8 个。macro GDS 使用一个 `RESULT` Cell，
保存当前 problem 重建出的完整候选 polygon，保留 context，不裁成最终 ownership 片段。

#### 单轮执行流程

对每个 macro：

1. 加载 problem；
2. 第一轮创建全零 current；第二轮加载同 macro 的第一轮 displacement；
3. `next = current.copy()`，创建 `written[S]=False`；
4. 按局部 core 行优先遍历：
   - 读取 `segments_for_core()` 作为 context；
   - 读取 `owner_segments_for_core()` 作为本 core 唯一可写段；
   - 对 owner segment 的位移累计当前 delta；
   - 标记 written，重复写立即失败；
5. 全部 core 完成后验证 owner 全写且 context 未写；
6. 用 `next` 重建一次当前完整候选 Region；
7. 对每个 core 的 `context_box` 构造居中 256×256 透光率 canvas，只保留当前一张；
8. 保存每 core transmission sum，证明所有 core 均执行；
9. 保存一个 MacroResult NPZ；
10. 保存一个含完整候选 polygon 的 macro GDS；
11. 释放当前 problem/state/Region，再处理下一个 macro。

第一轮所有 macro 完成后才允许进入第二轮。第二轮全部 macro 完成后，验证所有 owner
displacement 为零。迭代顺序改变不得影响结果。

#### 阶段边界与重复计算审计

阶段二**不会重复阶段一**的物化、merge、提边、分段、owner 和 CSR。阶段二必须做的工作：

- 从持久轮廓按当前位移重建 Region：每轮位移不同，不能缓存阶段一结果替代；
- 栅格当前 Region：每轮 mask 不同，属于迭代工作；
- canvas 居中补零：每个 core 当前输入必需。

固定不变的目标 raster 若以后真实求解器需要，可在明确测量磁盘/计算权衡后缓存；本次盲
`+2/-2` 迭代没有目标评价调用方，不提前保存巨量目标 canvas。

### 阶段 3：最终权威覆盖与全局 merge

#### 本阶段要做什么

从第二轮每个 macro 的完整候选 GDS 中选择不重叠的权威覆盖，消除 macro seam，写出一个
完整目标层版图。

#### 输入

- `round_002/gds/*.gds`；
- plan 中相同顺序的 macro ownership；
- `final_cell_mode`。

#### 输出

- `final_layout`；
- 合并前后 polygon 数、面积和耗时；
- 第二轮回零 XOR 验证结果。

#### 执行流程：`single_cell`（默认）

1. 逐 macro 读取 `RESULT` Region；
2. 与该 macro ownership 精确求交，只保留唯一权威覆盖；
3. 累加到目标层全局 Region；
4. 全部 macro 加入后设置一致的 coherence 并执行一次 `merged()`；
5. 拒绝无效 polygon、零面积异常或 coverage 面积改变；
6. 插入一个最终 Cell 并原子写出；
7. smoke/test 模式重新加载原始目标层，验证 XOR 面积为零。

这是本阶段明确接受的一次全局内存操作，用来换取最终无 seam 的单 Cell polygon。报告必须
记录 merge 前后 RSS；若真实整版超过 64 GiB，不能静默切换模式，必须失败并由用户决定
是否使用 `macro_cells` 或另行开发 boundary-stream merge。

#### 执行流程：`macro_cells`

1. 逐 macro 读取完整结果并裁到 ownership；
2. 每个 macro 建立一个独立子 Cell；
3. 顶层以单位变换引用全部 macro Cell；
4. 不做跨 Cell polygon merge；
5. 物理覆盖与 `single_cell` 相同，但表示层 seam 仍存在。

该模式用于调试和降低全局 merge 临时内存，不得在报告中描述为已经 normalize。

#### 阶段边界

最终合并只消费第二轮 GDS 和 plan；不读取 problem 来重新迭代，不重新提边。默认只输出
处理后的目标层，不复制其他 layer。

### 阶段 4：验证、报告和简化审计

#### 本阶段要做什么

证明两级网格、阶段缓存、双轮迭代、GDS 数量、回零、最终 merge 和架构清理全部符合计划。

#### 执行流程

1. 运行全部专项和全量测试；
2. 使用 `gcd_45nm.gds` 运行 `2×2` smoke；
3. 检查每轮 4 个 macro GDS、第二轮后 1 个 final GDS；
4. 检查第二轮所有 owner 位移为零，final XOR 为零；
5. 比较 single_cell 与 macro_cells 的覆盖 XOR 为零；
6. 记录阶段耗时、problem 字节、轮次 RSS 和全局 merge 峰值；
7. 搜索旧类型、未调用函数、重复 raster、兼容包装和异常吞噬；
8. 更新手册、专项报告和规划三文件；
9. 只做本地 commit，不推送。

## 13. 输出写入接口

`geometry/patch.py` 保留小规模 `PatchSet`，增加宏结果写出接口，支持两种 Cell 布局：

```python
@classmethod
def write_macro_results(
    cls,
    patches: Iterable[GeometryPatch],
    output_path: str | Path,
    dbu_um: float,
    *,
    cell_mode: Literal["single_cell", "macro_cells"],
    top_name: str = "OPC_RESULT",
) -> Path:
    """按单 Cell 全局 merge 或每 macro 子 Cell 两种方式原子写出权威 patch。"""
    ...
```

- `single_cell`：函数内部汇总同层 Region、全局 merge，再插入一个 Cell；
- `macro_cells`：每个 patch 裁 ownership 后直接插入独立子 Cell；
- 两者均信任 grid 产生的不重叠 ownership，不重复建立全局 ownership 冲突检测结构；
- 复用现有格式映射、临时文件和 `os.replace()`，不复制第二套 writer。

每轮单 macro 的完整候选 GDS 由 main 内部函数写出，因为它不属于最终 ownership patch：

```python
def _write_macro_gds(
    problem: MacroProblem,
    region: kdb.Region,
    path: Path,
) -> Path:
    """把单 macro 当前完整候选 Region 写入 RESULT Cell，供检查和最终合并。"""
    ...
```

## 14. 文件级改动

### 14.1 删除

| 文件 | 原因 |
|---|---|
| `opc/input/macro.py` | 从全局 core 反向组合 macro，与新路线冲突。 |
| `opc/input/preflight.py` | 全局问题容量估算会错误拒绝可按 macro 执行的版图，且无新调用方。 |
| `opc/input/edge/builder.py` | 旧全局 `MBOPCProblem` 被单 macro problem 取代。 |
| `opc/input/edge/macro.py` | `MacroPreparation` 与新 problem 重复。 |
| `opc/input/edge/ownership.py` | ownership 仅由 problem 准备调用，按用户要求合并进 `problem.py`。 |
| `opc/diagnostics.py` | 依赖旧 `CellRef/MBOPCProblem`，新树无生产调用方。 |

删除前必须 `rg` 搜索实际调用点；若发现本文未记录的新调用方，停止并报告。

### 14.2 新增

| 文件 | 核心内容 |
|---|---|
| `config/macro_pipeline.toml` | `gcd_45nm.gds` 的 2×2 smoke 配置。 |
| `opc/input/edge/problem.py` | MacroProblem、ownership 切线分段、owner/CSR、save/load。 |
| `main/run_macro_pipeline.py` | 阶段 0–3 的直接运行入口和性能摘要。 |
| `tests/opc/input/test_grid.py` | 两种 macro 模式、context/canvas、DBU 和覆盖测试。 |
| `tests/opc/input/test_macro_problem.py` | 跨界几何、owner/CSR、NPZ 和切线分段测试。 |
| `tests/main/test_macro_pipeline.py` | 两轮、每轮 macro GDS、最终 merge 和 CLI 测试。 |
| `doc/macro_core_pipeline_development_report.md` | 实际开发、删除和简化审计。 |
| `doc/macro_core_pipeline_test_report.md` | 命令、用例、结果、性能和 smoke 产物。 |

### 14.3 修改

| 文件 | 核心改动 |
|---|---|
| `opc/input/grid.py` | 重写为 CoreSpec、MacroSpec 和两种两级规划。 |
| `opc/input/mask.py` | 删除冗余 PhysicalMask；规范化函数直接返回 Region。 |
| `opc/input/raster.py` | context 局部栅格、居中 canvas、统一透光率和 ownership canvas。 |
| `opc/input/edge/reconstruction.py` | 消费 MacroProblem；其他拓扑保护保持。 |
| `opc/input/__init__.py` | 删除旧 macro/preflight/PhysicalMask 导出，导出新网格/raster。 |
| `opc/input/edge/__init__.py` | 删除旧 builder/macro/ownership 导出，导出 MacroProblem。 |
| `geometry/patch.py` | 支持 single_cell 全局 merge 与 macro_cells 两种最终写出。 |
| `tests/geometry/test_patch.py` | 两种 Cell 模式的覆盖和 polygon 数测试。 |
| `task_plan.md` | 批准实施后记录本重构阶段。 |
| `findings.md` | 记录接口、阶段边界和 kernel/canvas 事实。 |
| `progress.md` | 记录开发、测试、报告和本地提交。 |
| `doc/development_manual.md` | 若不存在则创建，记录直接运行和 problem 接口。 |
| `doc/test_manual.md` | 若不存在则创建，记录测试和产物检查方法。 |

### 14.4 明确不修改

- `00_PAST/**`；
- `layout/**`；
- `geometry/contour.py`、`geometry/raster.py`、`geometry/validate.py`；
- `opc/input/edge/sampling.py`；
- 用户 GDS/GLP、`.vscode/` 和无关工作树修改。

`edge/fragmentation.py` 仅在实际实现证明 `_split_segments_at_ownership_cuts()` 无法在
`problem.py` 使用公共 `SegmentBatch` 完成时才允许提出最小修改；不能未经再次说明直接改。

## 15. 测试矩阵

### 15.1 网格与配置

1. size：21/11/1 得到 11+10；
2. size：21.5/11/1 只有最外侧出现 0.5 缩短 core；
3. size：macro 不是 core 整数倍时失败；
4. count 2×2：21×17/core1 得到横向 11+10、纵向 9+8；
5. `macro_size_nm` 和 `macro_grid` 同时出现或同时缺失时失败；
6. macro 数超过 core 单元数时失败；
7. 负坐标 bbox；
8. macro/core ownership 无正面积重叠且面积和等于父框；
9. context 可以跨 macro/bbox，但不改变 ownership；
10. 不能精确转换 DBU 的 nm 参数失败；
11. core/context 不是 pixel 整数倍时失败；
12. `core+2context == canvas×pixel` 合法，超过时失败；
13. `context < max_displacement` 失败；
14. canvas 不是 256 时失败。

### 15.2 Canvas 与极性

1. 228×228 local mask 在 256 canvas 中四边各补 14 个零像素；
2. 奇数 padding 差值时，低侧 floor、高侧接收余量；
3. clear/opaque 均满足 1=透光、0=不透光；
4. opaque local context 背景为 1，但 canvas 外围 padding 仍为 0；
5. ownership_canvas 与居中 mask canvas 完全同 shape、同偏移；
6. global DBU probe 到 canvas 坐标包含 low-side padding；
7. 满 256 local 输入不再添加 padding；
8. 超 256 在分配 canvas 前失败；
9. 对照 `00_PAST` 已确认的 `_prepare_mask()` 规则，当前 256 输入不会被二次移动。

### 15.3 跨 macro/core 几何

动态生成：矩形、长条、凹多边形、孔洞、2 nm 窄环、多角度斜边、相接/重叠图形、
SREF 和 2×2 AREF。分别跨 macro 横边、竖边、角点和多个 core。

验证：

- query/context 框没有进入物理边段；
- ownership 切线分段后没有可写 segment 跨两个 owner；
- 共享斜边参数交点一致，不出现 33/34 DBU 分歧；
- 每个 segment 最多一个可写 owner；
- context segment 可以出现在多个 core CSR，但 owner 不重复；
- owner segment 必然属于 owner core membership；
- hole/ring 绕向和拓扑保持；
- 零位移按 ownership 汇总后全局 merge，XOR 为零。

### 15.4 双轮迭代

1. 第一轮所有 owner segment 位移为 +2 nm；
2. 第一轮每段恰写一次，context 段不写；
3. 第二轮从第一轮状态读取，不从零重启；
4. 第二轮累计 -2 nm 后 owner 位移全部为零；
5. 两轮都逐 core 生成有限 transmission sum；
6. 每个 macro 每轮恰好一个 result NPZ 和一个 GDS；
7. 2×2 时第一轮 4 GDS、第二轮 4 GDS；
8. 第二轮每个 macro GDS 的完整候选 Region 等于其零位移参考；
9. macro 正序、逆序执行，两轮状态和最终覆盖一致；
10. 阶段二通过 monkeypatch/调用计数证明没有调用 LayoutDB、materialize、extract、fragment
    或 owner 构造。

### 15.5 最终合并

1. single_cell 输出只有一个结果 Cell，跨 macro polygon merge 后不保留 seam；
2. macro_cells 输出含预期数量子 Cell；
3. 两种模式顶层物理覆盖 XOR 为零；
4. 第二轮 single_cell 与原始目标层 XOR 为零；
5. 第一轮用于检查的合并结果相对参考有非零变化，证明 +2 确实生效；
6. merge 前后覆盖面积不因 normalize 改变；
7. 无效 polygon、缺失 macro GDS、重复 macro ID 或 result round 不一致时明确失败；
8. 不复制未处理 layer。

### 15.6 持久化、性能和 smoke

- problem save/load 后所有标量与数组逐项相等；
- NPZ 无 object dtype，`allow_pickle=False` 可读；
- 错误版本、截断文件、错误位移长度直接失败；
- 原子保存失败不替换旧完整文件；
- 全流程只常驻当前 macro problem 和当前 core canvas；
- `gcd_45nm.gds` 使用 layer 11/0、2×2 macro 跑完整两轮；
- 记录阶段 1 最大 problem、阶段 2 最大 RSS、阶段 3 全局 merge 峰值；
- smoke 最终保存 8 个 macro GDS、1 个 final GDS 和 summary。

自动测试使用 `tmp_path` 动态生成 GDS，不依赖用户文件；`gcd_45nm.gds` 只用于最终 smoke，
不修改原文件，不提交 output。

## 16. 性能与内存统计

使用 `perf_counter()` 记录：配置/打开、网格、阶段一各 macro、第一轮、第二轮、最终 merge、
总耗时。使用 `psutil.Process().memory_info().rss` 记录阶段和单 macro 峰值；缺失 psutil 时
正常导入失败，不静默降级。

`summary.json` 保存：

```text
macro_count
core_count
problem_count
round_count = 2
round_001_macro_gds_count
round_002_macro_gds_count
segment_count_sum
membership_count_sum
maximum_problem_bytes
maximum_problem_macro_id
prepare_seconds
round_001_seconds
round_002_seconds
merge_seconds
total_seconds
prepare_peak_rss_bytes
iteration_peak_rss_bytes
merge_peak_rss_bytes
final_cell_mode
final_layout
final_xor_area
```

不按 core 保存大 JSON。每 core transmission sum 已在对应 result NPZ 中。

## 17. 验收命令

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\opc\input tests\main tests\geometry\test_patch.py
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc main tests\opc tests\main geometry\patch.py tests\geometry\test_patch.py
D:\app\miniforge\envs\myopc\python.exe -m compileall -q layout geometry opc main tests
D:\app\miniforge\envs\myopc\python.exe main\run_macro_pipeline.py config\macro_pipeline.toml
```

实施前基线为 `49 passed`。完整 Ruff 当前在未纳入本任务的 `geometry/contour.py` 有一个既存
导入空行告警；不借本任务修改，专项 Ruff 必须通过，完整结果在报告中如实记录。

## 18. 实施顺序、阶段提交与报告

只有用户明确批准本计划后才开始。

### 实施 A：配置、两级网格和居中 canvas

#### 要做什么

锁定两种 macro 入口、CoreSpec/MacroSpec、context 和 ICCAD13 256 canvas 数据契约。

#### 执行

1. 检查并保留用户工作树；
2. 实现严格 TOML/DBU 校验；
3. 重写 grid 和 raster；
4. 完成网格、极性、居中 padding 测试；
5. 本地提交：`refactor(opc-input): 重建两级网格与居中光刻画布`。

### 实施 B：持久化 MacroProblem

#### 要做什么

阶段一只计算一次完整参考边段、ownership 切线分段和 CSR，删除旧重复路线。

#### 执行

1. 新增 problem.py，简化 mask；
2. 完成 ownership 切线再分段、owner/CSR 和 NPZ；
3. 修改 reconstruction；
4. 删除旧 macro/preflight/builder/macro/ownership/diagnostics 文件；
5. 搜索旧符号和调用点；
6. 本地提交：`refactor(opc-input): 建立持久化 MacroProblem`。

### 实施 C：双轮迭代与逐 macro GDS

#### 要做什么

实现不依赖原 GDS 的两轮状态传递、逐 core 唯一写入和每轮 macro 汇总。

#### 执行

1. 实现 run_round；
2. 第一轮 +2 nm，第二轮 -2 nm；
3. 每 macro 每轮保存 result NPZ 和完整候选 GDS；
4. 验证阶段二没有重复阶段一昂贵函数；
5. 本地提交：`feat(main): 完成双轮 macro-core 迭代验证`。

### 实施 D：最终 merge 与两种 Cell 输出

#### 要做什么

实现 ownership 权威覆盖、single_cell 全局 normalize、macro_cells 调试输出和回零 XOR。

#### 执行

1. 扩展 PatchWriter；
2. 实现 merge_final；
3. 完成 seam、两种 Cell 模式和异常测试；
4. 用 gcd_45nm 跑 2×2 smoke；
5. 本地提交：`feat(geometry): 完成 macro 结果全局合并与双模式写出`。

### 实施 E：报告与简化审计

#### 要做什么

确认最终代码只保留当前调用方需要的结构和函数，并完整记录验证证据。

#### 执行

1. 更新 `task_plan.md`、`findings.md`、`progress.md`；
2. 创建/更新开发手册和测试手册；
3. 写专项开发报告和测试报告；
4. 审计重复字段、未调用函数、重复实现、异常入口和 bug 修复遗留；
5. 确认没有 Worker、注册器、双 problem、第二套 raster 或旧兼容包装；
6. 运行第 17 节全部命令；
7. 文档提交：`docs: 完成 macro-core 重构开发与测试报告`；
8. 不推送远端。

## 19. 最终调用关系

```text
run_macro_pipeline.run(config)
│
├─ 阶段 0/1 prepare_problems
│   ├─ LayoutDB.open(gcd_45nm.gds)
│   ├─ exact_dbu(...)
│   ├─ plan_macros(...) -> 2×2 MacroSpec
│   └─ 对每个 macro
│       ├─ materialize_intersecting(macro.query_box)
│       ├─ normalize_mask(...)
│       ├─ extract_contour(...)
│       ├─ fragment_edges(...)
│       ├─ _split_segments_at_ownership_cuts(...)
│       ├─ _build_macro_ownership(...)
│       └─ MacroProblem.save(...)
│
├─ 阶段 2 run_round(round=1, delta=+2nm)
│   └─ 每 macro：逐 core owner 更新 -> 重建 -> 居中 raster -> NPZ + GDS
│
├─ 阶段 2 run_round(round=2, delta=-2nm)
│   └─ 每 macro：读取 round1 -> 逐 core owner 更新 -> 回零 -> NPZ + GDS
│
└─ 阶段 3 merge_final(round=2)
    ├─ 逐 macro 读取完整候选 GDS
    ├─ 按 macro ownership 选择权威覆盖
    ├─ single_cell: 全局 Region.merged()
    │   或 macro_cells: 分别写入子 Cell
    └─ 完整目标层 GDS/OASIS + XOR 验证
```

依赖方向保持：

```text
layout -> geometry -> opc.input -> opc.input.edge -> main
```

基础层不得反向导入 main 或未来具体求解器。

## 20. 已知限制

### 20.1 巨大单 polygon

`materialize_intersecting()` 为避免虚假边，会把与 context 相交的完整 occurrence 带入
当前 macro。普通局部图形受 macro 限制，但一个横跨大量 macro、且自身顶点海量的 polygon
仍可能使单个 problem 超内存。当前没有验证过可保持 ring/hole 和跨 macro 更新的局部拓扑
格式，本阶段暴露限制，不增加未经证明的 polygon shard。

### 20.2 single_cell 全局 merge 内存

single_cell 为消除 seam 需要一次全局同层 Region merge，可能成为最终内存峰值。这是用户
本轮明确要求的正确性路径。macro_cells 提供较低临时内存但保留表示 seam。若 64 GiB 仍
无法完成 single_cell，必须报告并另行设计 boundary-stream merge，不能静默输出未 merge
结果冒充完成。

### 20.3 当前双轮不是真实 OPC

`+2/-2 nm` 只验证状态、owner、重建、canvas、轮次文件和 merge，不使用光刻输出决定边段
方向，不代表已实现 MB-OPC。以后真实求解仍必须遵守全局轮次屏障：所有 macro 从同一代
状态读取，全部完成后才能发布下一代，不能让某个 macro 独自跑完所有轮次。

### 20.4 Context 充分性

35×35 频域 kernel 不能给出有限空间光学半径。当前 context 参数在尺寸上与 ICCAD13 canvas
兼容，并覆盖最大几何位移，但是否达到光学收敛必须以后用增大 context 的结果差异测试确定。

## 21. 最终完成标准

- 配置不含 `macro_mode`，`macro_size_nm`/`macro_grid` 恰好一个；
- size 模式名义 macro 是 core 整数倍，21/11/1 得到 11+10；
- count 2×2 按 core 数平衡分配；
- `optical_range_dbu` 已完全替换为 `context_dbu`；
- CoreSpec、MacroSpec、MacroProblem 字段均有中文注释，所有函数有中文 docstring；
- core ownership/context/canvas 含义清晰，无 `input_box`；
- core+2context 不超过 256×pixel，局部内容居中，外围统一补 0；
- clear/opaque 输入始终遵守 1=透光、0=不透光；
- 阶段一昂贵参考计算每 macro 只执行一次；
- 阶段二不打开原 GDS、不重新物化、提边、分段或算 ownership；
- ownership 切线处分段，单 segment 不跨两个 owner；
- 第一轮全部 owner +2 nm，第二轮 -2 nm 后精确回零；
- 2×2 时每轮 4 个 macro GDS，两轮共 8 个；
- 每个 macro 每轮还有一个 result NPZ；
- macro GDS 保存完整候选 polygon；
- 最终按 ownership 选择权威覆盖，single_cell 全局 merge 后无 macro seam；
- macro_cells 模式也可用，并明确其表示差异；
- `gcd_45nm.gds` 完整 smoke 成功，最终 XOR 为零；
- 删除旧 macro/problem/ownership 路线，没有兼容包装、重复 raster 或一次性抽象；
- 完整测试、开发报告、测试报告、手册和规划三文件同步；
- 只做本地提交，不推送，不修改 `00_PAST/` 和用户数据。
