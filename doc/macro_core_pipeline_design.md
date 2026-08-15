# Macro–Core 两级任务流水线重构计划

> 状态：**等待用户审查，禁止开始实施**
> 本文件是后续实现的唯一决策依据。新上下文只需读取本文件、仓库根目录
> `AGENTS.md` 和实际代码，即可直接实施；若实际代码与本文记录不一致，必须先
> 报告差异，不得自行改变本文接口或扩大范围。

## 1. 重构目标

当前新树正处于逐模块从 `00_PAST/` 重新梳理的阶段。现有 OPC 输入代码采用
“先建立全局 core 网格，再把既有 core 切线分组为 macro”的路线，并同时保留
`MBOPCProblem`、`MacroPreparation` 等两套问题结构。该路线不再保留。

本次重构完成以下闭环：

1. 从完整目标层版图的 bbox 生成不重叠的 macro ownership 网格；
2. 在每个 macro 内独立生成不重叠的 core ownership 网格；
3. 当前主机按稳定顺序逐个 macro 生成一个可持久化 `MacroProblem`；
4. 当前阶段不接求解器，但必须逐 core 构造固定尺寸 mask canvas，随后生成零位移结果；
5. 逐个加载 problem/result，只回写 macro ownership，合并为完整目标层 GDS/OASIS；
6. 全流程可用一条 Python 命令直接运行，不需要安装本项目包。

本次允许删除新树中路线错误、没有有效调用方或只为旧路线服务的代码，不建立兼容层。
`00_PAST/` 全部只读，不得修改。

## 2. 已锁定的设计原则

### 2.1 Ownership 与 context

- macro ownership 之间没有正面积重叠，合起来精确覆盖目标 bbox，不向 bbox 外补齐；
- 同一 macro 内的 core ownership 之间没有正面积重叠，合起来精确覆盖该 macro；
- ownership 表示唯一写入范围；光学输入 context 可以跨越 core 和 macro 边界；
- macro 准备时查询该 macro 所有 core 输入框的并集；查询之间允许重叠；
- 最终合并时，每个结果必须先精确裁到自身 macro ownership，再写入输出；
- 相邻 macro 共享的 context 只能读取，不能形成第二个写入者。

### 2.2 防止 macro 边界产生虚假边

macro 查询必须调用 `ShapeQuery.materialize_intersecting()`：KLayout 的 ROI 只负责筛选
与 context 相交的 occurrence，返回的是完整相交图形，不与 context 框做布尔裁剪。
完整图形先合并为物理覆盖，再提取轮廓和切分边段。context 框和 macro ownership 框
均不得进入 `ContourBatch`，因此它们的四条边不会成为可移动边。

只有以下两个位置允许裁剪：

1. core 栅格化时，按该 core 的固定输入框读取像素；

### 2.3 当前阶段不实现的内容

- 不实现 MB-OPC、DiffOPC、ILT 或其他求解器；
- 不实现 Worker、任务队列、进程池、分布式调度或求解器注册器；
- 不实现断点恢复、失败重试和状态机；错误必须直接传播；
- 不实现全局 polygon merge/normalize；最终版图允许保留 macro seam 的几何碎片，
  但其物理覆盖必须与零位移输入严格一致；
- 不实现跨 macro 永久稳定的 segment ID；当前 problem 仅保证单次准备内的局部索引；
- 不修改 `layout/`；现有 `materialize_intersecting()` 已满足需求；
- 不为了未来求解器增加空接口或抽象基类。

## 3. 配置文件

新增 `config/macro_pipeline.toml`。所有生产参数显式填写，不提供代码内默认值。
路径相对于 TOML 文件所在目录解析。

```toml
[input]
layout = "../TestReticle/simple.gds"
top_cell = "TOP"                 # 可省略；省略时沿用 LayoutDB 的唯一顶层规则
layer = 1
datatype = 0
polarity = "clear"              # clear 或 opaque

[grid]
macro_mode = "size"             # size 或 count
macro_size_nm = 1200             # size 模式必填；count 模式禁止出现
# macro_grid = [2, 2]            # count 模式必填：[横向列数, 纵向行数]
core_size_nm = 400
pixel_nm = 8
canvas_pixels = 160              # 方形光刻输入的单边像素数
optical_range_nm = 256           # 单边光学影响半径

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24
miter_limit = 4.0

[output]
work_dir = "../output/macro_pipeline"
layout = "result.gds"            # 允许 .gds/.gds2/.oas/.oasis
```

`macro_mode` 两种模式互斥：

- `size`：必须且只能提供 `macro_size_nm`；
- `count`：必须且只能提供 `macro_grid`；
- 同时提供或同时缺失时直接报错，不猜测用户意图。

示例值满足：`1200 nm = 3 × 400 nm`，且
`8 × 160 = 1280 nm > 400 + 2 × 256 = 912 nm`。

## 4. 单位与整除规则

打开版图取得 `dbu_um` 后，使用以下规则把 nm 转换成整数 DBU：

```text
dbu_nm = dbu_um × 1000
value_dbu = value_nm / dbu_nm
```

不得用 `round()` 静默吸收误差。`value_dbu` 不是精确整数时直接报错，并在错误中写明
参数名、nm 值和当前 `dbu_nm`。

所有模式共有的校验：

```text
core_size_dbu > 0
pixel_dbu > 0
optical_range_dbu > 0
canvas_pixels 为正整数
core_size_dbu % pixel_dbu == 0
optical_range_dbu % pixel_dbu == 0
pixel_dbu × canvas_pixels > core_size_dbu + 2 × optical_range_dbu
```

`size` 模式额外校验：

```text
macro_size_dbu > core_size_dbu
macro_size_dbu % core_size_dbu == 0
```

注意：要求的是**配置中的名义 macro 大小**为 core 的整数倍，版图本身不需要是
macro 或 core 的整数倍。若版图某轴长 21 µm、`macro_size=11 µm`、`core_size=1 µm`，
该轴 macro 宽度必须为 `11 µm + 10 µm`；末端 10 µm macro 合法。若版图轴长为
21.5 µm，则为 `11 µm + 10.5 µm`，最后一个 macro 内允许出现一个 0.5 µm 的
缩短 core。

`count` 模式额外校验：

- `macro_grid=[columns, rows]` 两项均为正整数；
- 横向 macro 数不得超过横向 core 单元数；纵向同理，保证每个 macro 至少含一个 core；
- count 模式没有 `macro_size_nm`，因此不存在 macro/core 大小整除检查；
- macro 边界仍优先落在 core 切线上，只有版图最右/最上边缘允许形成缩短 core。

## 5. 两种 macro 切分算法

### 5.1 按 nm 大小切分（`macro_mode="size"`）

对 x/y 轴分别从 bbox 左/下边界开始，以 `macro_size_dbu` 为步长生成切线，最后一项
强制为 bbox 右/上边界。最后一个 macro 可以小于名义大小，但不能生成零宽 macro。

在每个 macro 内，再从该 macro 左/下边界开始，以 `core_size_dbu` 为步长生成局部
core 切线，最后一项强制为 macro 右/上边界。由于名义 macro 是 core 的整数倍，
只有版图最右/最上边缘的 macro 可能产生缩短 core。

### 5.2 按块数切分（`macro_mode="count"`）

`macro_grid=[columns, rows]` 表示完整 bbox 横向分为 `columns` 个 macro、纵向分为
`rows` 个 macro。这里的“均分”定义为**按 core 单元数量尽量均分**，不是直接对
DBU 几何长度做除法。

单轴算法：

1. 先计算从轴起点按 `core_size_dbu` 覆盖到终点所需的 core 单元数：
   `unit_count = ceil(axis_length / core_size_dbu)`；
2. `base, remainder = divmod(unit_count, macro_count)`；
3. 前 `remainder` 个 macro 分配 `base + 1` 个 core 单元，其余分配 `base` 个；
4. 逐项累加 core 单元宽度生成 macro 切线，最后一条切线强制等于轴终点；
5. 每个 macro 内再独立按 `core_size_dbu` 切 core；只有完整 bbox 的最终边缘允许
   出现缩短 core。

因此轴长 21 µm、core 1 µm、横向 2 块时，两个 macro 分别获得 11 和 10 个 core，
宽度为 11 µm 和 10 µm；不会得到两个 10.5 µm macro。

二维 macro 使用行优先稳定顺序：先按 y 从下到上，再按 x 从左到右，ID 为
`mr{row}c{column}`。

## 6. 固定 core 光刻输入框

每个 core 有两个不同矩形：

- `ownership_box`：唯一可写的几何范围；
- `input_box`：固定 `canvas_pixels × canvas_pixels` 的光刻输入范围。

对某一轴：

```text
canvas_span = pixel_dbu × canvas_pixels
extra = canvas_span - core_axis_length
low_padding = extra // 2
high_padding = extra - low_padding
```

输入框低坐标向外扩 `low_padding`，高坐标向外扩 `high_padding`。严格画布校验保证
名义完整 core 两侧 padding 均不少于 `optical_range_dbu`；缩短 core 的 padding
只会更大。输入框允许越出版图 bbox，框外按 mask 极性对应的背景处理。

一个 macro 的 `context_box` 是其全部 core `input_box` 的最小轴对齐包围框。macro
物化查询使用该 `context_box`，而不是简单使用 `ownership_box.expanded(R)`，因为固定
画布中可能存在大于光学最小范围的额外 padding。

## 7. 数据结构与函数定义

### 7.1 `opc/input/grid.py`

删除当前“全局 core 网格 + macro 反向分组”实现，定义以下唯一网格结构：

```python
@dataclass(frozen=True, slots=True)
class MacroSpec:
    """一个 macro 的唯一写入框、局部 core 切线和固定光刻输入参数。"""

    macro_id: str
    ownership_box: DbuBox
    x_cuts: NDArray[np.int64]
    y_cuts: NDArray[np.int64]
    pixel_dbu: int
    canvas_pixels: int
    optical_range_dbu: int

    @property
    def column_count(self) -> int: ...

    @property
    def row_count(self) -> int: ...

    @property
    def core_count(self) -> int: ...

    @property
    def context_box(self) -> DbuBox: ...

    def core_ownership_box(self, core_index: int) -> DbuBox: ...

    def core_input_box(self, core_index: int) -> DbuBox: ...

    def locate_owned_points(self, points: object) -> NDArray[np.int32]: ...
```

只保存切线，不常驻 `CoreSpec` 对象列表。`core_index` 使用 macro 局部行优先编号。
`locate_owned_points()` 对 macro ownership 外的点返回 `-1`；内部共享边按半开区间
稳定归右/上，整体最大边界归最后一列/行。

公开规划函数：

```python
def plan_macros(
    bounds: DbuBox,
    *,
    core_size_dbu: int,
    pixel_dbu: int,
    canvas_pixels: int,
    optical_range_dbu: int,
    macro_size_dbu: int | None = None,
    macro_grid: tuple[int, int] | None = None,
) -> tuple[MacroSpec, ...]: ...
```

`macro_size_dbu` 和 `macro_grid` 必须恰好提供一个。私有轴向函数只允许存在以下三个，
且必须同时被 x/y 两轴调用，不能再增加网格类：

```python
def _cuts_by_size(start: int, end: int, size: int) -> NDArray[np.int64]: ...

def _macro_cuts_by_count(
    start: int, end: int, core_size: int, count: int
) -> NDArray[np.int64]: ...

def _core_cuts(start: int, end: int, core_size: int) -> NDArray[np.int64]: ...
```

### 7.2 `opc/input/mask.py`

保留：

```python
class MaskPolarity(str, Enum):
    CLEAR = "clear"
    OPAQUE = "opaque"
```

删除 `PhysicalMask`。规范化入口改为：

```python
def normalize_mask(batch: RegionBatch, layer: LayerSpec) -> kdb.Region:
    """删除属性、合并物理覆盖、恢复孔洞并返回合法原生 Region。"""
```

`layer`、`polarity`、`query_box` 分别已由 problem、配置和 `MacroSpec` 持有，不在另一
结构中重复保存。

### 7.3 `opc/input/edge/ownership.py`

公开给 problem 构造器的唯一函数：

```python
def build_macro_ownership(
    segments: SegmentBatch,
    macro: MacroSpec,
) -> tuple[
    NDArray[np.int32],   # owner_indices，长度 S；-1 表示当前 macro 只读
    NDArray[np.int64],   # core_offsets，长度 C+1
    NDArray[np.int32],   # member_segment_indices，长度 M
]: ...
```

算法：

1. 从 `SegmentBatch` 参数区间批量计算 segment 起点、终点、bbox 和中点；
2. 用 `macro.locate_owned_points(midpoints)` 得到局部 owner；macro 外中点为 `-1`；
3. 对当前 macro 的每个 core，一次 NumPy 批量判断 segment bbox 是否与
   `core_input_box()` 接触；不逐 segment 进入 Python；
4. 两遍构造 CSR：第一遍计数并在分配前检查 `int32` 容量，第二遍填充；
5. 验证所有 `owner>=0` 的 segment 必须出现在其 owner core 的 membership 中；
6. 临时 S×2 数组在 CSR 大数组分配前释放。

当前阶段不提供可配置的 memory fallback；超过 `int32` 或实际内存时明确抛出
`OverflowError/MemoryError`，不得自动缩小 macro 或吞错重试。

### 7.4 `opc/input/edge/problem.py`

新增此文件，同时删除 `builder.py` 和 `edge/macro.py`。结构定义：

```python
@dataclass(frozen=True, slots=True)
class MacroProblem:
    """单个 macro 可独立持久化和加载的边段型 OPC 输入。"""

    macro: MacroSpec
    layer: LayerSpec
    polarity: MaskPolarity
    fragmentation: FragmentationConfig
    segments: SegmentBatch
    owner_indices: NDArray[np.int32]
    core_offsets: NDArray[np.int64]
    member_segment_indices: NDArray[np.int32]

    def segments_for_core(self, core_index: int) -> NDArray[np.int32]: ...

    def owner_segments_for_core(self, core_index: int) -> NDArray[np.int32]: ...

    def save(self, path: str | Path) -> Path: ...

    @classmethod
    def load(cls, path: str | Path) -> MacroProblem: ...
```

准备入口：

```python
def prepare_macro_problem(
    batch: RegionBatch,
    layer: LayerSpec,
    polarity: MaskPolarity | str,
    fragmentation: FragmentationConfig,
    macro: MacroSpec,
) -> MacroProblem: ...
```

处理顺序固定：

1. 要求 `batch.query_box == macro.context_box`；
2. `normalize_mask()` 得到当前完整相交候选的物理覆盖；
3. `extract_contour()` 一次提取完整真实轮廓；
4. `fragment_edges()` 按完整数学边确定性分段；
5. `build_macro_ownership()` 生成局部 owner 和 membership；
6. 构造 `MacroProblem`，不额外保留原生 `Region`。

为了完整轮廓重建，`SegmentBatch` 保留与 macro context 相交的完整候选 polygon 拓扑；
不能只保存 owned segment，否则跨 macro 的 ring 无法闭合。这一选择可能重复保存跨多个
macro 的巨大 polygon，属于本文第 14 节明确限制，当前阶段不增加不成熟的局部拓扑格式。

### 7.5 Problem NPZ 格式

使用 `np.savez`，不压缩，避免压缩造成额外 CPU 和峰值内存。先写同目录临时文件，成功
关闭后用 `os.replace()` 原子发布。读取必须 `allow_pickle=False`。

格式版本固定为 `1`，字段如下：

```text
format_version                  int32[1]
macro_id                        unicode[1]
macro_ownership_box             int64[4]
macro_x_cuts                    int64[X+1]
macro_y_cuts                    int64[Y+1]
pixel_dbu                       int64[1]
canvas_pixels                   int64[1]
optical_range_dbu               int64[1]
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

不保存 `PhysicalMask`、`Region`、重复 starts/ends/normals、owned segment 副本或全局
segment ID。读取时由现有 dataclass 不变量完成数组和拓扑校验，不再写第二套重复校验。

### 7.6 `opc/input/raster.py`

保留当前通用函数：

```python
def rasterize_region_canvas(
    region: kdb.Region, box: DbuBox, pixel_dbu: int, canvas: int
) -> NDArray[np.float32]: ...

def rasterize_mask_canvas(
    region: kdb.Region,
    box: DbuBox,
    pixel_dbu: int,
    canvas: int,
    *,
    polarity: MaskPolarity | str = MaskPolarity.CLEAR,
    field_box: DbuBox | None = None,
) -> NDArray[np.float32]: ...

def ownership_canvas(
    core: DbuBox, context: DbuBox, pixel_dbu: int, canvas: int
) -> NDArray[np.bool_]: ...
```

不新增“macro raster”或第二份栅格实现。主流程直接把
`macro.core_input_box(core_index)` 传给通用函数。opaque 模式的 `field_box` 使用当前
core input box，使画布内 `1` 始终表示透光。

### 7.7 `opc/input/edge/reconstruction.py`

仅把类型依赖从旧 `MBOPCProblem` 改为 `MacroProblem`：

```python
def reconstruct_contours(
    problem: MacroProblem, displacements: object
) -> ContourBatch: ...

def reconstruct_region(
    problem: MacroProblem, displacements: object
) -> kdb.Region: ...
```

现有 miter/bevel、ring 绕向、hole 越界、最大位移和合法性检查不重写；只删除对
`problem.physical_mask.layer` 的依赖，改读 `problem.layer`。

### 7.8 `geometry/patch.py`

保留现有小规模 `PatchSet/PatchWriter.write()`，新增当前大版图主流程的真实调用入口：

```python
@classmethod
def write_hierarchical(
    cls,
    patches: Iterable[GeometryPatch],
    output_path: str | Path,
    dbu_um: float,
    top_name: str = "OPC_MACROS",
) -> Path: ...
```

行为：

- 逐 patch 裁到 `ownership_box`；
- 每个 macro 建立一个独立子 Cell，子 Cell 中使用全局坐标；
- 顶层以单位变换引用全部 macro Cell；
- patch 插入后立即释放 Python `Region` 引用；
- 不构造全局 `PatchSet._regions` 和全局布尔 merge；
- 复用现有格式映射和同目录临时文件原子写出逻辑；
- 网格规划是可信内部输入，不再建立一份全局 ownership Region 做重复冲突检查。

输出 Layout 仍会在 KLayout 原生内存中持有最终全部 polygon，但不会额外持有一份合并
Region。真正的 GDS/OASIS record 流式写出不在本阶段实现。

## 8. 主流程定义

新增 `main/run_macro_pipeline.py`。文件开头按现有 main 演示入口的方式，把仓库根加入
`sys.path`，保证以下命令可直接运行：

```powershell
D:\app\miniforge\envs\myopc\python.exe main\run_macro_pipeline.py config\macro_pipeline.toml
```

### 8.1 配置结构

本文件只定义一个配置结构，避免按 input/grid/edge/output 再拆四个 dataclass：

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """主流程一次运行所需的全部显式用户配置。"""

    layout_path: Path
    top_cell: str | None
    layer: LayerSpec
    polarity: MaskPolarity
    macro_mode: str
    macro_size_nm: Decimal | None
    macro_grid: tuple[int, int] | None
    core_size_nm: Decimal
    pixel_nm: Decimal
    canvas_pixels: int
    optical_range_nm: Decimal
    corner_nm: Decimal
    segment_nm: Decimal
    max_displacement_nm: Decimal
    miter_limit: float
    work_dir: Path
    output_layout: Path
```

TOML 中的 nm 数值先通过 `Decimal(str(value))` 读取，避免二进制 float 参与精确整除判断。

函数定义：

```python
def load_config(path: str | Path) -> PipelineConfig: ...

def exact_dbu(value_nm: Decimal, dbu_nm: Decimal, name: str) -> int: ...

def prepare_problems(config: PipelineConfig) -> dict[str, object]: ...

def execute_problem(problem_path: Path, result_path: Path) -> dict[str, object]: ...

def execute_problems(plan: dict[str, object]) -> dict[str, object]: ...

def merge_results(plan: dict[str, object], output_path: Path) -> Path: ...

def run(config_path: str | Path) -> dict[str, object]: ...

def main() -> int: ...
```

不再拆 `main/configuration.py`、runner 类或阶段基类。以上函数均有当前调用方和独立阶段
含义；`run()` 是 Python 调用接口，`main()` 只解析一个位置参数 `config` 并打印摘要。

### 8.2 阶段一：准备 problem

`prepare_problems()` 固定执行：

1. 创建 `work_dir/problems`、`work_dir/results`；若目标目录已存在同名 problem/result，
   直接报错，不覆盖旧任务；
2. `LayoutDB.open()` 一次加载输入，取得 DBU、顶层、目标 bbox；
3. 精确换算配置并构造 `FragmentationConfig`；
4. `plan_macros()` 一次生成稳定 macro 列表；
5. 逐 macro 调用：
   `database.query([layer], macro.context_box).materialize_intersecting()`；
6. `prepare_macro_problem()`；
7. 原子保存 `problems/{macro_id}.npz`；
8. 记录当前 macro 的 segment/core/membership 数量、耗时、RSS，随后释放当前对象；
9. 所有 problem 成功后写 `plan.json`；任意失败时不写完成的 plan。

`plan.json` 至少包含：格式版本、输入路径、top cell、DBU、目标 bbox、目标层、极性、
全部已解析 DBU 配置、macro 稳定顺序、每个 problem 相对路径和各 macro/core box。
JSON 不保存大数组。

### 8.3 阶段二：占位执行

`execute_problem()` 每次只加载一个 problem：

1. 从 `segments.contours` 恢复一次候选 Region；
2. 为 macro 内每个 core 调用一次 `rasterize_mask_canvas()`；
3. 每次只保留当前 core 的一张 `float32[N,N]` canvas；
4. 保存每个 core 的透光率总和，用于证明全部 core 均被处理；
5. 创建长度等于 segment 数量的全零 `float64` 位移；
6. 原子保存 `results/{macro_id}.npz`；
7. 释放 problem、Region 和当前 canvas。

Result NPZ 字段：

```text
format_version          int32[1]，固定为 1
macro_id                unicode[1]
segment_displacements   float64[S]
core_transmission_sums  float64[C]
processed_core_count    int64[1]
```

Result 不定义 dataclass。`processed_core_count != problem.macro.core_count`、macro ID 不一致
或位移长度不一致时直接失败。

### 8.4 阶段三：合并完整目标层

`merge_results()` 按 plan 稳定顺序逐项：

1. 加载一个 problem 和对应 result；
2. `reconstruct_region(problem, displacements)`；
3. 构造 `GeometryPatch(macro_id, layer, region, macro.ownership_box)`；
4. 通过 `PatchWriter.write_hierarchical()` 的迭代器逐 macro 插入；
5. 输出只包含处理后的完整目标层，不复制输入中的其他 layer；
6. 写出后重新打开输出文件，确认顶层、目标层、bbox 和 macro Cell 数可读取；
7. 不在生产主流程重新加载原输入计算全版 XOR；全版 XOR 属测试职责，避免实际大版图
   为诊断重复加载和物化。

### 8.5 性能记录

使用 `perf_counter()` 记录：配置/打开、网格规划、problem 准备、占位执行、合并、总耗时。
使用 `psutil.Process().memory_info().rss` 记录各阶段和逐 macro 峰值。`psutil` 是明确依赖，
缺失时正常抛出导入错误，不做静默降级。

最终 `summary.json` 只保存聚合值和最大 macro 信息，不为每个 core 保存大字典：

```text
macro_count
core_count
segment_count_sum
membership_count_sum
maximum_problem_bytes
maximum_macro_id
prepare_seconds
execute_seconds
merge_seconds
total_seconds
peak_rss_bytes
output_layout
```

## 9. 文件删除、保留和修改清单

### 9.1 删除

| 文件                          | 删除原因                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `opc/input/macro.py`        | 旧逻辑从全局 core 反向组合 macro，与新两级规划冲突。                          |
| `opc/input/preflight.py`    | 全局 segment/membership 估算会拒绝本可分 macro 执行的版图；当前无生产调用方。 |
| `opc/input/edge/builder.py` | 旧全局`MBOPCProblem` 被单 macro `MacroProblem` 取代。                     |
| `opc/input/edge/macro.py`   | `MacroPreparation` 与新 `MacroProblem` 重复且保存两套 active 映射。       |
| `opc/diagnostics.py`        | 仍依赖旧`CellRef/MBOPCProblem`，新树无生产调用方；运行摘要由 main 负责。    |

删除前必须用 `rg` 搜索实际调用点；除上述已知导入外若发现新调用方，停止并把差异报告给用户。

### 9.2 新增

| 文件                                              | 内容                                    |
| ------------------------------------------------- | --------------------------------------- |
| `config/macro_pipeline.toml`                    | 可直接运行`simple.gds` 的显式配置。   |
| `opc/input/edge/problem.py`                     | `MacroProblem`、prepare、save、load。 |
| `main/run_macro_pipeline.py`                    | 准备、占位执行、合并的完整入口。        |
| `tests/opc/input/test_grid.py`                  | 两种 macro 模式及全部网格约束。         |
| `tests/opc/input/test_macro_problem.py`         | problem、ownership、持久化和跨界几何。  |
| `tests/main/test_macro_pipeline.py`             | 完整 CLI/Python 流程。                  |
| `doc/macro_core_pipeline_development_report.md` | 实施完成后记录实际改动和简化审计。      |
| `doc/macro_core_pipeline_test_report.md`        | 实施完成后记录命令、环境、用例和结果。  |

### 9.3 修改

| 文件                                 | 修改内容                                                   |
| ------------------------------------ | ---------------------------------------------------------- |
| `opc/input/grid.py`                | 重写为`MacroSpec + plan_macros()` 两级规划。             |
| `opc/input/mask.py`                | 删除`PhysicalMask`，直接返回规范化 Region。              |
| `opc/input/edge/ownership.py`      | 改为 macro 局部 owner/membership。                         |
| `opc/input/edge/reconstruction.py` | 改为消费`MacroProblem`。                                 |
| `opc/input/__init__.py`            | 删除旧 macro/preflight/PhysicalMask 导出，导出新网格接口。 |
| `opc/input/edge/__init__.py`       | 删除旧 problem/macro 导出，导出`MacroProblem`。          |
| `geometry/patch.py`                | 增加逐 macro 层级写出，复用原子输出逻辑。                  |
| `tests/geometry/test_patch.py`     | 增加层级写出覆盖测试。                                     |
| `task_plan.md`                     | 本任务经批准实施后增加独立阶段和状态。                     |
| `findings.md`                      | 记录新接口、两种 macro 模式和边界不变量。                  |
| `progress.md`                      | 记录实施、测试、报告和提交。                               |
| `doc/development_manual.md`        | 若不存在则创建；补充主流程与 problem 接口。                |
| `doc/test_manual.md`               | 若不存在则创建；补充运行与回归命令。                       |

### 9.4 明确不修改

- `00_PAST/**`
- `layout/**`
- `geometry/contour.py`、`geometry/raster.py`、`geometry/validate.py`
- `opc/input/edge/fragmentation.py`
- `opc/input/edge/sampling.py`
- 用户 GDS/GLP 和 `.vscode/` 文件

当前全局 Ruff 检查在 `geometry/contour.py` 有一个既存导入空行告警，本任务不借机修改；
专项 Ruff 对本次改动文件必须通过，完整 Ruff 结果在测试报告中如实记录该基线。

## 10. 测试设计

所有自动测试使用 `tmp_path` 中动态生成的 GDS，不依赖用户测试文件。`simple.gds` 只用于
最终人工/真实文件 smoke test，不修改、不提交生成产物。

### 10.1 网格单元测试

1. **size 基本切分**：轴长 21, macro 11, core 1，得到 macro `[0,11]`、`[11,21]`；
   分别含 11 和 10 个 core；
2. **size 末端缩短**：轴长 21.5, macro 11, core 1（换成可表达整数 DBU），第二个
   macro 为 10.5，且只在最外侧出现 0.5 的缩短 core；
3. **size 拒绝非整数倍**：macro 10.5、core 1 时配置失败；
4. **count 2×2**：21×17、core 1，横向分配 11+10 个 core，纵向分配 9+8；
5. **count 数量过大**：macro 列/行数超过对应 core 单元数时失败；
6. **负坐标 bbox**：切线锚定 bbox 左/下坐标，不错误锚定零点；
7. **覆盖不变量**：全部 macro/core 交集无正面积，面积和等于父框面积；
8. **固定画布**：完整和缩短 core 的 input box 宽高均为 `pixel×canvas`；
9. **光学余量**：四边最小 padding 均不少于 R；
10. **DBU 精确性**：不能整除 DBU 的 nm 参数失败；
11. **pixel 对齐**：core 或 R 不是 pixel 整数倍时失败；
12. **严格画布条件**：等于 `core+2R` 时失败，大于时通过；
13. **模式互斥**：size/count 参数同时出现或同时缺失时失败。

### 10.2 MacroProblem 几何矩阵

生成以下图形，并让其分别跨 macro 横边、竖边、角点和多个 core：

- 普通矩形；
- 长条矩形；
- 凹多边形；
- 带孔 polygon；
- 2 nm 窄环；
- 多角度斜边 polygon；
- 相接、角点接触和重叠图形；
- 一个层级 SREF 和一个 2×2 AREF occurrence。

逐项验证：

- macro context 框上没有新增边段；
- 零位移重建后裁 ownership，各 macro 合并与全局参考 XOR 面积为零；
- 所有 `owner>=0` 的 segment 恰好只有一个 macro/core 写入者；
- macro 外 context segment 的 owner 为 `-1`，但出现在相关 core membership；
- owner segment 一定属于 owner core membership；
- 斜边跨三个 macro 时，基于完整数学边的切分端点一致，不出现独立裁剪导致的
  33/34 DBU 分歧；
- hole、ring 数和绕向在零位移重建后保持；
- 一个 problem 至少覆盖多个 core 的场景得到不同的 CSR 范围；
- 正序、逆序准备 macro，最终 ownership 覆盖一致。

### 10.3 持久化测试

- save/load 后所有标量、切线和数组逐项相等；
- NPZ 不含 object dtype，`allow_pickle=False` 可正常读取；
- format version 错误时明确失败；
- 截断文件、macro ID 不一致、位移长度错误时明确失败；
- 保存失败不替换既有完整文件，不留下临时文件。

### 10.4 完整主流程测试

用生成式层级 GDS 和临时 TOML 调用 `run()`，并另有一个 subprocess 直接运行脚本：

- plan 中 macro 数量、顺序和 box 正确；
- 每个 problem/result 文件存在；
- 每个 result 的 processed core 数等于 problem core 数；
- 所有 core transmission sum 有限；
- 最终 GDS/OASIS 可重新打开；
- 输出目标层与输入目标层全局 XOR 面积为零；
- 输出顶层只含预期 macro 子 Cell；
- 输出不包含未处理层；
- size 和 count 两种模式各跑一次完整流程；
- clear 和 opaque 均验证 canvas 中 1 表示透光，但几何输出仍表示源 mask polygon；
- 在准备、执行或合并中注入明确异常，确认错误向上传播且不生成最终文件。

### 10.5 性能与内存验证

- 构造至少 4 个 macro、每 macro 至少 4 个 core 的图形；
- 报告每阶段时间、最大单 macro problem 字节和峰值 RSS；
- 确认执行阶段只存在一张当前 core canvas；
- 确认没有收集全版 `MacroProblem` Python 对象列表，只保留文件路径和小型 plan 元数据；
- 对 `TestReticle/simple.gds` 分别运行 size/count 配置并保存摘要；
- 不把性能阈值写成易受机器波动影响的单元测试，只在专项报告记录实测值。

## 11. 验收命令

实施完成后依次执行：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\opc\input tests\main tests\geometry\test_patch.py
D:\app\miniforge\envs\myopc\python.exe -m ruff check opc main tests\opc tests\main geometry\patch.py tests\geometry\test_patch.py
D:\app\miniforge\envs\myopc\python.exe -m compileall -q layout geometry opc main tests
D:\app\miniforge\envs\myopc\python.exe main\run_macro_pipeline.py config\macro_pipeline.toml
```

当前实施前测试基线为 `49 passed`。最终报告必须分别给出：完整 pytest、专项 pytest、
专项 Ruff、compileall、真实 `simple.gds` smoke test 的命令、结果、耗时和输出路径。

## 12. 实施顺序与本地提交

只有用户明确批准本计划后才执行以下步骤。

### 阶段 A：冻结设计与两级网格

1. 重新读取本文件和 `AGENTS.md`；
2. 检查工作树，记录并排除用户已有修改；
3. 更新规划三文件，状态设为 in progress；
4. 重写 `grid.py`，完成两种 macro 模式及网格测试；
5. 测试通过后提交：
   `refactor(opc-input): 重建 macro-core 两级网格规划`。

### 阶段 B：单 macro problem

1. 简化 mask；
2. 新增 `problem.py`，改 ownership/reconstruction；
3. 删除旧 macro/builder/preflight/diagnostics 路线并清理导出；
4. 完成几何矩阵与 NPZ 往返测试；
5. 搜索旧符号确保没有残留；
6. 提交：
   `refactor(opc-input): 以持久化 MacroProblem 取代全局问题`。

### 阶段 C：主流程与完整输出

1. 新增配置和 `run_macro_pipeline.py`；
2. 增加逐 macro 层级写出；
3. 完成准备、占位执行、合并、summary；
4. 完成完整流程测试和 `simple.gds` smoke test；
5. 提交：
   `feat(main): 完成 macro 任务准备执行与版图合并流程`。

### 阶段 D：报告和简化审计

1. 更新开发/测试手册、专项开发/测试报告、规划三文件；
2. 审计重复字段、未调用函数、重复实现、异常入口和 bug 修复遗留；
3. 特别确认没有 Worker、注册器、双 problem 类型、第二套 raster 和旧兼容包装；
4. 运行第 11 节全部验收命令；
5. 报告用户工作树保留情况和三个本地 commit；
6. 不推送远端。

## 13. 交付后的公共调用关系

```text
run_macro_pipeline.run(config)
│
├─ LayoutDB.open(input)
├─ exact_dbu(...)
├─ plan_macros(...)
│   └─ tuple[MacroSpec, ...]
│
├─ 对每个 MacroSpec：
│   ├─ LayoutDB.query(context_box).materialize_intersecting()
│   ├─ prepare_macro_problem(...)
│   │   ├─ normalize_mask(...)
│   │   ├─ extract_contour(...)
│   │   ├─ fragment_edges(...)
│   │   └─ build_macro_ownership(...)
│   └─ MacroProblem.save(problem.npz)
│
├─ 对每个 problem.npz：
│   ├─ MacroProblem.load(...)
│   ├─ contours_to_region(...)
│   ├─ 对每个 core：rasterize_mask_canvas(core_input_box)
│   └─ 保存零位移 result.npz
│
└─ merge_results(...)
    ├─ MacroProblem.load(...)
    ├─ reconstruct_region(...)
    ├─ GeometryPatch(..., macro ownership)
    └─ PatchWriter.write_hierarchical(...)
        └─ 完整目标层 GDS/OASIS
```

依赖方向仍为：

```text
layout -> geometry -> opc.input -> opc.input.edge -> main
```

基础层不得反向导入 main 或未来具体求解器。

## 14. 已知限制与未来求解器约束

### 14.1 巨大单 polygon

`materialize_intersecting()` 为防止虚假边，会把与 context 相交的完整 occurrence 带入
当前 macro。普通局部图形的内存由 macro 尺寸约束，但一个横跨大量 macro、且自身拥有
海量顶点的 polygon 仍可能让单个 problem 超过内存。当前没有足够需求和测试证明一种
可正确保留 ring/hole 的局部拓扑格式，因此本阶段必须暴露此限制，不能声称已经做到
严格 O(macro area) 内存，也不能增加未经验证的 polygon 切片算法。

### 14.2 最终输出内存

逐 macro 层级写出避免全局 Region 和全局 merge 的额外副本，但 KLayout 输出 Layout
仍需持有最终全部子 Cell 几何。若最终结果本身超过内存，需要另行设计底层 GDS/OASIS
record 流式 writer；不在本阶段假装解决。

### 14.3 后续迭代不能让 macro 各自跑完全部轮次

当前零位移占位执行允许 problem 逐个独立完成。未来接 MB-OPC、梯度 OPC 或 ILT 时，
若相邻 macro 的光学 context 会随优化改变，不能让 macro A 独立跑完 N 轮后再处理 B。
正确语义必须是：

```text
generation k 固定
    -> 所有 macro/core 从同一 generation k 只读
    -> 每个 owner 只写 generation k+1 中自己的范围
    -> 所有 macro 完成后执行全局屏障
    -> 原子发布 generation k+1
```

该全局轮次、状态文件和调度设计必须在引入真实求解器时另行评审；本次不提前实现。

### 14.4 Macro seam

最终每个 macro 只输出自身 ownership，覆盖没有空洞和正面积重叠，但跨 macro polygon
会在 seam 上成为多个子 Cell 内的片段。全局同层 merge/normalize 已记录在 `AGENTS.md`
未来优化栏目中；只有证明全版内存或流式算法可行后再实施。

## 15. 最终完成标准

只有同时满足以下条件才能称为完成：

- size/count 两种 macro 模式均实现且含完整测试；
- size 模式名义 macro 必须是 core 整数倍，21/11/1 示例得到 11+10；
- count 2×2 按 core 数平衡分配，而不是几何硬等分；
- macro/core ownership 无空洞、无正面积重叠；
- 固定 canvas 和单边光学半径规则严格执行；
- 一个 problem 包含多个 core，并完成逐 core 栅格化；
- macro 边界没有成为可移动边；
- 跨 macro/core 的矩形、斜边、孔洞和窄环零位移 XOR 为零；
- problem/result 可独立保存和读取；
- 主流程可直接从 TOML + GDS 运行到完整目标层 GDS/OASIS；
- 执行时只常驻当前 macro/problem 和当前 core canvas；
- 旧 macro/builder/problem 路线和无调用方包装全部删除；
- 所有新增/修改 Python 文件和函数均有中文 docstring，关键内部逻辑有详细中文注释；
- 测试、开发报告、测试报告和手册同步完成；
- 完成全局简化审计，没有因修 bug 留下多余函数、分支或重复字段；
- 只做本地提交，不推送远端，不修改 `00_PAST/` 和用户数据。
