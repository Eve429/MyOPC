# 最简 MB-OPC 迁移计划

> 状态：**等待用户审查，禁止开始实施**
> 日期：2026-08-16
> 本文件是本次最简 MB-OPC 迁移的唯一实施依据。重开上下文后，只需读取本文件、
> 仓库根 `AGENTS.md`、已批准的光刻迁移计划和实际代码即可实施。若实际代码、依赖环境、
> 用户工作树或光刻迁移结果与本文记录不一致，必须先报告差异，不得自行改变接口或扩大范围。

## 1. 本次目标

本次迁移 OpenILT `opc/simpleopc.py` 和只读归档
`00_PAST/opc/iteration/mbopc/solver.py` 中最基础的 MB-OPC 思想，但不原样复制旧实现。
目标是完成一个可以从 GDS/OASIS 直接运行的、固定步长、EPE 驱动的离散边移动方法：

1. 复用当前 Macro–Core 两级网格和每 macro 持久化 `MacroProblem`；
2. 复用已经分段的参考 `SegmentBatch`，迭代期间只更新一维法向位移向量；
3. 每个 core 使用 target mask、当前 mask 和 ICCAD13 光刻结果计算 EPE 移动方向；
4. inner 违规时沿公共外法向外移，outer 违规时内移，同时违规时不移动并记录歧义；
5. 每个 macro 独立完成自己的全部迭代；macro 内全部 core 完成当前轮评价后才发布该 macro 的下一状态；
6. L2 和 PVBand 作为诊断指标累计，但不改变 EPE 驱动方法的最佳状态选择；
7. 支持 CPU/CUDA，GPU 只保存当前 batch，禁止整张 reticle tensor 常驻；
8. 每个 macro 只保存自身 baseline、逐轮记录、最佳位移和最终候选 GDS；全部 macro 求解完成后只做一次全局 merge；
9. 最终输出由各 macro 已评价最佳状态组成的 ownership 权威覆盖，并明确它不是逐轮交换 context 的全局同步最优；
10. 新增单 macro 多 tile、以及多 macro 多 tile 两个可直接运行的 main；
11. 完成自动测试、真实光刻集成、开发/测试报告和简化审计；
12. 目录和接口允许后续增加梯度 MB-OPC 与 ILT，但不为未来方法写空实现。

“最简 MB-OPC”特指：

```text
固定参考边段
    -> 当前状态光刻仿真
    -> 固定 inner/outer EPE 探针
    -> {-1, 0, +1} 离散方向
    -> 固定/衰减步长更新绝对法向位移
    -> macro 内 tile 屏障
```

它不是梯度 MB-OPC，不优化连续像素参数，不添加 SRAF，也不保证 EPE 单调下降。

## 2. 本轮明确不做的内容

- 不迁移梯度 MB-OPC、DiffOPC、simple ILT、level-set ILT、multilevel ILT；
- 不迁移 OpenILT 的整张曝光图拼接、共享可变 Python list 和 Manhattan 专用端点修改；
- 不建立 `OPCMethod`、求解器注册器、工厂、插件、Worker、任务队列或分布式 RPC；
- 不定义强迫 MB-OPC、梯度 MB-OPC 和 ILT 使用相同优化变量的统一 Problem；
- 不让 ILT 依赖 `SegmentBatch`；ILT 的优化变量是像素/水平集；
- 不添加 SRAF，不修改迭代中的 segment 数量，不设计动态边段 ID；
- 不实现断点恢复、失败重试、跨机器调度或自动 batch 调参；
- 第一版不在每轮结束后合并全部 macro，也不让相邻 macro 交换上一轮位移；
- 不保存整张 reticle 的 nominal/max/min tensor；
- 不迁移旧 `ArrayTileCache` 独立文件，也不新增通用缓存框架；
- 不迁移 `estimate_rectangular_shots`，本轮算法不消费 shot 指标；
- 不修改 `layout/`、`geometry/`、`00_PAST/` 和用户 GDS；
- 不顺带修复“边恰好压在内部 macro 切线”的已知退化问题；
- 不宣称当前 `context_nm` 已由 35×35 频域 kernel 推导为充分光学范围。

## 3. 已核对事实与不确定性

### 3.1 当前可直接复用的接口

```text
opc.input.grid.MacroSpec/CoreSpec
opc.input.edge.MacroProblem
opc.input.edge.SegmentBatch
opc.input.edge.prepare_macro_problem()
opc.input.edge.edge_probe_points()
opc.input.edge.reconstruct_contours()/reconstruct_region()
opc.input.raster.rasterize_mask_canvas()/ownership_canvas()
geometry.PatchWriter.write_macro_results()
main/run_macro_pipeline.py
```

`MacroProblem` 已保存固定参考轮廓、参数化 segment、每段唯一 owner、每 core context
membership CSR，以及 macro/core/context/pixel/canvas/极性/位移上限。本轮不需要修改
`MacroProblem`、`SegmentBatch` 或 Problem NPZ 格式。

### 3.2 从归档保留的语义

- 全部 batch 读取同一份 `current`，方向先写 `next`；
- 全部 batch 完成后才发布下一状态，属于同步 Jacobi 更新；
- owner segment 产生方向，context segment 只参与光学上下文；
- target canvas 用有界 `uint8` 缓存，送设备时才转 `float32`；
- nominal、dose_max、defocus_min 一次 `forward_many()`；
- EPE 选择最佳状态，L2/PVBand 只诊断；
- 最后一次已发布状态必须额外评价；
- 候选必须经过 `reconstruct_region()` 的方向、hole 和有效性检查。

### 3.3 不能原样迁移的内容

旧求解器依赖已删除的 `MBOPCProblem.physical_mask/grid/config`、
`RectilinearCoreGrid.cores()` 和旧 raster 签名。旧 `_subset_contours()`、
`_polygon_ids_for_core()`、`_current_tile()` 用“参考 tile + polygon 差分”构造当前 mask。
第一版不迁移这条复杂快路：每个 macro 直接从自己的当前重建 Region 栅格化，边界 context
使用该 MacroProblem 中固定的参考图形。性能优化必须等基准证明瓶颈后进行。

### 3.4 主动暴露的不确定性

- `context_nm` 是否达到光学收敛尚未证明；
- 离散 EPE 更新是启发式算法，不能承诺任意版图收敛；
- 相邻 macro 的同边 segment 可能得到不同方向并形成真实 jog，本轮不做位移平滑；
- 多 macro 第一版在各自全部迭代期间不交换已移动 context，因此边界 core 使用的是邻区参考几何；这是用户为减少每轮全局 merge 开销选择的明确速度/边界精度权衡，不能在报告中描述成全局同步 MB-OPC；
- OpenILT 原始实现只移动 Manhattan 边；本项目使用已有单位外法向支持斜边，属于项目实现；
- 光刻迁移计划尚未实施；若最终 API 不符合本文件第 8 节，必须先停下对齐计划。

## 4. 总体架构分析

### 4.1 三类方法共有

```text
GDS/OASIS
  -> layout / ROI / Macro–Core 网格
  -> 局部透光率 canvas
  -> LithographyModel.forward/forward_many
  -> evaluation 指标
  -> macro 结果写出与最终 GDS 合并
```

### 4.2 边段型方法共有

```text
MacroProblem
  -> SegmentBatch
  -> owner/membership
  -> 一维 segment displacement
  -> reconstruct_region
```

简单 MB-OPC 与未来梯度 MB-OPC 共有这些；ILT 不经过这一层。

### 4.3 必须独立的优化部分

| 方法 | 优化变量 | 栅格路径 | 更新方式 |
|---|---|---|---|
| 简单 MB-OPC | NumPy `float64[S]` 位移 | KLayout Region 硬栅格 | EPE `-1/0/+1` 固定步长 |
| 梯度 MB-OPC | Torch `float32[S]` 位移 | 可微软边栅格 | autograd + optimizer |
| ILT | Torch `[H,W]` 像素/水平集 | 参数形成 mask | autograd + optimizer |

因此本轮只增加薄光刻模型 Protocol，不增加统一求解器接口。单/多 macro 两个 main 共用
应用流程函数，但不通过注册器选择方法；未来 ILT 只替换 Problem 准备与 solve 调用。

### 4.4 目录结论

```text
lithography/                    # 光刻模型一级目录
evaluation/                     # 评价一级目录
opc/input/                      # 方法无关网格/栅格
opc/input/edge/                 # 边段型共有输入与重建
opc/iteration/mbopc/            # 本次离散 EPE 方法
opc/iteration/diffopc/          # 未来目录，本轮不创建
opc/iteration/ilt/              # 未来目录，本轮不创建
main/                           # 直接入口、进度和 macro 结果编排
```

## 5. 第一版独立 macro 迭代与最终合并

### 5.1 本轮采用的执行顺序

```text
macro 0：baseline -> round 1 -> ... -> round N -> 保存局部 best GDS
macro 1：baseline -> round 1 -> ... -> round N -> 保存局部 best GDS
...
全部 macro 完成
    -> merge_macro_results(...)
    -> 最终全局 GDS
```

每个 macro 内仍保持严格同步：同一轮所有 tile 只读同一份 current，方向先写 next，全部 tile
完成后才发布该 macro 的下一状态。不同 macro 之间不设逐轮屏障、不写中间全局 GDS。

### 5.2 边界 context 的明确代价

`owner=-1` 的 segment 是当前 macro 的只读参考副本。独立迭代期间，它不会同步邻 macro 的
最新位移。因此：

- macro 内部 core 边界仍由 owner/membership 正确处理；
- macro 边界 core 的光刻 context 使用邻区零位移参考几何；
- 最终 merge 能消除 ownership 表示 seam，但不能补算缺失的邻 macro 动态光学影响；
- 多 macro 结果不要求与“单 macro 覆盖整 ROI”逐位一致；二者差异必须在测试报告量化。

这是第一版为避免每轮全局 merge/I/O 做出的明确取舍，不把它隐藏成当前能力。

### 5.3 独立 merge 函数与未来切换

最终几何只通过一个公共函数：

```python
def merge_macro_results(
    plan: dict,
    macro_gds_paths: Mapping[str, Path],
    output_path: Path,
    *,
    cell_mode: Literal["single_cell", "macro_cells"],
) -> Path:
    """选择各 macro ownership 权威覆盖并写出一个全局结果。"""
    ...
```

第一版在所有 macro 全部迭代完成后调用一次。未来改成逐轮同步时，复用同一函数对每轮的
macro GDS 调用；需要改的是 orchestration 的调用时机和 context 来源，不改 merge 的几何实现。

### 5.4 round 指标语义

baseline 先评价零位移状态，用它的方向产生 Round 1 位移；Round 1 候选通过重建后立即执行
光刻和 evaluation，得到的才是 `records[1]`。因此用户看到的 Round N 指标始终属于第 N 次
位移之后的几何。该评价同时产生下一轮方向，不会额外重复一次光刻。

底层 `evaluate_and_propose()` 仍遵守严格契约：返回指标属于函数刚刚评价的输入状态，返回
的 next 只是提案。上层把“移动后的下一次评价”登记为本轮结果，二者并不矛盾。

## 6. 输入、输出与不变量

### 6.1 单 macro 状态评价输入

```text
problem                 MacroProblem，固定参考几何/owner/membership
current_region          当前 macro 位移重建出的完整候选 Region
current_displacements   float64[S] 当前绝对位移
model                   LithographyModel
config                  SimpleMBOPCConfig
step_dbu                当前提案步长；仅评价时为 0
target_cache            有显式字节上限的 target uint8 LRU
```

### 6.2 底层状态评价输出

```text
next_displacements      float64[S]
epe                     当前状态 owner EPE 违规段数
l2                      当前状态 ownership 二值差异数
pvband                  当前状态 ownership 工艺带差异数
valid_probes            有效 owner 探针数
ambiguous_probes        inner/outer 同时违规数
moved_segments          提案相对 current 改变的 owner segment 数
```

这是底层函数的因果关系，不直接作为用户看到的 Round N 记录。`optimize_macro()` 会重建并
评价 next，随后把该次评价记为移动后的 Round N 指标；baseline 单独标记为 Round 0。

### 6.3 不变量

- current 长度必须等于 segment 数，且全部有限；
- `owner_indices==-1` 的 context 位移始终为 0；
- 每个 owner segment 每状态至多产生一次方向；
- 所有 batch 只读 current，不能读本轮 next；
- 提案裁到 `±max_displacement_dbu`；
- macro 内候选合法后才能发布为该 macro 下一状态；
- 任一候选非法时只终止并回滚当前 macro，错误原因不得吞掉；
- EPE 相同保留该 macro 较早状态，L2/PVBand 不打破平局；
- context/padding 不参与 L2/PVBand 重复计分；
- probe 只由 owner segment 产生；
- 数组行 0 是最低 Y，张量索引为 `[batch,y,x]`。

## 7. EPE 方向与居中坐标

参考 segment 法向统一为“透光区 → 不透光区”：

```text
inner = midpoint - normal * epe_distance_dbu
outer = midpoint + normal * epe_distance_dbu
```

| nominal 探针情况 | 含义 | direction |
|---|---|---:|
| inner 未打印、outer 未打印 | 印刷不足 | `+1` 外移 |
| inner 已打印、outer 已打印 | 印刷过量 | `-1` 内移 |
| inner 未打印、outer 已打印 | 双向冲突 | `0`，ambiguous |
| inner 已打印、outer 未打印 | 无违规 | `0` |

target 上只有 inner 透光、outer 不透光、二者不同且都在 canvas 内的探针才有效。
clear/opaque 不在求解器分支；法向已经由 `fragment_edges()` 统一。

旧求解器漏掉了 canvas 居中 padding。正确坐标为：

```text
x_canvas = (x_dbu - context.left) / pixel_dbu - 0.5 + low_x_padding
y_canvas = (y_dbu - context.bottom) / pixel_dbu - 0.5 + low_y_padding
```

本轮在 `opc/input/raster.py` 增加唯一公开换算函数，未来梯度 MB-OPC 也复用它。

## 8. 光刻模型与 evaluation 契约

### 8.1 `LithographyModel` 此时才合理

独立迁移 ICCAD13 时只有一个模型且没有通用消费者，所以光刻计划没有建立抽象。本轮
`evaluate_and_propose()` 成为第一个真实求解器调用方，Protocol 有当前实现和当前调用者，不再是
一次性抽象。它只描述求解器消费的能力，不包含 kernel、资产路径或 FFT 细节。

新增 `lithography/contracts.py`：

```python
@runtime_checkable
class LithographyConfigView(Protocol):
    """暴露求解器所需 canvas 和二值阈值。"""
    canvas: int
    print_threshold: float


@runtime_checkable
class LithographyModel(Protocol):
    """描述边段 OPC 与未来 ILT 消费的最小批量可微光刻接口。"""

    @property
    def device(self) -> torch.device: ...

    @property
    def config(self) -> LithographyConfigView: ...

    def condition(self, name: str) -> ProcessCondition: ...

    def forward_many(
        self, mask: torch.Tensor,
        conditions: Sequence[ProcessCondition],
    ) -> dict[str, torch.Tensor]: ...
```

`lithography/__init__.py` 增加 `LithographyModel` 导出；`LithographyConfigView` 不导出。

### 8.2 `evaluation/metrics.py`

本轮只迁移当前求解器消费的内容：

```python
@dataclass(frozen=True, slots=True)
class EPEEvaluation:
    """保存逐边段 EPE 有效性、违规类型和法向方向。"""
    valid: torch.Tensor
    inner_violations: torch.Tensor
    outer_violations: torch.Tensor
    ambiguous: torch.Tensor
    directions: torch.Tensor

    @property
    def violation_count(self) -> int: ...


def evaluate_binary_l2(
    target: torch.Tensor, nominal: torch.Tensor, threshold: float = 0.5,
    ownership_mask: torch.Tensor | None = None,
) -> int: ...


def evaluate_pvband(
    maximum: torch.Tensor, minimum: torch.Tensor, threshold: float = 0.5,
    ownership_mask: torch.Tensor | None = None,
) -> int: ...


def evaluate_edge_probes(
    target: torch.Tensor, nominal: torch.Tensor,
    batch_indices: torch.Tensor, inner_xy: torch.Tensor, outer_xy: torch.Tensor,
    threshold: float = 0.5,
) -> EPEEvaluation: ...
```

保留私有 `_aligned_images()`、`_selected_pixels()`，因为各有多个真实调用点；不迁移 shot。

## 9. 类与结构体定义

### 9.1 `SimpleMBOPCConfig`

位置：`opc/iteration/mbopc/simple.py`

```python
@dataclass(frozen=True, slots=True)
class SimpleMBOPCConfig:
    """保存已经转换到 DBU 的离散 EPE 迭代参数。"""
    iterations: int              # 最多发布更新次数
    initial_step_dbu: float      # 初始绝对法向步长
    decay_every: int             # 步长减半周期
    epe_distance_dbu: float      # inner/outer 探针距离
    batch_size: int              # 一次 forward 的 core 数
    target_cache_bytes: int      # CPU target uint8 LRU 上限

    def __post_init__(self) -> None: ...
```

不重复保存 `pixel_dbu/canvas/max_displacement/threshold/layer/polarity`，这些已有唯一来源。

### 9.2 `TargetCanvasCache`

同样放在 `simple.py`，不再建 `_cache.py`：

```python
class TargetCanvasCache:
    """按显式字节上限保存跨状态复用的只读 uint8 target canvas。"""

    def __init__(self, max_bytes: int) -> None: ...

    def get(self, macro_id: str, core_index: int) -> NDArray[np.uint8] | None: ...

    def put(
        self, macro_id: str, core_index: int, value: NDArray[np.uint8],
    ) -> None: ...
```

缓存 key 必须包含 macro ID；0 上限禁用，单项超过上限不缓存，不改变 dtype。

### 9.3 `SimpleMBOPCStep`

```python
@dataclass(frozen=True, slots=True)
class SimpleMBOPCStep:
    """保存一个 macro 已评价状态的指标及下一状态提案。"""
    next_displacements: NDArray[np.float64]
    epe: int
    l2: int
    pvband: int
    valid_probes: int
    ambiguous_probes: int
    moved_segments: int
```

### 9.4 `IterationRecord`

```python
@dataclass(frozen=True, slots=True)
class IterationRecord:
    """保存 baseline 或一次移动后状态的实际评价结果。"""
    round_index: int             # 0=baseline；1..N=对应位移完成后的状态
    step_dbu: float              # 产生本状态时使用的步长；baseline 为 0
    epe: int                     # 本状态实际 EPE
    l2: int                      # 本状态实际二值 L2
    pvband: int                  # 本状态实际 PVBand
    valid_probes: int            # 本状态有效探针
    ambiguous_probes: int        # 本状态歧义探针
    moved_segments: int          # 从上一状态移动到本状态的段数
    elapsed_seconds: float       # 重建并评价本状态的耗时
```

### 9.5 `SimpleMBOPCResult`

```python
@dataclass(frozen=True, slots=True)
class SimpleMBOPCResult:
    """保存单 macro 的最佳已评价位移、全部状态记录和停止原因。"""
    best_displacements: NDArray[np.float64]
    records: tuple[IterationRecord, ...]  # records[0] 固定为 baseline
    best_round: int                       # 0 表示零位移 baseline 最优
    stop_reason: str                      # zero_epe/no_update/invalid_geometry/iteration_limit
    stop_detail: str | None               # 非法候选的明确原因；正常停止为 None
```

改为独立 macro 完整迭代后，Record 和 Result 都有当前真实调用方，因此恢复这两个领域结构。
不再建立全局 state Result；多 macro main 只汇总各 macro 的现有 Result 到 summary。

## 10. 函数定义与核心算法

### 10.1 `opc/input/raster.py`

新增：

```python
def points_to_canvas(
    points_dbu: object, context_box: DbuBox,
    pixel_dbu: int, canvas_pixels: int,
) -> NDArray[np.float64]:
    """把全局 DBU 点转换为居中 canvas 的连续 `(x,y)` 像素坐标。"""
    ...
```

函数复用 `_center_padding()`，只换算，不 round、不 clip；越界和最近像素采样由 evaluation
处理。`ownership_canvas()` 签名和结果不变。

### 10.2 `opc/iteration/mbopc/simple.py`

公开两个算法函数：

```python
def evaluate_and_propose(
    problem: MacroProblem,
    current_region: kdb.Region,
    current_displacements: NDArray[np.float64],
    model: LithographyModel,
    config: SimpleMBOPCConfig,
    step_dbu: float,
    target_cache: TargetCanvasCache,
    *,
    can_update: bool,
    on_tiles_completed: Callable[[int], None] | None = None,
) -> SimpleMBOPCStep:
    """评价一个 macro 当前状态，并产生同步 owner 位移提案。"""
    ...


def optimize_macro(
    problem: MacroProblem,
    model: LithographyModel,
    config: SimpleMBOPCConfig,
    target_cache: TargetCanvasCache,
    *,
    on_tiles_completed: Callable[[int], None] | None = None,
) -> SimpleMBOPCResult:
    """让单个 macro 独立完成 baseline 和全部离散 EPE 迭代。"""
    ...
```

`evaluate_and_propose()` 按以下紧凑逻辑块实现，不拆出单调用点 batch 包装函数：

1. **入口契约**：校验位移 shape/有限/context=0、canvas 一致、step 语义；不重复校验
   `MacroProblem` 已保证的 owner/CSR。
2. **固定几何**：`segments.materialize()` 一次得到参考探针几何；core 按索引即时构造，
   不常驻 CoreSpec 列表、不建 polygon 子集。
3. **CPU batch**：预分配 target `uint8`、current mask `float32`、ownership `bool`。
   cache miss 时首次才用零位移重建 reference Region；current mask 直接栅格传入 Region。
4. **光刻**：target 转 device float32/255；在 `torch.no_grad()` 中一次
   `forward_many(nominal,dose_max,defocus_min)`。
5. **像素指标**：L2/PVBand 只在 ownership canvas 累计。
6. **EPE**：各 core 只取 owner segment，生成探针，经 `points_to_canvas()` 后批量评价；
   direction 只写 `next_values`，并由 `written[S]` 保证唯一写。
7. **释放和进度**：每 GPU batch 完成后只保留标量和方向，立即释放三张 printed 及输入
   张量；完成释放后才调用 `on_tiles_completed(batch_count)`，进度表示真正完成的 tile，
   不表示刚提交到 GPU 的 tile。
8. **出口**：核对写集、context=0、位移范围，返回指标和提案，不在 solver 发布 GDS。

立即累计指标不会提前移动边：`current_region/current_displacements` 在整个函数内只读，所有
方向只写独立 `next_values`。

`optimize_macro()` 的循环：

1. 零位移重建并执行 baseline 评价，写 `records[0]`；
2. 使用 baseline 返回的 next 重建 Round 1 候选；
3. 候选合法后评价它，把评价值写为 `records[1]`；
4. Round 1 评价同时产生 Round 2 next，继续循环；
5. 每轮只新增一次光刻评价，没有“移动前一次、移动后再重复一次”的额外计算；
6. EPE 更小才更新本 macro best，相同保留较早 round；
7. 候选非法时保留最后合法 best，停止原因写 `invalid_geometry` 并保留明确异常原因到 main summary；
8. 最多评价 baseline + N 个移动后状态。

`on_tiles_completed` 是当前 tqdm 需求的最小事件接口。迭代业务不导入 tqdm、不打印终端；
单/多 macro main 用同一回调显示进度，未来 ILT 也可按完成 tile 数调用相同形式的回调，
但本轮不为它定义基类或 Protocol。

## 11. 共享 Macro 应用流程

### 11.1 提取原因

当前 `main/run_macro_pipeline.py` 同时包含方法无关的 TOML/DBU/Problem 准备/GDS merge，
以及仅属于验证流程的固定 `+2/-2nm`。新增真实 MB-OPC 后已有两个当前调用者；复制代码或
让一个 runner 导入另一个 runner 的私有函数都不合理。因此只移动真实共有代码，不建立
未来注册器。

### 11.2 新增 `main/_macro_pipeline.py`

```python
@dataclass(frozen=True, slots=True)
class MacroPipelineConfig:
    """保存 Problem 准备和 macro 结果写出共同配置。"""
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
    work_dir: Path
    final_layout: Path
    final_cell_mode: Literal["single_cell", "macro_cells"]
```

公开函数：

```python
def load_macro_config(path: str | Path) -> MacroPipelineConfig: ...
def exact_dbu(value_nm: Decimal, dbu_nm: Decimal, name: str) -> int: ...
def atomic_write_json(path: Path, payload: dict) -> Path: ...
def prepare_problems(config: MacroPipelineConfig) -> dict: ...
def write_macro_gds(
    problem: MacroProblem, region: kdb.Region, path: Path, dbu_um: float,
) -> Path: ...
def merge_macro_results(
    plan: dict, macro_gds_paths: Mapping[str, Path], output_path: Path, *,
    cell_mode: Literal["single_cell", "macro_cells"],
) -> Path: ...
```

规则：代码从现有 runner 移动而非复制；`prepare_problems()` 不读取 `[iteration]`；
`merge_macro_results()` 不猜 result NPZ，也不关心调用发生在最终还是未来某一轮；各 runner
校验自己的结果文件；函数消费显式 `macro_id -> GDS Path`，不依赖某一种目录命名；不移动到
`opc.input`，因为 TOML/文件生命周期属于应用编排。`main/` 每一行继续有中文注释。

### 11.3 修改 `main/run_macro_pipeline.py`

删除并导入其 `PipelineConfig`、`exact_dbu`、`_atomic_write_json`、`prepare_problems`、
`_write_macro_gds` 和通用 merge 部分。保留：

```python
def load_validation_deltas(path: str | Path) -> tuple[Decimal, Decimal]: ...
def run_round(...): ...
def run(...): ...
def main(...): ...
```

已有 +2/-2 文件数量、位移 NPZ、错误语义和 gcd_45nm 最终 XOR 必须不变；验证 runner
把最终一轮 macro GDS 显式映射传给 `merge_macro_results()`。

## 12. 两个 main 与共享 MB-OPC 工作流

### 12.1 为什么是两个入口、一个共享工作流

新增两个直接运行入口：

```text
main/run_mbopc_single_macro.py    # 恰好一个 macro，macro 内多个 tile
main/run_mbopc_multi_macro.py     # 多个 macro，每个 macro 内多个 tile
```

两个入口只负责默认配置路径、命令行、异常转退出码和摘要打印；MB-OPC 共有业务放在
`main/_mbopc_workflow.py`。不再新增第三个 `main_test_mbopc.py`，避免三个入口重复。

以后迁移 ILT 时，layout/config/macro 规划/progress/最终 merge 的调用结构可以保持；方法相关的
Problem 准备和 solve 调用替换为 ILT 对应实现即可。本轮不为未来 ILT 建工厂或注册器。

### 12.2 `main/_mbopc_workflow.py` 运行配置

```python
@dataclass(frozen=True, slots=True)
class MBOPCRunConfig:
    """保存公共 Macro 配置和 simple MB-OPC 物理单位参数。"""
    pipeline: MacroPipelineConfig
    iterations: int
    initial_step_nm: Decimal
    decay_every: int
    epe_distance_nm: Decimal
    batch_size: int
    target_cache_mb: int
    device: str
    save_final_lithography: bool
    show_progress: bool


def load_config(path: str | Path) -> MBOPCRunConfig: ...
```

### 12.3 结果目录与 NPZ

```text
work_dir/
  plan.json
  problems/mr*.npz
  macros/
    mr0c0/
      result.npz
      best.gds
      metrics.json
    mr0c1/...
  final.gds
  final_lithography/
  summary.json
```

每个 `result.npz`：

```text
format_version            int32[1]
macro_id                  unicode[1]
best_round                int32[1]
best_displacements        float64[S]
stop_reason               unicode[1]
```

逐轮标量记录统一写 `metrics.json`；`records[0]` 是 baseline，`records[1]` 是第一次移动后的
评价，`stop_detail` 也只写 JSON（正常停止为 null）。NPZ 不重复保存 metrics；只用
`allow_pickle=False`，经同目录临时文件替换。

### 12.4 共享工作流函数

```python
def solve_macro(
    problem: MacroProblem,
    model: LithographyModel,
    config: SimpleMBOPCConfig,
    target_cache: TargetCanvasCache,
    output_dir: Path,
    *,
    show_progress: bool,
    progress_position: int,
    leave_progress: bool,
) -> tuple[SimpleMBOPCResult, Path]:
    """显示 tile 进度，让一个 macro 完成全部迭代并写出 best GDS。"""
    ...

def _run_mbopc(
    config_path: str | Path, *, require_multiple_macros: bool,
) -> dict:
    """执行两个入口共有的准备、逐 macro 求解、最终合并和结果保存。"""
    ...

def run_single_macro(config_path: str | Path) -> dict:
    """准备并求解恰好一个 macro、多个 tile，使用统一 merge 写最终结果。"""
    ...

def run_multi_macro(config_path: str | Path) -> dict:
    """逐个独立求解多个 macro，全部完成后只执行一次 merge。"""
    ...

def save_final_lithography(
    plan: dict, final_layout: Path, model: LithographyModel,
    batch_size: int, output_dir: Path,
) -> dict:
    """流式保存最终版图每 tile 的 nominal 连续/二值 PNG 和 manifest。"""
    ...
```

`_run_mbopc()` 有两个当前调用方，用布尔参数只表达入口已经确定的单/多 macro 约束，不选择
OPC/ILT 方法。`solve_macro()` 创建 `tqdm`，最大 total 为 `(iterations+1)×core_count`，因为
baseline 和每个移动后状态都要计算全部 tile。提前停止时按实际完成量关闭，不强行补到 100%。
`bar.update(batch_count)` 只通过 `on_tiles_completed` 在 batch 完成、张量释放后调用；自动
测试把 `show_progress` 设为 false，避免进度输出污染断言。

### 12.5 单 macro main

`main/run_mbopc_single_macro.py` 只定义 `main()`：加载配置并调用 `run_single_macro()`。
workflow 要求 `macro_count==1` 且 `core_count>1`，然后构造模型/cache、调用 `solve_macro()`，
最后把唯一 best GDS 交给 `merge_macro_results()`。单 macro 也走统一 merge 函数，用同一路径
验证 ownership 裁剪和写出，不维护第二套输出实现。

### 12.6 多 macro main

`main/run_mbopc_multi_macro.py` 只定义 `main()`：加载配置并调用 `run_multi_macro()`。
workflow 要求 `macro_count>1` 且每个 macro `core_count>1`，按稳定顺序逐 macro 完成全部迭代，
收集 `macro_id -> best.gds`，全部成功后只调用一次 `merge_macro_results()`。

多 macro 使用两层 tqdm：外层 `unit="macro"`，内层 `unit="tile"`；当前主机逐 macro 求解，
不并行运行多个 tile 进度条。候选几何非法属于已定义的算法停止：保留该 macro 较早 best 并
继续后续 macro；I/O、Torch、KLayout 和未知异常原样传播，不拿已完成 macro 拼半份最终 GDS。

### 12.7 main 对未来 ILT 的改动边界

两个入口文件本身不包含边段/EPE 逻辑。未来 ILT 新增自己的 workflow；若保持相同入口交互，
只替换 workflow import 和调用。公共 `_macro_pipeline.py` 的读取、网格描述、写 macro 和
`merge_macro_results()` 不改；ILT 不得伪造 MacroProblem。

### 12.8 最终光刻图案

`save_final_lithography=true` 时只对最终合并 GDS 运行一次；逐 macro/tile 流式 batch；每 tile
只保存 ownership 区域的 nominal 连续灰度 PNG 和阈值二值 PNG；manifest 记录 macro/tile、
ownership/context、pixel、padding 和文件名；每 batch 写完立即释放。完整 reticle 单张像素
拼图不在本轮实现。

## 13. 两个 main 的直接验证要求

两个入口都必须可从仓库外直接运行，不要求安装项目。自动测试生成小型 GDS 和临时 TOML：

- single：一个 macro、至少 2×2 tile，包含矩形、hole、斜边；
- multi：至少 2×2 macro，每个 macro 至少 2×2 tile，包含跨 macro polygon；
- 默认 CPU，可用时另测 CUDA；
- 展示 macro/tile 进度、baseline、移动后每轮指标、局部 best、最终 merge、GDS/PNG、耗时/内存；
- 未处理层不得进入输出。

运行：

```powershell
D:\app\miniforge\envs\myopc\python.exe main\run_mbopc_single_macro.py config\mbopc_single_macro.toml
D:\app\miniforge\envs\myopc\python.exe main\run_mbopc_multi_macro.py config\mbopc_multi_macro.toml
```

## 14. 配置文件

新增两个字段相同、macro 划分不同的配置：

```text
config/mbopc_single_macro.toml   # macro_grid=[1,1]
config/mbopc_multi_macro.toml    # 默认 macro_grid=[2,2]
```

以下为多 macro 示例：

```toml
[input]
layout = "../TestReticle/gcd_45nm.gds"
top_cell = "TOP"
layer = 11
datatype = 0
polarity = "clear"

[grid]
macro_grid = [2, 2]
# macro_size_nm = 4096
core_size_nm = 1024
context_nm = 400

[lithography]
pixel_nm = 8
canvas_pixels = 256

[edge]
corner_nm = 16
segment_nm = 32
max_displacement_nm = 24
miter_limit = 4.0

[mbopc]
iterations = 8
initial_step_nm = 8
decay_every = 4
epe_distance_nm = 16
batch_size = 8
target_cache_mb = 512
device = "auto"
save_final_lithography = true
show_progress = true

[output]
work_dir = "../output/mbopc_multi_macro"
final_layout = "../output/mbopc_multi_macro/gcd_45nm_mbopc.gds"
final_cell_mode = "single_cell"
```

新增校验：step/epe distance 精确换算 DBU；step≤max displacement；epe distance≤context；
iterations/decay/batch>0；cache≥0；device 只接受 auto/cpu/cuda[:index]；show_progress 为 bool；
single/multi main 分别校验自身 macro/tile 数量。

## 15. 文件级改动清单

### 15.1 新增文件

| 文件 | 类型 | 核心内容 |
|---|---|---|
| `evaluation/__init__.py` | 业务代码·包导出 | 导出 EPEEvaluation 和三项指标。 |
| `evaluation/metrics.py` | 业务代码·评价 | L2、PVBand、EPE 探针评价。 |
| `lithography/contracts.py` | 业务代码·接口契约 | 首个真实求解器需要的薄模型 Protocol。 |
| `opc/iteration/__init__.py` | 业务代码·包导出 | 中文模块 docstring，不建注册器。 |
| `opc/iteration/mbopc/__init__.py` | 业务代码·包导出 | 导出 Config、Cache、Step、Record、Result 和两个算法函数。 |
| `opc/iteration/mbopc/simple.py` | 业务代码·迭代算法 | 状态评价、提案和单 macro 完整离散 EPE 迭代。 |
| `main/_macro_pipeline.py` | 应用代码·公共编排 | 公共配置、Problem 准备、macro GDS、独立 merge 函数。 |
| `main/_mbopc_workflow.py` | 应用代码·MB-OPC 编排 | 两个入口共享的 solve、tqdm、结果和最终光刻输出。 |
| `main/run_mbopc_single_macro.py` | 运行入口 | 一个 macro、多个 tile 的直接运行 main。 |
| `main/run_mbopc_multi_macro.py` | 运行入口 | 多个 macro、每 macro 多 tile 的直接运行 main。 |
| `config/mbopc_single_macro.toml` | 配置文件 | 单 macro 默认运行参数。 |
| `config/mbopc_multi_macro.toml` | 配置文件 | 多 macro 默认运行参数。 |
| `doc/opc/mbopc_migration_design.md` | 设计文档 | 本文件；作为审核通过后的唯一实施依据。 |
| `tests/evaluation/__init__.py` | 测试代码·包标记 | 中文模块 docstring。 |
| `tests/evaluation/test_metrics.py` | 测试代码·单元测试 | 指标与方向测试。 |
| `tests/opc/iteration/__init__.py` | 测试代码·包标记 | 中文模块 docstring。 |
| `tests/opc/iteration/test_simple_mbopc.py` | 测试代码·算法测试 | solver、round 后指标、进度回调和真实模型测试。 |
| `tests/main/test_mbopc_runners.py` | 测试代码·端到端测试 | 单/多 macro 两个入口、最终一次 merge 和直接运行。 |
| `doc/opc/mbopc_development_report.md` | 开发报告 | 实施、偏差、性能、简化审计。 |
| `doc/opc/mbopc_test_report.md` | 测试报告 | 命令、场景、CPU/CUDA、coverage。 |

### 15.2 修改文件

| 文件 | 类型 | 核心改动 |
|---|---|---|
| `lithography/__init__.py` | 业务代码·包导出 | 增加 `LithographyModel` 导出，不改 ICCAD13 数值。 |
| `opc/input/raster.py` | 业务代码·坐标工具 | 增加 `points_to_canvas()`，不改变已有栅格结果。 |
| `opc/input/__init__.py` | 业务代码·包导出 | 导出新坐标函数。 |
| `main/run_macro_pipeline.py` | 运行入口·现有回归 | 移动共有代码并调用独立 merge，验证行为不变。 |
| 现有 raster 测试文件 | 测试代码·回归测试 | 增加 padding 坐标回归，不建单测试文件。 |
| `tests/main/test_macro_pipeline.py` | 测试代码·回归测试 | 锁定重构前后 +2/-2 和 XOR。 |
| `requirements.txt` | 依赖清单 | 增加 `tqdm`；不因进度条引入其他 UI 库。 |
| `doc/development_manual.md` | 开发手册 | 增加独立 macro 语义、接口、目录、进度和运行说明。 |
| `doc/test_manual.md` | 测试手册 | 增加两个 main、测试和 smoke 命令。 |
| `task_plan.md` | 项目计划 | lithography/evaluation 后增加 simple MB-OPC 子阶段。 |
| `findings.md` | 研究记录 | 记录独立 macro context 取舍、坐标和性能事实。 |
| `progress.md` | 进度记录 | 记录提交、测试、smoke 和偏差。 |

### 15.3 明确不修改

```text
00_PAST/**
layout/**
geometry/**
opc/input/edge/problem.py
opc/input/edge/fragmentation.py
opc/input/edge/reconstruction.py
用户 GDS
output/
.vscode/
```

若实施发现必须修改 `layout/` 或 `geometry/`，立即停止并说明最小改法，由用户逐次确认。

## 16. 自动测试矩阵

### 16.1 evaluation 与坐标

- `[H,W]`/`[B,H,W]`；ownership 外不计 L2/PVBand；
- inner-only `+1`、outer-only `-1`、ambiguous `0`、无违规 `0`；
- 越界、inner=outer、target 内外语义不成立为 invalid；
- shape/device 不一致明确失败；
- 无 padding、偶数 padding、奇数余量、右上像素中心；
- `points_to_canvas` 与 `ownership_canvas` 一致，x/y 不交换。

### 16.2 cache 与 solver

- cache hit、替换、LRU 驱逐、单项超限、0 禁用、跨 macro key；
- target 保持 uint8；cache hit 后不重复栅格 target；
- 单 core 矩形向外/向内/ambiguous；context 不移动；
- 两 core 同步读取；batch_size=1/2 结果相同；
- owner 恰写一次；最终只评价态不提案；位移 clip；
- 每 batch 只调用一次三条件 forward_many；
- L2/PVBand 不影响 direction；居中 padding 探针正确；
- 空 macro；真实 ICCAD13 CPU；CUDA 可用时 direct path。

### 16.3 图形矩阵

- 普通矩形；
- 2nm 窄壁中空图形、8nm 探针；
- 外线可能越过内线的 hole 候选；
- 左线可能越过右线的矩形候选；
- 凹多边形、多 polygon、多 hole；
- 45° 斜边、斜边跨 core/macro；
- 一个 polygon 横跨至少三个 core；
- SREF/AREF 展开后跨 macro；
- clear/opaque。

非法候选必须停止当前 macro、保留最后合法 best，并以 `invalid_geometry` 明确记录；多 macro
流程仍可用该 best 继续，但 summary 必须暴露停止原因。代表测试保存边段/owner/probe/方向
PNG 到 pytest 临时目录并在报告引用；默认测试不得写仓库 `output/`。

### 16.4 独立 macro、round 语义和最终 merge

- 每个 macro 内全部 tile 同轮读取同一 current，不能看到本轮已生成的 next；
- 多 macro 第一版不读取邻 macro 已移动状态，边界 context 固定为参考几何；
- `records[0]` 是 baseline，`records[1]` 必须是第一次移动后的实际评价；
- `iterations=1` 恰好包含 baseline 和一次移动后评价；
- 指标改善/恶化与记录对应 GDS/位移一致；
- macro 正序/逆序产生相同局部结果，最终 merge 后 XOR=0；
- batch 大小不改变位移；
- 一个 macro `invalid_geometry` 保留其较早 best，并在 summary 明确记录；
- 所有 macro 完成前不得调用最终 merge；
- single 和 multi 都只调用 `merge_macro_results()`，multi 全流程调用恰好一次；
- 多 macro 与单 macro 整 ROI 结果允许不同，但差异面积和边界带位置必须被量化；
- zero_epe/no_update/invalid_geometry/iteration_limit 停止原因准确；
- tqdm 每个完成 batch 按真实 tile 数更新，提前停止不伪造 100%。

### 16.5 现有流程和直接运行

- `run_macro_pipeline.py` +2/-2 输出、文件数、XOR 不变；
- 全量 layout/geometry/opc.input/main 回归；
- subprocess 从仓库外分别运行 single/multi 两个 main；
- 检查 summary、每 macro result/best GDS、最终 GDS、PNG manifest；不要求 pip install；
- 捕获终端输出证明进度描述包含 macro ID、tile 单位和实际完成数；关闭进度时无 tqdm 输出。

## 17. 性能与内存

CPU：一次只加载一个 MacroProblem；当前 Region 由该 macro 位移在内存重建，不写每轮全局
GDS；target `uint8` cache 严格有界；current mask 只保留当前 batch；不复制 polygon/edge/
segment 为 Python 对象列表。KLayout 第一版仍逐 tile 栅格，这是明确基线。

GPU batch 主要包含 target/mask/ownership、三张 printed 和光刻临时场。simple MB-OPC 使用
`no_grad()`，每 batch 立即释放，不保存整张 reticle。

报告必须记录：准备、每 macro baseline/round 的 raster/lithography/evaluation/reconstruct、
最终一次 merge 时间，cache hit/miss/peak，tqdm 实际 tile 数，RSS、CUDA peak，segment/
membership/tile/macro 数，以及 gcd_45nm 单/多 macro 数据。本轮不预设速度达标数字。

## 18. 实施阶段与提交

只有用户明确批准后开始；每阶段检查用户工作树，只做本地 commit，不推送、不提交用户数据。

### 阶段 A：前置模块与契约

完成 ICCAD13 迁移；迁移最小 evaluation；新增有当前调用方的 LithographyModel；前置测试
全绿后才能进入下一阶段。

建议提交：

```text
feat(evaluation): 迁移 simple MB-OPC 所需指标
feat(lithography): 增加首个求解器消费的模型契约
```

### 阶段 B：居中坐标

新增 `points_to_canvas()` 和奇偶 padding/像素中心回归，证明已有 raster 逐值不变。

```text
feat(opc-input): 统一居中画布点坐标换算
```

### 阶段 C：共享 Macro 生命周期

移动通用配置/准备/写出/merge；运行现有全测和 gcd_45nm 回零 smoke；任何行为变化都不得
进入阶段 D。

```text
refactor(main): 提取两个真实流程共用的 macro 生命周期
```

### 阶段 D：单 macro 完整优化

实现 Config/Cache/Step/IterationRecord/Result、`evaluate_and_propose()` 和 `optimize_macro()`；
完成初态评价、移动后评价、同步 batch、进度回调、图形合法性和真实模型测试；审计无旧差分
快路、重复坐标和未使用 helper。此阶段只处理一个 macro，不读取或合并其他 macro 的中间状态。

```text
feat(mbopc): 实现同步 EPE 驱动的单 macro 完整优化
```

### 阶段 E：两个主流程与最终合并

实现共享 `_mbopc_workflow.py`、单 macro 入口、多 macro 入口、各自配置、macro/tile 进度条、
per-macro best、独立 `merge_macro_results()`、final GDS/PNG。多 macro 流程中，每个 macro
独立完成全部迭代并写出 best，全部结束后只合并一次；测试 macro 执行顺序不影响最终物理覆盖，
并明确量化边界 context 固定在参考状态带来的精度限制。

```text
feat(main): 增加单宏与多宏 simple MB-OPC 入口
```

### 阶段 F：验证与报告

直接运行 `run_mbopc_single_macro.py` 和 `run_mbopc_multi_macro.py`；CPU 全量、可用时 CUDA；
gcd_45nm 至少完成初态评价和一次合法更新尝试，时间允许再跑默认全部迭代；更新手册/报告/
规划；做差异、未调用函数、重复实现、异常入口、coverage 和 bug 补偿逻辑审计。

```text
test(mbopc): 完成端到端验证与迁移报告
```

## 19. 验收命令

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\evaluation
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\opc\iteration
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\main\test_mbopc_runners.py
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m ruff check evaluation lithography opc main tests
D:\app\miniforge\envs\myopc\python.exe -m compileall -q evaluation lithography opc main tests
D:\app\miniforge\envs\myopc\python.exe main\run_mbopc_single_macro.py config\mbopc_single_macro.toml
D:\app\miniforge\envs\myopc\python.exe main\run_mbopc_multi_macro.py config\mbopc_multi_macro.toml
```

coverage 必须命中三种 EPE/ambiguous、cache 全路径、初态与移动后指标、进度回调、合法发布/
非法候选停止、最终只合并一次、固定边界 context、CPU real model 和可用时 CUDA；不以总
百分比掩盖关键分支。

## 20. 报告要求

开发报告记录：实际文件、计划偏差、OpenILT/归档取舍、独立 macro 策略及其边界精度取舍、
Protocol 引入理由、Config/Cache/Step/IterationRecord/Result 的当前调用方和必要性、
`merge_macro_results()` 的独立边界、旧差分快路未迁原因、性能、异常边界、简化审计、是否
修改 layout/geometry（应为否）、本地 commit 和未推送。

测试报告记录：环境版本、全部命令、图形矩阵、baseline/每轮移动后指标/best、batch 与 macro
顺序不变性、进度条实际计数、最终合并调用次数、hole/矩形越界拒绝、斜边/跨边界图、固定
边界 context 与同步全局 context 的差异样例、clear/opaque、真实 ICCAD13 CPU/CUDA、
gcd_45nm 规模/耗时/内存/停止原因/产物，以及诚实的未覆盖分支。

## 21. 最终调用关系

```text
main/run_mbopc_single_macro.py::main()
└─ run_single_macro(config_path)
   └─ _run_mbopc(config_path, require_multiple_macros=False)
      ├─ load_config() -> load_macro_config()
      ├─ prepare_problems() -> prepare_macro_problem()   # 恰好一个 macro
      ├─ ICCAD13Lithography(...)
      ├─ solve_macro(problem, model, progress)
      │  └─ optimize_macro(...)
      │     ├─ evaluate_and_propose(zeros)               # baseline/round 0
      │     │  ├─ target/current/ownership canvas
      │     │  ├─ model.forward_many()
      │     │  ├─ L2/PVBand
      │     │  ├─ edge_probe_points()
      │     │  ├─ points_to_canvas()
      │     │  └─ evaluate_edge_probes()
      │     └─ for round_i
      │        ├─ reconstruct_region(next)
      │        ├─ evaluate_and_propose(candidate)        # round_i 移动后指标
      │        └─ 更新 macro best/下一轮提案
      ├─ write_macro_gds(best)
      ├─ merge_macro_results(plan, {id: best}, final)    # 统一 ownership/写出路径
      ├─ save_final_lithography()
      └─ atomic_write_json(summary.json)

main/run_mbopc_multi_macro.py::main()
└─ run_multi_macro(config_path)
   └─ _run_mbopc(config_path, require_multiple_macros=True)
      ├─ load_config() -> load_macro_config()
      ├─ prepare_problems() -> prepare_macro_problem()   # 每 macro 一次
      ├─ ICCAD13Lithography(...)
      ├─ for macro                                      # macro 间不交换中间状态
      │  ├─ 建立该 macro 的 tile tqdm
      │  ├─ solve_macro() -> optimize_macro()           # 独立完成全部轮次
      │  └─ write_macro_gds(best)
      ├─ merge_macro_results(plan, best_paths, final)   # 全部完成后只调用一次
      ├─ save_final_lithography()
      └─ atomic_write_json(summary.json)
```

依赖保持：`layout -> geometry -> opc.input -> opc.input.edge -> opc.iteration.mbopc`；
iteration 可消费顶层 lithography/evaluation；基础层不得反向依赖具体方法；main 只做应用编排。

## 22. 后续兼容性边界

梯度 MB-OPC 可复用 MacroProblem、owner/membership、points_to_canvas、光刻/evaluation 和
`merge_macro_results()`；需独立实现可微软 raster、Torch 参数、optimizer、loss、Problem
扩展数据和自己的 Config。不得继承 simple Config 或调用 simple step。

ILT 可复用配置加载、macro/core 网格、canvas、光刻/evaluation、进度显示和最终合并生命周期；
不能复用 segment/owner/EPE 重建，应有自己的 raster Problem/state 和 workflow。两个 MB-OPC
入口保持薄层；若以后希望同一入口切换 ILT，只替换 workflow 的导入与调用，不在本轮增加
注册器或工厂。

如果以后确认需要“每轮同步跨 macro 的动态光学 context”，编排层可在每轮收齐各 macro
候选后调用同一个 `merge_macro_results()`，再从合并状态生成下一轮 context；该升级会改变
调用时机和 context 来源，不要求改写几何合并函数。第一版不实现这一同步模式，也不能宣称
边界结果等价于全版图同步求解。

动态 SRAF 会改变 polygon/segment 数，必须另行设计稳定 ID、owner/membership、state format
version 和跨 macro 发布。本轮不预留空字段假装支持。

## 23. 研究过程错误记录

| 错误 | 原因 | 处理 |
|---|---|---|
| 读取 `00_PAST/opc/input/problem.py` 失败 | 旧问题实际位于 `edge/builder.py` | 改读搜索确认的真实路径。 |
| Windows 下 `rg 00_PAST/main/*mbopc*.py` 失败 | ripgrep 未接收 PowerShell 通配展开 | 改用 `Get-ChildItem` 和精确路径。 |
| 首次完整计划补丁校验失败 | 一行遗漏新增行前缀 | 原文件未部分修改；记录后用修正补丁重建。 |
| 组合更新第 11～14 节补丁失败 | 同一补丁跨越已修改章节，旧上下文不再匹配 | 拆成按章节的小补丁；文件未发生部分修改。 |

## 24. 最终完成标准

- [ ] 用户明确批准本计划；
- [ ] 光刻 forward/backward/CPU/CUDA 测试通过；
- [ ] evaluation 三项指标完成；
- [ ] LithographyModel 有 ICCAD13 实现和当前 MB-OPC 调用方；
- [ ] points_to_canvas 修正 padding，原 raster 逐值不变；
- [ ] +2/-2 runner 提取前后行为不变；
- [ ] simple solver 只有 Config/Cache/Step/IterationRecord/Result 五个有当前调用方的必要结构；
- [ ] 无旧 MBOPCProblem/PhysicalMask/polygon 差分快路兼容层；
- [ ] 每个 macro 独立完成全部轮次，macro 内 tile 同轮只读同一 current；
- [ ] 多 macro 完成后只调用一次 `merge_macro_results()`，不做逐轮全局合并；
- [ ] 跨 macro 边界使用固定参考 context 的限制已在报告中量化，未宣称等价于全局同步；
- [ ] EPE 选全局 best，L2/PVBand 只诊断；
- [ ] round 0 是 baseline，round N 指标属于第 N 次位移后的候选，最终候选已完整评价；
- [ ] inner/outer/ambiguous 全测试；
- [ ] 矩形、窄壁 hole、越界、凹形、多 hole、斜边、跨 core/macro、SREF/AREF 覆盖；
- [ ] batch 和 macro 顺序不改变结果；
- [ ] CPU 真实 ICCAD13 端到端通过，可用时 CUDA 通过；
- [ ] 两个 main 的 tile/macro tqdm 计数正确，`show_progress=false` 时完全关闭；
- [ ] run_mbopc_single_macro.py 可从 GDS/OASIS 直接运行并拒绝非单 macro 配置；
- [ ] run_mbopc_multi_macro.py 可从 GDS/OASIS 直接运行并拒绝非多 macro/macro 内单 tile 配置；
- [ ] final GDS、summary、best nominal PNG manifest 存在；
- [ ] 不修改 layout/geometry/00_PAST/用户数据；
- [ ] pytest、ruff、compileall 全绿；
- [ ] 报告、手册、task_plan/findings/progress 同步；
- [ ] 审计无未调用函数、重复实现、吞错、一次性抽象和旧 bug 补偿；
- [ ] 关键阶段本地 commit，未推送远端。
