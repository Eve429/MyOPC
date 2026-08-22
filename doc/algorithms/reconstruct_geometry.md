# reconstruct geometry：边段位移到合法版图几何

## 1. 算法定位

`reconstruct geometry` 是边段型 OPC 的几何重建算法，代码位于：

```text
opc/input/edge/reconstruction.py::_reconstruct_geometry
```

它不负责光刻模型、不负责 EPE 评价，也不负责跨 macro 合并。它只回答一个
问题：

> 给定一个 `MacroProblem` 和每个控制 segment 的法向位移，如何恢复出保留
> 原始 polygon/hole 拓扑、且可以被 KLayout 接受的候选几何？

公开的两个消费接口是：

```python
reconstruct_region(problem, displacements) -> kdb.Region
reconstruct_region_with_midpoints(problem, displacements) -> (kdb.Region, midpoints)
```

其中：

- `reconstruct_region` 给 Simple MB-OPC、公共 macro 管线和最终 GDS 写出使用；
- `reconstruct_region_with_midpoints` 给 Gradient MB-OPC 使用，在同一次几何
  计算中同时返回 Region 和当前 segment 的连续中点，保证 forward 几何与
  backward 梯度采样位置一致。

## 2. 输入与输出

### 2.1 输入

| 输入 | 类型 | 含义 |
|---|---|---|
| `problem` | `MacroProblem` | 一次 prepare 阶段生成的参考轮廓、数学边、segment 参数区间、owner 和 macro 契约 |
| `displacements` | 一维数值数组 | 长度必须等于 segment 数量；单位为 DBU；每个值表示沿该 segment 外法向的位移 |
| `fragmentation.max_displacement_dbu` | `float` | 单段位移绝对值上限 |

`SegmentBatch` 是几何的参考真源：

- `contours.vertices` 保存原始整数顶点；
- `segment_edge_ids` 把每个控制段映射回数学边；
- `edge_next_ids` 给出数学边终点；
- `t0/t1` 给出控制段在数学边上的参数区间；
- `edge_normals` 给出统一方向的单位外法向。

位移不会改数学边方向，也不会改变 ring/polygon 数量；它只移动控制段的
端点和内部拼接位置。

### 2.2 输出

`_reconstruct_geometry` 内部返回 `ReconstructionResult`：

| 字段 | 类型 | 语义 |
|---|---|---|
| `contours` | `ContourBatch` | 经过整数化、去重和拓扑校验后的候选轮廓 |
| `segment_midpoints` | `float64[S,2]` 或 `None` | 每个控制段在当前位移几何中的连续中点；仅梯度路径请求 |

`reconstruct_region` 随后把 `ContourBatch` 转成 `kdb.Region`，并检查
`has_valid_polygons()`。

## 3. 全流程

```text
MacroProblem + displacement[S]
        │
        ├─ 位移 shape / finite / max displacement 校验
        │
        ├─ SegmentBatch.materialize()
        │       └─ 得到每个 segment 的连续 starts / ends / normals
        │
        ├─ 按 ring 建立 previous segment
        │
        ├─ 判断相邻段关系
        │       ├─ 同一数学边且位移相同：共用一个拼接点
        │       ├─ 同一数学边但位移不同：保留两个端点，形成 jog
        │       └─ 不同数学边：计算拐角交点，必要时 bevel
        │
        ├─ 可选：按当前拼接几何计算 segment midpoint
        │
        ├─ 生成连续轮廓点
        │       └─ np.rint → DBU 整数点
        │
        ├─ 去除连续重复点和重复闭合点
        │
        ├─ 校验 ring 数量、方向、hole-hull 关系和轮廓合法性
        │
        └─ ContourBatch → kdb.Region → valid polygon 校验
```

重要的是：重建阶段不会把每个 core 的局部图形分别裁剪再拼接，也不会根据
query/context 边界重新生成边。输入 problem 已经保存了完整相交几何；重建只
根据全局 segment 位移恢复完整候选。

## 4. 第一步：位移规范化和参考几何物化

### 4.1 位移守卫

`_validated_displacements()` 执行三项检查：

1. 转换为连续 `float64` 一维数组；
2. 长度必须等于 `segments.segment_count`；
3. 所有值必须有限，且不能超过 `max_displacement_dbu`。

失败分别表示输入接口错误或候选几何越过算法允许范围。异常不会被静默
吞掉：非有限值和超限位移会阻止候选发布。

### 4.2 SegmentGeometry

`segments.materialize(values)` 根据数学边端点和 `t0/t1` 插值出每个控制段的
起点和终点，并沿法向增加位移：

```text
start = reference_start + normal × displacement
end   = reference_end   + normal × displacement
```

这里的 `starts/ends` 仍然是连续 `float64` 坐标。此时还没有取整，也没有
生成 KLayout Polygon。

## 5. 第二步：建立 ring 内前后段关系

控制段数组按 ring 连续存储，`ring_segment_offsets` 是 CSR 偏移。算法构造：

```python
previous[i] = i - 1
previous[ring_start] = ring_end - 1
```

因此每个 segment 都能找到同一 ring 内的前一个 segment，ring 首段会回接到
ring 末段。这个数组只描述当前 ring 的拓扑顺序，不依赖 segment 是否跨过
core 边界。

接着比较相邻段的数学边编号：

```python
same_edge = segment_edge_ids[previous] == segment_edge_ids[current]
same_position = same_edge & isclose(
    displacement[previous], displacement[current]
)
```

`same_edge` 与 `same_position` 必须区分：同一条数学边可能因为 macro 边界
切分或 fragment 分段而包含多个控制段。

## 6. 第三步：处理普通拼接点和拐角

### 6.1 同一数学边

如果相邻两个控制段来自同一条数学边：

- 位移相同：两段在同一条偏移直线上，可以共用一个 junction；
- 位移不同：两段偏移线不再连续，必须保留前一段终点和后一段起点，形成
  一个真实 jog，不能强行用中点抹平。

这就是：

```python
two_points = same_edge & ~same_position
```

### 6.2 不同数学边的拐角

如果相邻段来自不同数学边，算法把它们视为一个真实拐角。

设：

```text
P = previous segment 的位移后终点
Q = current segment 的位移后起点
a = previous 数学边方向向量
b = current 数学边方向向量
```

两条无限直线写成：

```text
P + t·a = Q + s·b
```

通过二维叉积求 `t`，得到解析交点：

```text
intersection = P + t·a
```

这样得到的是 miter 角点，而不是简单把 `P` 和 `Q` 取平均。

### 6.3 平行边与 miter 过长

存在两种情况不能使用解析交点：

1. 两条边平行或近似平行，叉积接近 0；
2. 交点距离原始角点太远，形成尖刺。

算法使用 `miter_limit` 判断第二种情况：

```text
distance(intersection, original_corner)
    > miter_limit × max(abs(previous_displacement), abs(current_displacement), 1)
```

满足任一条件时启用 bevel：保留两个端点 `P`、`Q`，而不是使用交点。

最终拐角的 `bevel` 行也会进入：

```python
two_points = (same_edge & ~same_position) | bevel
```

## 7. 第四步：Gradient 中点的计算

只有 `with_midpoints=True` 时才执行这一步，Simple MB-OPC 默认不承担这部分
内存和计算开销。

中点必须对应“当前位移后的实际 segment 几何”，而不是原始参考边中点。对每个
segment：

- 普通共用 junction：segment 起点和终点使用同一拼接点；
- jog 或 bevel：起点使用后一段起点，终点使用前一段终点；
- ring 尾段通过 `following` 回接 ring 首段。

伪代码：

```python
if two_points[i]:
    segment_start = current_start[i]
else:
    segment_start = junction[i]

if two_points[following[i]]:
    segment_end = previous_end[following[i]]
else:
    segment_end = junction[following[i]]

midpoint[i] = (segment_start + segment_end) / 2
```

这些中点保留为连续 `float64`，不会经过 `np.rint`。Gradient MB-OPC 后续把
它们映射到 canvas 坐标，在 `_EdgeGradientMask.backward()` 中采样当前边位置的
mask 梯度。正向光刻使用完整 Region；中点只用于反向的 midpoint STE 采样。

## 8. 第五步：生成整数轮廓

每个 segment 输出点数由以下规则决定：

```python
if same_position:
    output_count = 0
elif two_points:
    output_count = 2
else:
    output_count = 1
```

含义是：

- `same_position`：当前控制段内部拼接点与相邻段完全重合，不重复写点；
- `two_points`：jog 或 bevel 必须写两个端点；
- 普通拐角：写一个解析 junction。

算法先在连续坐标中生成所有点，再统一执行：

```python
integer_vertices = np.rint(points).astype(np.int64)
```

取整只发生在输出轮廓阶段，避免斜边或 miter 计算过程中提前量化导致不同
tile 产生 33/34 DBU 一类的分歧。

## 9. 第六步：去重和拓扑验证

### 9.1 点清理

整数化后可能出现：

- 相邻连续点取整到同一个 DBU 点；
- ring 尾点与首点取整后重复。

算法会：

1. 删除普通连续重复点；
2. 强制保留每个 ring 的首点；
3. 删除重复闭合点；
4. 按原 ring 分组重新建立 `ring_offsets`。

这些操作只清理几何表示，不改变正常情况下的物理覆盖；如果取整导致
ring 退化，后续验证会拒绝候选。

### 9.2 拓扑验证

`_validate_reference_topology()` 对照参考轮廓检查：

- ring 数量不变；
- polygon 与 ring 的 CSR 关系不变；
- ring 有向面积符号不翻转；
- hole 仍然位于所属 hull 内。

这一步防止“几何仍能被 KLayout 解析，但 polygon 已穿越或 hole 已逃逸”的
情况继续进入光刻模型。

随后 `validate_contours()` 检查轮廓本身是否合法。最后转换成 `kdb.Region`，
并用 `has_valid_polygons()` 做 KLayout 原生有效性检查。

## 10. 总体伪代码

```text
function reconstruct_geometry(problem, displacement, with_midpoints):
    d = validate_shape_finite_and_limit(displacement)
    geometry = materialize_segments(problem.segments, d)

    if segment_count == 0:
        return original_contours, optional_empty_midpoints

    previous = build_ring_previous(problem.segments.ring_segment_offsets)
    same_edge = edge_id[previous] == edge_id[current]
    same_position = same_edge and close(d[previous], d[current])

    for every corner where not same_edge:
        solve intersection of two displaced edge lines
        if parallel or miter_too_long:
            mark bevel
        else:
            save analytic junction

    two_points = (same_edge and not same_position) or bevel

    if with_midpoints:
        calculate current segment start/end from junction/jog/bevel rules
        midpoint = (start + end) / 2

    output_counts = 0 for same_position, 2 for two_points, else 1
    points = emit junctions / previous ends / current starts
    points = round_to_integer_dbu(points)
    points = remove_consecutive_duplicates_and_duplicate_closure(points)

    contours = build_contour_batch(points, original_polygon_ring_offsets)
    validate_reference_topology(reference_contours, contours)
    validate_contours(contours)
    return contours, midpoint

function reconstruct_region(problem, displacement):
    contours = reconstruct_geometry(problem, displacement, false)
    region = contours_to_region(contours)
    require region.has_valid_polygons()
    return region
```

## 11. 性能与内存

- 核心数组操作按 segment 向量化，避免逐 segment 的 Python 热循环；
- `SegmentGeometry` 只在一次重建期间存在；
- Simple MB-OPC 不请求中点，避免额外的 following、segment start/end 和
  midpoint 数组；
- Gradient MB-OPC 请求中点时，Region 与中点来自同一次 `_reconstruct_geometry`
  调用，避免重复重建；
- 主要空间开销是 `O(S)`，其中 `S` 为当前 macro 的 segment 数量；corner
  解析只对真实拐角子集处理；
- 算法不持有整张 reticle，也不负责跨 macro 的最终合并。

## 12. 失败语义

| 条件 | 结果 |
|---|---|
| 位移 shape 不匹配 | `ValueError` |
| 位移非有限 | `ReconstructionError` |
| 位移超过最大值 | `ReconstructionError` |
| ring/polygon 拓扑改变 | `ReconstructionError` |
| ring 方向翻转 | `ReconstructionError` |
| hole 越出 hull | `ReconstructionError` |
| 轮廓或 Polygon 退化 | `ReconstructionError` 或 KLayout `ValueError` |

调用方不得把这些异常转换成“优化成功”或静默跳过。Simple/Gradient 求解器
只会在各自规定的候选状态边界捕获已定义的几何失败，并将原因写入停止状态。

## 13. 相关测试

- `tests/opc/input/test_macro_problem.py`：零位移覆盖、斜边、jog、bevel、
  ring/hole、拓扑守卫和中点几何契约；
- `tests/opc/iteration/test_simple_mbopc.py`：Simple 候选重建、非法几何和
  状态发布；
- `tests/opc/iteration/test_gradient_mbopc.py`：Gradient 中点调用、当前状态
  中点、forward/backward 采样和非法候选发布。
