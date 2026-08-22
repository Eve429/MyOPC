# Contract — opc.input.edge（边段输入与重建）

边段型方法（simple/梯度 MB-OPC）共有的参考输入构造与位移重建。
锚点：`opc/input/edge/{fragmentation,problem,reconstruction,sampling}.py`。

## 分段

```python
class FragmentationConfig:                        # fragmentation.py
    corner_length_dbu / max_segment_length_dbu / max_displacement_dbu: float
    miter_limit: float                            # 构造期校验（正长度等）

def fragment_edges(contours, polarity, config) -> SegmentBatch

class SegmentBatch:                               # frozen；参考几何唯一数组真源
    contours: ContourBatch                        # 顶点/环/多边形两级 CSR
    edge_ids: Int32Array[S]                       # 段所属数学边起点索引
    edge_next_ids / edge_polygon_ids / edge_normals   # 边级缓存[E]
    ring_segment_offsets: IntArray[R+1]
    t0 / t1: FloatArray[S]                        # 参数区间 0≤t0<t1≤1
    segment_count -> int
    def materialize(self, displacements=None) -> SegmentGeometry
        # 端点=顶点+边向量×t（可非整数）；None=零位移参考；
        # 位移时 starts/ends += normals×d（法向不随位移变）

class SegmentGeometry:                            # starts/ends/normals 均 float64[S,2]
```

- **法向**：`_outward_normals` 材料→空区单位外法向；opaque 翻转 → 全局
  统一"透光区→不透光区"；求解器无极性分支。
- **切线分裂**：ownership 切线处斜边交点参数由整数端点+整数切线计算；
  分裂碎片沿用原段数学边号；同一切线交点去重（零长碎段回归）。

## MacroProblem（problem.py，NPZ format v3）

```python
class MacroProblem:                               # frozen，构造期校验全部不变量
    macro / layer / polarity / fragmentation / segments
    owner_indices: Int32Array[S]                  # 唯一可写 core；-1=只读 context
    core_offsets: Int64Array[C+1]                 # CSR
    member_segment_indices: Int32Array[M]         # own ⊆ membership
    def segments_for_core(i) / owner_segments_for_core(i)
    def save(dir) / MacroProblem.load(path)       # v3；v2（含 dark_box）与 v1 显式拒绝
def prepare_macro_problem(batch, layer, polarity, fragmentation, macro,
                          *, data_bounds: DbuBox) -> MacroProblem
```

不变量（构造期强制）：owner ∈ [-1,C)；CSR 单调闭合；own ⊆ membership
（空 membership 不短路）；段中点归属即 owner。**负板补铬（2026-08-22
几何方案）**：data_bounds 是全局数据包络（layer bbox，须显式传参）；
opaque 在提边之前补画包络外到查询边界的不透光图形——补区与既有铬
共线相接处经布尔并 + merged() 融合（表示层共线边消除）；补区外缘落在
查询边界上，恒为 context-only 段（owner=-1，不可动、不进输出）；包络
边有透光缺口时缺口处形成真实铬|石英 owned 段。clear 忽略 data_bounds
（包络外无图形天然恒暗）。

## 重建（reconstruction.py）

```python
def reconstruct_contours(problem, displacements) -> ContourBatch
def reconstruct_region(problem, displacements) -> kdb.Region
```

守卫（ReconstructionError）：ring 拓扑改变、环绕翻转、hole 越出 hull、
位移超上限、无效多边形；**几何退化也可能以 ValueError 冒出**（KLayout
数组校验，如共线 ring 少于三顶点）——调用方按非法候选处理。位移 shape/
有限性属上游契约（evaluate_state 入口先行拦截）。

## 探针（sampling.py）

```python
def edge_probe_points(starts, ends, normals, distance_dbu) -> (inner[N,2], outer[N,2])
    # 围绕参考边中点：inner=mid−normal×d（材料侧）、outer=mid+normal×d
```

## 事实核对锚点

`tests/opc/input/test_macro_problem.py`（不变量组）；`tests/opc/iteration/
test_simple_mbopc.py`（真构造越界两用例）。
