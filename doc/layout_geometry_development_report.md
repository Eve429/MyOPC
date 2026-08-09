# Layout / Geometry 开发报告

## 1. 当前结论

Layout/Geometry 基础层以 KLayout 0.30.x C++ `Layout`、`RecursiveShapeIterator`、`Region` 为唯一后端，以整数 DBU 为几何单位。Python 负责只读生命周期、Layer/ROI 批次和少量连续 NumPy 数组，不默认 flatten 层级，也不逐 polygon/edge 跨解释器热循环。

项目无需安装，直接运行：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py `
  TestReticle\simple.gds --layer 1/0 --arrays --diagnostics
```

## 2. 数据流

```text
GDS/OASIS
  -> LayoutDB（一次读取、保留层级）
  -> ShapeQuery（Cell + Layer[] + ROI）
  -> RegionBatch（每层一个精确裁到 ROI 的 KLayout Region）
     -> ContourBatch / EdgeBatch（明确请求才转连续数组）
     -> 灰度 raster（明确请求才生成）
     -> GeometryPatch / PatchSet / PatchWriter（独立输出）
```

### 2.1 Layout

- `LayoutDB.open` 只解析一次文件，多 top 时要求显式选择；
- `query` 只创建轻量惰性描述，`materialize` 才执行原生递归查询；
- 原生 shape flags 只把 Box/Path/Polygon 放入 Region，Text/Edge 只在显式 diagnostics 中统计；
- `preserve_properties=True` 只启用属性导入，不使用会过滤普通图形的 `SProperties`；
- `RegionBatch` 只复制很小的 Layer 映射，Polygon 保留在 C++ 内存。

`RecursiveShapeIterator` 的 box 用于层级候选筛选，不会自动截断跨边界 Polygon。当前 `ShapeQuery.materialize` 在每层构造 Region 后立即做一次原生交集，统一保证 `RegionBatch` 精确落在 planner ROI。属性模式使用 `Region.and_(clip, NoPropertyConstraint)`，否则 KLayout 普通 `&` 会丢弃 Polygon 属性；两条路径均为每 Layer 一次 C++ 批处理。

### 2.2 Geometry

- `ContourBatch` 用 `int64` 顶点、CSR ring offsets、polygon ID 和 hole 标记保存轮廓；
- `EdgeBatch` 向量化生成闭环数学边，OPC 切段策略不进入基础层；
- `validate_contours` 检查空边、重复点、ring 数量和拓扑输入不变量；
- `render_region_batch`/`render_layout_region` 使用 KLayout 面积栅格，输出顶部朝上的 `uint8` 图；
- `PatchSet` 按 Layer 累计原生 ownership Region，拒绝正面积冲突，并精确裁剪 patch；
- `PatchWriter` 写临时文件后原子替换目标，不修改源数据库。

## 3. 本次架构减法

在整图 MB-OPC 真实工作流稳定后重新搜索调用点，删除了：

- `GeometryEngine`：无状态固定后端门面，唯一生产调用只是根演示程序的重复 ROI 裁剪；
- `UniformGridIndex`：只有自身测试/基准，当前 OPC 已由 core-to-segment CSR membership 解决实际邻域需求；
- `RegionBatch.backend` 和 `BackendMismatchError`：所有数据类型都直接持有 `kdb.Region`，不存在第二后端；
- `CoordinateSystemError`：只服务已删除门面；
- `EdgeBatch.bboxes`：只服务已删除索引；
- `DbuBox.overlaps`：只是无调用方的 `intersection(...) is not None` 糖衣；
- 对应的 2 个模块、2 份测试及空间索引 benchmark/CLI 参数。

批量 Region 能力没有重新包装成另一组无调用方函数；实际调用点直接使用 KLayout 原生集合运算。未来出现第二个真实后端或任意 edge-neighbor 查询消费者时，再从真实需求提取接口。

`LayoutDB.hierarchy_summary` 作为早期明确交付的只读层级检查 API 保留：它不复制图形、不进入迭代热路径，也没有并列实现；外部 planner 可直接消费该公共结果。

## 4. 性能设计

1. 层级优先：AREF/SREF 不 flatten；
2. ROI 优先：原生层级筛选后每 Layer 一次精确 Region 相交；
3. 数组按需：只有重复数值计算才提取轮廓/边；
4. 诊断显式：默认不进行图形分类、PNG 或完整几何物化；
5. 栅格分块：原生临时 float 像素约束在 1,000,000 以内，最终只保留 `uint8`；
6. 输出隔离：源版图只读，结果写独立 GDS/OASIS。

精简后百万逻辑实例 ROI 精确物化中位数 0.10435 ms，P95 0.16723 ms，RSS 增量 0.484 MiB；历史中位数 0.1058 ms，没有性能退化。删除的索引即使历史测试很快，也没有实际调用方，因此不再把它的速度当项目门槛。

## 5. 当前公共入口

- `layout.LayoutDB`、`DbuBox`、`LayerSpec`、`RegionBatch`、`PatchWriter`；
- `geometry.extract_contours`、`extract_edges`、`extract_edge_batches`、`contours_to_region`；
- `geometry.validate_contours`、`render_region_batch`、`render_layout_region`；
- `geometry.GeometryPatch`、`PatchSet`。

## 6. OPC/ILT 复用边界

MB-OPC 使用精确 ROI `RegionBatch`、轮廓和数学边；ILT 可复用层级查询、Region 和 raster，不必依赖边段；Patch/output 对所有方法通用。任何方法都不应直接长期持有 Python polygon 列表。

版图层只表达数据和查询，不预建 GPU/求解器接口。几何层只提供当前使用的 Region、数组、验证、栅格和 patch 能力，不再为假设中的后端或邻域算法保留空抽象。

## 7. 像素展示

```python
from geometry import render_layout_region
from layout import DbuBox, LayerSpec, LayoutDB

with LayoutDB.open("TestReticle/gcd_45nm.gds") as database:
    pixels = render_layout_region(
        database, DbuBox(11400, 13150, 317300, 308850), LayerSpec(11, 0),
        pixel_size_nm=5, output_path="gcd_45nm.png", show=False)
```

0 为背景，255 为完整 mask，灰度为像素内精确面积覆盖。像素尺寸必须能精确换算为整数 DBU；默认最大输出 64,000,000 pixels。

## 8. 维护规则

每个 bug 必须保留最小回归；修复后删除旧 wrapper/分支/字段。性能基准只覆盖实际生产能力，不为已删除模块保留“墓碑测试”。关键节点本地 commit，未经授权不 push。
