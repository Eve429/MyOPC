# Layout / Geometry 测试报告

## 1. 当前结果

Layout/Geometry 专项 38 项通过，综合语句/分支覆盖率 91%，高于 90% 门槛。Ruff、compileall 和严格性能基准通过。完整仓库结果见项目测试手册及本次架构精简测试报告。

## 2. 环境

| 项目 | 值 |
|---|---|
| Python | 3.12.0 |
| KLayout | 0.30.10 |
| NumPy | 2.5.1 |
| CPU | AMD Ryzen 7 4800H，8 核/16 线程 |
| 内存 | 15.37 GiB |
| 使用方式 | 源码直接运行，无 editable install 要求 |

## 3. 自动测试范围

### 3.1 Layout

- 文件/top/Cell/Layer 不存在、关闭后惰性查询失效；
- DBU box、Layer、层级 bbox、R90/镜像/AREF；
- Box/Path/Polygon 物化与 Text/Edge 诊断隔离；
- 属性开关前后几何集合不变，带属性图形的键值保留；
- 跨 ROI Polygon 被精确裁剪，不返回仅“候选相交”的完整图形；
- 属性图形跨 ROI 裁剪后 bbox 变化但属性不丢失。

### 3.2 Geometry

- hull/hole 连续数组与 Region round-trip XOR=0；
- 数学边顺序、ring/polygon/hole 元数据和连续 `int64` 布局；
- 零长度边、非法 ring 和验证报告；
- 精确灰度覆盖、坐标翻转、孔洞、重叠合并、二维分块；
- 跨 core 图片拼接与整图逐像素一致；
- Patch ID、ownership 冲突、跨 core 图形互补裁剪；
- GDS/OASIS 输出 round-trip XOR=0。

已删除仅验证 `GeometryEngine` 与 `UniformGridIndex` 自身的 5 项测试；新增 1 项精确 ROI 回归。测试净减少 4 项是生产 API 删除的直接结果，不是降低现有路径覆盖。

## 4. ROI 精确裁剪回归

构造 `Box(0,0,100,100)`，查询 `DbuBox(25,20,75,80)`：结果 bbox 必须恰为查询框，面积必须为 3,000 DBU²。

属性回归构造普通图形 `(0,0;10,10)` 和 tagged 图形 `(20,0;30,10)`，查询 `x=[5,25]`：结果变为 `(5,0;10,10)` 与 `(20,0;25,10)`，数量仍为 2，后者仍携带 `{7: "tagged"}`。该测试防止未来用普通 `Region & clip` 再次丢属性。

## 5. 跨 core Patch

原图 `Box(25,20,75,80)` 跨 `x=50`：左右 ownership 各得到 1,500 DBU²，正面积交集为 0，合并后与原图 XOR=0。共享边界允许，ownership 正面积重叠拒绝。

OPC 最终矢量不使用这种 patch 裁剪拼接；该能力只用于明确的 patch 输出。MB-OPC core 负责计算/写权限，最终从全局参考边统一重建。

## 6. 严格性能基准

```powershell
D:\app\miniforge\envs\myopc\python.exe `
  benchmarks\benchmark_layout_geometry.py --strict
```

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 百万逻辑实例 ROI polygons | 25 | 必须 25 |
| 文件打开 | 10.237 ms | 记录项 |
| 精确物化中位数 | 0.10435 ms | ≤50 ms |
| 精确物化 P95 | 0.16723 ms | 记录项 |
| ROI RSS 增量 | 0.484 MiB | ≤64 MiB |
| 2048² raster | 499.59 ms | ≤5,000 ms |
| raster RSS 增量 | 6.379 MiB | ≤128 MiB |
| raster coverage | exact | 必须 exact |

历史精确查询中位数为 0.1058 ms，当前没有退化。空间索引基准已删除，因为对应模块无生产调用方。

## 7. 覆盖率

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest `
  tests\layout tests\geometry --cov=layout --cov=geometry `
  --cov-branch --cov-report=term-missing -q
```

结果：38 passed in 1.50 s，TOTAL 91%。`geometry/raster.py` 98%，`geometry/contour.py` 96%，`geometry/edge.py`/`patch.py` 95%，`layout/query.py` 90%。未命中分支主要为异常入口。

## 8. 真实版图历史基线

`gcd_45nm.gds` Layer 11/0：1,776 polygons、21,590 vertices/edges、bbox `[11400,13150,317300,308850]` DBU。5 nm/pixel 全图为 6,118×5,914，历史端到端 4.663 s、峰值 RSS 149.36 MiB。源 GDS 未纳入提交。

## 9. 结论

删除无调用方门面/索引后，现有查询、属性、数组、栅格和 patch 能力保持，ROI 语义反而在所有入口统一；性能与覆盖率门槛通过。
