# Contract — geometry

方法无关的 Region↔轮廓转换、校验、Patch 写出与共享栅格原语。
锚点：`geometry/`（contour.py、patch.py、raster.py、validation.py）。

## 公开接口

```python
def extract_contour(source) -> ContourBatch       # geometry/contour.py
class ContourBatch:                               # 顶点 int + 两级 CSR（ring/polygon）
    vertices: IntArray                            # 全局顶点（整数 DBU）
    ring_offsets / polygon_ring_offsets           # 环/多边形两级 CSR

def contours_to_region(contours) -> kdb.Region    # 轮廓 -> Region
def validate_contours(contours, ...)              # 环绕向/闭合/有效性校验

def iter_region_coverage_tiles(region, box, pixel_dbu,
                               shape, dtype)       # geometry/raster.py
    # 生成 (y0, x0, areas)：原生分块面积覆盖率，左下原点，像素中心=原点+半像素

class GeometryPatch:                              # geometry/patch.py
    GeometryPatch(macro_id, layer, region, box)
class PatchWriter:
    @classmethod
    def write_macro_results(cls, patches, output_path, dbu_um, *,
                            cell_mode: "single_cell" | "macro_cells",
                            top_name: str = "OPC_RESULT") -> Path
```

## 契约

- **左下原点**：一切数组行 0 = 最低 Y；PNG/显示翻转仅在 I/O 边界。
- **覆盖率语义**：像素值 = 图形与像素框交面积比（0~1 连续），不移动几何
  边界迎合像素网格；轴长非像素整数倍时向上取整最小覆盖。
- **Patch 写出**：patches 的 region 应已裁到各自 box（ownership 权威覆盖）；
  `single_cell` 全局 merge 消 seam，`macro_cells` 每 macro 一子 Cell；
  两种模式物理覆盖一致。
- **整数量纲**：ContourBatch 顶点是整数 DBU；浮点仅出现在段参数化与探针层。

## 已知保留

`render_layout_region`：零生产引用但有直接回归与演示使用（明确保留决策）。

## 事实核对锚点

`tests/geometry/`（25 例）；`geometry/patch.py::PatchWriter.write_macro_results`
（merge 消费样例）。
