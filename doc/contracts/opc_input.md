# Contract — opc.input（网格与栅格化）

方法无关的两级网格规划、居中光刻画布与坐标换算。
锚点：`opc/input/grid.py`、`opc/input/raster.py`、`opc/input/mask.py`。

## 网格规划

```python
def plan_macros(bounds: DbuBox, *,
                core_size_dbu: int, context_dbu: int,
                pixel_dbu: int, canvas_pixels: int,
                macro_size_dbu: int | None = None,
                macro_grid: tuple[int, int] | None = None) -> tuple[MacroSpec, ...]

class MacroSpec:                                  # opc/input/grid.py（frozen）
    macro_id: str                                 # "mr{r}c{c}" 行优先
    ownership_box: DbuBox                         # 半开不重叠；面积和 == bounds 面积
    x_cuts / y_cuts: IntArray                     # core 全局切线，严格递增
    context_dbu / pixel_dbu / canvas_pixels: int
    core_count -> int
    def core(self, index) -> CoreSpec             # 即时构造
    def locate_owned_points(points) -> IntArray   # macro 外的点返回 -1

class CoreSpec:
    core_id: str                                  # "c_r{r}c{c}"
    ownership_box: DbuBox                         # 唯一可计分/回写
    context_box: DbuBox                           # ownership 四边扩 context
```

契约：macro_size 与 macro_grid 恰好一个非空；macro 尺寸严格大于 core；
context ≥ max_displacement；core+2×context 的像素数 ≤ canvas（超限在分配前
失败）；[1,1] 网格 = 单 macro 覆盖全 ROI、内部仍切 core。

## 栅格化与坐标（opc/input/raster.py）

```python
def rasterize_mask_canvas(region, context_box, pixel_dbu, canvas_pixels, *,
                          polarity) -> np.float32[canvas, canvas]
def ownership_canvas(ownership_box, context_box, pixel_dbu,
                     canvas_pixels) -> np.bool_[canvas, canvas]
def points_to_canvas(points_dbu, context_box, pixel_dbu,
                     canvas_pixels) -> np.float64[N, 2]
def rasterize_region_window(region, box, pixel_dbu) -> np.float32[H, W]  # 无 padding 底层
```

- **透光率语义**：1=透光；clear=coverage、opaque=1−coverage；canvas 外围
  padding 恒 0。
- **居中 padding**：`_center_padding` 差值均分、奇数余量归高坐标侧；三个
  公开函数与 lithography `_prepare_mask` 共用同一偏移（同尺寸输入同一布局）。
- **坐标换算**：`points_to_canvas` 返回 (x,y) 连续坐标
  `= (x−left)/pixel − 0.5 + low_x`（float64 必须——int32 DBU 域超 2²⁴ 后
  float32 丢整数；探针坐标禁止手写公式）；不 round 不 clip，取整与越界
  由评价层处理。
- **ownership_canvas 与 points_to_canvas 互为反函数**（全部 True 像素中心
  整数回映，测试锁定）。

## 极性

`MaskPolarity.CLEAR/OPAQUE`（`opc/input/mask.py::normalize_mask`）；
配置层归一化，下游只消费枚举。

## 事实核对锚点

`tests/opc/input/test_grid.py`（TestMacroPlanning/TestCenteredCanvas/
TestPointsToCanvas）；`opc/input/raster.py` 注释内的映射公式。
