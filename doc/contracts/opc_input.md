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
  padding 恒 0。窗口外的暗场由几何保证（2026-08-22 起：负板在 prepare
  阶段已补画包络外不透光图形、正板包络外无图形天然 coverage=0），
  本函数不再有暗界参数。
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

## 像素宏问题（opc/input/pixel，2026-08-19 随 Simple ILT 迁移）

```python
def prepare_pixel_macro_problem(batch, layer, polarity, macro, *,
                                planning_bounds: DbuBox,
                                data_bounds: DbuBox) -> PixelMacroProblem
    # planning_bounds=plan_macros 所用规划边界（field，ownership 四向包含校验）；
    # data_bounds=全局数据包络（layer bbox）：负板栅格化前补画包络外到
    # 查询边界的不透光图形（2026-08-22 几何方案，与 edge 路径同一语义）；
    # clear 忽略之。双参数必填无默认。
class PixelMacroProblem:                             # frozen；NPZ v1
    macro / layer / polarity
    target_u8: np.uint8[Hq, Wq]                      # query box transmission 0..255
    def save(path) / load(path)                      # allow_pickle=False
    def target_canvas(core_index) -> np.uint8[256, 256]
    def ownership_canvas(core_index) -> np.bool_[256, 256]
    def trainable_index_canvas(core_index) -> np.int64[256, 256]  # macro 外 -1（int64：2^31 宏像素防溢出）
def reconstruct_pixel_region(problem, binary_ownership) -> kdb.Region
```

- **一次栅格化**：prepare 对每个 macro 恰一次 `rasterize_region_window`
  （query box 整除 pixel）；NPZ 不存每 core 重复画布，core canvas 按需切片
  并经 `_center_padding` 居中（与 mask/ownership 画布同布局）。
- **实际 box 整像素契约**：macro 与全部 core 的 ownership 宽高必须是
  pixel_dbu 整数倍（等价于 layer bbox 宽高整像素）；否则栅格化前
  ValueError，不产 partial ownership pixel。这是像素管线比 edge 管线
  更严的输入对齐要求。
- **trainable 索引**：非负值是 [Hm,Wm] macro 参数行主序扁平下标；同一
  物理像素在任何 core 画布中同值（索引定义在 macro 网格）。
- **回写**：binary transmission 按行游程合并 Box 后每宏恰一次 merge；
  极性逆变换只在回写边界（clear 输出透光像素，opaque 输出不透光像素）。

事实核对锚点：`tests/opc/input/test_pixel_problem.py`。
