# Contract — layout

只读层级版图访问层。锚点：`layout/`（database.py/types.py/query.py/source.py）。

## 公开接口

```python
class LayoutDB:                                  # layout/database.py
    @classmethod
    def open(cls, path, top_cell: str | None = None) -> LayoutDB   # 上下文管理器
    top_cell_name -> str                          # 实际顶层（None 参数时要求唯一顶层）
    dbu_um -> float                               # 如 0.001（1nm/DBU）
    layers() -> tuple[LayerSpec, ...]
    layer_bbox(layer: LayerSpec) -> DbuBox | None # 原生逐层包络（不物化）；空层 None
    cell_hierarchy() -> dict[str, tuple[str, ...]] # 顶层邻接表（each_child_cell 去重）
    def query(self, layers, box: DbuBox) -> ShapeQuery

class ShapeQuery:                                 # layout/query.py
    def materialize(self) -> RegionBatch           # 框内物化
    def materialize_intersecting(self) -> RegionBatch  # 完整相交物化（不裁剪 occurrence）

class RegionBatch:                                # layout/types.py
    RegionBatch(regions: dict[LayerSpec, kdb.Region], box: DbuBox, stats=None)
    def region(self, layer: LayerSpec) -> kdb.Region

LayerSpec(layer: int, datatype: int)              # layout/types.py（值对象）
DbuBox(left, bottom, right, top)                  # 整数 DBU；.to_native() 转 kdb.Box
read_glp(path, layer_map)                         # layout/source.py（分派在 open）
```

## 契约

- **打开方式**：`LayoutDB.open` 上下文管理器；`materialize*` 必须在 with 内
  调用（窗口 Region 依赖打开的 DB；已物化 RegionBatch 独立存活）。
- **查询窗口**：全量查询用 `layer_bbox`（不使用 ±2^30 魔法框——int32 域
  外图形会静默丢失）。
- **格式分派**：GDS/OASIS/GLP 在 open 内决定；GLP 需显式层映射，误用拒绝。
- **全 str cell 名**：无 CellRef 凭证类型；名称查找即校验。
- **SREF/AREF**：物化时按 occurrence 完整展开（不裁剪），跨边界图形完整。

## 异常

- 未知顶层/多顶层、GLP 误用、空层 bbox：显式 ValueError/None，不静默。

## 事实核对锚点

`tests/layout/`（27 例）；`main/_macro_pipeline.py::prepare_problems`
（生产消费样例：open→layer_bbox→query→materialize_intersecting）。
