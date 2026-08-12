# 版图极性、GLP 与运行配置开发报告

## 1. 交付结果

本轮增加显式 `clear/opaque` 版图极性、严格 GLP 输入和六套 TOML 运行配置。`LayoutDB` 现在只解析源文件一次，容量预检复用同一原生数据库；`geometry/` 未修改。

## 2. 关键接口

- `LayoutDB.open(path, top_cell=None, glp_layer_map=None)`：GDS/OASIS 继续由 KLayout 读取，`.glp` 由严格解析器直接构造内存 Layout，无临时 GDS。
- `MaskPolarity.CLEAR/OPAQUE`：`PhysicalMask.region` 始终保存源多边形；所有光学数组统一以 1 表示透光。
- `rasterize_mask_canvas(...)`：clear 返回源覆盖率；opaque 只在显式处理框内返回 `1-覆盖率`，处理框外为 0。
- `prepare_problem(..., polarity=...)`：opaque 的边法向反向，使法向始终从透光侧指向不透光侧，正位移始终扩大透光区域。
- `ConfiguredArgumentParser`：按“默认 common→默认 entry→自定义 common→自定义 entry→CLI”合并，未知 section/key/type 直接失败。

## 3. 性能与内存

容量预检不再重新 `layout.read`。大型 GDS/OASIS 和文本 GLP 均只有一个源数据库；预检仍只遍历层级 ROI，不提前构造完整 Region/SegmentBatch。opaque 反相按当前 tile 生成一个 field coverage 临时数组，不构造处理框补集 Region，因此不会增加可移动边或全局几何常驻量。

TOML 只在 CLI 启动时读取一次，优化迭代热路径不访问配置文件。配置不进入算法基础层，Python API 继续接受原参数和 dataclass。

## 4. 格式范围

GLP 第一版支持 `BEGIN`、`EQUIV ... MICRON [+X,+Y]`、`CNAME`、`LEVEL`、`CELL [PRIME]`、`RECT N`、`PGON N` 和 `ENDMSG`。名称末尾数字默认映射 layer/0；非数字承载层要求 `--glp-layer NAME=LAYER/DATATYPE` 或 TOML `glp_layers`。未使用的辅助 LEVEL 可保留，未知语句、非整数坐标、非 N 图形和非法多边形拒绝。GLP 输入当前只允许版图输出为 GDS。

离线 raster 协议升级为 v2、segment 协议升级为 v3，新增 polarity metadata；loader 继续读取 v1/v2，并把缺失极性解释为历史 clear 语义。

## 5. 简化审查

没有新增格式注册器、求解器包装基类或 SRAF 占位实现。GLP 只有一个当前调用方 `LayoutDB.open`；配置只有一个 CLI 合并器；普通 Region coverage 保留为底层函数，极性只增加一个光学语义包装。旧重复版图读取逻辑已从 preflight 删除，没有保留兼容分支。
