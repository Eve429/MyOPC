# Layout / Geometry 开发报告

## 1. 开发结论

本阶段已经形成可直接运行、面向局部批处理的 Layout / Geometry 基础层。实现以 KLayout C++ `Layout`、`RecursiveShapeIterator` 和 `Region` 为计算内核，以整数 DBU 作为唯一几何单位；Python 只负责生命周期、数据契约、批次调度和少量 NumPy 边数组计算。该结构不会在默认路径展开全版图，也不会逐 Polygon 跨越 Python/C++ 边界。

项目不要求安装自身包。用户可在仓库根目录直接执行：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\simple.gds --layer 1/0 --arrays --diagnostics
```

`pyproject.toml` 仅描述第三方依赖和开发工具；运行入口依赖当前文件所在仓库，不依赖 `pip install -e .`。已实际卸载 editable 安装并完成测试。

## 2. 架构与数据流

```text
GDS / OASIS
    │ 单次读取，保留 Cell/AREF/SREF 层级
    ▼
LayoutDB ── ShapeQuery(Cell, Layer[], ROI)
    │ KLayout 原生递归迭代与坐标变换
    ▼
RegionBatch（按 Layer 保存原生 Region）
    ├─ GeometryEngine：clip / union / difference / xor / offset / merge
    ├─ ContourBatch：连续 int64 顶点与 CSR 风格环索引
    ├─ EdgeBatch：向量化闭环边及 polygon/ring 元数据
    ├─ UniformGridIndex：局部边候选查询
    └─ PatchSet：按 core ownership 裁剪、冲突检查、确定性聚合
            │
            ▼
       PatchWriter（原子写出 GDS / OASIS）
```

### 2.1 Layout 层

- `LayoutDB.open()`：版图只解析一次；多 top Cell 时要求调用方明确选择，避免静默选错。
- `LayoutDB.query()`：只创建轻量查询描述，不立即物化坐标。
- `ShapeQuery.materialize()`：每层一次原生递归查询，实例变换和 Region 构造均在 C++ 内完成。
- 原生 `shape_flags` 只允许 Box、Path、Polygon 进入 mask Region；Text 和 Edge 不进入正常几何路径。
- 诊断默认关闭。只有显式指定 `diagnostics=True` 时，才进行额外遍历并分配计时、分类统计对象。
- `RegionBatch` 只复制很小的 Layer 映射；Polygon 数据继续保留在原生内存中。

### 2.2 Geometry 层

- `GeometryEngine` 的接口粒度是 `RegionBatch`，不提供逐图形回调，避免算法实现意外退化为 Python 热循环。
- 布尔运算、裁剪、偏置和合并使用 KLayout `Region`；`combine` 与 `union` 分开，避免调用方只想拼接时支付隐式 merge 成本。
- `ContourBatch` 使用连续 `int64` 数组保存顶点，使用 `ring_offsets` 表达变长环，并保留孔洞、Polygon ID。
- `EdgeBatch` 从闭环轮廓向量化生成起止点，供 OPC 算法做边定位；碎片化规则不塞入基础几何层。
- `UniformGridIndex` 用于 tile-local 边候选查询。超大边单独维护，避免一条长边占据过多网格桶。

### 2.3 Patch 与输出

- `GeometryPatch` 描述 `patch_id + layer + region + ownership_box`。
- `PatchSet.add()` 先检查同层 ownership 正面积重叠，再用 ownership 精确裁剪输入 Region。
- 一个完整 Polygon 可以同时传给相邻 core；每个 core 只获得自己 ownership 内的部分。共享边界允许存在，正面积重叠会被拒绝。
- 同层已占 ownership 和已裁剪结果分别在原生 `Region` 中累计，避免 Python O(n²) Patch 扫描和每次插入排序。
- `PatchWriter` 不修改源 `LayoutDB`，只写 Patch。临时文件与目标位于同一目录，完整写出后使用 `os.replace` 原子替换。

## 3. 面向不同 OPC 方法的扩展方式

当前边界按“数据能力”划分，而不是绑定某一种 OPC 算法：

| OPC 方法 | 推荐输入 | 可复用能力 | 算法层自行负责 |
|---|---|---|---|
| Rule-Based OPC | `EdgeBatch` + 局部索引 | 邻边候选、offset、布尔运算 | 规则匹配、边分段、移动量 |
| Model-Based OPC | `ContourBatch` / `EdgeBatch` | ROI、批量数组、Patch ownership | 成像模型、梯度/优化器 |
| ILT | `RegionBatch` 或栅格适配器 | 层级读取、局部裁剪、输出 | 栅格化、反演和约束 |
| SRAF / Hotspot | Region 与局部索引 | 布尔关系、候选搜索 | 插入/分类策略 |

后续算法模块应依赖 `DbuBox`、`LayerSpec`、`RegionBatch`、`ContourBatch` 或 `EdgeBatch`，不要直接控制 `LayoutDB` 内部的 KLayout 对象。需要 GPU、栅格或其他多边形内核时，可在批次边界增加适配器；当前代码只标注 `backend="klayout"` 并做一致性检查，没有提前构建复杂插件注册系统。

## 4. 性能设计

1. 保留层级：AREF/SREF 不默认 flatten，一百万逻辑实例的测试文件仍只保存一个叶 Cell 和一个 AREF。
2. ROI 优先：先用层级 bbox 索引筛选候选，再在原生 Region 中精确裁剪跨边界图形。
3. 批量跨语言：每层一次原生调用；仅在 OPC 重复计算确实需要数组时显式转换轮廓和边。
4. 整数坐标：核心数据始终使用 DBU `int64`，避免浮点舍入和反复单位转换。
5. 可选诊断零常态开销：正常路径不统计 Text/Edge，也不构造 `MaterializationStats`。
6. 局部索引：重复邻域查询时用网格索引替代全量边 bbox 扫描。
7. 写出隔离：源版图只读，减少深复制和误修改；输出仅含变更 Patch。

最终基准中，一百万逻辑实例的小 ROI 查询与裁剪中位数为 0.1058 ms，额外 RSS 为 0.48 MB；100,000 条边的网格查询中位数为 0.0208 ms，相对完整 NumPy bbox 扫描加速 15.99 倍。具体环境和方法见测试报告。

## 5. 直接运行与公共入口

查询默认 top bbox 和全部已有 Layer：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\simple.gds
```

查询负坐标 ROI、输出数组统计和 JSON：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\JustPoly.gds --layer 1/0 --box -2500 -600 500 1600 --arrays --json
```

把单 ROI 精确裁剪结果写为 Patch：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\simple.gds --layer 1/0 --box 0 0 1000 1000 --output result.gds
```

可重复使用的 Python 主接口是：

- `layout.LayoutDB`、`DbuBox`、`LayerSpec`、`PatchWriter`
- `geometry.GeometryEngine`、`extract_contours()`、`extract_edge_batches()`
- `geometry.UniformGridIndex`、`GeometryPatch`、`PatchSet`

## 6. 过度设计与逻辑复杂度审查

最终审查重点检查了抽象层数、热路径对象分配、错误修复特例和重复状态。

已简化项目：

- 删除“逗号连接 Box”的自定义解析逻辑，CLI 使用 argparse 原生四整数参数，负坐标无需特殊分支。
- 诊断统计完全退出正常热路径，避免为了测试信息影响实际运行。
- Patch ownership 冲突由每层一个原生 Region 维护，删除 Python 逐 Patch 扫描和每次插入排序。
- 图形类型过滤固定在原生迭代器，未增加 Python 逐 Shape 补丁逻辑。
- 没有实现通用依赖注入容器、插件注册器、全局缓存或提前抽象的 OPC 算法接口。

保留的必要复杂度：

- 生产 materialize 使用“正面积相交”，诊断遍历使用“接触或相交”。这是为了让零面积 Text 出现在诊断中，同时不污染 mask Region，属于明确隔离的语义差异。
- `combine` 和 `union` 分开，是为避免隐式 merge 的性能与语义变化，不是重复 API。
- Region 与数组两种表达并存：前者适合原生布尔运算，后者适合 OPC 数值计算；转换是显式的单向边界。

审查未发现循环依赖、死抽象或为单个测试硬编码的生产分支。Ruff 规则检查发现的 10 项维护性问题已经手工消除；未使用 Ruff/Black 自动格式化，因为它会破坏用户要求的紧凑排版。

## 7. 已知边界与后续建议

- 当前输出是 patch-only，不负责把 Patch 合并回完整源版图；合并策略应由上层流程明确决定。
- ownership 冲突判定基于轴对齐 core box，同层仅允许零面积边界接触。
- `UniformGridIndex` 返回 bbox 候选，精确距离或相交仍应由算法层进行。
- 当前轮廓数组是整数 DBU；需要物理单位时在配置/报告边界乘以 `dbu_um`。
- 大规模多进程运行时，建议每个 worker 打开自己的只读版图句柄，并以 ownership box 分派任务；不要共享可变 KLayout 对象。
- 下一阶段应先选择一个真实 OPC 方法做窄接口验证，再决定是否需要栅格/GPU 后端，避免提前建设用不到的扩展框架。

## 8. planner 区域像素图

系统现在提供两个紧凑入口：

```python
from geometry import render_layout_region
from layout import DbuBox, LayerSpec, LayoutDB

with LayoutDB.open("TestReticle/gcd_45nm.gds") as database:
    pixels = render_layout_region(
        database,
        DbuBox(11400, 13150, 317300, 308850),
        LayerSpec(11, 0),
        pixel_size_nm=5,
        output_path="gcd_45nm.png",
        show=True,
    )
```

- `render_layout_region()` 接受复用中的只读数据库和 planner `DbuBox`，负责查询、裁剪、栅格化、保存与显示。
- `render_region_batch()` 接受已经提取的 `RegionBatch`，适合 planner/core 流水线直接展示现有结果，避免重复查询。
- 两个函数都返回顶部朝上的二维 `uint8` 数组；0 表示空，255 表示完整覆盖，中间灰度表示 Polygon 在像素中的精确面积覆盖率。
- 核心使用 KLayout `Region.rasterize()`，不在 Python 中逐 Polygon 绘制。由于原生接口不采用 merged semantics，局部区域在栅格前显式合并，避免重叠面积重复计数。
- 原生浮点结果按最多 1,000,000 个临时像素切成二维块，最终只保留八位数组；默认在分配前拒绝超过 64,000,000 像素的图片。
- 物理像素尺寸必须能精确换算为整数 DBU，避免静默舍入改变 OPC 采样网格。默认值为 5 nm/pixel。
- Pillow 只负责灰度 PNG 和可选系统查看器；不引入独立 GUI、图层混色或交互框架。

直接保存真实版图完整 Layer：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\gcd_45nm.gds --layer 11/0 --pixel-size-nm 5 --png gcd_45nm.png
```

增加 `--show-image` 可同时调用系统图片查看器。PNG 模式要求且只允许一个 Layer；多个 Layer 必须分别输出，从而保持 mask 语义明确。
