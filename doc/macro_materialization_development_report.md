# Macro 未裁剪物化与双 Halo 开发报告

## 1. 目标与交付边界

本轮解决大版图按多个 CPU macro 物化时，精确 ROI 裁剪把查询框四边误当成可移动物理边的问题。实现遵循三条边界：现有 `ShapeQuery.materialize()` 的精确裁剪语义保持不变；仅最小扩展已授权的 `layout/query.py`，`geometry/` 零修改；交付首个可用 macro 前端验证，不把尚未实现的磁盘 shard 或多轮 out-of-core solver 描述为当前能力。

## 2. 数据流与接口

`ShapeQuery.materialize_intersecting()` 仍使用 KLayout 层级 ROI 空间索引筛选相交 occurrence，但不与查询框做布尔相交。相交图形完成 top 坐标变换后在原生侧一次 `flatten()`，既保留完整真实边，也能在 `LayoutDB` 关闭后安全消费；完全不相交图形不会进入内存。

`macro_boxes(tile_grid, maximum_span_dbu)` 只组合现有 tile，macro ownership 边界严格落在 tile cuts 上。单 tile 大于请求跨度时保留整个 tile，不能为了满足 macro 尺寸制造新的 ownership 边界。

`prepare_macro()` 的处理顺序固定为：完整候选物理合并、真实轮廓提取、边段切分、查询 context 活跃段筛选、全局 tile owner 分配、当前 macro 局部 membership CSR。`MacroPreparation` 只保存当前消费者需要的七项数据；owned 段由活跃段和 owner 按需推导，不保存重复数组，也不引入跨 macro 稳定 ID。处理 ROI 外、但落入边缘 tile halo 的真实边以 `owner=-1` 保留为固定只读 membership，不参与任何更新发布。

## 3. 两层 Halo

- `tile_halo_nm`：tile 光刻与 EPE 的只读 context，不拥有输出写权限；原含糊的 `halo_nm` 公共 CLI/Python 参数已显式迁移。
- `roi_halo_nm`：CPU macro 查询完整相交图形的外扩范围，必须至少为 `tile_halo_nm + max_displacement_nm`，确保 tile context 及允许移动后的来源图形仍在候选范围内。

ICCAD13 的 35×35 数据是频域 Hopkins 核，不是 17-pixel 有限空间卷积核。本轮没有从资产数组尺寸伪造光学半径；`tile_halo_nm` 仍需由用户针对版图和精度做收敛测试。ILT 不提取可移动边，因此继续使用精确 `materialize()` 生成像素 ROI，不经过 `prepare_macro()`。

## 4. 前端验证器

`main/run_mbopc_frontend.py --macro-verify` 在全局预检后逐 macro 执行局部预检、完整相交物化与边段准备。每个 tile 仅在栅格化时截到 `context_box`，并与既有精确 ROI 物化的像素结果逐像素比较。macro 完成后立即释放 Region、边段、membership 和临时像素数组。

验证器不保存整张版图的几何签名集合；macro 唯一写入由不重叠的 owner tile 集合保证，只常驻一个 `bool[core_count]` 覆盖位图和标量计数。内存摘要仅保存峰值 macro 快照，不按 macro 数累积字典。局部预检复用已经打开的 `LayoutDB`，通过 `include_layout_load_bytes=False` 避免把同一源文件解析成本重复计入每个 macro。

## 5. 性能与内存设计

- 层级筛选、实例变换、Region 构造/展平和布尔合并均在 KLayout 原生侧批量完成；Python 不逐 polygon 读取版图。
- 分段后按 NumPy 数组批量筛选；membership 只覆盖当前 macro 所含 tile，并在最终 CSR 分配前执行硬上限检查。
- 全局常驻验证状态为 O(tile 数)，当前 macro 工作集为 O(相交完整候选 + 活跃边段 + 局部 membership + 单 tile 像素)。tile `CoreSpec` 按索引即时构造，不先展开整张网格对象元组；参考端点直接从紧凑参数计算，不复制不需要的法向数组。
- 真实多轮 solver 尚未接入该生命周期；现有 Simple MB-OPC 与 DiffOPC 仍使用完整内存 `MBOPCProblem`。

## 6. 简化与缺陷清理

实现过程中删除了两项可能造成过度设计或内存回退的写法：一是 ownership-box 重算的第二份归属掩码，归属现在只以全局 owner tile 为真源；二是所有 owned 边段的 Python tuple/set 查重，改为 tile 覆盖位图。未增加基类、注册器、磁盘格式、兼容包装或空目录。

复核还纠正了把频域核尺寸解释成空间影响半径的错误方案，相关 Protocol 属性、runner 校验、测试和文档均已删除，没有保留失效分支。旧 `--halo-nm` 被明确拒绝而非静默映射，避免两种 halo 语义长期并存。
