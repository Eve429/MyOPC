# FAQ 契约修复开发报告

## 1. 修复范围

本次只处理开发手册审查确认的现实问题：Geometry/OPC raster 公共数组方向不一致、simple MB-OPC 最后一次更新被丢弃、Region 生命周期描述错误，以及 Ruff 扫描用户 notebook 的配置缺口。用户明确授权修改 `geometry/`；`layout/` 生产代码没有修改。

## 2. Raster 统一

`geometry.iter_region_coverage_tiles` 继续作为唯一面积覆盖率底层，KLayout 裁剪、合并、分块和归一化逻辑没有变化。`geometry.render_region_batch` 与 `opc.input.rasterize_region_canvas` 的公共返回数组现在都以第 0 行表示最低 Y：前者为可变图幅 `uint8`，后者为固定 canvas `float32`。

上下翻转只发生在三类图片边界：原子保存 PNG、Pillow 查看器和 OPC 边界标注底图。算法不再需要根据 raster 来源判断方向，也不会把图片坐标误传给探针或光刻模型。该改动没有增加大型临时数组：`np.flipud` 返回负步长视图，Pillow 编码才消费图片方向数据。

## 3. MB-OPC 迭代语义

旧实现把 `iterations` 同时当作评价循环次数和更新上限，末轮在发布前退出，导致 `iterations=N` 最多提交 N−1 次，且 `iterations=1` 只评价初态。

当前契约为：`iterations=N` 表示最多提交 N 次全局同步更新。求解器评价初态，每次合法候选通过全局重建守卫并跨屏障发布后，再评价更新态。完整执行产生 N+1 条状态记录；最终记录不提出下一候选，因此 `step_dbu`、`moved_segments` 和 `rejected_segments` 均为 0。最佳位移只从这些已实际经过光刻/EPE 的状态中选择。

这一实现保持既有性能边界：GPU 仍只保存当前 batch，CPU 仍只常驻 `current/next/best/written` 等一维状态；额外代价是一轮最终状态的 tile 光刻评价，这是保证最后一次更新可比较所必需的计算，不保存整张 reticle tensor。

## 4. 生命周期与工具配置

实测和回归确认：未执行的 `ShapeQuery` 依赖打开的 `LayoutDB`；`materialize()` 返回的 `RegionBatch` 已独立持有 ROI Region，数据库关闭后 Polygon 数、面积和后续计算保持有效。开发手册已删除“已物化 Region 会变空”的错误说法。

`pyproject.toml` 新增 Ruff 排除项 `Test/klayout.ipynb`，并把门禁固定为语法、未定义名称和确定性导入错误规则 `E4/E7/E9/F`；`E402/E702` 对应项目直接运行入口和紧凑式排版的既有约束。这样 Ruff 升级不会把中文标点、命令行 `print` 等风格偏好扩展成全仓重写任务，用户 notebook 也不会被扫描或自动修复。

## 5. 简化审计

修复没有新增坐标枚举、raster 包装类、兼容函数或第二套求解器。Geometry 只移动翻转边界；MB-OPC 继续使用同一个循环、缓存、owner 过滤和拓扑重建函数。旧 N−1 特例测试被直接替换为单次更新真实提交回归，没有保留旧行为开关。
