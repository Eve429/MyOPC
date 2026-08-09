# MyOPC 项目开发规则

## 代码与注释
- 所有第一方 Python 文件必须有中文模块 docstring；所有函数、方法和测试函数必须有中文 docstring。
- 函数内部对坐标方向、数据不变量、性能路径、内存上界、边界归属和异常原因提供详细中文注释。
- 注释放在紧凑逻辑块之前，解释原因和约束，不逐行复述语法；保持项目现有紧凑式排版，不运行会大幅展开代码的自动格式化。

## 架构与性能
- 依赖方向保持为 `layout -> geometry -> opc.input -> opc.input.edge`；`opc.iteration.<method>` 可依赖输入层、顶层 `lithography` 和 `evaluation`，这些基础层不得反向依赖具体迭代方法。
- 未经用户逐次确认，不得修改现有 `layout/` 和 `geometry/`；如果新功能无法仅通过公共接口实现，必须先停止并说明必要性、影响和最小改法。
- 保留层级和局部 ROI；批量跨越 Python/KLayout 边界，避免逐 polygon、逐 edge 或逐 segment 的解释器热循环。
- 重复迭代的数据必须缓存；诊断、PNG、GDS 和完整几何物化只在明确请求时执行。
- 新抽象必须有当前调用方；不得为了假设中的未来方法建立空接口、注册器或无实现目录。

## 未来优化内容
- 本节记录已经确认但尚未全部实现的方向，不得在报告中描述为当前能力；实际开发仍须满足测试、文档、简化审计和用户逐次授权要求。
- 大 reticle 的普通 OPC 轮次固定参考边段、全局索引、法向、拓扑顺序和 owner，不重新提边或切分；迭代只更新相对固定参考边界的全局绝对位移。只有显式 remesh 才能改变分段，并必须同步重建归属和优化器状态。
- 全部 tile 必须基于同一只读 `d_current` 计算，owner 是 segment 的唯一写入者；非 owner core 只能通过 halo 读取或提交只读误差贡献。所有 tile 完成后才能发布 `d_next`，禁止边计算边修改本轮状态造成顺序相关结果。
- 大 reticle 使用流式 GPU batch：GPU 只保留当前若干 core+halo 的 mask、光刻中间量和局部评估结果；CPU 保留紧凑参考边段、归属、membership、`d_current/d_next` 和少量状态，不得把整张 reticle 像素图常驻 GPU。
- 更新合并应预分配 `d_next` 并按 owner segment index 直接 scatter，使用 epoch/bitset 检测重复写入；当前 segment 身份仅在一次已准备问题内有效，不为尚不存在的跨进程提交或 checkpoint 常驻稳定 key。
- 普通迭代不得每轮重建完整 reticle Region、PNG 或 GDS；只物化当前 tile 所需边段，并按 dirty polygon/tile 做局部重建或栅格化。第一版为保证正确性可每轮遍历所有 tile，确认光学影响半径后才能增加 dirty tile 跳过。
- halo 必须覆盖光刻模型的有效影响半径和最大允许边位移；边段即使跨 core 也不按 core 边界重新分段，固定 owner 的更新在轮次屏障后供所有 context 读取。
- 当前 OPC 物化会把 SREF/AREF occurrence 转为 top 全局坐标，并在物理合并后失去 master cell、源 shape、instance path 和 transform 映射；因此当前对一个 occurrence 的修正不会修改源 cell，也不会自动传播到其他引用。
- 未来层级 OPC 输入应按需保留 `source_cell_id`、`source_shape_id`、`instance_path`、`instance_transform` 和 `occurrence_id`。默认仍按物理 occurrence 及其上下文独立优化，不得直接修改 master cell 后无条件传播到所有引用。
- 未来层级输出优先按上下文等价性复用修正结果：相同修正的实例共享 OPC cell，不同修正克隆 cell variant 并重定向对应引用，无法安全复用的局部区域允许受控 flatten；源 GDS 始终只读。

## Bug 修复与交付
- 每个 bug 修复必须有可复现的回归测试，并在修复后搜索调用点，删除仅服务于旧错误的函数、包装层、分支和变量。
- 最终交付前执行完整差异、未调用函数、重复实现、异常入口和覆盖率未命中分支审计，并在开发报告记录清理结果。
- 每项功能同步更新项目开发手册、测试手册、专项开发/测试报告以及 `task_plan.md`、`findings.md`、`progress.md`。
- 关键阶段只做本地 Git commit；未经用户明确授权不得推送远端。用户 GDS、图片和无关工作树修改必须保留并排除在功能提交之外。
