# 大 Reticle 流式处理独立开发方案

## 1. 当前能力与缺口

当前求解器已经做到 GPU 只保存当前 batch 的 core+halo 张量，CPU 保存紧凑全局边段和一维位移；`gcd_45nm` 的 870 core 全流程已验证。阶段 1 已增加物化前层级容量扫描、`int32`/内存预算拒绝和进程内存检查点，但通过预检后仍会一次物化所选完整 ROI，`RectilinearCoreGrid.cores()` 仍展开全部规则 core，最终候选拓扑检查仍遍历全局轮廓。

因此当前实现适合预检通过且内存能容纳完整物理 Region 和边段数组的 reticle，不应宣称已经覆盖任意超大、极稀疏整版。24 GiB GPU 不是主要限制；真正的上界通常是 CPU 上完整 ROI Region、全局 segment/membership 和每轮全局重建。超限时当前只安全拒绝，以下 macro/shard 仍是未实施阶段 2。

## 2. 目标不变量

- 固定参考边段、全局下标、法向、拓扑顺序和 owner；普通轮次不 remesh；
- 所有 tile 基于同一个只读 `d_current`，只由 owner 写 `d_next`，轮次屏障后发布；
- halo 覆盖光学有效半径与最大位移，跨 core 边段不按 core 边界再次切分；
- GPU 只保留当前 batch，CPU 不常驻整张 reticle 像素图；
- 源层级版图只读，诊断 PNG/GDS 只在明确请求时生成；
- 不因 macro 边界改变 segment 身份、整数坐标或拓扑顺序。

## 3. 分阶段设计

### 3.1 稀疏 active core

输入扫描先按层级 bbox/原生 ROI 查询得到有图形或可能受 halo 影响的 active core ID，只为 active core 建立 CSR。规则 `x_cuts/y_cuts` 继续定义全局坐标和 owner 规则，避免引入第二种几何语义；求解器以 active ID 批量生成 `CoreSpec`，不展开空 core 对象。

该阶段必须测量 active ID 表、CSR 和层级查询的额外成本。若版图接近满铺，稀疏表不得比当前密集路径显著更慢；可以保留一个由实际密度阈值选择的内部快路，但不能暴露两个 problem 类型。

### 3.2 macro ROI 准备

planner 将整版切成远大于光刻 tile 的 macro，每个 macro 带准备 halo：

1. 只物化 macro+halo 的层级 Region；
2. 规范化并提取候选数学边；
3. 用全局坐标产生确定性几何签名，仅用于相邻 macro 去重；
4. 中心 macro 负责发布边身份，halo 只补上下文；
5. 完成去重后追加到全局紧凑数组，立即释放当前 macro Region。

签名不能作为跨 remesh 永久 key。它只在一次 prepare 的相邻 macro 合并阶段使用，至少包含 Layer、规范化 ring 拓扑位置、全局整数端点和方向；斜边不能经过两次独立 Region 裁剪后再按裁剪端点匹配，否则会重现 33/34 DBU 分点差异。

### 3.3 分块拓扑与局部物化

全局数组继续保存 segment→edge/ring/polygon 的紧凑索引，但 polygon 顶点可按 macro 分块存储。每轮只重建当前 batch 关联的 dirty polygon；发布屏障按 dirty polygon 做绕向/hole 检查。最终 GDS 以 macro 顺序流式写入独立结果版图，边界 polygon 只能由固定 owner macro 写出。

第一版仍遍历全部 active tile，待光学影响半径和 dirty 传播得到验证后才允许跳过 clean tile。不能依据“本轮没有 owner 更新”直接跳过相邻 tile，因为其光学上下文可能受邻近 dirty polygon 影响。

## 4. 接口迁移原则

- 不新增 `SparseMBOPCProblem`；在当前 `MBOPCProblem` 有真实调用需求时，把 core ID/CSR 表达扩展为紧凑 active 映射；
- 不修改 `layout`/`geometry` 私有实现来绕过公共 ROI 查询，除非先证明现有每 macro 批量接口不足并逐次取得授权；
- `opc.input.edge` 负责 macro 边合并和固定身份，`opc.iteration.mbopc` 只消费准备结果；
- ILT 可复用 active tile/macro raster 调度，但不依赖 segment、owner 或拓扑重建。

## 5. 验收测试

- 空白占比 90% 以上的合成整版：active CSR 内存相对密集 core 显著下降；
- 满铺整版：稀疏管理开销不超过当前准备时间的 10%；
- 单个矩形、孔洞、斜边跨 2×3 core 且跨两个 macro：segment 数、owner 和零位移 XOR 与一次性准备完全一致；
- SREF/AREF 多 occurrence：默认按物理 occurrence 独立，不能误改 master 后传播；
- 两个 macro 完成顺序互换：`d_next`、EPE、L2/PVBand 和输出 GDS 一致；
- 24 GiB GPU/64 GiB CPU 目标机上，用逐级放大的合成版图记录 CPU/GPU 峰值并验证失败前有明确内存预检。

## 6. 实施顺序

1. 只读测量真实目标 reticle 的空 core 比例、Region/segment/membership 峰值；
2. 实现 active core ID 与当前结果等价测试；
3. 实现 macro prepare、相邻 macro 去重和一次性 prepare 对照；
4. 实现 dirty polygon 局部重建，保持全 tile 光刻遍历；
5. 最后再评估 dirty tile 跳过、层级 cell variant 和 checkpoint。

每一步独立提交并可回退。没有完成一次性结果对照、跨 macro 斜边/孔洞测试前，不进入下一步。
