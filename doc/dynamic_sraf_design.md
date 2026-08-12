# 动态 SRAF 接入设计

## 1. 当前边界

本文件描述后续把 DiffOPC 风格 SRAF 生成接入 MB-OPC/DiffOPC 的实施方案，不代表当前版本已经能够生成 SRAF。当前实现只优化准备阶段已有的主图形边段；没有为本方案增加占位接口、注册器或空目录。

## 2. 为什么不能直接修改现有位移数组

SRAF 是新多边形，会增加 ring、数学边、segment、owner 和 core membership。现有 `MBOPCProblem` 的 segment ID 只在一次已准备问题内有效，Adam 的参数和一、二阶矩也与数组长度逐项对齐。因此动态加入 SRAF 必须发生在全局轮次屏障，不能在某个 GPU batch 计算过程中追加数组，否则其他 tile 会读取不同的问题版本。

目标图形与优化 mask 必须分离：目标始终是设计意图，不因 SRAF 改变；当前 mask 则由主图形和已经发布的 SRAF 共同组成。EPE 只对主图形探针计分，SRAF 通过连续 L2、PVBand 和规则检查优化，避免把辅助图形误当成必须打印的目标边。

## 3. 一轮更新的原子流程

1. 本轮所有 tile 基于同一只读问题版本和 `d_current` 计算，仍遵守 owner 唯一写入和全局屏障。
2. 流式累计热点摘要，只保存候选位置、强度、工艺角差异和来源 tile，不保留整张 reticle tensor。
3. 屏障处统一生成候选 SRAF，多批完成最小宽度、间距、主图形间距、处理框和跨 tile 去重校验。
4. 合法 SRAF 先按全局 Polygon 顺序追加轮廓，再追加数学边和 segment；旧 segment 保持稳定前缀，旧 ID 不改变。
5. 为新增 segment 一次性计算 owner 和 halo membership；全部数组构造与交叉校验成功后才原子发布新问题版本。
6. 位移、梯度和优化器状态扩容：旧前缀逐项复制，新 segment 的位移、梯度、Adam `exp_avg/exp_avg_sq` 均初始化为零。
7. 只失效受新 SRAF 光学影响的当前 mask tile；固定设计目标缓存不失效。下一轮所有 tile 同时看到新版本。

如果候选 SRAF 与既有图形接触或合并，旧拓扑前缀不再成立。第一版必须拒绝这类候选；未来若确实需要合并，只能显式 full remesh，并同步重建所有 segment ID、owner、membership、位移映射和优化器状态，不能用局部补丁掩盖拓扑变化。

## 4. 数据与内存约束

- 不增加逐 segment 的字符串 role 字段。发布版本只需记录 `primary_segment_count`，其前缀为主图形，后缀为 SRAF；Polygon/ring 的分组仍由现有 CSR 推导。
- SRAF 候选在 CPU 以紧凑整数框或多边形批次存在，验证完成后才转为 `ContourBatch`；禁止每轮为整张 reticle 重建完整 Region、PNG 或 GDS。
- 热点跨 core 时按全局坐标和确定性 key 去重，最终 owner 仍由候选 segment 的参考中点唯一确定。非 owner tile 只能通过 halo 读取。
- 主 target cache 与目标光刻数组不变；当前 mask cache 按 dirty tile 失效。halo 必须继续覆盖光学影响半径、最大位移和新增 SRAF 的影响范围。

## 5. 与不同 OPC/ILT 方法的关系

- Simple MB-OPC 可以把 SRAF 生成当作“输入构造阶段的周期性扩展”，原有边移动与全局屏障不变。
- DiffOPC 可在发布后把新增 segment 直接加入可微位移张量；旧 Adam 状态按稳定前缀复用。
- 像素 ILT 不需要 segment 身份，但可复用同一候选生成和 MRC 验证结果作为 mask 初始化；不得让 ILT 基础层反向依赖 MB-OPC 求解器。
- GLP/GDS/OASIS 只影响 `LayoutDB` 输入，不改变 SRAF 的内部全局坐标契约。输出第一版仍统一写 GDS。

## 6. 实施验收矩阵

- 单个 SRAF、多个 SRAF、孔洞邻近、斜边邻近和处理框边缘候选。
- 候选跨 2/3/4 个 core，改变 batch size、tile 数和遍历顺序后结果一致。
- 追加前后旧 segment ID、位移、Adam 两个矩量逐项相同；新状态严格为零。
- 主图形 EPE probe 数不因 SRAF 增加，L2/PVBand 能感知 SRAF。
- MRC 拒绝、跨 tile 重复和原子发布失败均不改变当前问题版本。
- 受控内存测试证明峰值与“当前热点批次 + 新增边段”相关，而不是整张 reticle 像素数。

