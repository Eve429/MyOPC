# DiffOPC 第四阶段测试报告

## 1. 专项覆盖

`tests/opc/test_diffopc.py` 共 14 项：

- 零位移软栅格与参考 mask 严格相等，非饱和位移的 autograd 与中心有限差分一致；
- 外轮廓与孔洞法向翻转后，占据变化方向同步翻转；
- DBU probe 按左下原点像素中心采样，无旧版半像素偏移；
- batch=1/2 的最佳位移和逐轮损失一致，且同一问题按 1-core/2-core 划分仍一致，验证逐 batch backward 与 owner-only 计分不改变数学结果；
- 最佳记录与保存位移来自同一个评价快照；
- 孔洞、斜边、多 core membership 的小位移可统一重建为合法 Region；
- 模拟全局拓扑失败时只保留已评价合法状态；
- 无效轮数、探针距离、chunk、缓存和全零损失权重在求解前拒绝；
- 直接 GDS 入口保存结果 NPZ、GDS、JSON 和最终光刻 manifest。

相关 simple MB-OPC、离线输入及共享缓存 37 项回归通过，证明拓扑检查归位和缓存去重未改变原方法；最终全仓 208 项通过（68.28 s）。

专项覆盖率为 80%（solver 88%、contracts 94%、rasterizer 80%、runner 60%）；runner 未覆盖集中在 CLI 错误打印、重复输入拒绝和设备异常分支。本报告不把入口行覆盖等同于算法分支质量，也不虚报未覆盖路径。

## 2. 真实 Hopkins 验证

输入：`TestReticle/simple.gds` 的 `Layer 1/0`，ROI `(-520,-20)-(-180,320)` DBU；4 cores、44 segments、60 memberships。参数为 pixel=2 nm、tile=256 nm、halo=32 nm、两轮、batch=2、raster chunk=4。

| 设备 | 最佳轮次 | L2 | PVBand | EPE | 优化耗时 | GPU 峰值分配 |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 773→687 | 350→247 | 2→0 | 2.059 s | 0 |
| CUDA | 1 | 773→687 | 350→247 | 2→0 | 2.508 s | 133,264,384 bytes |

CPU/CUDA 连续损失只有浮点尾差，二值指标和最佳轮次完全一致。CUDA 首轮包含运行时初始化，且样本很小，因此表中耗时只证明流程与资源上界，不作为 GPU 加速比。

## 3. 产物与资源验证

CPU/CUDA 入口均保存 `diffopc_result.npz`、`diffopc_result.gds`、`summary.json` 和 4 个 ownership-only 最终光刻 tile；重建 Region 合法，固定 512² shot 估计为 103。summary 分开记录输入、模型、优化、重建输出时间，进程内存检查点和 CUDA 峰值。

初次整图 CPU 冒烟使用 1 nm 像素形成约 26×26 cores，属于不合适的快速验收配置，已主动终止且未作为通过结果。随后使用覆盖真实图形的多 core ROI 完成 CPU/CUDA 验收；这不改变当前仍需完整 CPU 边段常驻的能力边界。
