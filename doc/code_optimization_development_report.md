# 代码性能与可读性优化开发报告

## 1. 范围与结论

本轮只优化已有真实调用链，没有引入新结构体、注册器或算法接口。修改覆盖 Layout/Geometry API 收敛、轮廓与栅格临时内存、ICCAD13 三工艺角计算、MB-OPC tile batch 和 owner 索引。大 reticle 稀疏化没有混入本轮代码，见[独立方案](large_reticle_streaming_plan.md)。

两个关键本地提交为：

- `ef34cf3`：收敛 Layout/Geometry 依赖与输出职责；
- `937cbdc`：复用光刻频谱并压缩 OPC 批处理内存。

没有推送远端，没有修改用户 GDS 或既有输出图片。

## 2. Layout/Geometry 收敛

### 2.1 依赖与文件减法

`layout/writer.py` 原先导入 `geometry.PatchSet`，使基础版图层反向依赖几何输出层；`layout/layer.py` 只有 `LayoutDB.query` 一个调用方。本轮执行直接收敛：

- `PatchWriter` 移入 `geometry/patch.py`，公共入口改为 `geometry.PatchWriter`；
- 删除 `layout/writer.py`、`layout/layer.py`，不保留兼容别名；
- Layer 去重、规范化和排序合入 `LayoutDB.query`；
- 关闭状态校验合入 `_native_layout`，删除重复 `_assert_open` 包装；
- `ShapeQuery.database` 使用 `TYPE_CHECKING` 下的 `LayoutDB` 类型，不在运行时制造循环导入。

修改后单独 `import layout` 不会加载任何 `geometry` 模块。`PatchSet.add` 复用已经构造的 ownership `Region` 完成裁剪，不再重复创建同一矩形 Region。

### 2.2 连续轮廓与原位栅格

`extract_contour` 原先为每个 ring 创建 Python 点列表和 NumPy 小数组，最后 `concatenate`。现在一次追加到三个 `array('q')` 连续缓冲，再通过 `np.frombuffer` 建立 `int64` CSR 视图。`ContourBatch` 会持有缓冲引用，返回后不会出现悬空内存。

20,000 个矩形的只读对照中，耗时约从 563.74 ms 降到 456.30 ms，tracemalloc 峰值从 6,679,859 bytes 降到 1,845,632 bytes，下降约 72%。

两个栅格入口都改为在 KLayout 返回的面积矩阵上原位除法和裁界；PNG 路径继续原位乘 255 和取整，只在最终写入 `uint8` 图片时转换。坐标方向、部分像素覆盖、孔洞和 ROI 裁剪语义未改变。

## 3. ICCAD13 光刻热路径

优化前 nominal、maximum、minimum 分别执行 `fft2(dose * mask)`。由于 FFT 线性且 aerial 强度为场幅平方，本轮改为：

1. 准备 mask 后只执行一次单位剂量 FFT；
2. focus kernel 传播一次，单位剂量强度分别乘 `dose_nominal²`、`dose_max²`；
3. defocus kernel 传播一次，强度乘 `dose_min²`；
4. 三个工艺角继续独立执行 sigmoid 和尺寸恢复。

该实现不缓存跨调用 tensor，不预展开 kernel，不增加持久显存，并保留 PyTorch autograd。CUDA batch=8、256×256、20 次中位数从 25.0241 ms 降到 16.4560 ms，提升 1.5207 倍；峰值显存均为 277,296,128 bytes，最大逐像素误差为 5.82e-7。

## 4. MB-OPC tile batch

### 4.1 target 生命周期

`_target_tile` 现在始终返回并缓存 `uint8`。每个 batch 预分配一块 `uint8 target`、一块 `float32 current mask` 和一块 `bool ownership`，逐 tile 原位填充；target 只在一次性传到模型设备时转为 `float32` 并原位除以 255。

以 batch=64、256×256 估算，CPU target 临时量从 32 MiB 降到 4 MiB。微基准中该传输路径从约 20.565 ms 降到 9.582 ms。current mask 仍保留未量化面积覆盖率，零位移有图形的 core 继续精确栅格化参考 Region，不把 target 量化值偷换为 current。

### 4.2 owner 索引

`_owner_indices` 原先为每个 core 对完整 `owner_indices` 做一次 `flatnonzero`，复杂度为 `core_count × segment_count`。现在从已有 `core_offsets/member_segment_indices` CSR 取当前 core 的 owner+halo membership，再筛选唯一 owner，复杂度与总 membership 数成正比。

223,553 segment、870 core 的有界合成对照从 45.908 ms 降到 13.178 ms。segment 全局下标、owner 唯一写、halo 只读以及轮次屏障均未改变。

## 5. 可读性与过度设计复查

- 净删除两个源码文件，没有新增生产模块或数据结构；
- 没有为未来大 reticle、ILT 或其它 OPC 方法预建空接口；
- 保留 `ContourBatch`、`SegmentBatch`、`MBOPCProblem` 三个职责不同且有当前调用方的数据契约；
- 没有保留 `layout.PatchWriter`、`normalize_layers`、`_assert_open` 等兼容包装；
- 修复已漂移的 MB-OPC 前端基准字段，并增加直接 CLI 回归；
- owner、target 和光刻优化都直接落在现有热路径，没有建立第二套求解流程。

## 6. 当前限制

当前完整 ROI 仍会物化为一个 `RegionBatch`，规则网格仍会展开全部 core；这对现有 `gcd_45nm` 验证合理，但不是任意整张大 reticle 的最终内存架构。该问题需要独立处理 active core、macro ROI 和跨 macro segment 身份，不能靠本轮局部函数优化宣称解决。
