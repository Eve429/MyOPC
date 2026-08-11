# 离线光刻与 MB-OPC 工作台开发报告

## 1. 目标与结果

本阶段在 `tests/workbench/` 建立可直接运行的离线专项工作台。版图像素化或边段构造只执行一次，后续可以单独修改、分析和计时光刻模型或 OPC 迭代，不重复读取 GDS、物化 Region 或切分边段。

初版没有修改 `layout/`、`geometry/` 和现有 OPC 数据结构；后续在用户授权的数据契约收敛中同步升级边段归档，加载后仍直接恢复唯一的 `MBOPCProblem`，没有建立兼容 problem 类型。

## 2. 公共测试接口

| 接口 | 输入 | 输出 | 主要约束 |
|---|---|---|---|
| `materialize_raster_input` | GDS/OASIS、Layer、ROI、pixel/canvas | 内存 mask、metadata | 预检通过后才物化，不写中间文件 |
| `prepare_raster_input` | GDS/OASIS、Layer、ROI、pixel/canvas | raster version 1 NPZ | ROI 必须直接放入单个 canvas |
| `load_raster_input` | raster NPZ | `float32` mask、metadata | 左下原点、值域 `[0,1]`、禁止 pickle |
| `resolve_raster_input` | 版图或 raster NPZ | `float32` mask、metadata | 按 `.npz` 自动分派，两个 runner 共用 |
| `prepare_segment_input` | 版图、ROI、分段与 core 配置 | MBOPC version 2 NPZ | 预检后才物化和构造 owner |
| `load_segment_input` | MBOPC NPZ | `MBOPCProblem`、metadata | 不依赖源 GDS，不重新分段 |

`run_lithography_test` 返回按工艺条件名称索引的张量字典；`run_mbopc_iteration_test` 返回现有 `SimpleMBOPCResult`。没有为工作台新增结果结构、算法基类、注册器或占位方法目录。

## 3. 数据格式

像素归档保存固定 `float32[canvas,canvas]` mask、有效宽高、Layer、ROI、DBU、pixel 和 `bottom_left` 方向。模型输入不执行 PNG 的上下翻转；只有保存人眼图片时才 `flipud`。

边段归档保存：

- contour 顶点、ring offsets 和 polygon ring offsets；
- edge next/polygon 两个 `int32` 缓存和单位外法向；
- segment 的 ring offsets、edge ID、`t0/t1`；
- grid x/y cuts、halo、唯一 owner 和 membership CSR；
- Layer、ROI、DBU、分段配置、tile/halo 与规模统计。

边段 NPZ 不压缩，避免大版图保存时同时承担压缩 CPU 和临时内存；像素数组固定较小且通常稀疏，使用压缩 NPZ。两者都在同目录写临时文件后用 `os.replace` 原子发布。

`save_problem_npz` 继续是不可恢复诊断快照，当前为 version 3。工作台 segment version 2 是独立的可恢复协议；version 1 明确提示重新生成，不把跨版本 remesh 身份问题塞入转换分支。

## 4. 内存与异常边界

准备前依次执行：

1. 源文件大小上限，默认 4 GiB；
2. 像素 ROI 的精确宽高，默认不得超过 256×256；
3. KLayout 原生递归迭代统计层级展开图形和顶点；
4. 按当前 `fragment_edges` 公式估算 segment；
5. 按原始边 bbox、规则网格和 halo 估算 membership 上界；
6. 按文件解析、原生几何和 NumPy 临时数组估算峰值，默认上限 8 GiB。

严格预检额外读取一次版图，但不会构造目标 Region；通过后才使用现有 `LayoutDB` 公共路径正式读取。这个额外 I/O 只发生在一次性测试数据准备，不进入模型或迭代热路径。物化后的真实数组规模还会二次复核。

读取端先利用 ZIP 成员声明检查总解压量，再以 `allow_pickle=False` 读取。恢复时先检查格式版本，再验证 nested contour CSR、两个 edge cache、单位法向、segment 全局顺序、每条边的 `[0,1]` 连续覆盖、ring 归属、membership 越界/重复、owner context 唯一出现和 metadata 计数。

## 5. 直接运行与性能路径

相关文件均用自身路径解析仓库根，可从外部工作目录直接执行，不需要 `pip install`。光刻与 SimpleILT 可直接把版图 ROI 物化为 CPU mask 后送入模型，也可加载已保存 mask；直接路径不落隐式 NPZ。离线迭代入口只加载 problem 和配置，之后保持原求解器的 `current/next` 轮次屏障、唯一 owner 写入和 halo 只读语义。

模型输出和 tile tensor 仍按 batch 释放；新增工作台不会把整张 reticle tensor 常驻 GPU。最佳结果继续通过全局参考边一次重建，不按 core 裁剪拼接。

## 6. 过度设计复查

- 只新增一个数据模块、两个当前可运行入口和一组测试，没有生产目录或空接口；
- metadata 使用普通字典，不新增与 `MBOPCProblem` 重复字段的数据类；
- 预检 helper 均有当前像素或边段准备调用方，归档 helper 同时服务两个入口；
- 两个生产 runner 只改输入边界，模型、优化、评价和结果保存仍各保留唯一实现；
- 直接版图支持只增加一个已有模块内的自动分派函数，两个真实 runner 共用，不增加输入类或注册器；
- 加载器的额外分支均对应内存安全或“损坏文件静默算错”的明确风险；
- 完成调用点和重复实现审计后，没有发现仅为修复旧错误保留的包装或变量。

测试和实测细节见 [离线工作台测试报告](offline_workbench_test_report.md)。
