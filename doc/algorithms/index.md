# 算法文档目录

本目录解释当前源码中具有独立数学语义的算法实现。文档关注“输入如何变成
输出、关键公式、状态不变量和失败边界”，不替代 `contracts/` 中的接口契约，
也不替代 `architecture/` 中的模块依赖说明。

## 当前算法

| 算法 | 文档 | 代码入口 | 用途 |
|---|---|---|---|
| 边段几何重建 | [`reconstruct_geometry.md`](reconstruct_geometry.md) | `opc/input/edge/reconstruction.py::_reconstruct_geometry` | 将参考边段与法向位移恢复为合法轮廓/Region |
| Abbe 光刻成像 | [`abbe.md`](abbe.md) | `lithography/torchlitho/model.py::TorchLithoLithography._abbe_aerial` | 有效光源逐点相干成像叠加（含 R2 缺陷修正记录） |
| Hopkins 光刻成像 | [`hopkins.md`](hopkins.md) | `lithography/torchlitho/tcc.py` + `model.py::_hopkins_aerial` | TCC 构造、本征核分解与点源 rank-1 证明 |

## 阅读顺序

1. 先读 `doc/architecture/system.md`，了解该算法在整体系统中的位置；
2. 再读 `doc/contracts/edge.md`，确认 `SegmentBatch`、位移和重建失败契约；
3. 最后读具体算法文档和对应测试，核对公式与实现细节。

## 算法文档约定

- 坐标统一使用 GDS/KLayout DBU；连续中点和法向位移可以是 `float64`，写回
  轮廓前才进行整数化；
- 伪代码只描述当前源码已经实现的行为，不描述尚未实现的优化方向；
- 如果算法同时服务 Simple MB-OPC、Gradient MB-OPC 或公共管线，会明确标出
  每个调用方使用的输出。
