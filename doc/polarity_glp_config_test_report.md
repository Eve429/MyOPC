# 版图极性、GLP 与运行配置测试报告

## 1. 测试范围

- GLP：矩形、多边形、显式/数字层映射、多 top、未使用辅助 LEVEL、非法语句/坐标/极性。
- 极性：clear/opaque 像素互补、跨处理框 halo、法向反转、正位移几何、零位移源图形恢复。
- 配置：六份默认文件、自定义 common/entry、CLI 覆盖、相对路径、GLP 映射、未知键/类型/section/文件。
- 集成：Layout/Geometry、preflight、Simple MB-OPC、DiffOPC、离线 raster/segment、全部 main 入口。

## 2. 真实格式验证

OpenILT `benchmark/ICCAD2013/M1_test1.glp` 通过直接运行：DBU `0.001um`，读取 10 个 Polygon、10 个 ring、52 条数学边，面积 `215344 DBU²`。同一文件通过 MB-OPC preflight：估算 288 segment/288 membership，准备峰值 67680 bytes，扫描完整且接受。源文件只解析一次。

## 3. 自动测试结果

- 极性与 MB-OPC/DiffOPC/离线联动：59 项通过。
- 配置与 CLI 回归：23 项通过。
- GLP、配置、极性、离线与入口最终专项：46 项通过。
- 全仓最终测试：226 项通过，62.59 秒。
- Ruff、compileall 和 `git diff --check` 作为最终静态门禁执行。

opaque 测试显式检查处理框外 halo 为 0，且处理框没有进入 contours；因此不会产生四条虚假可移动边。旧 clear 测试和历史 NPZ 契约继续通过。
