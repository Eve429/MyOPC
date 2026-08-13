# Macro 未裁剪物化与双 Halo 测试报告

## 1. 环境与范围

- 日期：2026-08-13
- Python：`D:\app\miniforge\envs\myopc\python.exe`
- 平台：Windows，CPU 功能与全量回归
- 保护范围：`geometry/` 零修改；`layout/` 只有授权的未裁剪物化接口

专项测试覆盖完整相交图形生命周期、macro/tile 边界归属、虚假边、逐 tile 栅格、两层 halo、容量保护和 ILT 路径隔离。

## 2. 几何与归属矩阵

- 跨查询框矩形：只保留真实水平边，不产生查询框上的竖直边；
- 斜多边形：跨多个 macro 后不在 context 四边形成整段，单 macro 与多 macro 的 owned 真实边集合一致；
- 中空图形与窄环：外环、孔洞均来自完整 occurrence，查询边界不成为新环；
- 重叠图形：先物理合并再提边，内部重叠边不进入 segment；
- 2×3 tile：每个活跃 segment 具有唯一全局 owner，halo 只增加只读 membership；
- 处理 ROI 外真实边：落入边缘 tile halo 时以 `owner=-1` 保留，只参与 membership、不进入 owned 发布；
- 层级 SREF：完整 occurrence 在 top 坐标物化，数据库关闭后 Region 仍有效；
- membership 上限：在最终 CSR 大数组创建前抛 `MemoryError`；
- ILT：真实 GDS 的 LevelSetILT 一轮运行成功，monkeypatch 证明未调用 `prepare_macro()`。

## 3. 入口与参数回归

`--macro-verify` 必须提供真实版图；`roi_halo_nm < tile_halo_nm + max_displacement_nm` 时立即拒绝；`tile_halo_nm` 必须与像素格对齐。旧 `--halo-nm` 由 argparse 明确拒绝，配置文件和 Python API 均使用 `tile_halo_nm`。普通 frontend、离线 segment、Simple MB-OPC 与 DiffOPC 的调用点同步迁移。

## 4. 真实版图冒烟

对 `TestReticle/simple.gds` 的 Layer 1/0、完整 bbox 执行：

```powershell
D:\app\miniforge\envs\myopc\python.exe main/run_mbopc_frontend.py `
  TestReticle/simple.gds --layer 1/0 --tile-size-nm 512 `
  --tile-halo-nm 64 --roi-halo-nm 88 --macro-size-nm 1024 `
  --pixel-nm 8 --max-displacement-nm 24 --macro-verify --json
```

结果：4×7 共 28 tile，分为 8 macro；全局预检估算 929 segment/2039 membership；逐 macro 实际累计 885 owned segment/1399 membership；栅格差异 0 像素，重复 owner 0。`macro_prepare_and_raster=0.452 s`，总流程 `0.574 s`；报告的峰值 macro NumPy 常驻数组为 12,156 bytes，进程 peak working set 从起始约 406.0 MiB 到约 409.7 MiB。该数字是此小版图、此机器上的功能冒烟，不外推为大 reticle 性能承诺。

## 5. 质量门

精简后的核心定向测试为 `39 passed in 40.24s`，覆盖 macro 几何、入口、ILT、MB-OPC 和 preflight；修改文件 Ruff 全部通过。增加属性、ROI 外只读 halo 边和按需 core 回归后，最终全仓为 `248 passed in 81.83s`。全目录 Ruff、compileall、中文 docstring、重复函数体、调用点、Markdown 链接/围栏与 `git diff --check` 均通过。

测试过程发现三类设计问题并在最终实现前清除：deep Region 关闭数据库后为空，改为原生展平并保留生命周期回归；KLayout `flatten()` 会静默丢属性，属性模式改用保持 shape class 的原生 `merged` 并增加回归；全局边段签名 set 会破坏有界内存，改为 O(tile 数) 覆盖位图。另有一次模型概念复核否定“35×35 频域核等于 17 pixel 空间半径”，最终代码不包含该错误校验。
