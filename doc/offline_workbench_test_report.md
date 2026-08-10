# 离线光刻与 MB-OPC 工作台测试报告

## 1. 测试环境与范围

- 日期：2026-08-10；
- Python：项目 `myopc` Conda 环境；
- KLayout 0.30.x、NumPy 2.x、PyTorch CUDA；
- 真实文件：`TestReticle/simple.gds`，Layer 1/0，DBU 0.001 µm；
- 自动测试：`tests/workbench/test_offline_workbench.py`。

测试不修改用户 GDS，不把 `output/` 产物纳入 Git。本次用户授权的结构收敛修改了 `geometry` 轮廓契约，并由完整几何/OPC 回归覆盖。

## 2. 自动测试矩阵

| 类别 | 用例 | 结果 |
|---|---|---|
| 像素一致性 | 离线 mask 与 `rasterize_region_canvas` 逐像素比较 | 完全一致 |
| canvas 保护 | 4096×4096 DBU ROI/4 nm 在公开物化前拒绝 | 通过 |
| 复杂度保护 | 层级展开图形上限为 1，在 Region 物化前拒绝 | 通过 |
| 多图形恢复 | 矩形、孔洞、斜边、层级引用、跨 2×4 core | XOR 0、合法 |
| 归档损坏 | 缺少 edge next cache | 加载时明确拒绝 |
| 索引损坏 | membership 等于 segment count | 加载时明确拒绝 |
| 版本迁移 | 带旧字段缺失特征的 segment v1 | 在字段检查前提示重新生成 |
| 元数据损坏 | counts 缺少必需字段 | 统一转换为 ValueError |
| 读取内存 | 极小 `max_archive_gib` | NumPy 分配前拒绝 |
| 光刻入口 | CPU ICCAD13 + 数值 NPZ + 四张 PNG | 通过 |
| OPC 入口 | 离线跨 core problem + 一轮 ICCAD13 | GDS/NPZ/PNG/JSON 完整 |
| 直接运行 | 三个脚本从仓库外执行 `--help` | 全部退出码 0 |

此外从 `C:\Windows\Temp` 使用绝对脚本路径实际完成 raster 准备、CUDA 光刻前向和一轮离线 MB-OPC；三个命令均退出码 0，证明不是只有参数解析能在未安装项目时工作。

合成图形包括 20 nm 孔洞壁、非正交斜边、跨 x=128 DBU 的长边和两种层级变换。零位移重建与加载后的物理 Region XOR 面积为 0，membership 数大于 segment 数且 owner 跨多个 core。

## 3. 自动化结果

```text
Ruff: All checks passed
compileall: passed
pytest: 130 passed in 24.75 s
workbench/相关契约: 23 passed in 12.60 s
workbench statement/branch coverage: 74%
```

结构迁移后全仓库 130 项全部通过。成功路径、两种物化前保护、v1 拒绝、主要损坏输入、真实光刻和真实迭代均已运行。

## 4. `simple.gds` 像素与光刻实测

完整 bbox 为 `[-2000,-1100,-200,2200]` DBU。8 nm 下完整高度需要 413 像素，准备函数正确拒绝整个 bbox；选择 `[-2000,-1100,-200,948]` 后有效尺寸为 225×256。

| 指标 | 结果 |
|---|---:|
| raster 输入文件 | 3,793 bytes |
| 光刻输出 NPZ | 787,806 bytes |
| 执行设备 | cuda:0 |
| 模型前向 | 0.477 s |
| GPU 峰值 | 34,026,496 bytes |
| mask 范围 | 0.0–1.0 |
| nominal 范围 | 1.35e-5–1.0 |

已保存并人工查看 mask、nominal、maximum、minimum PNG。图像方向与版图一致，孔洞保持为空，三个工艺角均无 NaN/Inf。

## 5. `simple.gds` 边段与三轮迭代实测

默认配置为 1024 nm core、512 nm halo、32 nm 最大段长、8 nm pixel。归档关闭源版图后重新加载，再独立运行三轮。

| 数据规模/资源 | 结果 |
|---|---:|
| polygons / rings / edges | 10 / 14 / 107 |
| segments / cores / memberships | 885 / 8 / 2,658 |
| 边段输入 NPZ | 50,681 bytes |
| 归档加载 | 0.021 s |
| 三轮迭代 | 1.244 s |
| 全流程 | 1.513 s |
| GPU 峰值 | 271,534,080 bytes |

| 轮次 | EPE | L2 | PVBand | moved |
|---:|---:|---:|---:|---:|
| 0 | 338 | 2822.466 | 388.928 | 338 |
| 1 | 203 | 1766.541 | 415.595 | 203 |
| 2 | 113 | 1309.422 | 436.144 | 113 |

结果与既有完整入口的历史 `338→203→113` EPE 基准一致，证明归档恢复没有改变 segment 顺序、owner 或轮次屏障。PVBand 上升已原样记录。结果 Region 合法，GDS、最佳位移 NPZ、摘要 JSON 和标注 PNG 均生成。

另以 512/256 nm core/halo 验证 28 cores、3,305 memberships，EPE 为 `339→212→112`；差异来自切分/光学上下文配置变化，不是离线恢复误差。

人工查看标注图后确认：跨 core 外轮廓连续，斜边端点连接，孔洞内外探针方向正确，halo 不产生第二套回写边。

## 6. 最终审计

- `git diff --check` 通过；
- `geometry/` 的授权差异仅为两级 CSR 与删除 EdgeBatch，`layout/` 无差异；
- 新增模块、函数、测试函数均有中文 docstring；
- 所有 helper 有当前调用方，无注册器、空基类或重复 problem/result 结构；
- 归档写入原子化，读取禁止 pickle 并有解压尺寸限制；
- 用户版图和 `output/` 测试产物未进入提交；
- 未推送任何远端。
