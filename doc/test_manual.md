# MyOPC 测试手册

## 1. 环境与直接运行

仓库不需要安装为 Python 包。当前已验证解释器：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
```

运行依赖 KLayout、NumPy、Pillow 和 PyTorch；测试依赖 pytest、pytest-cov、Ruff 和 psutil。

## 2. 前端输入验证

默认合成案例：

```powershell
& $python run_mbopc_frontend.py
```

真实版图：

```powershell
& $python run_mbopc_frontend.py TestReticle\gcd_45nm.gds `
  --layer 11/0 --grid 2 1 --probe-distance-nm 16 `
  --output-dir .benchmarks\gcd_frontend --skip-geometry-suite --json
```

该入口验证规范化、分段、唯一 owner、halo membership、固定参考位移、全局重建和 core 覆盖。输出为：

- `summary.json`：计数、耗时、内存和几何一致性；
- `segments.npz`：格式版本 2、无稳定 key、按全局 segment 下标对齐；
- `overview.png`：core/owner/法向/inner/outer probe 标注；
- `reconstruction.gds`：参考与重建结果。

`--probe-distance-nm` 只控制展示探针，必须与 `--corner-nm` 相互独立。

## 3. 完整 MB-OPC

默认运行：

```powershell
& $python run_mbopc.py
```

整张 `gcd_45nm` 三轮复现：

```powershell
& $python run_mbopc.py TestReticle\gcd_45nm.gds `
  --layer 11/0 --iterations 3 --batch-size 8 --device cuda `
  --output-dir .benchmarks\gcd_mbopc --preview --json
```

完整入口只输出 `summary.json`、结果 GDS 和可选 PNG，不输出 NPZ。`--box` 可选择 DBU ROI；`--tile-size-nm` 和 `--halo-nm` 必须能由版图 DBU 精确表示且满足像素对齐。显存不足优先降低 `--batch-size`，不会改变 owner 和轮次屏障语义。

## 4. 自动测试与静态检查

```powershell
& $python -m pytest -q
& $python -m pytest tests\opc tests\lithography tests\evaluation `
  --cov=opc --cov=lithography --cov=evaluation --cov-branch `
  --cov-report=term-missing -q
& $python -m ruff check layout geometry opc lithography evaluation tests `
  benchmarks\benchmark_mbopc_frontend.py run_layout_geometry.py `
  run_mbopc_frontend.py run_mbopc.py
& $python -m compileall -q layout geometry opc lithography evaluation tests `
  run_layout_geometry.py run_mbopc_frontend.py run_mbopc.py
```

当前结果：全仓库 117 项通过；OPC/光刻/评价专项 77 项通过，综合语句/分支覆盖率 92%；Layout/Geometry 专项 38 项通过，综合覆盖率 91%。阶段 29 新增 3 项热路径回归，分别锁定 ownership 不物化法向、零位移不重建 contour/Region，以及 current mask 不被 uint8 target 偷换。

## 5. 严格性能基准

```powershell
& $python benchmarks\benchmark_mbopc_frontend.py --strict
```

严格门槛检查 5,000 个图形、110,000 个 segment：零位移 XOR、最大段长、owner 完整性、halo 稀疏性，以及紧凑常驻数组相对展开表示至少节省 60%。当前结果：

| 指标 | 阶段 29 当前值 | 本轮修改前基线 |
|---|---:|---:|
| prepare | 115.55 ms | 126.69 ms |
| materialize | 12.19 ms | 12.31 ms |
| zero reconstruct（显式 API） | 576.01 ms | 579.56 ms |
| persistent arrays | 2.441 MiB | — |
| 相对展开节省 | 69.38% | 69.38% |

基准不再测试 key lookup，因为生产迭代直接使用数组下标且已删除该路径。

补充热点等价基准：11 万 segment 的 `build_ownership` 中位耗时从 40.20 ms 降到
37.15 ms，tracemalloc 峰值从 24.56 MiB 降到 17.85 MiB，三组归属数组逐项一致；
500 polygons/11,000 segments 的零位移 `_current_tile` 从 44.35 ms 降到 11.06 ms，
输出像素逐项一致。`zero reconstruct` 仍保留为显式诊断/最终输出 API，但已从求解器零位移初始化路径移除。

## 6. 图形与边界必测矩阵

每项图形逻辑至少检查：零位移 XOR 为 0、段长不超配置、法向为单位向量、每段恰有一个 owner、全局重建合法。

- 重叠矩形与角接触：重叠内边消失，角接触组件保持正确；
- 含孔图案：外环/内环法向与探针材料语义正确；
- 2 nm 中空壁、8 nm 探针：穿过窄壁的长边 probe 无效，不产生错误更新；
- 外轮廓越过内轮廓：拓扑保护拒绝 hole 逃逸并整轮回滚；
- 矩形左边越过右边：ring 绕向翻转被拒绝并整轮回滚；
- 斜边跨多 core：所有 tile 读取同一轮状态，最终不按 core 裁矢量，连接连续；
- 负坐标、多行多列 core、ROI 边界：内部半开、最大外边界闭合且无重复 owner；
- probe 冲突/越界/同像素：方向为 0，并分别记录 ambiguous 或 invalid。

图形标注图位于 `doc/images/mbopc/`。颜色说明：owner/core 使用不同颜色；黄色为外法向；青色为 inner probe；红色为 outer probe。

## 7. 同步迭代必测不变量

- 同一轮所有 batch 读取相同 `current`，`next_values` 只在轮次屏障后发布；
- 同一 segment 仅由唯一 owner 更新，halo 只读；
- target LRU 首次与命中结果一致，浮点范围为 `[0,1]`；
- L2/PVBand 只累计 core ownership 像素，不重不漏；
- batch 结束后不持有整张 reticle tensor；
- 非法候选整轮回滚，合法结果从固定全局参考边统一重建；
- 根脚本能从仓库外工作目录直接运行。

## 8. 真实版图验收基线

`gcd_45nm.gds` Layer 11/0 前端结果：1,776 polygons、21,590 edges、223,553 segments；常驻 segment 数组 4,830,716 bytes，相比旧实现 12,675,300 bytes 减少 61.89%；阶段 28 prepare 152.82 ms；重建 XOR 为 0。

阶段 29 再次执行三轮 CUDA 完整流程：870 cores、880,801 memberships；EPE `129645 -> 74592 -> 48348`，L2 `1038629.522 -> 563335.522 -> 440251.431`；GPU 峰值分配 271,544,320 bytes；总耗时 79.117 s（阶段 28 为 85.892 s）；结果 GDS 合法且无全流程 NPZ。

PVBand 在三轮中上升，必须在报告中原样保留，不能只报告改善指标。

## 9. 最终交付审计

交付前执行：完整 diff、`git diff --check`、受保护目录差异、删除符号调用点、AST 中文 docstring、未调用函数/重复实现/异常入口、覆盖率未命中分支和 Markdown 链接/代码围栏检查。用户 GDS、图片和无关工作树修改不得进入提交。
