# MyOPC 测试手册

## 1. 运行环境

本仓库不需要安装为 Python 包。以下示例使用当前项目已验证的解释器：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
```

运行时依赖 KLayout、NumPy、Pillow 和 PyTorch；开发测试另需 pytest、pytest-cov、Ruff 和 psutil。Windows 直接运行环境 Python 时，`lithography.iccad13` 会在导入 PyTorch 前加入当前环境的 CUDA DLL 目录。

## 2. 最快功能验证

在仓库根目录直接运行：

```powershell
& $python run_mbopc_frontend.py
```

无参数模式同时验证重叠、孔洞、凹角、斜边、跨 core 长边、owner 更新、采样、全局重建、core 覆盖、NPZ、PNG 和 GDS，并生成 5 个多图形图集案例。

主产物默认位于 `.benchmarks/mbopc_frontend_demo/`：

- `summary.json`：计数、耗时、内存和 XOR 检查。
- `segments.npz`：不含 pickle 的纯数值边段数据。
- `overview.png`：mask、owner、core、法向与采样点标注图。
- `reconstruction.gds`：`REFERENCE` 和 `RECONSTRUCTED` 两个顶层 Cell。

## 3. 真实版图验证

完整 simple MB-OPC 可直接运行，无参数时默认读取 `TestReticle/simple.gds`：

```powershell
& $python run_mbopc.py
```

默认按 1024 nm core、512 nm halo、8 nm/pixel、batch 8 运行最多 8 轮，并输出 `output/mbopc/summary.json`、`mbopc_result.npz` 和 `mbopc_result.gds`。只有明确传入 `--preview` 才额外物化诊断数据并保存 PNG。

完整 `gcd_45nm` 三轮复现命令：

```powershell
& $python run_mbopc.py TestReticle\gcd_45nm.gds `
  --layer 11/0 --iterations 3 --batch-size 8 `
  --output-dir .benchmarks\mbopc_gcd_full_3 --preview --json
```

处理自定义 ROI 时使用 `--box LEFT BOTTOM RIGHT TOP`，坐标为输入版图 DBU。`--tile-size-nm` 和 `--halo-nm` 必须可由 DBU 精确表达且是 `--pixel-nm` 的整数倍；显存不足时只需减小 `--batch-size`，不会改变 owner 或轮次屏障语义。

仅验证输入前端、分段与产物而不运行光刻迭代时，仍可使用：

```powershell
& $python run_mbopc_frontend.py `
  TestReticle\gcd_45nm.gds `
  --layer 11/0 --grid 2 1 `
  --output-dir .benchmarks\gcd_45nm_mbopc `
  --skip-geometry-suite --json
```

`--skip-geometry-suite` 只跳过已独立验证的合成图集，不跳过当前版图的分段、owner、更新、全局重建、core 覆盖或产物输出。处理自定义 ROI 时使用 `--box LEFT BOTTOM RIGHT TOP`，坐标单位为输入版图 DBU。

## 4. 自动测试

运行 OPC 测试与覆盖率：

```powershell
& $python -m pytest -q tests\opc
& $python -m pytest --cov=opc --cov-branch --cov-report=term-missing -q tests\opc
```

运行全仓库回归：

```powershell
& $python -m pytest -q
```

静态与语法门槛：

```powershell
& $python -m ruff check layout geometry opc lithography evaluation tests benchmarks\benchmark_layout_geometry.py benchmarks\benchmark_mbopc_frontend.py run_layout_geometry.py run_mbopc_frontend.py run_mbopc.py
& $python -m compileall -q layout geometry opc lithography evaluation tests run_layout_geometry.py run_mbopc_frontend.py run_mbopc.py
```

`Test/klayout.ipynb` 是既有用户笔记，不属于当前自动修改范围。

## 5. 严格性能基准

```powershell
& $python benchmarks\benchmark_mbopc_frontend.py --strict
```

严格模式检查：

- 5,000 个图形、110,000 个控制段的准备与重建耗时。
- 50,000 个稳定 key 批量查找结果与速度。
- 零位移 XOR、最大段长、owner 完整性。
- 紧凑常驻数组相对完全展开表示至少节省 40%。
- halo membership 不得退化为 segment×core 稠密矩阵。

## 6. 标注图阅读

- 白色：物理 mask。
- 红、蓝、黄、绿等边线：segment 唯一 owner 和 core 边框。
- 黄色线/点：从边界指向空区的外法向。
- 红色点：正法向外部采样。
- 青色点：负法向材料侧采样。
- `Snn/Cn`：segment 序号与 owner core 序号；大版图只抽样文字和采样点，但所有边段仍绘制。

## 7. 新增测试规则

新图形逻辑至少同时检查零位移 XOR 为 0、segment key 唯一、段长不超配置、法向为单位向量且 owner 唯一。任何 bug 修复必须留下最小回归用例，不得只用一个大 GDS 做人工观察。

simple MB-OPC 还必须覆盖：

- 同轮不同 batch 读取相同 `current`，跨 core segment 只由唯一 owner 写入一次。
- target LRU 首次生成与缓存命中的浮点值完全一致并限制在 `[0,1]`。
- halo 像素不计入 L2/PVBand，core ownership 像素不重不漏。
- 2 nm 中空窄壁配 8 nm 探针时，无效长边探针不产生更新；拐角短段按局部 target 语义单独判定。
- inner/outer 同时违规时方向为 0 并计入 ambiguous；越界、同像素或 target 语义错误的探针无效。
- 矩形左边越过右边导致 ring 绕向翻转时整轮回滚；外轮廓越过 hole 时整轮回滚。
- 斜边跨多个 core 时不按 core 裁最终矢量，最佳位移从固定全局参考边统一重建。
- 真实 ICCAD13 模型至少完成一轮，根脚本必须能从仓库外工作目录直接执行。

当前专项覆盖命令：

```powershell
& $python -m pytest tests\opc tests\lithography tests\evaluation `
  --cov=opc --cov=lithography --cov=evaluation --cov-branch `
  --cov-report=term-missing -q
```

专项实测 75 项通过，综合语句/分支覆盖率 92%；全仓库 119 项通过。

## 8. 目录重构回归

阶段 20 移动 OPC 文件时执行了以下检查：

- `import opc.input`、`import opc.input.edge` 和根目录主程序导入成功。
- `opc.common`、`opc.mbopc` 不再可导入，避免旧路径被缓存目录误保留。
- Ruff、compileall、37 项 OPC 专项测试和 81 项全仓库测试全部通过。
- `layout/`、`geometry/` 没有内容差异，函数体 AST 对比没有算法差异。
- 函数调用文档中的相对源码链接全部存在。

当时的重构基线为：OPC 37 项通过、综合语句/分支覆盖率 93%，全仓库 81 项通过；当前完整基线见第 7 节。
