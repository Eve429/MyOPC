# MyOPC 测试手册

## 1. 运行环境

本仓库不需要安装为 Python 包。以下示例使用当前项目已验证的解释器：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
```

运行时依赖 KLayout、NumPy 和 Pillow；开发测试另需 pytest、pytest-cov、Ruff 和 psutil。

## 2. 最快功能验证

在仓库根目录直接运行：

```powershell
& $python run_mbopc_frontend.py
```

无参数模式同时验证重叠、孔洞、凹角、斜边、跨 core 长边、owner 更新、采样、重建、Patch 拼接、NPZ、PNG 和 GDS，并生成 5 个多图形图集案例。

主产物默认位于 `.benchmarks/mbopc_frontend_demo/`：

- `summary.json`：计数、耗时、内存和 XOR 检查。
- `segments.npz`：不含 pickle 的纯数值边段数据。
- `overview.png`：mask、owner、core、法向与采样点标注图。
- `reconstruction.gds`：`REFERENCE` 和 `RECONSTRUCTED` 两个顶层 Cell。

## 3. 真实版图验证

```powershell
& $python run_mbopc_frontend.py `
  TestReticle\gcd_45nm.gds `
  --layer 11/0 --grid 2 1 `
  --output-dir .benchmarks\gcd_45nm_mbopc `
  --skip-geometry-suite --json
```

`--skip-geometry-suite` 只跳过已独立验证的合成图集，不跳过当前版图的分段、owner、更新、重建、拼接或产物输出。处理自定义 ROI 时使用 `--box LEFT BOTTOM RIGHT TOP`，坐标单位为输入版图 DBU。

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
& $python -m ruff check layout geometry opc tests benchmarks\benchmark_layout_geometry.py benchmarks\benchmark_mbopc_frontend.py run_layout_geometry.py run_mbopc_frontend.py
& $python -m compileall -q layout geometry opc tests run_layout_geometry.py run_mbopc_frontend.py
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
