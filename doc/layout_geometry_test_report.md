# Layout / Geometry 测试报告

## 1. 测试结论

最终验证全部通过：43 个自动化测试通过，`layout/` 与 `geometry/` 合并语句/分支覆盖率为 92%，Ruff 静态规则检查、Python 字节码编译、仓库外工作目录直接执行、严格性能门槛均通过。Patch 和像素图测试都明确覆盖了“一个图形跨越两个 core 边界”的情况，左右结果无丢失、无正面积重叠。

## 2. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 10 10.0.19045 |
| Python | 3.12.0 |
| 解释器 | `D:\app\miniforge\envs\myopc\python.exe` |
| KLayout | 0.30.10 |
| NumPy | 2.5.1 |
| psutil | 7.2.2 |
| CPU | AMD Ryzen 7 4800H，8 核 / 16 线程 |
| 内存 | 15.37 GB |
| 项目安装状态 | 未安装 editable 项目包，直接从源码运行 |

## 3. 自动化测试范围

### 3.1 Layout

- 文件不存在、无 top、多 top 未指定、top/Cell/Layer 不存在和关闭后访问。
- `DbuBox` 整数规范化、面积、扩展、相交、非法框。
- GDS/OASIS 单次读取、层列表、Cell 引用、层级 bbox 和层级摘要。
- 惰性查询直到 `materialize()` 才生成 Region。
- 原生 Box/Path/Polygon 物化和 Text 排除策略。
- 可选 diagnostics 对 polygon-like、Text、Edge、other 的分类。
- R90、镜像、AREF 等层级变换后的坐标与 ROI 裁剪。
- 属性保留开关和无诊断时 `stats is None`。

### 3.2 Geometry

- clip、combine、union、intersection、difference、xor、offset、merge。
- 跨 Cell 坐标系与后端不匹配保护。
- 外轮廓和孔洞转连续数组，并从轮廓重建 Region，XOR 面积为零。
- 闭环边提取、ring/polygon/hole 元数据和连续 `int64` 内存布局。
- 零长度边、零面积轮廓和 hull 数量等验证结果。
- UniformGridIndex 的候选完整性、超长边处理、重复查询和非法参数。

### 3.3 Patch 与输出

- Patch ID 唯一性。
- ownership 冲突按 Layer 隔离；相邻边界允许，正面积重叠拒绝。
- 跨 core Polygon 精确分割。
- GDS 与 OASIS 两种格式写出后重新读取，和期望 Region 的 XOR 面积均为零。
- 未知后缀、非法输出目录/参数保护。

### 3.4 直接入口

- 从仓库根目录直接执行并解析中文/JSON 输出。
- 从仓库之外的临时工作目录调用入口的绝对路径，证明不依赖当前目录或项目安装。
- 负坐标 `--box LEFT BOTTOM RIGHT TOP`。
- `--arrays`、`--diagnostics`、`--output` 组合流程。
- 可预期领域错误返回退出码 2，而不是打印 Python traceback。

### 3.5 灰度像素图

- 精确的完整、部分和空白像素覆盖率。
- 版图 `+Y` 到图片向上的坐标方向转换。
- 孔洞、重叠 Polygon 合并、空 ROI 和二维分块一致性。
- ROI 宽高不能整除像素尺寸时的右侧/顶部灰度补齐。
- 跨 core Polygon 的左右图片横向拼接后与完整 ROI 图片逐像素一致。
- PNG 八位灰度模式、原子保存和可选系统查看器调用。
- 物理像素到整数 DBU 的精确换算、最大像素保护、缺失/多 Layer 和输出路径错误。

## 4. 测试数据

### 4.1 仓库现有数据

| 文件 | 基线 |
|---|---|
| `TestReticle/simple.gds` | 用户可编辑样例，DBU 0.001 μm，top `TOP`，Layer 1/0；不再对其坐标和计数设置固定自动化断言 |
| `TestReticle/JustPoly.gds` | 单 top，两个 Polygon，包含负坐标 |
| `TestReticle/test1.gds` | 两个 top；明确选择 `cell1` 后 Layer 2/0、1/0、3/0 分别为 22、13、1 个 polygon-like 图形 |

### 4.2 测试时生成数据

自动生成的临时 GDS/OASIS 包含多 Layer、Box、Path、Text、带孔 Polygon、R90、镜像和 AREF。另生成一个叶 Cell 加 1000 × 1000 AREF 的基准文件；逻辑实例为一百万，但文件和内存结构保持层级，不把全部实例平铺写出。

精确 bbox、图形数和 CLI 结果统一使用这些生成式数据作为确定性基线，因此用户修改 `simple.gds` 等样例不会造成自动化测试误报。

### 4.3 真实版图只读冒烟测试

对用户提供且仍未纳入 Git 的 `TestReticle/gcd_45nm.gds` 进行了只读直接入口验证：

| 项目 | 结果 |
|---|---:|
| 文件大小 | 229,658 bytes |
| top Cell | `TOP` |
| DBU | 0.0001 μm |
| Layer | 11/0 |
| top bbox | `[11400, 13150, 317300, 308850]` DBU |
| Polygon 数 | 1,776 |
| 总面积 | 28,594,652,500 DBU² |
| 顶点 / 环 / 边 | 21,590 / 1,776 / 21,590 |
| diagnostics | polygon-like 1,776；Text/Edge/other 均为 0 |
| 进程总墙钟时间 | 461.001 ms |

墙钟时间由 PowerShell 测量，包含 Python 进程启动、KLayout/NumPy 导入、GDS 读取、Region 物化、数组转换、诊断额外遍历和 JSON 输出，不等同于纯 ROI 内核时间。

## 5. 跨 core Patch 专项结果

专项测试构造完整 Polygon `Box(25, 20, 75, 80)`，边界位于 `x=50`：

- 左 core ownership：`Box(0, 0, 50, 100)`。
- 右 core ownership：`Box(50, 0, 100, 100)`。
- 原图面积：3,000 DBU²。
- 左、右 Patch 面积：各 1,500 DBU²。
- 左右 Patch 正面积交集：0 DBU²。
- 两个 Patch 聚合结果与原图 XOR：0 DBU²。

因此，共享边界不会触发冲突；跨边界图形被互补裁剪，没有几何丢失，也没有重复 ownership。

## 6. 性能基准

执行命令：

```powershell
D:\app\miniforge\envs\myopc\python.exe benchmarks\benchmark_layout_geometry.py --strict
```

### 6.1 百万逻辑实例层级 ROI

| 指标 | 结果 | 严格门槛 |
|---|---:|---:|
| 逻辑实例数 | 1,000,000 | 固定场景 |
| ROI Polygon 数 | 25 | 必须为 25 |
| 文件打开 | 10.0135 ms | 记录项 |
| 查询 + 精确裁剪中位数 | 0.1058 ms | ≤ 50 ms |
| 查询 + 精确裁剪 P95 | 0.1164 ms | 记录项 |
| RSS 增量 | 0.4805 MB | ≤ 64 MB |

结果说明查询保持层级，没有因一百万逻辑实例而全量展开。

### 6.2 100,000 条边局部索引

| 指标 | 结果 | 严格门槛 |
|---|---:|---:|
| 索引构建 | 425.4770 ms | 一次性成本 |
| 查询次数 | 1,000 | 固定场景 |
| 索引查询中位数 | 0.0208 ms | 记录项 |
| 索引查询 P95 | 0.0264 ms | 记录项 |
| 完整 NumPy bbox 扫描中位数 | 0.3325 ms | 对照项 |
| 查询加速 | 15.9855 × | ≥ 2 × |
| 与暴力扫描结果一致 | 是 | 必须为是 |

严格模式返回码为 0，全部门槛通过。性能数字是本机基线，不应视为所有机器的绝对承诺；适合后续作为相同环境下的回归比较。

## 7. 覆盖率与质量检查

测试命令：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q --cov=layout --cov=geometry --cov-report=term-missing
```

结果：43 passed，合并覆盖率 92%，达到不低于 90% 的验收线；新增 `geometry/raster.py` 覆盖率为 98%。

其他检查：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m ruff check layout geometry tests benchmarks run_layout_geometry.py
D:\app\miniforge\envs\myopc\python.exe -m compileall -q layout geometry tests benchmarks run_layout_geometry.py
git diff --check
```

Ruff 静态规则检查和 compileall 均通过。Ruff formatter 不是本项目门槛：其自动展开规则与用户要求的紧凑排版冲突，因此只采用规则审查、编译和测试验证。

## 8. 缺陷回归与风险判断

开发期间发现并修正的主要问题：

- ROI 迭代器可能把 Text 纳入 `Region.count()`：改用原生 `shape_flags`，没有增加 Python 逐图形过滤。
- 零面积 Text 在 diagnostics 中消失：仅诊断迭代器使用接触语义，正常物化仍使用面积相交。
- 负数逗号 Box 与 argparse 冲突：删除自定义格式，改为四个整数。
- Region 未 merge 时的面积理解错误：修正测试，让 raw count 和几何集合面积分别验证。
- Patch 冲突检查最初为 Python O(n²)：改为按层原生 ownership Region。

这些修复均通过回归测试。最终审查没有发现因修正单个错误而形成的层层特判；诊断语义差异被限制在一个私有统计函数中，CLI 解析也已回归标准 argparse 行为。

## 9. 最终判定

- 功能正确性：通过。
- 跨 core 几何完整性：通过。
- GDS/OASIS 输出一致性：通过。
- 直接 Python 文件运行：通过。
- 性能与内存门槛：通过。
- 可维护性和中文注释要求：通过。
- 不安装项目包运行：通过。
- 用户真实 GDS 只读验证：通过，文件未提交。

## 10. 像素图专项结果

### 10.1 真实 `gcd_45nm` 完整版图

对未跟踪的用户文件执行：

```powershell
D:\app\miniforge\envs\myopc\python.exe run_layout_geometry.py TestReticle\gcd_45nm.gds --layer 11/0 --pixel-size-nm 5 --png .benchmarks\gcd_45nm_full_5nm.png --json
```

| 指标 | 结果 |
|---|---:|
| Layer | 11/0 |
| Polygon 数 | 1,776 |
| 物理像素尺寸 | 5 nm/pixel |
| PNG 尺寸 | 6,118 × 5,914 |
| 总像素数 | 36,181,852 |
| 端到端时间 | 4.663 s |
| 峰值进程 RSS | 149.36 MB |
| PNG 文件大小 | 78,464 bytes |

视觉检查确认图片顶部方向、完整布局结构、白色 mask 和黑色背景正确。生成图片位于被 Git 忽略的 `.benchmarks` 目录，源 GDS 和 PNG 都没有提交。

### 10.2 可重复严格栅格基准

严格 benchmark 增加 2,048 × 2,048、像素网格对齐的纵横 Region：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 栅格化耗时 | 482.839 ms | ≤ 5,000 ms |
| RSS 增量 | 4.32 MB | ≤ 128 MB |
| 白像素数与 Region 面积一致 | 是 | 必须为是 |

基准覆盖多次原生二维分块，严格模式返回码为 0。
