# MyOPC 当前架构与精简性评审

## 1. 评审结论

当前项目的核心分层总体正确，没有发现循环依赖，也没有发现 `ContourBatch`、`SegmentBatch`、`MBOPCProblem` 重复拥有同一批数组。此前对边段结构的收敛是有效的：拓扑、分段参数、owner/membership 分别只保存在一个层级，引用关系不会复制 NumPy 数据。

真正需要收敛的部分主要位于 `main/` 和跨算法公共边界，而不是继续压缩 `layout/geometry/opc.input.edge` 的核心数据结构。当前最值得优先处理的是：修正生产依赖声明、统一重复的 SimpleILT 入口、把被多个模块使用的私有 helper 变成明确公共边界、给可替换光刻模型建立最小能力契约，以及修复预检跨包访问 `LayoutDB` 私有成员的问题。

本轮是只读架构评审，没有修改生产 Python、`layout/`、`geometry/`、用户版图或输出图形。

## 2. 评审范围和方法

评审覆盖 `layout/`、`geometry/`、`opc/`、`lithography/`、`evaluation/`、`main/`、配置、项目元数据、测试和当前事实文档。采用以下证据：

- 57 个生产 Python 文件的模块、类、函数、dataclass 字段清单；
- 第一方 import 图、循环依赖和跨层私有成员访问扫描；
- 函数体 AST 重复扫描、公共符号与异常调用点扫描；
- 所有 runner 的输入、模型、优化、评价、产物和配置流程对照；
- 226 项测试收集结果及既有大 reticle 方案、接口参考和开发手册对照；
- Git 跟踪产物、依赖元数据和历史阶段状态核对。

问题等级含义：

- P1：确定错误或会持续破坏模块边界，建议下一轮优先处理；
- P2：确定冗余、职责漂移或可读性问题，适合在 P1 后集中收敛；
- P3：命名、文档或仓库卫生问题，不阻塞运行但会增加理解成本；
- 保留：看似重复但有明确性能、内存或算法独立性理由，不建议删除。

## 3. 当前模块结构评价

```text
layout
  └─ geometry
      └─ opc.input
          └─ opc.input.edge

lithography ─┐
evaluation  ─┼─> opc.iteration.{ilt, mbopc, diffopc}
opc.input   ─┘

main 负责选择输入、构造具体模型、调用算法和保存产物
```

静态扫描结果符合规定依赖方向：基础层没有反向依赖具体迭代方法，`main` 是组合根，各算法目录彼此独立，全项目没有 import cycle。`main.run_mbopc` 的 fan-out 较大，但它本来就是完整流程入口，不需要为了降低数字引入 runner 基类。

### 3.1 核心数据所有权

| 结构 | 唯一职责 | 是否应保留 |
|---|---|---|
| `PhysicalMask` | 源多边形 Region、Layer、处理框、极性 | 保留 |
| `ContourBatch` | Polygon→Ring→Vertex 两级 CSR 拓扑 | 保留 |
| `SegmentBatch` | 数学边缓存、ring→segment CSR、segment 参数区间 | 保留 |
| `MBOPCProblem` | grid、owner、core→segment membership 及上述对象引用 | 保留，建议改名 |
| `SegmentGeometry` | 一次计算中按需物化的端点和法向 | 保留，不应常驻 |

`SegmentBatch.contours` 是对象引用，不是顶点数组副本；`MBOPCProblem.segments` 也是对象引用。当前没有再次保存 `layer/ring_id/polygon_id/is_hole` 的逐段冗余字段，owner 和 membership 也没有被包装成第二个重复结构。

`PhysicalMask.region` 与 `SegmentBatch.contours` 确实是同一几何的两种表示：前者服务 KLayout 布尔运算和普通 tile 栅格化，后者服务边段重建和可微栅格化。删除任意一个都会迫使多轮热路径反复转换，因此在当前完整内存 problem 中是合理的速度换内存。对十亿级边段，应该实施既定 macro shard/按块物化方案，而不是在现有对象中删除其中一种表示。

## 4. P1：优先修复的问题

### 4.1 `psutil` 被错误声明为开发依赖

`opc/input/preflight.py` 在模块顶层导入 `psutil`，多个生产 runner 调用其内存接口；但 `pyproject.toml` 只在 `dev` 可选依赖列出它，`requirements.txt` 也把它归入“开发、测试与性能基准”。这会使标准运行依赖契约与代码现实不一致。

建议：把 `psutil>=7` 移入 `[project].dependencies`，并在 `requirements.txt` 归到运行依赖。该修改不影响算法行为。

### 4.2 预检跨包访问 `LayoutDB` 私有成员

`opc.input.preflight` 直接访问：

- `database._native_layout`；
- `database._native_cell(...)`；
- `database._native_layer_index(...)`。

预检必须使用原生 `RecursiveShapeIterator`，否则会在容量判断前物化完整 Region，所以需求本身合理；问题是上层绕开了 `layout` 公共接口。任何 `LayoutDB` 内部重构都可能静默破坏 OPC 预检。

建议：在 `layout/` 提供一个受控的只读层级图形迭代接口，由它封装 cell、layer index、shape flags 和数据库生命周期；预检只消费该公共接口。该项必须修改受保护 `layout/`，实施前仍需用户逐次授权。不要在 `opc` 再复制一层 KLayout database 访问代码。

### 4.3 光刻迭代接口绑定具体 `ICCAD13Lithography`

SimpleILT、LevelSet、CurvMulti、Multilevel、Simple MB-OPC 和 DiffOPC 的类型标注均直接依赖 `ICCAD13Lithography`。但测试已有多个结构兼容的替身模型，实际最小需求只是：

- `device`；
- `config.canvas`，边段 OPC 还需要 `config.print_threshold`；
- `condition(name)`；
- `forward_many(mask, conditions)`。

项目当前已经有多种算法消费者，并明确要求后续替换光刻模型，因此这是现实接口而不是假想抽象。

建议：在顶层 `lithography` 定义零运行期开销的最小 `Protocol`，迭代层依赖该能力契约；runner 仍显式实例化 ICCAD13。不要增加模型注册器、工厂或空 backend 目录。若未来模型仍共用 `ProcessCondition`，可继续使用现有类型；否则 condition 应作为不透明 token，求解器不得读取 kernel/dose 实现字段。

### 4.4 公共 `prepare_problem()` 自身不是有界构造器

真实版图 runner 会先调用 preflight，但公共 `prepare_problem()` 可以被直接调用。其 `_build_ownership()` 会在 `np.repeat` 前按总 membership 一次分配数组，没有显式字节或数量上限。因此当前安全契约实际是“真实版图入口必须先预检”，不是 builder 自身保证有界。

建议：短期把接口和文档明确为 `in-memory problem builder`，不在其中重复实现一套估算器；未来 shard 路径使用单独的、真正按 macro 有界的构造流程。若希望公共 API 对任意调用都自保护，可增加显式 `max_memberships/max_bytes` 参数，但必须由真实调用方传入，不能设置一个隐藏全局默认值。

## 5. P2：确定的冗余和职责漂移

### 5.1 `run_simpleilt.py` 与 `run_ilt.py --method simple` 重复完整流程

两者都执行 raster 输入解析、SimpleILT、三工艺条件评价、NPZ/PNG、最终光刻图和 summary；目前已经出现以下漂移：

- 默认迭代数和步长不同；
- 输出格式名分别为 `myopc.simpleilt-result` 和 `myopc.ilt-result`；
- 时间、进程内存和 summary 字段不一致；
- 一处返回 `(SimpleILTResult, summary)`，另一处只返回 summary。

建议：以 `run_ilt(method="simple")` 为唯一实现；`run_simpleilt.py` 如需兼容历史命令，只保留参数适配和委托，不再保存第二套产物流程。迁移前应固定两种入口的兼容字段和默认值，避免“精简”改变用户脚本结果。

### 5.2 `main/offline_inputs.py` 职责过多

该文件 858 行，同时承担：

1. 原子 JSON/NPZ/PNG 写入；
2. 最终光刻结果和 tile manifest 保存；
3. 版图到 raster/segment 的内存物化；
4. NPZ 格式版本、加载、损坏校验和旧版本兼容；
5. 离线准备 CLI。

更明确的信号是 6 个 runner 从它导入 `_atomic_json/_atomic_npz/_atomic_png/_exact_dbu` 等私有函数。下划线接口已经成为事实上的跨模块公共 API。

建议只做一次有现实调用方的最小拆分：新增 `main/artifacts.py`，公开原子 JSON/NPZ/PNG 和最终光刻产物保存；`offline_inputs.py` 保留输入物化、归档版本和 CLI。暂不把 raster/segment 再拆成两个文件，因为两者共享预检、metadata、归档上限和损坏校验不变量。

### 5.3 Layer 参数解析存在三份实现和一层转发

`main.configuration.parse_layer_spec` 已提供正式实现，但 `run_layout_geometry.py`、`run_mbopc_frontend.py` 又复制了解析逻辑，`offline_inputs.parse_layer` 只是薄包装，`run_mbopc.py` 再依赖这个包装。

建议所有入口直接使用 `parse_layer_spec`，删除三份复制/转发函数。该项改动小、风险低，并能消除错误类型捕获不一致。

### 5.4 ILT 公共 helper 放在 `simple.py` 私有命名空间

LevelSet、CurvMulti 和 Multilevel 都从 `ilt.simple` 导入 `_image_batch/_resize_image/_curvature_loss/_smooth_sigmoid_mask`。这些已经是多算法共享实现，不再是 SimpleILT 私有细节。

建议迁移到紧凑的 `opc/iteration/ilt/_common.py`，只供该包内部导入。不要为此建立求解器继承层次；各算法 config 的重复字段具有不同默认值和校验，应继续独立。

### 5.5 两个边段求解器重复 owner 查询

Simple MB-OPC 的 `_owner_indices()` 和 DiffOPC 的 `_owner_segments()` 语义及实现相同：对每个 core 的 membership 切片按 `owner_indices` 过滤。

建议把稳定查询放到问题对象，例如 `owner_segments_for_core(core_index)`；两个 solver 可自行决定是否一次构造 tuple。`_target_tile` 虽也相似，但绑定各自 config 和后续数值路径，不必为了少量代码强行统一。

### 5.6 预检和生产切分各自保存同一计数公式

`preflight._fragment_counts` 与 `fragment_edges` 都计算一条数学边应产生多少 segment。预检不能调用完整切分函数，否则会失去物化前保护；但复制公式可能在策略调整时产生估算漂移。

建议抽取一个只返回 counts 的纯 NumPy helper，预检和生产切分共同使用。它应位于输入层而非通用 `utils` 大杂烩，并且不得分配 SegmentBatch。

## 6. P3：命名、文档和仓库卫生

### 6.1 `MBOPCProblem` 已是通用边段 OPC 问题

该类型同时被 Simple MB-OPC 和 DiffOPC 使用，`myopc.mbopc-input` 归档名也已落后于职责。建议直接迁移为 `EdgeOPCProblem` 和新的 edge-opc 归档名；旧 v2/v3 归档继续兼容读取。不要新增包装类或保留两套字段。由于它影响公共 API、文档和归档，此项优先级低于无行为变化的精简。

### 6.2 项目元数据仍停留在 Layout/Geometry 阶段

`pyproject.toml` 名称为 `myopc-layout-geometry`，描述和 `task_plan.md` 顶部目标也只强调 Layout/Geometry；当前项目已经包含 lithography、evaluation、四种 ILT/OPC 路径和多个 runner。建议更新为整个 MyOPC 项目身份。

### 6.3 存在无实现的 `OwnershipError`

`OwnershipError` 只有定义和包级导出，没有抛出、捕获或测试；owner/membership 校验实际抛 `ValueError`。从精简角度建议删除。`GeometryError` 和 `OPCError` 基类则被 runner 用来统一捕获真实子类，应保留。

### 6.4 当前事实文档发生状态漂移

- `task_plan.md` 总表仍把阶段 64 标为 in progress，但阶段 66–77 和现实代码已全部完成；
- `module_interface_reference.md` 仍把 DiffOPC 写成“原型、尚未完成连续 EPE/完整产物验收”，与当前 runner、专项报告和测试矛盾；
- 主手册同时存在 `development_manual.md` 和超长 `项目开发手册.md`，没有明确哪个是当前事实源。

建议立即修正前两项错误；对文档体系只增加导航和“当前/阶段归档”标识，不机械合并历史报告。

### 6.5 Git 跟踪生成产物

`.gitignore` 已排除 `output/`，但 Git 仍跟踪 `output/mbopc` 下四个结果文件；根目录还有 `gcd_45nm.png` 和 `result.gds`。这些可能是用户有意保留的基线，本轮不删除。建议后续逐项确认用途：正式夹具迁到明确的 test fixture 目录，普通运行产物从索引移除。

## 7. 不建议实施的“精简”

- 不把 `ContourBatch/SegmentBatch/MBOPCProblem` 合并成一个巨型结构；这会混淆拓扑、控制变量和 tile 归属的生命周期。
- 不重新建立独立 `Ownership` 数据类；现有三个 CSR 数组均有热路径，包装不会省内存或减少字段。
- 不合并 ILT 配置基类；算法默认、优化器和约束不同，继承只会增加耦合。
- 不建立通用 runner 基类或算法注册器；当前只需复用稳定的产物 I/O 和参数解析。
- 不删除 `PhysicalMask.region` 或 `SegmentBatch.contours` 中任意一种表示；当前转换成本会进入每轮每 tile 热路径。
- 不抽取 9 个入口相同的 `sys.path` 引导代码；直接运行深层脚本时必须在第一方 import 前执行，放进项目模块反而无法导入。
- 不因 `layout/source.py` 在第三方文件读取边界使用 `except Exception` 就机械缩窄；它保留了原异常 cause，统一包装为领域错误是合理的 I/O 边界行为。

## 8. 推荐实施顺序

### 第一批：低风险收敛

1. 修正 `psutil` 运行依赖和项目元数据；
2. 统一 Layer parser；
3. 删除未使用 `OwnershipError`；
4. 修正文档状态漂移；
5. 抽取 ILT 包内公共 helper 和 owner 查询 helper。

### 第二批：入口去重

1. 抽出 `main/artifacts.py`；
2. 让 `run_simpleilt.py` 委托统一 `run_ilt`；
3. 对齐产物格式、默认值、summary 和回归测试；
4. 完成私有跨模块 import 为零的静态审计。

### 第三批：明确扩展边界

1. 增加最小 lithography Protocol，并让所有 solver 类型标注依赖它；
2. 经用户授权后，为 `LayoutDB` 增加受控只读层级扫描公共接口；
3. 评估 `EdgeOPCProblem` 名称和归档迁移；
4. 再进入已单独设计的 macro shard 大 reticle 阶段。

第一、二批主要是结构整理，不应改变数值结果或性能；第三批涉及公共接口，应设置兼容迁移测试。任何 `layout/` 改动仍需用户明确授权。

## 9. 验收建议

后续实施时至少增加以下门禁：

- import 图无环，基础层不得依赖 `main` 或具体 iteration；
- 除同包内部 `_common` 外，跨模块不得导入下划线私有符号；
- 所有 solver 对一个结构化假模型通过接口契约测试，不要求具体 ICCAD13 类型；
- `run_simpleilt.py` 与 `run_ilt.py --method simple` 在相同参数下产物数值和 summary 核心字段一致；
- preflight 与真实 fragmentation 对多种边长、角段阈值和随机边长给出完全相同的 segment count；
- 全量行为回归、真实 CUDA/CPU 光刻回归、中文 docstring、重复函数体、未调用符号、文档链接和差异审计继续通过。

## 10. 总体判断

当前代码并不是“结构体太多、文件太多”的失控设计。核心数据面已经比较克制，性能关键路径也有明确的缓存、CSR、owner-only 和轮次屏障语义。主要问题是项目从 Layout/Geometry 逐步长成完整 OPC 平台后，入口层和公共契约没有完全同步升级，形成了私有 helper 外泄、双入口重复、具体模型耦合和历史命名/文档漂移。

因此下一轮优化应以“整理边界、删除真实重复、保持数据面不动”为原则，而不是再次大规模重写核心几何或把所有流程塞进统一框架。
