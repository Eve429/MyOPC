# 当前架构评审验证报告

## 1. 验证目的

本报告验证[当前架构与精简性评审](current_architecture_review.md)所依据的代码基线仍可正常工作，并确认本轮只产生评审与规划文档，不修改生产实现、受保护目录或用户数据。

## 2. 环境与范围

- 日期：2026-08-12；
- Python：项目既有 Conda 环境 `D:\app\miniforge\envs\myopc\python.exe`；
- 生产范围：`layout/`、`geometry/`、`opc/`、`lithography/`、`evaluation/`、`main/`；
- 测试范围：`tests/` 全量；
- 工作树基线：评审开始前干净。

## 3. 静态结构验证

### 3.1 模块和依赖

- 生产 Python 文件 57 个；
- import 图无循环；
- 未发现基础层依赖具体 iteration 或 `main`；
- 唯一确定的跨包私有对象访问是 `opc.input.preflight -> LayoutDB._native_*`；
- 6 个 runner 从 `main.offline_inputs` 导入私有 helper；
- ILT 三种扩展算法从 `ilt.simple` 导入包内共享私有 helper。

### 3.2 数据结构和重复实现

- dataclass 字段逐项核对后，`ContourBatch/SegmentBatch/MBOPCProblem` 没有重复拥有同一 NumPy 数组；
- AST 规范化函数体扫描没有发现完全相同的生产函数体；
- Layer parser 存在三份解析实现和一层薄包装；
- SimpleILT 存在两个完整 runner 流程；
- `OwnershipError` 没有真实抛出、捕获或测试调用点；
- 其他异常类均有实际用途，不能按“只定义一次”删除。

### 3.3 仓库和文档

- 发现 Git 仍跟踪已被 `.gitignore` 排除的 `output/mbopc` 生成产物；本轮未删除；
- 发现阶段 64 状态与 DiffOPC 接口参考存在历史描述漂移；本轮只在评审报告记录；
- 新增/修改文档相对链接全部存在，Markdown 围栏数量成对；
- `git diff --check` 通过。

## 4. 行为回归

执行：

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
& $python -m ruff check layout geometry opc lithography evaluation main tests
& $python -m compileall -q layout geometry opc lithography evaluation main tests
& $python -m pytest -q
```

结果：

- Ruff：通过；
- compileall：通过；
- pytest：`226 passed in 60.08s`；
- 总命令耗时：63.8 秒。

测试覆盖 Layout/Geometry、跨 core patch、raster 方向、孔洞/斜边/窄图形、边段归属与重建、极性、容量预检、光刻 forward/backward、Simple/LevelSet/CurvMulti/Multilevel ILT、Simple MB-OPC、DiffOPC、直接脚本和离线归档。

## 5. 差异和保护范围

- 生产 Python：零修改；
- `layout/`：零修改；
- `geometry/`：零修改；
- `Test/`、`TestReticle/`、`output/`：零修改；
- 用户 GDS、PNG 和无关工作树文件：零修改；
- 本轮变更仅为专项评审、专项验证、手册导航和三份规划记录。

## 6. 结论

当前基线功能回归全部通过，评审发现的问题主要是依赖声明、封装边界、入口重复和文档/命名漂移，并非现有数值路径已经失败。后续可按评审报告分三批实施，每批都应保持本报告所列全量门禁；涉及 `layout/` 的预检接口修复必须先取得用户逐次授权。
