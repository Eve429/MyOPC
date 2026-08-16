# Architecture — 系统结构与依赖

当前系统的事实结构（与源码一致；不包含未来方案）。

## 模块划分

| 模块 | 职责 | 不承担 |
|---|---|---|
| `layout/` | GDS/OASIS/GLP 只读层级读取、层查询、Region 物化、DBU/坐标类型 | 任何几何算法、栅格化、迭代 |
| `geometry/` | Region↔轮廓↔边转换、校验、Patch 裁剪/缝合/写出、共享栅格原语 | 版图 I/O、光刻、迭代 |
| `opc/input/` | 方法无关的两级网格规划（`grid.py`）、居中光刻画布与坐标换算（`raster.py`）、极性（`mask.py`） | 边段语义、求解器 |
| `opc/input/edge/` | 边段型方法共有输入：提边/分段（`fragmentation.py`）、MacroProblem 构造与 NPZ 持久化（`problem.py`）、重建守卫（`reconstruction.py`）、探针采样（`sampling.py`） | 像素级优化变量（ILT）、光刻、评价指标 |
| `lithography/` | ICCAD13 Hopkins 模型（`iccad13.py`）与求解器消费的薄契约（`contracts.py`） | 版图、栅格化、评价指标 |
| `evaluation/` | L2/PVBand/EPE 三项指标（`metrics.py`） | 光刻前向、迭代决策 |
| `opc/iteration/mbopc/` | 最简 MB-OPC 求解器（`simple.py`） | 光刻实现、合并、进度库 |
| `main/` | 应用编排：共享宏管线生命周期（`_macro_pipeline.py`）、MB-OPC 工作流（`_mbopc_workflow.py`）、各直接运行入口 | 领域算法 |

## 依赖方向（唯一合法拓扑）

```text
layout -> geometry -> opc.input -> opc.input.edge
opc.iteration.mbopc -> opc.input(.edge) + lithography + evaluation
main -> 上述全部（仅应用编排）
```

禁止（enforced by review，无 import 门禁）：

```text
基础层（layout/geometry/opc.input/lithography/evaluation）
  -X-> opc.iteration.* / main
lithography -X-> layout/geometry/opc/evaluation/main   （只 import torch+标准库+包内）
evaluation  -X-> layout/geometry/opc/lithography/main  （只 import torch+标准库）
main 之间互不 import（_macro_pipeline/_mbopc_workflow 是被入口 import 的共享模块，非入口）
```

## 关键边界

- **layout ↔ 几何**：layout 只交付原生 `kdb.Region` 与坐标类型；一切
  Region→轮廓→边的解释在 geometry/opc.input.edge。
- **参考 vs 迭代态**：`MacroProblem` 全部数组只读；唯一迭代状态是调用方
  持有的一维 `displacements`。
- **方法层**：`opc/iteration/` 下每个方法子包自足（各自的 config/state/
  solve），不建注册器；diffopc、ilt 目录尚未创建（无空目录）。
- **入口薄层**：`main/run_*.py` 只做参数/调用/摘要打印；两个 MB-OPC 入口
  共享 `_mbopc_workflow.py`，验证管线与 MB-OPC 共享 `_macro_pipeline.py`。

## 外部依赖

klayout（Region/多边形）、numpy（参考数组）、torch（光刻+张量）、
pillow（PNG 留档）、psutil（RSS）、tqdm（入口进度条，业务层不导入）。
版本基线见 `requirements.txt`；解释器 myopc conda env。
