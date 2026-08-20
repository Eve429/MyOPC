# Architecture — 系统结构与依赖

当前系统的事实结构（与源码一致；不包含未来方案）。

## 模块划分

| 模块 | 职责 | 不承担 |
|---|---|---|
| `layout/` | GDS/OASIS/GLP 只读层级读取、层查询、Region 物化、DBU/坐标类型 | 任何几何算法、栅格化、迭代 |
| `geometry/` | Region↔轮廓↔边转换、校验、Patch 裁剪/缝合/写出、共享栅格原语 | 版图 I/O、光刻、迭代 |
| `opc/input/` | 方法无关的两级网格规划（`grid.py`）、居中光刻画布与坐标换算（`raster.py`）、极性（`mask.py`） | 边段语义、求解器 |
| `opc/input/edge/` | 边段型方法共有输入：提边/分段（`fragmentation.py`）、MacroProblem 构造与 NPZ 持久化（`problem.py`）、重建守卫（`reconstruction.py`）、探针采样（`sampling.py`） | 像素级优化变量（ILT）、光刻、评价指标 |
| `opc/input/pixel/` | 像素型方法共有输入：query box 一次栅格化的 PixelMacroProblem、core 画布/参数索引映射、像素→Region 回写 | 边段语义、光刻、评价指标 |
| `lithography/` | ICCAD13 Hopkins 模型（`iccad13.py`）与求解器消费的薄契约（`contracts.py`） | 版图、栅格化、评价指标 |
| `evaluation/` | L2/PVBand/EPE 三项指标（`metrics.py`） | 光刻前向、迭代决策 |
| `opc/iteration/mbopc/` | MB-OPC 求解器（simple 离散 + gradient 梯度） | 光刻实现、合并、进度库 |
| `opc/iteration/ilt/` | ILT 求解器（Simple sigmoid 像素优化 + LevelSet SDF/STE/宏 Adam + 共享 record/result/loss） | 光刻实现、合并、进度库 |
| `main/` | 应用编排：共享宏管线生命周期（`_macro_pipeline.py`）、MB-OPC 公共工作流与适配器（`_mbopc_workflow.py` + `_simple/_gradient_mbopc_workflow.py`）、像素 ILT 公共工作流与适配器（`_ilt_workflow.py` + `_simple/_levelset_ilt_workflow.py`）、各直接运行入口 | 领域算法 |

## 依赖方向（唯一合法拓扑）

```text
layout -> geometry -> opc.input -> opc.input.edge / opc.input.pixel
opc.iteration.mbopc -> opc.input(.edge) + lithography + evaluation
opc.iteration.ilt -> opc.input.pixel + lithography
main -> 上述全部（仅应用编排）
```

禁止（enforced by review，无 import 门禁）：

```text
基础层（layout/geometry/opc.input/lithography/evaluation）
  -X-> opc.iteration.* / main
lithography -X-> layout/geometry/opc/evaluation/main   （只 import torch+标准库+包内）
evaluation  -X-> layout/geometry/opc/lithography/main  （只 import torch+标准库）
opc.iteration.ilt -X-> opc.input.edge / layout / geometry / main
main 内单向依赖：入口 → 方法适配器 → _mbopc_workflow/_ilt_workflow → _macro_pipeline
（共享模块被 import，非入口；同级模块不横向互调）
```

## 关键边界

- **layout ↔ 几何**：layout 只交付原生 `kdb.Region` 与坐标类型；一切
  Region→轮廓→边的解释在 geometry/opc.input.edge。
- **参考 vs 迭代态**：`MacroProblem` 全部数组只读；唯一迭代状态是调用方
  持有的一维 `displacements`。
- **方法层**：`opc/iteration/` 下每个方法子包自足（各自的 config/state/
  solve），不建注册器；ILT 已有 simple 与 levelset 实现；diffopc 目录尚未创建。
- **入口薄层**：`main/run_*.py` 只做参数/调用/摘要打印；MB-OPC 入口经
  各自方法适配器共享 `_mbopc_workflow.py` 生命周期，验证管线与 MB-OPC
  共享 `_macro_pipeline.py`。

## 外部依赖

klayout（Region/多边形）、numpy（参考数组）、torch（光刻+张量）、
pillow（PNG 留档）、psutil（RSS）、tqdm（入口进度条，业务层不导入）。
版本基线见 `requirements.txt`；解释器 myopc conda env。
