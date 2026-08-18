# Glossary — 项目统一术语

每个术语只定义一次；文档中使用时以本文件为准。括号内为同义词/英文。

## 版图与坐标

- **DBU**（Database Unit）：版图最小整数坐标单位，GDS 头部 dbu_um 定义
  （如 0.001 µm = 1 nm）。所有几何坐标最终是整数 DBU；nm→DBU 换算必须用
  Decimal 精确整除（`main/_macro_pipeline.py::exact_dbu`），不能整除即失败。
- **左下原点**：所有版图/模型数组第 0 行 = 最低 Y；PNG/显示翻转只发生在
  I/O 边界。
- **Layer**：`layout.LayerSpec(layer, datatype)`，一次处理唯一目标层；
  其他层不复制进产物。
- **bbox**：目标层的整体包络（`layout.LayoutDB.layer_bbox`），也是全量
  查询的正确窗口（不使用 ±2^30 魔法框）。
- **极性**（polarity）：`clear` = 源图形透光（coverage 即透光率）；
  `opaque` = 源图形不透光（field − coverage）。数组值统一 1=透光。

## 两级网格（Macro–Core）

- **Macro**：版图一级划分，ownership 半开不重叠、面积和恰等于层 bbox。
  每个 macro 独立构造/持久化/求解一个 `MacroProblem`。
- **Core / Tile**：macro 内二级划分（同义词，文档统一用 core）；
  `ownership_box` 唯一可计分/可回写，`context_box` 是四边扩 context 后的
  只读计算范围。
- **Context / Halo**（同义词，本项目统一用 context）：core 四边扩展的
  只读光学上下文宽度；吸收 FFT 循环卷积污染；context ≥ max_displacement。
- **Canvas**：ICCAD13 固定 256×256 光刻画布；core 局部窗口（context 决定）
  经居中 padding 放入 canvas，外围填 0。
- **居中 padding**：差值均分、奇数余量归高坐标侧
  （`opc/input/raster.py::_center_padding`）；raster/ownership/探针坐标
  共用同一偏移。

## 边段（edge）

- **Segment**：参考轮廓的边段，迭代的最小操作单元；长度 ≤ 配置、每段
  唯一 owner。参考（零位移）几何固定不变。
- **Owner**：`owner_indices[s]` = 唯一可写该段的 core 编号；-1 表示只因
  context 可见的只读副本。
- **Membership**：core 视角 CSR（`core_offsets`/`member_segment_indices`），
  一段可因 context 出现在多个 core，但 owner 唯一；own ⊆ membership。
- **Displacement**：一维 `float64[S]` 绝对法向位移数组，是唯一迭代状态；
  context 段恒 0；|值| ≤ max_displacement。
- **法向**：材料→空区的单位外法向，opaque 极性下翻转，最终统一为
  "透光区→不透光区"方向；求解器无极性分支。

## 光刻（lithography）

- **ICCAD13**：唯一当前光刻模型（Hopkins 部分相干，35×35×24 核、256 画布、
  四资产 buffer）。资产 SHA-256 是模型身份硬断言。
- **工艺条件**（ProcessCondition）：`nominal`(focus×1.00²)、
  `dose_max`(focus×1.02²)、`defocus_min`(defocus×0.98²)；一次
  `forward_many` 共享 mask FFT。
- **透光率画布**：模型输入，1.0=透光、0.0=不透光、边缘像素保留连续覆盖率。

## 评价（evaluation）

- **EPE**（Edge Placement Error）：inner/outer 探针对的违规计数；
  探针 = 参考边中点 ∓ 法向×epe_distance；方向 ∈ {-1,0,+1}。
- **探针有效性**：target 上 inner 透光、outer 不透光、二者不同、都在画布内，
  四者同时成立才有效；全无效 = 无法评价（insufficient_probes）。
- **PVBand**：dose_max 与 defocus_min 二值图不一致的 ownership 像素数
  （工艺窗诊断，不参与 simple 求解器决策）。
- **L2**：nominal 二值图与 target 二值图不一致的 ownership 像素数（诊断）。
- **Ownership 像素**：`ownership_canvas` 标记的唯一计分像素；context/padding
  不重复计分。

## 迭代（simple MB-OPC）

- **Baseline**（Round 0）：零位移状态的首次评价；records[0]。
- **Round**：一次位移后的已评价状态；Round N 指标属于第 N 次位移后的几何
  （评价同时产生下一轮提案）。
- **Proposal**：评价产生的 next 位移，仅是提案；经重建守卫验证后才成为
  下一状态（Jacobi 同步：同轮所有 tile 只读同一 current）。
- **停止状态**：zero_epe / no_update / invalid_geometry / insufficient_probes /
  iteration_limit。
- **独立 macro**：macro 间不交换中间状态，边界 context 固定为邻区参考几何；
  全部完成后恰一次显式映射 merge。不是全局同步最优，差异须量化。
- **Change**：一次完整的开发变更单元（CHG-xxx），生命周期
  spec → 实现 → 测试 → 两报告 → completed。
