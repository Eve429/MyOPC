# Contract — ilt

像素型 ILT：优化变量是宏 ownership 像素参数（不经过 SegmentBatch/owner/
边段重建）。当前实现为 Simple ILT 与 LevelSet ILT；共享问题/结果/workflow
契约供后续 CurvMulti、Multilevel 复用。

## 输入与数据契约

- 问题：`opc/input/pixel.py::PixelMacroProblem`（见
  `doc/contracts/opc_input.md` 像素宏问题节）——query box 一次栅格化的
  uint8 transmission、core 画布/参数索引映射、像素→Region 回写。
- 场边界：`prepare_pixel_macro_problem` 必填 `layout_bounds`（plan_macros
  所用版图层 bbox，00_PAST field_box 契约的迁移等价）——query 超出 bbox
  的环带 transmission 恒置 0（不透光）：opaque 的 1−coverage 不得把无几何
  环带反成虚假透光；clear 处为逐位 no-op。场边界只作用于 transmission
  数组，绝不作为图形进入 Region（不产生虚假可动边）。
- 记录：`opc/iteration/ilt/_common.py::ILTStateRecord`
  （state_index/stage_index/stage_state_index/scale + 四项损失 + 耗时；
  Simple/LevelSet 恒写 0/state_index/state_index/1）。
- 结果：`ILTMacroResult(best_parameters/soft_mask/binary_mask/
  best_state_index/records)`；workflow 只读消费。
- 损失：`owned_continuous_losses`（nominal/process/pvband，ownership 求和）
  + `curvature_loss`（3×3 零和核 valid 卷积）+ `weighted_macro_loss`
  （nominal 权重恒 1）。

## Simple 求解器（`opc/iteration/ilt/simple.py`）

```python
class SimpleILTConfig:  # [simple_ilt] 段直接注册（8 字段，无派生换算）
    iterations / step_size / sigmoid_steepness / weight_process_l2 /
    weight_pvband / curvature_weight / mask_threshold / batch_size

def optimize_simple_macro(problem, model, config, *,
                          on_tiles_completed=None) -> ILTMacroResult
```

语义保证：

- **初始化**：`params = 2·T − 1`（OpenILT 同式；P1-1 修复，取代早期
  logit+eps 方案）。0/1 像素 sigmoid 斜率 = β·σ(β)σ(−β)（β=4 时 0.0707），
  内部像素保持可优化（拓扑/开孔/SRAF 前提）。性质：
  σ(β(2T−1)) ≥ 0.5 ⟺ T ≥ 0.5——mask_threshold=0.5 时 state0 二值掩膜
  与 T 二值化一致（其他阈值不构成该对齐不变量）；
  state0 soft = σ(β(2T−1)) 不精确等于 T（1e-6 恢复契约已废除，
  见 CHG-20260819-simple-ilt-openilt-init）。
- **宏同步状态**：同一 state 全部 core/batch 读同一宏参数快照；core 只限制
  loss ownership，不截断 context 内可训练像素的梯度；梯度 scatter-add
  求和（绝不平均）；全部 core 完成后恰一次同步 SGD step；N 次更新对应
  N+1 个已评价状态（末状态纯评价）。
- **transmission 单一定义（三值语义）**：trainable→当前 soft；真实
  context（macro 外、终评画布，含物理 T=0 像素）→初始 soft
  σ(β(2T−1))，无梯度；context window 外的数值 padding→恒 0
  （`context_valid_canvas` 判据）。监督/指标目标是 raw T。同一物理
  像素在不同宏画布中的初始值一致（跨宏 seam 测试锁定）。
- **macro best**：只按完整宏总损失严格更低选择，batch 切分/顺序不变。
- **联合约束**：curvature_weight > 0 要求 context ≥ 1 像素（入口拒绝；
  否则 valid 卷积的 ownership 边缘曲率随 core 切分变化）。
- **异常**：非有限 loss/梯度/参数 → FloatingPointError，不发布当前宏。

## LevelSet 求解器（`opc/iteration/ilt/levelset.py`）

```python
class LevelSetILTConfig:  # [levelset_ilt] 段直接注册（6 字段，无派生换算）
    iterations / step_size / weight_process_l2 /
    weight_pvband / curvature_weight / batch_size

signed_distance_initialization(target_u8, pixel_nm=1.0) -> np.ndarray
macro_gradient_magnitude(problem, initial_query_phi, macro_phi, pixel_nm=1.0)
build_levelset_final_context_canvas(problem, core_index, config)
optimize_levelset_macro(problem, model, config, *, pixel_nm=1.0,
                        on_tiles_completed=None) -> ILTMacroResult
```

语义保证：

- **物理单位事实源**：LevelSet 的 `phi` 与 `step_size` 均以 nm 表示；生产入口
  的 `pixel_nm` 只来自 `[lithography].pixel_nm`，不得在 `[levelset_ilt]`
  重复配置。solver/test 的 `pixel_nm=1.0` 默认仅用于保持独立单元测试的
  单位归一语义，正式 workflow 会显式绑定真实 `pixel_nm`。
- **SDF once/macro**：`target_u8/255 >= 0.5`（127/128 分界）为唯一阈值
  事实源；背景为 +物理像素中心距离（nm）、前景取负；SciPy EDT 使用
  `sampling=(pixel_nm,pixel_nm)`。全前景/全背景常量场为
  `±max(Hq,Wq)·pixel_nm`。raster 像素中心定义，刻意不与 OpenILT
  polygon-edge 初值逐值对齐。
- **macro-global phi**：唯一可训练参数 [Hm,Wm] float32（SDF 的 ownership
  crop，单位 nm）；CPU 常驻 + `torch.optim.Adam`（betas=(0.9,0.999)、
  eps=1e-8、weight_decay=0、amsgrad=False）。`config.step_size` 是该 nm-SDF
  的 Adam 学习率，因此像素从 4nm 改为 2nm 时不再隐式把同一超参数解释为
  不同物理长度。
- **唯一梯度系数**：每 backward state 恰一次 `macro_gradient_magnitude`：
  [Hm+2,Wm+2] halo（外围=initial_query_phi 固定物理 context，中心=当前
  快照），中心差分除以 `2·pixel_nm`，因此标准 SDF 上 `|grad(phi)|≈1`
  且系数无量纲。同一参数跨 core 恒同 phi 同系数；末纯评价 state 不调用。
- **STE**：`_LevelSetBinarize`——forward `(phi < 0)`（phi==0 不透光），
  backward `-|grad(phi)|·grad_output`（系数只读返回 None，内部零空间差分）。
- **context ≥ 1 像素**：无条件入口拒绝（中心差分需真实物理邻域，与
  curvature_weight 无关）。
- **固定 context**：训练与终评同一套三值语义——真实 context 取 hard
  `target >= 0.5`，数值 padding 恒 0；终评不重跑 SDF。
- **宏同步状态/梯度**：同 Simple——同一 state 全 core 同快照、scatter-add
  raw sum 不平均、屏障后恰一次 Adam step；N 更新 N+1 评价态、best 严格
  更低平局保早。
- **best 物化**：`best_parameters`=nm 单位 phi、`binary_mask=(phi<0)`、
  `soft_mask=sigmoid(-phi)`（仅诊断）；result NPZ 中 `best_parameters`
  语义同样为 nm-SDF。
- **步长标度**：推荐配置需按物理 nm 进行 benchmark；当前 smoke 以
  `step_size=3.2nm` 作为 OpenILT `4nm/pixel × 0.8` 的尺度起点，不将该值
  宣称为最终最优参数。

## 公共 workflow（`main/_ilt_workflow.py`）

`ILTMethod(method_name/config_type/optimize_macro/evaluated_states/
build_fixed_context_canvas)` 五字段注入；`run_ilt_workflow` 负责 prepare、逐宏
求解、best binary 终评、产物、merge 与 summary。LevelSet 方法适配器在进入
公共 workflow 前仅从同一 TOML 的 `LithographyConfig.pixel_nm` 读取一次物理
尺度并绑定到 solver；公共 `ILTMethod` 仍不感知 phi、sigmoid 或方法数学字段。

入口：`python main/run_simple_ilt.py [config.toml]`（默认
`config/simple_ilt.toml`）；`python main/run_levelset_ilt.py [config.toml]`
（默认 `config/levelset_ilt.toml`）。

## 已知限制

- macro 间不交换优化后参数；macro seam 仍可能存在（独立宏 + 固定 context）。
- 输出是 pixel-grid stair-step 几何；无 MRC/shot/最小线宽约束。
- 目标层 bbox 宽高必须整像素（比 edge 管线严），否则 prepare 前置失败。
- EPE/shot 不属于 ILT 评价指标。

## 后续方法入口

CurvMulti/Multilevel 规格见
`doc/changes/active/CHG-20260818-{curvmulti,multilevel}-ilt/`；实施须以本契约为
基础零修改或向后兼容扩展共享层。

## 事实核对锚点

`tests/opc/input/test_pixel_problem.py`、`tests/opc/iteration/test_simple_ilt.py`、
`tests/opc/iteration/test_levelset_ilt.py`、
`tests/opc/iteration/test_levelset_physical_units.py`、
`tests/main/test_simple_ilt_runner.py`、`tests/main/test_levelset_ilt_runner.py`；
smoke `config/simple_ilt.toml`、`config/levelset_ilt.toml`。
