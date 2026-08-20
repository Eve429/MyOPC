# Contract — ilt

像素型 ILT：优化变量是宏 ownership 像素参数（不经过 SegmentBatch/owner/
边段重建）。当前实现为 Simple ILT 与 LevelSet ILT；共享问题/结果/workflow
契约供后续 CurvMulti、Multilevel 复用。

## 输入与数据契约

- 问题：`opc/input/pixel.py::PixelMacroProblem`（见
  `doc/contracts/opc_input.md` 像素宏问题节）——query box 一次栅格化的
  uint8 transmission、core 画布/参数索引映射、像素→Region 回写。
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

signed_distance_initialization(target_u8) -> np.ndarray          # once/macro
macro_gradient_magnitude(problem, initial_query_phi, macro_phi)  # once/backward-state
build_levelset_final_context_canvas(problem, core_index, config)  # 终评 hard context
optimize_levelset_macro(problem, model, config, *,
                        on_tiles_completed=None) -> ILTMacroResult
```

语义保证：

- **SDF once/macro**：`target_u8/255 >= 0.5`（127/128 分界）为唯一阈值
  事实源；背景 +像素中心距离、前景取负；全前景/全背景 ±max(Hq,Wq) 有限
  常量场。SciPy `distance_transform_edt` 生产路径（旧纯 Python EDT 仅作
  测试 oracle）。raster 像素中心定义，刻意不与 OpenILT polygon-edge
  初值对齐（规格 §2.2 冻结）。
- **macro-global phi**：唯一可训练参数 [Hm,Wm] float32（SDF 的 ownership
  crop）；CPU 常驻 + `torch.optim.Adam`（betas=(0.9,0.999)、eps=1e-8、
  weight_decay=0、amsgrad=False 与 PyTorch reference 逐位一致）。
- **唯一梯度系数**：每 backward state 恰一次 `macro_gradient_magnitude`：
  [Hm+2,Wm+2] halo（外围=initial_query_phi 固定物理 context，中心=当前
  快照）中心差分 /2 → sqrt(dx²+dy²)。同一参数跨 core 恒同 phi 同系数
  （core-local 差分/replicate padding 为契约违例）；末纯评价 state 不调用。
- **STE**：`_LevelSetBinarize`——forward `(phi < 0)`（phi==0 不透光），
  backward `-|grad(phi)|·grad_output`（系数只读返回 None，内部零空间
  差分，任意形状张量成立）。
- **context ≥ 1 像素**：无条件入口拒绝（中心差分需真实物理邻域，与
  curvature_weight 无关）。
- **固定 context**：训练与终评同一套三值语义——真实 context 取 hard
  `target >= 0.5`，数值 padding 恒 0；终评不重跑 SDF。
- **宏同步状态/梯度**：同 Simple——同一 state 全 core 同快照、scatter-add
  raw sum 不平均、屏障后恰一次 Adam step；N 更新 N+1 评价态、best 严格
  更低平局保早。
- **best 物化**：`best_parameters`=phi、`binary_mask=(phi<0)`、
  `soft_mask=sigmoid(-phi)`（仅诊断）；`best_parameters` 在 result NPZ 中
  语义为 phi。
- **步长提示**：像素中心 SDF 的 |phi| ≥ 1，Adam 首步 |Δ| < lr——lr ≤ 1
  时边界像素需多状态累积才可能越过 0 等值线（smoke 对照见
  CHG-20260818-levelset-ilt development_report）。

## 公共 workflow（`main/_ilt_workflow.py`）

`ILTMethod(method_name/config_type/optimize_macro/evaluated_states/
build_fixed_context_canvas)` 五字段注入；`run_ilt_workflow` 负责 prepare
（ilt_plan.json 键集兼容 merge/final-litho）、逐宏求解（进度 total =
core×states、双层 try/finally）、best binary 终评（ownership 二值
L2/PVBand；固定 context 画布由方法策略 `build_fixed_context_canvas
(problem, core_index, config)` 提供，本层不读 sigmoid_steepness/mask_
threshold/phi 等方法数学字段）、产物（best.gds + `<method>_result.npz`
+ metrics.json）、merge 恰一次（空宏候选按零覆盖容忍）、可选最终光刻
与 summary（`seam_strategy=macro_independent_fixed_context` 显式入档）。
config 鸭子契约只要求 `batch_size`（终评分批）。

入口：`python main/run_simple_ilt.py [config.toml]`（默认
`config/simple_ilt.toml`）；`python main/run_levelset_ilt.py [config.toml]`
（默认 `config/levelset_ilt.toml`）。

## 已知限制

- macro 间不交换优化后参数；macro seam 仍可能存在（独立宏 + 固定 context）。
- 输出是 pixel-grid stair-step 几何；无 MRC/shot/最小线宽约束。
- 目标层 bbox 宽高必须整像素（比 edge 管线严），否则 prepare 前置失败。
- EPE/shot 不属于 ILT 评价指标（本 change 明确不迁）。

## 后续方法入口

CurvMulti/Multilevel 规格见
`doc/changes/active/CHG-20260818-{curvmulti,multilevel}-ilt/`；
实施须以本契约为基础零修改或向后兼容扩展共享层（ILTMethod 五字段含
`build_fixed_context_canvas`）。

## 事实核对锚点

`tests/opc/input/test_pixel_problem.py`、`tests/opc/iteration/test_simple_ilt.py`、
`tests/opc/iteration/test_levelset_ilt.py`、`tests/main/test_simple_ilt_runner.py`、
`tests/main/test_levelset_ilt_runner.py`；smoke `config/simple_ilt.toml`、
`config/levelset_ilt.toml`。
