# Dataflow — Simple ILT（像素型宏同步优化）

宏 ownership 像素参数的 sigmoid 优化：OpenILT 初始化、core 批梯度
scatter-add、屏障后单次 SGD、macro best 严格更低；不经过边段/owner/
EPE 重建。

入口：`python main/run_simple_ilt.py [config/simple_ilt.toml]`

## 函数级流向

```text
main/run_simple_ilt.py::main（可选位置参数，默认 config/simple_ilt.toml）
└─ main/run_simple_ilt.py::run_simple_ilt（SIMPLE_ILT_METHOD 适配器同文件）
   └─ main/_ilt_workflow.py::run_ilt_workflow(SIMPLE_ILT_METHOD)
      ├─ configuration.py::load_config（[layout][partition][lithography]
      │    [simple_ilt][output]；不读取 [edge]）
      ├─ _ilt_workflow.py::prepare_pixel_problems
      │  ├─ layout.LayoutDB.open（单次）→ layer_bbox
      │  ├─ configuration.py::resolve_grid_config（无 edge 的网格换算）
      │  ├─ opc/input/grid.py::plan_macros（面积守恒复核）
      │  └─ 逐 macro：query(query_box).materialize_intersecting
      │     → opc/input/pixel/problem.py::prepare_pixel_macro_problem
      │        实际 macro/core box 整像素校验（否则栅格化前 ValueError）
      │        → normalize_mask → rasterize_region_window（恰一次）
      │        → transmission（clear 直用 / opaque 取反）→ uint8
      │     → pixel_problems/<macro>.npz（NPZ v1，不存每 core 画布）
      │     → ilt_plan.json（键集兼容 merge/final-litho）
      ├─ ICCAD13Lithography(device)
      ├─ 逐 macro（独立；外层条 try/finally）：
      │  ├─ pixel/problem.py::PixelMacroProblem.load
      │  ├─ opc/iteration/ilt/simple.py::optimize_simple_macro
      │  │  ├─ 初始化：params = 2·T − 1（OpenILT 方案；σ(β(2T−1)) ≥ 0.5
      │  │  │    ⟺ T ≥ 0.5，二值化与目标逐格一致且内部像素可优化）
      │  │  ├─ for state in 0..N（N+1 已评价宏状态）：
      │  │  │  ├─ 组批：target_canvas / ownership_canvas /
      │  │  │  │    trainable_index_canvas / context_valid_canvas
      │  │  │  ├─ 快照参数 → leaf 张量（requires_grad 仅 state<N）
      │  │  │  │    → soft = σ(β·local)
      │  │  │  ├─ 固定 context = σ(β(2T−1))（由 target 推导，无梯度）；
      │  │  │  │    三值语义：window 外数值 padding 恒 0
      │  │  │  ├─ mask = where(trainable≥0, soft, context)
      │  │  │  ├─ forward_many（三条件一次 FFT）
      │  │  │  ├─ _common.owned_continuous_losses + curvature_loss
      │  │  │  │    （3×3 零和核 valid 卷积；weight=0 不构建）
      │  │  │  ├─ weighted loss.backward() → np.add.at 宏梯度
      │  │  │  │    （求和不平均）→ 释放 → on_tiles_completed
      │  │  │  ├─ record（ILTStateRecord：state/stage/scale 通用坐标）
      │  │  │  │    → macro best 严格更低
      │  │  │  └─ state<N：梯度有限检查 → 单次 SGD step
      │  │  │        （params −= step_size×梯度；无 Adam 状态）
      │  │  └─ best 物化：soft/binary（CPU float32，阈值 mask_threshold）
      │  ├─ _ilt_workflow.py::_evaluate_best_binary（REQ 二值终评）
      │  │    每批一次 forward_many → ownership 二值 L2 / PVBand
      │  ├─ pixel/problem.py::reconstruct_pixel_region（行游程 + 每宏
      │  │    恰一次 merge；极性逆变换只在回写边界）
      │  └─ write_macro_gds → simple_ilt_result.npz + metrics.json
      ├─ merge_macro_results（恰一次；空 macro 候选按零覆盖容忍）
      ├─ save_final_lithography（可选）
      └─ summary.json（seam_strategy=macro_independent_fixed_context 入档）
```

## 伪代码

```text
prepare：for macro: 一次栅格化 query box → uint8 transmission → NPZ → ilt_plan
solve（每宏独立）：
  T = 宏 ownership 的 target_u8/255
  params = 2·T − 1                                     # OpenILT 初始化
  for state in 0..N:
    grad = zeros[宏参数]
    for batch in cores(batch_size):                    # 同一参数快照
        mask = σ(β·params[trainable])  于 trainable 位
               其余 = 窗口内 σ(β(2T−1)) / 窗口外恒 0   # 三值语义
        printed = forward_many(mask, 三条件)
        L = L_nom + w_proc·L_proc + w_pv·L_pv + w_curv·曲率   # ownership 求和
        L.backward(); np.add.at(grad, trainable索引, local梯度)
        释放
    record + macro best（严格更低，完整已评价宏状态）
    if state < N: params −= step_size × grad           # 屏障后单次 SGD
  best → soft/binary（mask_threshold）
final：best binary 每批一次 forward → ownership 二值 L2/PVBand
       → reconstruct_pixel_region → best.gds + result/metrics
merge：全部宏完成后恰一次（空宏=零覆盖）→ 可选最终光刻 → summary
```

## 本工作流边界

- KLayout 只在 prepare（查询/一次栅格化）、pixel→Region 回写、GDS 写/merge。
- 宏参数/梯度/best 常驻 CPU float32；GPU 每批一图 + backward 后释放；
  极性只在 problem 构造与 Region 回写两边界出现。
- 宏间不交换优化后参数（macro seam 已知限制，summary 显式入档）；
  目标层 bbox 宽高必须 pixel 整数倍（比 edge 管线严）。
