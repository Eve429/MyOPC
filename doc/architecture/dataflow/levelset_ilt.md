# Dataflow — LevelSet ILT（SDF 参数化 + 外部梯度 STE + 宏 Adam）

宏 ownership 像素 phi（带符号距离场）的 hard 优化：SciPy EDT 初始化
once/macro、macro 域唯一 |grad(phi)| 代理系数、`-|grad|·上游` STE、
跨 core raw-sum 梯度、屏障后单次 Adam。

入口：`python main/run_ilt_levelset.py [config/levelset_ilt.toml]`

## 函数级流向

```text
main/run_ilt_levelset.py::main（可选位置参数，默认 config/levelset_ilt.toml）
└─ main/run_ilt_levelset.py::run_levelset_ilt（LEVELSET_ILT_METHOD 适配器同文件）
   └─ main/_ilt_workflow.py::run_ilt_workflow(LEVELSET_ILT_METHOD)
      ├─ configuration.py::load_config（[layout][partition][lithography]
      │    [levelset_ilt][output]；不读取 [edge]）
      ├─ prepare_pixel_problems（与 Simple 完全共享，见 simple_ilt.md）
      ├─ ICCAD13Lithography(device)
      ├─ 逐 macro（独立；外层条 try/finally）：
      │  ├─ pixel/problem.py::PixelMacroProblem.load
      │  ├─ opc/iteration/ilt/levelset.py::optimize_levelset_macro
      │  │  ├─ 入口契约：画布一致；context_dbu >= pixel_dbu 无条件
      │  │  │    拒绝（与 curvature_weight 无关）
      │  │  ├─ signed_distance_initialization（once/macro：
      │  │  │    target_u8/255>=0.5 唯一阈值；SciPy EDT outside/inside
      │  │  │    顺序执行即释放；全前景/全背景 ±max(Hq,Wq) 常量场）
      │  │  │    → initial_query_phi[Hq,Wq] → ownership crop
      │  │  │    → macro_phi[Hm,Wm]（CPU float32 torch 参数）
      │  │  ├─ torch.optim.Adam（CPU；契约超参 betas=(0.9,0.999)/
      │  │  │    eps=1e-8/weight_decay=0/amsgrad=False）
      │  │  └─ for state in 0..N（N+1 已评价宏状态）：
      │  │     ├─ state<N：macro_gradient_magnitude（恰一次/state：
      │  │     │    [Hm+2,Wm+2] halo，外围=initial 固定 context、
      │  │     │    中心=当前快照，中心差分 /2 → sqrt(dx²+dy²)）
      │  │     ├─ _skeleton.pack_batches（每 macro 打包一次；组批 target/ownership/trainable_index/
      │  │     │    context_valid 四画布
      │  │     ├─ 快照 gather → local 叶子（requires_grad 仅 state<N）
      │  │     │    → _LevelSetBinarize.apply(local_phi, local_grad)
      │  │     │    （forward=hard phi<0；backward=-|grad|·上游，
      │  │     │    系数返回 None，内部零空间差分）
      │  │     ├─ 固定 context：窗口内 hard target>=0.5、padding 恒 0
      │  │     │    → mask = where(trainable>=0, hard, context)
      │  │     ├─ forward_many（三条件一次 FFT）
      │  │     ├─ _common.owned_continuous_losses + curvature_loss
      │  │     │    （作用于 hard mask；weight=0 不构建）
      │  │     ├─ weighted loss.backward() → np.add.at 宏梯度
      │  │     │    （raw sum 不平均）→ 释放 → on_tiles_completed
      │  │     ├─ record（ILTStateRecord 恒 0/state/1）
      │  │     │    → macro best 严格更低（平局保早）
      │  │     └─ state<N：梯度有限检查 → param.grad=宏梯度 →
      │  │          optimizer.step() 恰一次 → zero_grad(set_to_none=True)
      │  │          → phi/Adam 态有限检查
      │  │  └─ best 物化：binary=(phi<0)、soft=sigmoid(-phi)（仅诊断）
      │  ├─ _ilt_workflow.py::_evaluate_best_binary（REQ 终评；
      │  │    固定 context 由 levelset.py::build_levelset_final_context_canvas
      │  │    提供——hard target，不重跑 SDF）
      │  ├─ pixel/problem.py::reconstruct_pixel_region（行游程 + merge）
      │  └─ write_macro_gds → levelset_ilt_result.npz + metrics.json
      │       （best_parameters 语义为 phi）
      ├─ merge_macro_results（恰一次；空 macro 候选按零覆盖容忍）
      ├─ save_final_lithography（可选）
      └─ summary.json（seam_strategy=macro_independent_fixed_context 入档）
```

## 伪代码

```text
prepare：与 Simple 完全共享（逐 macro 一次栅格化 → NPZ → ilt_plan）
solve（每宏独立）：
  phi = SDF(target_u8)[ownership crop]                  # SciPy EDT once/macro
  adam = Adam(phi, lr)                                  # CPU 契约超参
  for state in 0..N:
    if state < N:
        mag = 中心差分(halo = 初始 context 环 + phi 快照)  # 恰一次/state
    for batch in cores(batch_size):                     # 同一参数快照
        hard = STE(phi[trainable], mag[trainable])      # forward=phi<0
        context = 窗口内 (T>=0.5) / 窗口外恒 0           # hard 三值语义
        mask = where(trainable, hard, context)
        printed = forward_many(mask, 三条件)
        L = L_nom + w_proc·L_proc + w_pv·L_pv + w_curv·曲率(hard mask)
        L.backward(); np.add.at(宏梯度, trainable索引, local梯度)  # raw sum
        释放；on_tiles_completed(批大小)
    record + macro best（严格更低，完整已评价宏状态）
    if state < N: adam.step 恰一次 → 有限检查            # 屏障后单步
  best → binary=(phi<0) / soft=sigmoid(-phi)
final：best binary 每批一次 forward（hard context 策略，不跑 SDF）
       → ownership 二值 L2/PVBand → reconstruct → best.gds + result/metrics
merge：全部宏完成后恰一次（空宏=零覆盖）→ 可选最终光刻 → summary
```

## 本工作流边界

- 与 Simple 共享全部 prepare/终评/产物/merge 生命周期；唯一差异是
  参数化（phi vs sigmoid 参数）、代理梯度（外部系数 vs 直接 sigmoid）、
  优化器（Adam vs SGD）与 context 的 transmission 定义（hard vs soft）。
- phi/Adam m/v/宏梯度常驻 CPU float32；GPU 每批仅 local 叶子图，批后
  释放；SDF once（EDT float64 中间量即用即弃），mag 每次 [Hm+2,Wm+2]。
- 同一参数跨 core 恒同 phi/同系数（INV-003）：core-local SDF/差分为
  契约违例；context < 1 像素无条件入口拒绝。
- 像素中心 SDF |phi| ≥ 1，Adam 首步 |Δ| < lr：lr ≤ 1 时边界像素需
  多状态累积才可能越过 0 等值线（选步长时注意）。
