# Dataflow — Gradient MB-OPC（midpoint STE 梯度 + Adam，含可微 EPE loss）

owner 边段法向位移的连续优化：midpoint STE 代理梯度、四项连续 loss
（nominal/process/PVBand + 可选 EPE profile）、同步 Adam。

入口：`python main/run_mbopc_gradient.py config/gradient_mbopc.toml`

## 函数级流向

```text
main/run_mbopc_gradient.py::main
└─ main/run_mbopc_gradient.py::run_gradient_mbopc（GRADIENT_METHOD 适配器同文件）
   └─ main/_mbopc_workflow.py::run_mbopc_workflow(GRADIENT_METHOD)
      ├─ configuration.py::load_config（[gradient] 段，尾部可选
      │    weight_epe=0.0 / epe_steepness=4.0，旧 TOML 兼容）
      ├─ prepare_problems（共享生命周期，见 macro_pipeline.md）
      ├─ resolve_gradient_config（跨段校验 + nm→DBU；lr 超位移上限仅警告）
      │    → GradientMBOPCConfig
      ├─ ICCAD13Lithography(device) + TargetCanvasCache
      ├─ 逐 macro：MacroProblem.load
      │  └─ _solve_macro → opc/iteration/mbopc/gradient.py::optimize_gradient_macro
      │     ├─ 入口契约：画布一致；epe_distance ≤ context；
      │     │    weight_epe>0 时 epe_distance 必须为 pixel 整数倍（R=Q/2≥1）
      │     ├─ 无 owner 段 → no_owned_segments 空结果（不建 optimizer）
      │     ├─ _prepare_macro_context → _GradientMacroContext（静态，一次）
      │     │    owner_ids / segment_to_parameter / 参考几何（materialize 一次）
      │     │    零位移 Region+采样中点 / _batching.pack_macro_statics
      │     │    （计分画布/EPE 探针坐标/target 源，每 macro 一次）
      │     │    / 逐 core sampling membership / EPE profile+段长+L_sum（仅启用时）
      │     ├─ parameters[O]（device，requires_grad）+ Adam（固定超参）
      │     └─ for state in 0..iterations:
      │        ├─ _evaluate_state（同参数快照的全部 core 批）
      │        │  ├─ 组批：_batching.cached_target_canvas
      │        │  │    + rasterize_mask_canvas（当前候选）+ 静态打包
      │        │  │    ownership 画布（逐态不重算）+ 已发布段中点
      │        │  ├─ gradient.py::_EdgeGradientMask.apply（STE：forward 数值
      │        │  │    =hard 栅格，backward 在段中点双线性采样 2·g_mid/pixel_dbu）
      │        │  ├─ forward_many(nominal, dose_max, defocus_min)（一次 FFT）
      │        │  ├─ owned 三 loss + weight_epe>0 时：
      │        │  │    _profile_d_s(nominal_error, slots, xy)
      │        │  │    → penalty=2(σ(γ·d_s)−0.5) → 批 L_epe=Σlen·pen/Σlen
      │        │  ├─ weighted batch_loss.backward()（梯度经 _EdgeGradientMask
      │        │  │    的 autograd 边自动 scatter-add 累加进 parameters.grad）
      │        │  │    → _batching.discrete_batch_diagnostics（L2/PV/EPE 探针）
      │        │  └─ 释放批张量 → on_tiles_completed
      │        ├─ 非有限检查 → record（epe_loss 字段）→ best 严格更低
      │        │    → zero_loss / iteration_limit 判断 → 梯度有限检查
      │        └─ _take_optimizer_step：step → clamp ±max → torch.equal
      │             判 no_update（None）→ 展开 candidate_full →
      │             reconstruct_region_with_midpoints（候选 Region+采样中点
      │             成对产出；ValueError/ReconstructionError 上抛由主函数
      │             置 invalid_geometry）→ 成对发布
      │  └─ _solve_macro 尾部：write_macro_gds → save_macro_result
      │     （gradient_result.npz + metrics.json 含逐 state epe_loss）
      ├─ merge_macro_results 恰一次 → 可选 save_final_lithography
      └─ summary.json（loss_weights 含 epe、epe_steepness、best_epe_loss）
```

## 伪代码

```text
ctx = _prepare_macro_context(problem)                  # 静态一次
params = zeros[O]; adam = Adam(params, lr)
for state in 0..N:
    zero_grad（仅 state<N）
    for batch in cores(batch_size):                     # 同一参数快照
        mask = STE(候选栅格, 段中点)                    # forward=hard，backward=2·g_mid/pixel
        printed = forward_many(mask, 三条件)
        L = w_nom·L_nom + w_proc·L_proc + w_pv·L_pv
            + w_epe·Σ_s len_s·2(σ(γ·Σ_q D)−0.5) / Σ len_s    # D=(Z_nom−T)²
        L.backward(); np.add.at(宏梯度, trainable索引, local梯度)   # 求和
        释放；on_tiles_completed(批大小)
    record（四项 loss + 离散诊断）；best ← 严格更小的完整已评价状态
    if state == N: break                                 # 末状态纯评价
    梯度有限检查 → adam.step → clamp ±max →
    (region, midpoints) = reconstruct_region_with_midpoints(候选)
        失败 → invalid_geometry（保留历史 best，原因入 stop_detail）
    成对发布 region 与 midpoints（下一状态栅格/采样同源）
```

## 本工作流边界

- 同 state 全部批读同一 CPU 参数快照；宏梯度 scatter-add 求和不平均，
  屏障后恰一次 Adam step（Jacobi 语义）。
- EPE profile 预计算常驻 CPU O(O·Q)；`D` 复用 nominal 误差张量；启用
  EPE 不增加 forward_many 次数。weight_epe=0 时逐值兼容旧三 loss 路径。
- 采样中点恒来自与栅格化同一次合法重构（Region+midpoint 成对发布）。
