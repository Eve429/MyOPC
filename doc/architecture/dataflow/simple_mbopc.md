# Dataflow — Simple MB-OPC（离散 EPE 驱动）

固定步长、EPE 驱动的最简 MB-OPC：方向 ∈ {−1,0,+1}×step（clip 到
±max_displacement），EPE 严格更小才更新 best，平局保留较早轮。

入口：`python main/run_mbopc.py config/mbopc_single_macro.toml`（macro 数由 config 决定）

## 函数级流向

```text
main/run_mbopc.py::main
└─ main/run_mbopc.py::run_mbopc（SIMPLE_METHOD 适配器同文件）
   └─ main/_mbopc_workflow.py::run_mbopc_workflow(SIMPLE_METHOD)
      ├─ configuration.py::load_config（[mbopc] 段）
      ├─ _macro_pipeline.py::prepare_problems（共享生命周期，见 macro_pipeline.md；
   │    bounds 经 resolve_field_bounds 处理 [layout] field_box/field_size）
      ├─ configuration.py::resolve_mbopc_config（跨段校验 + nm→DBU；入口导入）
      │    → SimpleMBOPCConfig
      ├─ lithography.ICCAD13Lithography(device)（auto；资源统计起量）
      ├─ mbopc/_cache.py::TargetCanvasCache(target_cache_bytes)（跨 macro 共享）
      ├─ 逐 macro（稳定顺序；macro 间不交换状态；外层条 try/finally）：
      │  ├─ edge/problem.py::MacroProblem.load(NPZ)
      │  ├─ _mbopc_workflow.py::_solve_macro（tqdm total=(iterations+1)×core_count）
      │  │  └─ opc/iteration/mbopc/simple.py::optimize_macro
      │  │     ├─ segments.materialize()（参考几何整迭代一次）
      │  │     ├─ reconstruct_region(零位移) → baseline 评价 + Round1 提案
      │  │     └─ 逐轮：候选 reconstruct 守卫 → evaluate_and_propose（末轮纯评价）
      │  │        └─ evaluate_and_propose 批循环：
      │  │           ├─ CPU 逐 tile 组批：TargetCanvasCache.get/put
      │  │           │    （miss → 零位移参考栅格化 uint8 回填）
      │  │           ├─ rasterize_mask_canvas（当前候选）+ ownership_canvas
      │  │           ├─ edge/sampling.py::edge_probe_points（参考中点±法向）
      │  │           │    → opc/input/raster.py::points_to_canvas（居中换算）
      │  │           ├─ lithography.forward_many(nominal, dose_max, defocus_min)
      │  │           │    （no_grad；一次 mask FFT 共享）
      │  │           ├─ evaluation.evaluate_binary_l2 / evaluate_pvband
      │  │           │    （只在 ownership 像素）+ evaluate_edge_probes
      │  │           │    （全批探针一次；阈值跟随模型 PrintThresh）
      │  │           └─ direction×step 只写 next_values（owner 唯一写；
      │  │                整批一次 .cpu()）→ 写集核对 → clip ±max
      │  │     出口：records[0]=baseline；best 位移 → reconstruct_region
      │  └─ _solve_macro 尾部：write_macro_gds(problem.layer, best_region)
      │     → run_mbopc.py::save_macro_result（result.npz + metrics.json）
      │     → macro_summary
      ├─ merge_macro_results（全部完成后恰一次，显式 macro_id→GDS 映射）
      ├─ save_final_lithography（可选）
      └─ atomic_write_json(summary.json)（方法/资源公共键 + 适配器附加键）
```

## 伪代码

```text
step(r) = initial_step × 0.5^((r−1) // decay_every)          # 步长衰减
reference = segments.materialize()                            # 唯一一次
baseline = evaluate(零位移)；records[0] = baseline
if owner>0 且 valid_probes==0: 停止 insufficient_probes       # 先于 best 比较
elif baseline.epe==0: 停止 zero_epe
else:
  for r in 1..iterations:
     candidate = 上一评价的提案
     if 提案无变化: 停止 no_update                            # 不重复评价同状态
     candidate_region = reconstruct_region(candidate)          # 守卫（含 KLayout
         ValueError 退化形态 → 停止 invalid_geometry，原因入 stop_detail）
     proposal = evaluate(candidate, can_update=(r<iterations)) # 末轮纯评价
     records[r] = proposal（指标属第 r 次位移后状态）
     if owner>0 且 valid_probes==0: 停止 insufficient_probes   # 先于 best
     if proposal.epe < best_epe: best ← candidate              # 严格更小，平局保早
     if proposal.epe == 0: 停止 zero_epe
     if can_update 且 提案无变化: 停止 no_update
  未触发停止 → iteration_limit
每次 evaluate（批循环）：
  for batch in cores(batch_size):
     targets = LRU 命中或零位移参考栅格（uint8/255）
     masks = 当前候选栅格；ownership = 唯一计分像素
     probes = 参考中点 ± epe_distance×法向 → canvas 坐标
     printed = forward_many(masks, 三条件)                     # no_grad 一次 FFT
     l2/pvband 只计 ownership；epe 全批探针一次评价
     next_values[owner段] += direction×step（written 恰写一次）
     释放全部张量 → on_tiles_completed(真实批大小)
  写集核对 + clip ±max_displacement → 返回指标与提案
```

## 本工作流边界

- 离散方法全程 `no_grad`；GPU 上只有批张量与三条件前向，批后释放。
- target 走 CPU uint8 LRU（key 含 macro id）；当前候选每状态重新栅格化。
- L2/PVBand 仅诊断；EPE 是唯一驱动 best 的指标。
