# Architecture — 端到端数据流

## 总管线（GDS 到最终 GDS）

```text
GDS/OASIS/GLP
  -> main/_macro_pipeline.py::load_macro_config        TOML 六段校验，nm 参数 Decimal 保存
  -> prepare_problems                                  打开源版图一次（LayoutDB.open）
     ├─ layer_bbox                                     目标层包络（原生，不物化）
     ├─ exact_dbu × N                                  nm→DBU 精确整数换算（不能整除即失败）
     ├─ opc/input/grid.py::plan_macros                 两级网格（Macro→Core，契约校验）
     └─ 逐 macro：query(context).materialize_intersecting
        -> opc/input/edge/problem.py::prepare_macro_problem
           提边/分段/切线分裂/owner/CSR -> MacroProblem.save(NPZ)
     全部成功后写 plan.json
  -> 求解（见下）
  -> merge_macro_results(plan, {macro_id: best.gds}, out, cell_mode)
     逐 macro 回读候选 -> 裁 ownership -> PatchWriter.write_macro_results
     -> 逐 macro 窗口回读验证面积守恒
  -> summary.json（atomic_write_json）
```

## MB-OPC 求解流（main/_mbopc_workflow.py::run_mbopc_workflow + 方法适配器）

```text
load_config（算法段经适配器 algo_config_type 请求：[mbopc]/[gradient]）
  -> prepare_problems
  -> 适配器 build_solver_config（跨段校验 + nm→DBU）
  -> ICCAD13Lithography(device)（auto=有 CUDA 用 CUDA；资源统计起量）
  -> 逐 macro（稳定顺序，macro 间不交换状态；外层条 try/finally 收尾）：
       MacroProblem.load(NPZ)
       -> _solve_macro 公共包装：tqdm(total=(iterations+1)×core_count, unit=tile)
          -> method.optimize_macro（simple/gradient 算法本体）
             baseline(零位移) 评价 -> 逐状态：提案/更新 -> reconstruct 守卫 -> 评价
          -> reconstruct_region(best_displacements) -> write_macro_gds(best.gds)
       -> 适配器 save_macro_result（result.npz|gradient_result.npz + metrics）
  -> merge_macro_results（全部完成后恰一次，显式映射）
  -> save_final_lithography（可选；独立规整 tile 网格流式 PNG + manifest）
  -> summary.json（method/资源统计公共键 + 适配器附加键）
```

## 单次状态评价（simple.py::evaluate_and_propose 批循环）

```text
每批 batch_size 个 core：
  CPU 逐 tile：target（uint8 LRU 命中或零位移参考栅格）·
               current mask（当前候选 Region 栅格）· ownership · owner 探针坐标
  GPU 一次：target/255 + mask -> forward_many(nominal, dose_max, defocus_min)
            （no_grad；一次 mask FFT 共享）
  指标：L2/PVBand 只在 ownership 像素；EPE 全批探针一次评价（threshold 0.499）
  回写：direction×step 只写 next_values（owner 唯一写），整批一次 .cpu()
  释放：del 全部张量 -> on_tiles_completed(batch_count)（进度=真完成）
出口：写集核对（owner 恰写一次）· clip ±max_displacement · 返回指标+提案
```

## 验证管线（main/run_macro_pipeline.py，±2nm 双轮回零）

```text
load_macro_config + load_validation_deltas（[iteration] 冻结 [+2,-2]nm）
  -> prepare_problems -> run_round ×2（读旧写新、owner 恰写一次守卫、
     逐 core transmission sum、result NPZ + 完整候选 GDS）
  -> collect_round_macro_gds（result 轮次校验 -> 显式映射）
  -> merge_macro_results -> 回零 XOR 验证（最终 vs 原始目标层）
```

## 跨界标注

- Python/KLayout 边界：Region 构造/布尔/写出/读取（layout、geometry、
  merge 裁剪）；栅格化在 `geometry.iter_region_coverage_tiles` 原生分块。
- CPU/GPU 边界：批组装与指标累计在 CPU（numpy）；前向与 EPE 探针评价在
  模型 device；方向/掩码张量整批一次搬运。
- 文件边界：NPZ（problem/result，allow_pickle=False）、GDS（KLayout 原子
  写）、JSON（plan/metrics/summary/manifest，临时文件替换）。


## Simple ILT 流（main/_ilt_workflow.py::run_ilt_workflow + 方法适配器）

```text
load_config（[layout][partition][lithography][simple_ilt][output]，不读 [edge]）
  -> prepare_pixel_problems    LayoutDB 单次 open；resolve_grid_config（无 edge）
     -> 逐 macro：query(query_box).materialize_intersecting
        -> prepare_pixel_macro_problem（实际 box 整像素校验 → normalize_mask
           → 一次 rasterize_region_window → uint8 transmission）
        -> pixel_problems/<macro>.npz
     -> ilt_plan.json（键集兼容 merge/final-litho）
  -> ICCAD13Lithography(device)
  -> 逐 macro（独立；外层条 try/finally）：
       PixelMacroProblem.load
       -> optimize_simple_macro：N+1 个宏状态 × core 批（同一宏参数快照；
          scatter-add 梯度求和；屏障后单次 SGD step；macro best 严格更低）
       -> best binary 终评（每批一次 forward_many；ownership 二值 L2/PVBand）
       -> reconstruct_pixel_region → write_macro_gds(best.gds)
       -> simple_ilt_result.npz + metrics.json
  -> merge_macro_results（恰一次；空 macro 候选按零覆盖容忍）
  -> save_final_lithography（可选）
  -> summary.json（seam_strategy 显式入档）
```

跨界标注补充：像素 ILT 的 KLayout 边界只在 prepare、pixel→Region、GDS
写/merge；宏参数/梯度/best 常驻 CPU，GPU 每批一图 backward 后释放。
