# CurvMulti ILT 数据流（多尺度控制网格 + nominal wafer 曲率）

与 simple_ilt.md 共用像素 ILT 数据契约（PixelMacroProblem 四画布、三值
context、merge 恰一次）；本文只描述 CurvMulti 特有的参数化与 stage 循环。

## 函数级流向

```text
main/run_curvmulti_ilt.py::main（可选位置参数，默认 config/curvmulti_ilt.toml）
└─ main/run_curvmulti_ilt.py::run_curvmulti_ilt（CURVMULTI_ILT_METHOD 适配器同文件）
   └─ main/_ilt_workflow.py::run_ilt_workflow(CURVMULTI_ILT_METHOD)
      ├─ configuration.py::load_config（[layout][partition][lithography]
      │    [curvmulti_ilt][output]；不读取 [edge]）
      ├─ _ilt_workflow.py::prepare_pixel_problems（含 layout_bounds 场边界）
      ├─ lithography.ICCAD13Lithography(device) → 逐 macro：
      │  └─ curvmulti.py::optimize_curvmulti_macro
      │     ├─ 入口校验（画布一致/整除全部 scale/最粗网格≥kernel/
      │     │    曲率 context≥1px）
      │     ├─ 初值 = 宏 ownership raw T（[Hm,Wm]）
      │     └─ 逐 stage（scales 粗→细，每 stage 独立 SGD）：
      │        ├─ stage0 参考 _common.resize_image(area)；
      │        │    后续 stage 初值 = resize_image(上一 stage best, nearest)
      │        └─ 逐 state（评价→屏障 step→评价…共 N+1 态）：
      │           └─ 逐 core batch（_skeleton.run_state_batches，静态画布每 macro 打包一次）：
      │              ├─ _common.smooth_sigmoid_mask(control, k, β, offset)
      │              ├─ _common.resize_image(nearest) 上采样回 [Hm,Wm]
      │              ├─ trainable 索引 gather → 三值画布组装
      │              │    （context=σ(β(2T−1))、padding=0）
      │              ├─ lithography.forward_many（完整物理网格）
      │              ├─ _common.owned_continuous_losses
      │              │    + curvature_loss(printed nominal, ownership)
      │              │    （曲率作用于 wafer——CurvMulti 的算法差异）
      │              └─ backward 梯度经可微链自然累加进 control.grad
      ├─ 终评：_evaluate_best_binary（build_curvmulti_final_context_canvas）
      ├─ 产物：best.gds + curvmulti_ilt_result.npz + metrics.json
      │    （records 写 stage_index/stage_state_index/scale 真值）
      └─ merge_macro_results（全部完成后恰一次）
```

## 阶段化伪代码

```text
control0_s = area(T_ownership, (Hm/s, Wm/s))          # 仅首 stage
for stage, s in enumerate(scales):                     # 粗 → 细
    control = warm(control_prev_best, s) if stage else control0_s
    optimizer = SGD([control], lr)                     # 每 stage 独立
    for k in 0..N:                                     # N+1 个评价态
        对每个 core batch:
            mask_ctrl = σ(β·(avgpool(control, k) − offset))
            full = nearest(mask_ctrl → [Hm,Wm])
            canvas = where(trainable, gather(full), context)  # padding 0
            printed = forward_many(canvas)
            loss = owned_losses + w·curvature(printed_nominal)
            if k < N: loss.backward() → control.grad 累加
        record(state 全局编号, stage, k, s, 各 loss)
        best/ stage_best: 严格更低更新（平局保早）
        if k == N: break
        屏障 finite 检查 → optimizer.step() 恰一次
    control_prev_best = stage_best（丢弃 optimizer/图）
输出：全局 best 控制网格 nearest 上采样 [Hm,Wm] → soft/binary → Region
```

## 本工作流边界

- 与 Simple/LevelSet 共用 `_ilt_workflow` 生命周期、PixelMacroProblem、
  场边界（layout_bounds）与 merge；方法差异只在参数化（控制网格 +
  平滑 sigmoid）与曲率对象（printed nominal wafer）。
- 粗尺度只减参数自由度：光刻 forward 恒在完整物理网格（最近邻上采样后）。
- 产物命名 `curvmulti_ilt_result.npz`（公共 ILT 结果格式，
  best_parameters=best 控制网格的 nearest 上采样）。
