# Dataflow — 总管线（共享宏生命周期与验证管线）

GDS 到最终 GDS 的权威生命周期：prepare（逐 macro 持久化 problem）→ 求解
（由具体工作流文件描述）→ merge（ownership 权威覆盖恰一次）。验证管线
`run_macro_pipeline` 是它的完整直接消费者（±2nm 双轮回零）。

入口：`python main/run_macro_pipeline.py config/macro_pipeline.toml`

## 函数级流向

```text
main/run_macro_pipeline.py::main → run(config_path)
├─ main/configuration.py::load_config(..., EdgeConfig, ValidationConfig, OutputConfig)
│    六段 TOML 单次读取；deltas = validation.round_deltas_nm（冻结 [+2,−2]nm）
├─ main/_macro_pipeline.py::prepare_problems                    [阶段 1]
│  ├─ layout.LayoutDB.open（全程唯一一次打开源版图）
│  ├─ layer_bbox → resolve_prepare_config（nm→DBU 精确整数换算）
│  ├─ opc/input/grid.py::plan_macros（两级网格契约校验；面积守恒复核）
│  └─ 逐 macro（行优先）：
│     ├─ database.query([layer], macro.query_box).materialize_intersecting
│     └─ opc/input/edge/problem.py::prepare_macro_problem
│        ├─ opc/input/mask.py::normalize_mask（合并物理覆盖、恢复孔洞）
│        ├─ geometry.extract_contour → edge/fragmentation.py::fragment_edges
│        ├─ problem.py::_split_segments_at_ownership_cuts（段内部不跨 owner）
│        └─ problem.py::_build_macro_ownership（owner/CSR membership）
│           → MacroProblem.save → problems/<macro>.npz
│     全部成功后 atomic_write_json(plan.json)
├─ run_round(plan, r, delta_dbu) ×2                             [阶段 2]
│  ├─ MacroProblem.load → 读上一轮位移 NPZ（首轮全零）
│  ├─ 逐 core：owner_segments_for_core 累加 delta（written 恰写一次守卫）
│  ├─ edge/reconstruction.py::reconstruct_region（新位移候选）
│  ├─ 逐 core：rasterize_mask_canvas → transmission sum（有限性检查）
│  └─ atomic_write_npz(result) + write_macro_gds(round gds)
├─ collect_round_macro_gds(plan, 2)（result 轮次校验 → 显式映射）
├─ main/_macro_pipeline.py::merge_macro_results                 [阶段 3]
│  ├─ 逐 macro：LayoutDB.open(best.gds) → layer_bbox
│  │    （层缺失 = 空候选，按零覆盖 Region 处理）→ clip 到 ownership
│  ├─ geometry/patch.py::PatchWriter.write_macro_results（双 cell 模式）
│  └─ 逐 macro 窗口回读验证覆盖面积守恒
├─ 回零 XOR 验证（最终版图 vs 原始目标层，面积必须为 0）        [阶段 4]
└─ summary.json

可选尾部（其他工作流复用）：
main/_macro_pipeline.py::save_final_lithography
  独立规整 tile 网格（macro_grid=(1,1)）逐批 forward → nominal/binary PNG + manifest
```

## 伪代码

```text
阶段 0  配置：load_config 六段；deltas 冻结 [+2,−2]nm；exact_dbu 换算（落不了格点即失败）
阶段 1  准备：with LayoutDB.open：
            plan_macros(bounds)                        # 两级网格
            for macro in macros（行优先）:
                batch = query(query_box).materialize_intersecting
                problem = prepare_macro_problem(...)   # 提边/分段/切线分裂/owner
                problem.save(NPZ)
            # 全部成功后才写 plan.json（失败不留"已完成"计划）
阶段 2  双轮：for r, delta in [(1,+2nm), (2,−2nm)]:
            for macro: 读旧位移 → 逐 core owner 累加 delta（恰写一次守卫）
                       → reconstruct_region → 逐 core transmission sum
                       → result NPZ + 完整候选 GDS
阶段 3  合并：for macro: 回读候选 → 裁自身 ownership → Patch（空候选=零覆盖）
            PatchWriter 写出最终版图 → 逐窗口面积守恒回读验证
阶段 4  回零：final XOR 原始目标层面积 == 0 → summary.json
```

## 本工作流边界

- KLayout 只在阶段 1（查询/物化）、阶段 2（候选写出）、阶段 3（回读裁剪/
  合并/验证）出现；阶段 2 的 transmission 求值在 torch。
- 单遍消费者 `main/run_single_pass.py`：同一套 resolve/plan/prepare/
  reconstruct 内联执行 + 单一位移 + `PatchWriter.write_macro_results` 直出
  最终版图（无 NPZ、无迭代、无光刻前向）。
