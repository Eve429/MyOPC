# Contract — opc.iteration.mbopc（最简 MB-OPC）

固定步长、EPE 驱动的离散边段移动求解器。
锚点：`opc/iteration/mbopc/simple.py`；编排 `main/_mbopc_workflow.py`
（simple 方法适配器宿主 `main/_simple_mbopc_workflow.py`）。

## 数据结构（frozen/slots）

```python
SimpleMBOPCConfig(iterations, initial_step_dbu, decay_every,
                  epe_distance_dbu, batch_size, target_cache_bytes)
    # 构造期校验正值；跨层约束（step≤max_disp、epe≤context）在 optimize 入口复验

class TargetCanvasCache(max_bytes)                # LRU；get/put(macro_id, core_index)
    # 0 上限禁用；单项超限不缓存；key 必含 macro id

SimpleMBOPCStep(next_displacements, epe, l2, pvband,
               valid_probes, ambiguous_probes, moved_segments)
IterationRecord(round_index, step_dbu, epe, l2, pvband,
                valid_probes, ambiguous_probes, moved_segments, elapsed_seconds)
SimpleMBOPCResult(best_displacements, records, best_round,
                  stop_reason, stop_detail)
```

## 算法函数

```python
def evaluate_and_propose(problem, current_region, current_displacements,
                         model, config, step_dbu, target_cache, *,
                         can_update, reference=None,
                         on_tiles_completed=None) -> SimpleMBOPCStep

def optimize_macro(problem, model, config, target_cache, *,
                   on_tiles_completed=None) -> SimpleMBOPCResult
```

- **入口契约**：位移长度=段数、有限、context 段=0；模型画布=问题画布；
  step≥0。reference 参数复用整迭代一次物化的参考几何（None 现算）。
- **批语义**：每批一次三条件 forward_many（no_grad）；方向 = current +
  {-1,0,+1}×step 只写 owner 段（written 恰一次守卫）；EPE 回切整批一次
  .cpu()；批后释放张量再报 on_tiles_completed(batch_count)。
- **轮次语义**：records[0]=baseline；Round N 指标属于第 N 次位移后状态，
  评价同时产生下轮提案（末轮纯评价不提案）；无变化提案直接停止不重复评价
  （no_update 时 records 只含 baseline）。
- **步长衰减**：产生 Round r 的步长 = initial × 0.5^((r−1)//decay_every)。
- **best 选择**：EPE 严格更小才更新，平局保留较早轮；L2/PVBand 只诊断。
- **五种停止**：zero_epe / no_update / invalid_geometry（重建守卫，含
  KLayout ValueError 退化形态，原因进 stop_detail 不吞错）/
  insufficient_probes（有 owner 段但 valid_probes==0——"无法评价"不是
  "零违规"；检查先于 best 比较）/ iteration_limit。空 macro（零段）= zero_epe。

## 编排契约（main/_mbopc_workflow.py）

```python
run_mbopc_workflow(method, config_path)            # 公共生命周期：加载→prepare
                                                   # →model/cache→macro 循环→merge
                                                   # →留档→summary（资源统计公共）
MBOPCMethod(method_name, algo_config_type,         # 方法适配器（simple 宿主
  build_solver_config, optimize_macro,             #   _simple_mbopc_workflow.py）：
  save_macro_result, macro_summary,                #   配置解析/optimizer/序列化/
  summary_extras)                                  #   摘要钩子注入公共层
resolve_mbopc_config(algo, partition, edge, dbu_nm)  # [mbopc] 跨段校验 + nm→DBU
                                                     # （configuration.py）
_solve_macro(method, problem, model, cfg, cache, out_dir, *, dbu_um,
            show_progress, progress_position, leave_progress)
    # 公共包装：tqdm total=(iterations+1)×core_count，unit=tile，
    # try/finally 收尾（内层条与外层 macro 条均有 finally 纪律）
save_final_lithography(plan, final_layout, model, batch_size, output_dir)
    # 独立规整 tile 网格；with 内逐 tile 窗口物化；PNG+manifest
```

- **独立 macro**：macro 间不交换中间状态；全部完成后恰一次
  `merge_macro_results`（显式 macro_id→GDS 映射）；这不是全局同步最优，
  差异须量化（gcd_45nm：single 总 EPE 比 multi 之和小 236 段、覆盖 XOR
  34650860 DBU²）。
- **macro 数量**：不加人为约束——macro_grid/macro_size_nm 是几就按几求解
  （单/多共用一个入口，数量模式配置层校验、size 模式 plan 后兜底）。
- **产物**：macros/<id>/{result.npz(format v1), best.gds, metrics.json} +
  final.gds + 可选 final_lithography/ + summary.json。
- **内存**：target uint8 LRU；GPU 批后释放；最终 PNG/merge 验证逐窗口物化
  （merge patches 持有全部 clipped 为已知上界——PatchWriter 接口在
  geometry/）。

## 事实核对锚点

`tests/opc/iteration/test_simple_mbopc.py`（53 例）、
`tests/main/test_mbopc_runners.py`（23 例）；gcd_45nm smoke 记录于
`changes/completed/CHG-20260816-simple-mbopc/`。
