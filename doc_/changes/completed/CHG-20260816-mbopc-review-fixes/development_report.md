# Development Report — CHG-20260816-mbopc-review-fixes

## 实际交付（提交链）

```text
3725c0e  fix(mbopc): P1 三项（insufficient_probes / 几何流式与 layer_bbox /
         _as_int 严格整数校验）
acfcab0  perf(mbopc): P2 组（reference 复用、整 batch 回切、跳过重复评价、
         末轮纯评价、前置校验、tqdm finally、真构造越界用例）
e289f2c  docs(mbopc): 同步审查修复轮与已知限制
```

## 关键实现事实

- 2nm 壁 + 8nm 探针实测：修复前 zero_epe 误报 → 修复后
  insufficient_probes（valid=0/epe=0/保留 baseline/原因在案）；
- merge 窗口化验证需显式裁回 ownership（materialize_intersecting 不裁剪，
  跨界 polygon 会被相邻窗口重复计数）；
- 实测发现并记录：−25/−30 边交叉会被 miter 解析成反向合法 ring（守卫不
  触发），−20 共线退化是最先触发的守卫形态（ValueError）；
- **行为变化**：no_update 时 records 只含 baseline（无变化提案不再重复
  评价一轮）；metrics.json 消费方须知；
- gcd_45nm smoke 三版本（迁移后/P1 后/P2 后）四 macro best_epe 逐位一致
  （7263/5904/5625/4884）——零算法漂移。

## 涉及文件

`opc/iteration/mbopc/simple.py`、`main/{_mbopc_workflow,_macro_pipeline,
run_macro_pipeline}.py`、三测试文件、手册与报告（见 e289f2c）。
