# Test Report — CHG-20260816-mbopc-review-fixes

## 验证结论

- 全量 330 → **341 passed**（+11：insufficient_probes 真构造 1、配置类型
  注入 8、真构造越界 2）；
- ruff / compileall 全绿；gcd_45nm multi smoke 复跑：best_epe 与修复前
  逐位一致、870 tile PNG 照常（几何流式与窗口化验证零输出变化）；
- 停止路径补全覆盖：insufficient_probes（2nm/8nm 复现场景）、
  invalid_geometry 真构造 ×2（hole 越出 hull、共线退化 ValueError 形态）；
- no_update 行为变化用例更新（records 2→1 条，断言新语义）。

## 环境

myopc conda env；审查复现与修复验证同环境（torch 2.5.1+cu124）。
