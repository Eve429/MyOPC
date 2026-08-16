# Test Report — CHG-20260815-macro-core-pipeline

原始完整报告：`doc_/archive/reports/macro_core_pipeline_test_report.md`。

## 验证结论

- 全量回归（当时）115 passed（layout 27 + geometry 25 + opc/input 55 不含
  后续套件口径变化）；专项 coverage 审计 84%；
- gcd_45nm smoke 通过标准全部满足：摘要 XOR==0、plan.json/problems×macro/
  两轮 results+gds ×macro/summary.json、final_xor_area == 0；
- 审查轮新增：复杂几何矩阵（斜边/跨 core/凹形/SREF 展开对照）、正逆序
  双轮位移与最终覆盖一致、未处理层不复制、空 macro 合法、切线交点去重
  回归。

## 环境

myopc conda env（解释器与依赖见 `requirements.txt`）；生成式 GDS/TOML
（tmp_path），不依赖 TestReticle 坐标。
