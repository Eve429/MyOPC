# Test Report — CHG-20260816-simple-mbopc

原始完整报告：`doc_/archive/reports/mbopc_test_report.md`。

## 验证结论

- 全量 330 passed（+106 新用例）：evaluation 25（100%）、
  opc/iteration 51、runner 21；ruff/compileall 全绿；
- 关键机制证据（monkeypatch 计数）：每批恰一次三条件 forward_many、
  cache 命中免重栅格（8→4）、恰一次 merge、进度 =（iterations+1）×core；
- batch 不变性、macro 正逆序覆盖 XOR==0、L2 不打破 EPE 平局；
- 图形矩阵（真实 ICCAD13 CPU）：矩形/窄壁 hole/凹形/多 polygon/45° 斜边/
  跨 core/跨 macro/opaque/空 macro；
- 端到端 smoke：两入口 gcd_45nm CUDA 各 ~126s（数字见 development_report）。

## 环境

myopc conda env（torch 2.5.1+cu124 / GTX 1650）；生成式 GDS/TOML。
