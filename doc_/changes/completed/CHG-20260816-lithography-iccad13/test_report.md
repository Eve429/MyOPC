# Test Report — CHG-20260816-lithography-iccad13

原始完整报告：`doc_/archive/reports/lithography_test_report.md`。

## 验证结论

- 专项 81 passed（配置 22 / 资产 10 / 条件 3 / 形状与 padding 11 / CPU
  数值 8 / 共享计算 2 / backward 7 / CUDA 4 / main 直跑 3 等类）；
  全量当时 224 passed；
- **数值身份**：CPU 三工艺角 sums 与 OpenILT 同资产基线逐位相等
  （nominal 25802.533203125 / dose_max 26009.16796875 /
  defocus_min 25675.23828125，差 0.0）；
- 有限差分 rtol=atol=2e-2 通过；batch 与逐张 atol 1e-6；
- CUDA（GTX 1650，2.5.1+cu124）：parity 1e-4；main 三条件前向 172.4ms /
  peak 32MiB；
- coverage 100%，无豁免分支。

## 环境

myopc conda env；资产随包分发，SHA-256 硬断言。
