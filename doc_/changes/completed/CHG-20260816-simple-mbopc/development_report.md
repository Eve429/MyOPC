# Development Report — CHG-20260816-simple-mbopc

原始完整报告（含审查修复轮）：`doc_/archive/reports/mbopc_development_report.md`。

## 实际交付（提交链）

```text
2b9194a  lithography 契约（LithographyModel/LithographyConfigView）
a5509bc  evaluation 三指标（不迁 shot；EPE threshold 保留 0.499）
c596d70  points_to_canvas 居中坐标（已有栅格逐值不变）
71d42ba  共享宏管线生命周期（±2/-2 与 gcd XOR 验证不变）
986cbfd  求解器 simple.py（五结构 + 两算法函数）
84407e5  单/多 macro 两入口 + workflow + 两 TOML + tqdm
0b21e54  端到端验证与报告
```

## 关键实现事实

- 全量 224 → 330 passed；evaluation coverage 100%、simple.py 99%；
- gcd_45nm CUDA 实测：multi（2×2）126.0s，四 macro EPE 逐轮单调下降
  （37743→7263 等）；single 126.6s（128227→23440）；
- 独立 macro 代价量化：single 总 EPE 比 multi 之和小 236 段（~1%），
  覆盖 XOR 34650860 DBU²；
- 实施中真实 bug 一枚：方向写入漏乘步长（±1 DBU 而非 ±step），测试拦截。

## 与规格偏差

原始报告 §3 列 8 项（threshold 0.499、solve_macro 补 dbu_um、
save_final 独立网格、§3.4 过时、"漏 padding"措辞、simpleopc 不在归档、
tqdm 已在环境、plan 去 round_deltas_dbu）。
