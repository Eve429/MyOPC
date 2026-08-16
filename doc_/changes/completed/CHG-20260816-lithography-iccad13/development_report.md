# Development Report — CHG-20260816-lithography-iccad13

原始完整报告：`doc_/archive/reports/lithography_development_report.md`。

## 实际交付（提交链）

```text
6338710  配置/资产（ICCAD13Config/ProcessCondition/四 .pt + LICENSE）
8773e37  可微批量前向（iccad13.py ~370 行）
5f0747a  main 验证入口（六阶段演示）
ab7129e  迁移报告
b70eca3  演示入口追加阶段 6 matplotlib 可视化（用户追加需求）
```

## 关键实现事实

- `lithography/`（三公共类型）+ `main/main_test_lithography.py` +
  `tests/lithography/` 81 例（coverage 100%，204/204 语句）；
- requirements.txt 建立（klayout/matplotlib/numpy/pillow/psutil/torch，
  实测版本注明）；
- Windows DLL 修复：实施中实际复现 nvrtc DLL 缺失（直跑不经 conda run），
  按设计 §11.7 授权加回模块级 DLL 目录注册（必须先于 import torch），
  回归测试即子进程直跑用例。

## 与规格偏差

见原始报告 §12（DLL 补充、基线数字更新、main 阶段 6 追加等四项）。
