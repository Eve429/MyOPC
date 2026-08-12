# 最终光刻结果输出测试报告

## 覆盖

- 在线 MB-OPC：验证 manifest、tile NPZ、四类 PNG。
- 离线 MB-OPC：验证跨两个 core 的 ownership-only 输出。
- SimpleILT：验证最终 NPZ、四张 PNG，以及 PNG 关闭时 NPZ 保留。
- 静态检查：compileall 与 Ruff。

## 结果

命令：

```powershell
& 'D:\app\miniforge\envs\myopc\python.exe' -m pytest tests/opc/test_mbopc_cli.py tests/workbench/test_offline_workbench.py -q
```

结果：19 passed；四个修改模块 compileall 和 Ruff 均通过。
