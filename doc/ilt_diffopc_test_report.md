# ILT 与 DiffOPC 迁移测试报告

定向命令：

```powershell
& 'D:\app\miniforge\envs\myopc\python.exe' -m pytest tests/opc/test_new_methods.py -q
```

结果：3 passed。覆盖水平集/多尺度统一结果、软边段有限梯度和离线问题消费；既有光刻/工作台回归 26 passed，Ruff 与 compileall 通过。

后续补充孔洞/斜边跨 core 的数值梯度、MRC/SRAF 和大版图 CUDA 峰值测试。
