# 代码性能与可读性优化测试报告

## 1. 测试环境

- Python 3.12.0；KLayout 0.30.10；NumPy 2.5.1；
- Windows 10，16 逻辑 CPU，测试进程可见内存约 15.37 GiB；
- CUDA 设备执行 ICCAD13 与真实版图三轮验证；
- 解释器：`D:\app\miniforge\envs\myopc\python.exe`。

## 2. 自动回归与静态检查

```powershell
$python = 'D:\app\miniforge\envs\myopc\python.exe'
& $python -m ruff check layout geometry opc lithography evaluation benchmarks tests `
  run_layout_geometry.py run_mbopc.py run_mbopc_frontend.py
& $python -m compileall -q layout geometry opc lithography evaluation benchmarks tests `
  run_layout_geometry.py run_mbopc.py run_mbopc_frontend.py
& $python -m pytest -q
```

结果：Ruff 与 compileall 通过，135 tests passed in 23.76 s。

专项覆盖率命令对 `tests/opc`、`tests/lithography`、`tests/evaluation` 执行 branch coverage，结果为 81 tests passed，综合 statement/branch coverage 92%。

新增回归覆盖：

- `layout` 独立导入不加载 `geometry`；
- 连续轮廓缓冲的 dtype、连续性、孔洞和重建生命周期；
- 三工艺角共享频谱与优化前独立 FFT 公式逐像素对照；
- nominal/maximum/minimum 组合损失可向 mask 传播有限非零梯度；
- target cache 命中和未命中都保持 `uint8`；
- membership CSR owner 结果与全局扫描逐 core 完全一致；
- MB-OPC 前端严格基准从仓库外工作目录直接运行。

## 3. 性能结果

| 项目 | 修改前 | 修改后 | 结论 |
|---|---:|---:|---|
| 百万实例 ROI 中位数 | 0.10375 ms | 0.10170 ms | 无退化 |
| 2048² 栅格 | 502.317 ms | 483.731 ms | 精确、无退化 |
| 20k 矩形 contour 峰值 | 6,679,859 B | 1,845,632 B | 下降约 72% |
| ICCAD13 batch=8 中位数 | 25.0241 ms | 16.4560 ms | 1.5207× |
| ICCAD13 GPU 峰值 | 277,296,128 B | 277,296,128 B | 未增加 |
| owner 索引合成对照 | 45.908 ms | 13.178 ms | 约 3.48× |

ICCAD13 三个工艺角最大逐像素误差分别为 nominal 0、maximum 5.82e-7、minimum 1.50e-7，均小于约定的 5e-6。

5,000 图形严格 MB-OPC 前端基准：110,000 segments、134,734 memberships；prepare 122.358 ms、materialize 13.097 ms、零位移重建 398.755 ms；紧凑数组节省 67.46%，XOR、超长段和无 owner 检查全部通过。

## 4. 真实版图结果

### 4.1 `simple.gds`

默认 1024/512 nm core/halo、8 nm pixel、batch=8、CUDA 三轮：

| 轮次 | EPE | L2 | PVBand | valid | ambiguous | moved/rejected |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 338 | 2822.4661 | 388.9287 | 880 | 0 | 338/0 |
| 1 | 203 | 1766.5405 | 415.5956 | 880 | 0 | 203/0 |
| 2 | 113 | 1309.4220 | 436.1449 | 880 | 0 | 113/0 |

停止原因为 `iteration_limit`，最佳轮次为 2，结果 GDS 可由 KLayout 重新读取。

### 4.2 `gcd_45nm.gds`

完整 Layer 11/0：1,776 polygons、21,590 edges、223,553 segments、870 cores、880,801 memberships。

| 轮次 | EPE | L2 | PVBand | valid | ambiguous | moved/rejected |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129645 | 1038629.5250 | 115627.0156 | 223298 | 51 | 129594/0 |
| 1 | 74592 | 563335.5260 | 134541.1511 | 223298 | 5 | 74587/0 |
| 2 | 48348 | 440251.4335 | 147187.1436 | 223298 | 2 | 48346/0 |

计数、EPE、valid/ambiguous、移动/拒绝和停止原因与历史结果一致；L2/PVBand 差异在约定容差内。problem 常驻数组 9,802,180 bytes，GPU 峰值 267,334,656 bytes，总耗时 79.834 s；历史阶段 28 为 85.892 s。输出 GDS 含非空 `REFERENCE`、`RECONSTRUCTED` 顶层 Cell，可被 KLayout 重新读取。

## 5. 结论

所有功能、数值、性能和架构门槛通过。PVBand 随轮次上升仍按原值报告；本轮没有把单一指标改善误写为整体收敛，也没有把独立的大 reticle 方案描述为已实现能力。
