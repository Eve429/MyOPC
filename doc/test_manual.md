# MyOPC 测试手册（迁移期）

## 1. 运行方式

```bash
# 全量（当前 411 用例）
D:/app/miniforge/envs/myopc/python.exe -m pytest -q tests
# 单套件 / 单用例
D:/app/miniforge/envs/myopc/python.exe -m pytest -q tests/opc/input/test_grid.py
D:/app/miniforge/envs/myopc/python.exe -m pytest -q tests/main/test_macro_pipeline.py::TestTwoRounds
```

`pyproject.toml` 已固定 `testpaths=["tests"]`、`addopts="-ra"`。

## 2. 套件与职责

| 套件 | 职责 |
|---|---|
| tests/layout | 版图打开/查询/物化/GLP（27） |
| tests/geometry | 轮廓提取、校验、Patch、栅格化（25） |
| tests/opc/input | 两级网格规划与校验、居中 canvas 与极性、points_to_canvas、MacroProblem 与 NPZ（49） |
| tests/main | 管线配置校验、阶段产物、双轮状态机、最终合并、单遍入口（34） |
| tests/lithography | 配置解析、资产哈希/布局、前向数值参考、性能计数、backward 有限差分、CUDA parity、main 直跑（81） |
| tests/evaluation | L2/PVBand/EPE 指标与方向表、ownership 屏蔽、阈值边界、光刻契约 isinstance（25） |
| tests/opc/iteration | simple MB-OPC：cache 全路径、入口契约、stub 方向/全部停止路径（含 insufficient_probes 与两个真构造越界）、batch/进度/计数、真实 ICCAD13 图形矩阵、CUDA 直通（54）；gradient MB-OPC：surrogate 2·g_mid 公式与越界/重复索引、真实 ICCAD13 ±1 DBU 有限差分方向一致（clear/opaque）、loss 独立复算与 halo 屏蔽、batch 不变与 Adam 屏障事件序、状态/best 快照、共线退化真构造、几何矩阵、调用计数、跨 core membership 采样计数（40 条）与梯度 SUM 累加（P1-1 回归）、CPU/CUDA（45） |
| tests/main/test_mbopc_runners | 单/多 macro 入口端到端：产物与 records 语义、恰一次 merge、正逆序、batch 不变性、invalid 保留 best、差异上界量化、配置类型注入、仓库外直跑、进度开关（23） |
| tests/main/test_gradient_mbopc_runner | 梯度入口端到端：[gradient_mbopc] 配置契约（类型/权重/Decimal/epe 整除）、产物与 summary（§8.2 键全集、RSS/CUDA 字段）、多 macro 一次合并、正逆序 XOR==0、进度计数与异常收尾、仓库外直跑（25） |

## 3. 测试纪律

- **全生成式数据**：GDS/TOML/NPZ 一律在 `tmp_path` 内动态生成；不依赖
  `TestReticle/*.gds` 用户数据（该目录仅 gcd_45nm 供最终 smoke，只读）。
- 每个几何不变量成组断言：零位移 XOR == 0、owner 唯一（`0≤o<C`，不是
  owners 值互异）、own⊆membership、ring 拓扑保持、法向单位向量。
- 阶段边界用 monkeypatch 调用计数证明，不用注释或口头约定。
- bug 修复必须携带可复现回归用例；构造期不变量（如 CSR 边界）在
  `__post_init__` 校验，测试负责注入破坏值验证拒绝路径。
- lithography 数值纪律：CPU 参考值（三工艺角 sums）与 OpenILT 同资产
  基线绑定（实测逐位相等）；资产 SHA-256 是硬断言，漂移即说明数值参考
  全部失效。

## 4. 光刻模型直跑验证

```bash
D:/app/miniforge/envs/myopc/python.exe main/main_test_lithography.py
```

通过标准：退出码 0；输出包含 device、三工艺角 range/sum/曝光像素、
batch `(2, 256, 256)`、`梯度 finite=True`、`阶段 6 · 可视化` 与
`已保存`；CUDA 时附 elapsed 与 peak allocated。从仓库外工作目录执行
同样必须成功（sys.path 自引导）。

阶段 6 可视化（2026-08-16 追加）：2×2 灰度面板（输入 mask + 三工艺角
连续胶图，origin=lower 保持左下原点显示），PNG 留档到
`output/lithography/main_test_lithography.png`（gitignored、锚定仓库根），
随后 `plt.show()` 弹窗——**手工直跑会等待窗口关闭**；测试子进程设
`MPLBACKEND=Agg`（show 无操作不阻塞）。

coverage：

```bash
D:/app/miniforge/envs/myopc/python.exe -m coverage run --source=lithography -m pytest -q tests/lithography
D:/app/miniforge/envs/myopc/python.exe -m coverage report -m
```

## 5. MB-OPC 入口 smoke（2026-08-16）

```bash
D:/app/miniforge/envs/myopc/python.exe main/run_mbopc_single_macro.py config/mbopc_single_macro.toml
D:/app/miniforge/envs/myopc/python.exe main/run_mbopc_multi_macro.py config/mbopc_multi_macro.toml
```

通过标准：退出码 0；摘要含 device、每 macro `best_round/best_epe/stop`、
合并耗时与最终版图；`work_dir` 下 plan.json、problems/、macros/<id>/
{result.npz,best.gds,metrics.json}、summary.json、final.gds；
`save_final_lithography=true` 时 final_lithography/ 有逐 tile PNG 与
manifest。gcd_45nm 默认参数 CUDA 实测约 126s（multi 870 tile，EPE 逐轮下降，
报告见 `doc/opc/mbopc_test_report.md`）。产物目录不提交。

### 梯度入口 smoke（2026-08-17）

```bash
D:/app/miniforge/envs/myopc/python.exe main/run_gradient_mbopc.py config/gradient_mbopc.toml
```

通过标准：退出码 0；摘要含 device、loss 权重、每 macro
`best_state/best_total_loss/stop`、合并与总耗时、峰值 RSS/CUDA、最终版图；
`work_dir/macros/<id>/` 下三件产物文件名为 `gradient_result.npz`（键
format_version/macro_id/best_state_index/best_displacements/stop_reason）、
`gradient_metrics.json`（records 含 state_index 与三项连续 loss）、`best.gds`；
summary.json 顶层含 `method="gradient_mbopc"`、`loss_weights`、
`rss_start_bytes/rss_after_prepare_bytes/peak_rss_bytes`、`cuda_peak_bytes`
（CPU 运行为 null）。产物目录不提交；实测数字见
`doc/opc/gradient_mbopc_test_report.md`。

## 6. 管线 smoke 验收

```bash
D:/app/miniforge/envs/myopc/python.exe main/run_macro_pipeline.py config/macro_pipeline.toml
```

通过标准：

- 摘要打印 `最终 XOR 面积：0（应为 0）`；
- `output/macro_pipeline/` 下：`plan.json`、`problems/*.npz` ×macro 数、
  每轮 `round_00N/results/*.npz` 与 `round_00N/gds/*.gds` 各 ×macro 数
  （2×2 → 两轮共 8 个 GDS）、`summary.json`；
- `summary.json` 的 `final_xor_area == 0`。

产物目录不提交；smoke 最终版图按 TOML 相对路径落在 `config/` 下时验证后
删除。

## 7. 已知口径

- `merge_peak_rss_bytes` 为合并完成后即时采样（psutil 无历史峰值接口），
  如实反映在测试报告。
- 完整 `ruff check .` 在未纳入本任务的 `geometry/contour.py` 有一个既存
  导入空行告警；专项范围（layout/geometry/opc/lithography/main/tests）
  必须全绿。
