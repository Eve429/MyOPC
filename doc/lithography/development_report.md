# lithography 迁移开发报告（ICCAD13 Hopkins 模型）

> 依据：`doc/lithography/lithography_migration_design.md`（用户批准实施）。
> 实施日期：2026-08-16；提交 `6338710`（A 资产）→ `8773e37`（B 前向）→
> `5f0747a`（C main 入口）→ 报告提交（D）。

## 1. 实际复制与重写的文件

**自 `00_PAST/lithography/` 复制（零修改，只读归档未动）**：

| 文件 | 说明 |
|---|---|
| `lithography/assets/iccad13/focus.pt` | complex64[35,35,24] |
| `lithography/assets/iccad13/defocus.pt` | complex64[35,35,24] |
| `lithography/assets/iccad13/focus_scale.pt` | float32[24] |
| `lithography/assets/iccad13/defocus_scale.pt` | float32[24] |
| `lithography/config/iccad13.txt` | 九字段“名称 值”配置 |
| `lithography/OPENILT_LICENSE.txt` | MIT（© 2023 Stanley Zheng） |

**重写（对照旧 `iccad13.py` 过滤迁移）**：`lithography/__init__.py`、
`lithography/iccad13.py`（约 370 行）。

**新增**：`main/main_test_lithography.py`、`tests/lithography/`（81 用例）、
`requirements.txt`、本报告与测试报告、两份手册更新。

## 2. 资产身份（SHA-256，测试硬断言）

```text
focus.pt          204bee39d8225c6d3cda52ea2d13b7c6f6cf4e4244de2ce960576d1bc741438f
defocus.pt        df624de9e17485d819e488ccada7edff133690cfa01370bfabec8f9e7cb8d532
focus_scale.pt    4e6f6136d419bdf0b56e9b461471c72d97ab3ba582fa19fe65ffaf25d188dab6
defocus_scale.pt  4ce70debf23593594c2fda1bd0cadf427abb0c132b16a047f6589f51148c8dc8
```

与设计文档 §3.2 声明逐字一致（复制后 sha256sum 复核）。

## 3. 为什么没有迁移（简化决策）

| 未迁移内容 | 理由 |
|---|---|
| `contracts.py` 的 `LithographyModel`/`LithographyConfigView` Protocol | 只有唯一具体模型，无第二实现与生产调用方；等真实第二模型再抽契约（设计 §4.1） |
| OpenILT `exact.py` 手写 backward / CT / combo kernel | 前向全原生可微算子，PyTorch 自动生成 backward，forward/backward 同一公式；CT kernel 仅为手写导数服务 |
| TorchLitho（Abbe、动态 TCC、任意 NA/波长/光源） | 固定 ICCAD13 资产不需要；引入参数系统、插值与 TCC 生成循环是另一个模型的评审任务 |
| `resolution != canvas` 的 nearest 插值分支 | 画布/分辨率冻结 256，配置层直接拒绝其他值（设计 §7.1）；不做插值伪装支持 |
| 资产布局 shape 猜测分支 | 只接受已声明的 [H,W,K] 方阵 + [K] 布局，kernel 维数量与 scale 严格一致，否则拒绝 |
| `_restore_size()` / `_kernel_bank()` 单调用点包装 | crop 与 bank 选择直接内联在 `forward_many` 对应紧凑块（设计 §8.1） |

## 4. 公共接口与 tensor 契约

```text
ICCAD13Config.from_file(path) -> 冻结配置（九字段 + 数值契约）
ProcessCondition(name, kernel, dose)   # kernel ∈ {"focus","defocus"}
ICCAD13Lithography(config_path?, asset_dir?, device?) -> nn.Module
model.device                            # buffer 设备
model.condition("nominal"|"dose_max"|"defocus_min")
model.forward(mask, condition)          # [H,W]/[B,H,W] → 同 shape float32
model.forward_many(mask, conditions)    # → dict[name, Tensor]
```

- 输入/输出：`1.0=透光`，行 0=最低 Y（与输入同向，不翻转 Y），范围 (0,1)；
- kernel/scale 是 buffer（`named_parameters()` 为空），`.to(device)` 四个
  buffer 同步移动；
- 输入 CPU/GPU 均可（模型搬到自身 device）；计算 dtype 固定
  float32/complex64。

## 5. 与当前 raster 的对齐

- `_prepare_mask` 的居中补零公式与 `opc.input.raster._center_padding`
  逐位一致（差值均分、奇数余量归高坐标侧）；满 256 输入 padding 全零，
  不二次移动（测试 `test_raster_canvas_passes_through_directly` 用
  `rasterize_mask_canvas(context 1824/pixel 8)` 直传验证）；
- 极性解释全部在 raster 侧完成（clear=coverage、opaque=1−coverage），
  模型只消费 1=透光的最终画布。

## 6. 性能不变量证据（monkeypatch 调用计数）

`test_single_fft_and_per_bank_propagation`：默认三条件一次
`forward_many` 中 `torch.fft.fft2` 恰 **1** 次、`_propagate` 调用序列恰
`[focus, defocus]` 各 1 次；`test_forward_many_matches_independent_calls`
证明共享路径与三次独立 forward 数值一致（atol 1e-6）。
`_kernel_multiply` 四象限四次批量赋值，无逐 kernel/batch/pixel Python
循环（代码审查确认）。

## 7. 数值验证结果

- **CPU 固定数值**：确定性两张 mask（[2,200,150]，旧测试移植）三工艺角
  sums 与 OpenILT 基线**逐位相等**（最大差 0.0，验收容差 0.05 未用到）：
  nominal 25802.533203125 / dose_max 26009.16796875 /
  defocus_min 25675.23828125；
- **有限差分**：非均匀权重（linspace −0.7~1.3）下检查点 (9,8)、ε=1e-3，
  autograd 与中心差分一致（rtol=atol=2e-2，旧测试同参数移植）；
- **batch 一致性**：batch 与逐张逐像素 atol 1e-6。

## 8. CUDA 实测（2026-08-16）

- 环境：torch 2.5.1+cu124，NVIDIA GeForce GTX 1650，CUDA 可用；
- CPU/GPU parity：三条件 rtol=atol=1e-4 通过；
- CUDA forward/backward 完成且梯度有限；
- main 入口实测：三条件前向 172.4 ms，peak allocated 32.0 MiB；
- 测试只记录 elapsed/peak，不设机器相关阈值。

## 9. DLL 修复（对设计的唯一实施期补充）

设计 §11.7 预设“只有实际复现缺 DLL，才能提出最小 Windows 修复”——
实施 B 期间**实际复现**：pytest/子进程以环境 python.exe 直跑时
`torch.cuda.is_available()` 为 True，首次 CUDA FFT 抛
`nvrtc: failed to open nvrtc-builtins64_124.dll`（`conda run` 会把
`<env>/bin` 放入搜索路径，直跑不会）。按设计授权加回旧版已验证的最小
修复：模块级 `os.add_dll_directory(<env>/bin)` + PATH 前置，句柄进程内
保留（`lithography/iccad13.py` 模块头，含 why 注释）。回归测试即
`test_direct_environment_python_loads_cuda_runtime`（子进程直跑 CUDA）。

## 10. 测试与 coverage

- lithography 专项 **81 passed**；全量 **224 passed**（基线 143 + 81）；
- coverage（`--source=lithography`）：`iccad13.py` 202/202 语句、
  `__init__.py` 2/2，**100%**，无未覆盖分支需要说明；
- 详见 `test_report.md`。

## 11. 简化审计

- 无未调用函数：全部公开方法（`condition/forward/forward_many/device`）
  与私有方法（`_prepare_mask/_kernel_multiply/_propagate/_load_kernel_bank`）
  均有测试与内部调用方；
- 无重复前向：forward 委托 forward_many，单一路径；
- 无 Protocol、注册器、抽象基类、插件系统；
- 无第二套 raster：模型不做任何栅格化；
- 无手写 backward / CT / combo / TorchLitho 代码；
- `__init__.py` 只导出三个真实公共类型；
- `00_PAST/`、`layout/`、`geometry/`、`opc/`、`config/macro_pipeline.toml`、
  `main/run_macro_pipeline.py`、用户工作树（`doc/opc/`、`.vscode/`）零改动。

## 12. 与设计文档的偏差清单

| 偏差 | 原因 |
|---|---|
| 增加 Windows DLL 注册（§11.7 默认不迁） | 实测复现 nvrtc DLL 缺失，属设计预设的授权路径（§9 本报告） |
| 基线数字 115 → 实际 143 | 设计自注“实施时先跑新鲜基线”；143 为 run_single_pass 之后的事实基线 |
| 补齐 §11.4 之外的 ProcessCondition 校验测试 | 旧版 `__post_init__` 行为保留（防止未知名静默落到 defocus 分支），需要注入测试覆盖 |

其余接口、公式、资产、测试矩阵、提交划分与设计文档一致。

## 13. 已知限制

承设计 §17：非完整物理参数模型（无 NA/波长/光源可调）；context 充分性
不由模型证明（35×35 是频域核尺寸）；固定 256 画布；不做 full-reticle
调度；forward 输出是连续胶图，二值化与 EPE/PVBand 属 evaluation/求解器。
