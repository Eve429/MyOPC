# CHG-20260823-torchlitho：TorchLitho-2.0 光刻模型迁移（Abbe + Hopkins）

状态：completed（2026-08-23）。实施四批：A 模型核心（6e1f779）→ B golden 多
图案一致性（82f5789）→ C 配置分派与接线（9065555）→ D 文档与报告（本批）。

## 1. 目标与用户裁决

把外部库 TorchLitho-2.0（`D:\00_WorkSpace\02_CodeStorage\01_OPC\TorchLitho-2.0`，
Apache 2.0）的 Abbe 与 Hopkins 两光刻方法迁入 `lithography/torchlitho/`，满足
现有 `LithographyModel` 协议（求解器零改动）；在 `doc/algorithms/` 详解两算法；
**交付多图案一致性测试报告证明迁移前后结果一致**。

用户三项裁决（会话记录）：

1. Hopkins 走 **Option 2**：忠实迁移 genTCC + randomized SVD + 源形状参数化
   （point/disk/dipole/quadrupole），为多极照明预留真实语义的 SVD 机器；
2. **R2 修正**：原库 Abbe 源点为范数标量（瞳同心放大而非平移，非点源物理
   错误且无法表达多极），迁移改为向量源点；一致性证明分层（点源逐位锚点
   不受影响，盘源差异量化报告）；
3. 测试报告为点名交付物，多图案覆盖。

## 2. 接口设计

- `TorchLithoConfig`（`[torchlitho]` 段，全默认）：method / source_shape /
  sigma / pole_center / wavelength_nm / na / refractive_index / defocus_min_nm /
  dose 三值 / 胶模型三参数（默认对齐 iccad13.txt）。
- `TorchLithoCondition(name, defocus_nm, dose)`：条件令牌。求解器全程把
  conditions 当不透明令牌（`model.condition(name)` 产生后原样传回
  `forward_many`，零字段访问——已 grep 核实），故自有条件类型与
  `ProcessCondition` 鸭子类型并存，`contracts.py` 不动。
- `TorchLithoLithography(config, canvas, pixel_nm, device)`：满足协议；
  居中 padding 复制 ICCAD13 约定（差值均分、奇数余量归高侧，测试锁定逐位
  一致）；出口 `sigmoid(steepness·(I·dose² − target))` 对齐 ICCAD13 语义；
  同 defocus 共享一次成像；全程保留 autograd 图。

## 3. 现有代码改动（获批清单，逐项实施）

| 文件 | 改动 | commit |
|---|---|---|
| `main/configuration.py` | LithographyConfig +model 可选键；TorchLithoConfig 注册 [torchlitho]；build_lithography_model 工厂 | 9065555 |
| `main/_mbopc_workflow.py` / `main/_ilt_workflow.py` | load_config 加 TorchLithoConfig；实例化改工厂 | 9065555 |
| `main/main_test_lithography.py` | +--model flag（默认 iccad13） | 9065555 |
| `lithography/__init__.py` | 导出 3 公共名 | 9065555 |
| 其余（contracts.py / iccad13.py / _macro_pipeline.py / 现有 8 个 config） | **零改动** | — |

## 4. 不迁移清单（理由）

Simulator ABC（接口体系与协议冲突）；interpolate ×pixel 上采样（MyOPC 输出与
输入同形）；AbbeFunc/AbbeGradient 手写 vjp（前向全原生可微算子，autograd
覆盖；原 vjp 无标准 backward 且只以全一上游梯度展示——00_PAST/findings
§258 旧审计结论延续）；glp.py/imageTools/example/（MyOPC 有自己的 mask
管线）；cv2 依赖（torch F.interpolate 同构替换，实测差 float32 ulp 级）；
任意旋转角/annular 源（后续增量）。

## 5. 验收标准

1. 门禁四件套全绿（本批范围）；基线 695+1 只增（终态 786+10 并行现场+1）；
2. 点源 8 图案 × 2 方法 × 2 离焦对原库 golden 一致（abbe 逐位、hopkins
   float32 包络）；
3. disk 源 hopkins 与原库 TCC 数值链逐位；
4. resize 两分支（2048nm 纯零嵌入 / 4096nm 真插值）对原库（含 cv2 路径）一致；
5. `run_ilt_simple.py config/torchlitho_abbe.toml` CPU/CUDA 冒烟贯通。

结果全部满足，证据见 test_report.md。
