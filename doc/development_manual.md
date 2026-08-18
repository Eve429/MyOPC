# MyOPC 开发手册（迁移期）

面向项目所有者与后续开发的操作手册。规则以仓库根 `AGENTS.md` 为最高准绳。

## 1. 环境与门禁

- 解释器固定：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12，
  依赖见 `requirements.txt`：klayout / numpy / pillow / psutil / torch）。
- 门禁（范围必须显式，绝不 `ruff check .`）：

```bash
python -m pytest -q tests
python -m ruff check layout geometry opc lithography main tests
python -m compileall -q layout geometry opc lithography main tests
```

- Bash on Windows：路径用正斜杠。

## 2. 依赖方向

```text
common -> stdlib + numpy（runtime.py 例外需 torch；不依赖任何业务包）
layout -> geometry
geometry -> layout
opc.input -> geometry/layout + common（arrays）
opc.input.edge -> opc.input + common（arrays）
lithography -> torch + 标准库（不导入 common/layout/geometry/opc/evaluation/main）
evaluation -> torch + 标准库（不导入 common/layout/geometry/opc/lithography/main）
opc.iteration.mbopc -> opc.input(.edge) + lithography + evaluation + torch/numpy/kdb
main -> 上述全部 + common（io/units/runtime）
```

`common/`（2026-08-18 新建）集中跨模块辅助：`arrays`（as_vector/as_matrix/
as_points）、`io`（atomic_write_json/atomic_write_npz）、`units`（exact_dbu）、
`runtime`（resolve_device）。layout/geometry/lithography 不引入 common
（lithography 按用户裁决完全不动）。

基础层不得反向依赖；`opc.iteration.<method>` 可依赖输入层、`lithography` 与
`evaluation`（消费 `LithographyModel` 契约而非 ICCAD13 具体类型）；
`main/_macro_pipeline.py` 是全部真实流程共用的 macro 生命周期（problem
准备/候选写出/最终合并/最终光刻留档）；MB-OPC 两方法共用
`main/_mbopc_workflow.py` 的公共生命周期（配置加载→prepare→device/model/
cache→macro 循环→merge→留档→summary），算法差异以 `MBOPCMethod` 适配器
注入：`main/_simple_mbopc_workflow.py` 与 `main/_gradient_mbopc_workflow.py`
只保留各自的 solve/序列化/摘要钩子（2026-08-18 拆分防大 workflow 复燃，
同日上提公共层——注入式而非机械合并，新增方法只写一个 adapter 文件）。

## 3. Macro–Core 管线（直接运行，无需安装）

```bash
D:/app/miniforge/envs/myopc/python.exe main/run_macro_pipeline.py config/macro_pipeline.toml
```

流程（阶段 0–3）：

1. **阶段 0** `load_config` + `plan_macros`：严格 TOML（未知段/键拒绝，
   `macro_grid` 与 `macro_size_nm` 恰好一个）、`exact_dbu` Decimal 精确
   nm→DBU 换算、像素整除/画布容量/context≥位移校验、目标层 bbox 流式
   扫描（不物化）。
2. **阶段 1** `prepare_problems`：逐 macro 一次完成完整相交物化 →
   `normalize_mask` → `extract_contour` → `fragment_edges` →
   ownership 切线分裂 → owner/CSR → `MacroProblem.save`（NPZ）；全部成功
   后才写 `plan.json`。
3. **阶段 2** `run_round` ×2：每 macro 逐 core owner 唯一写入（重复写即
   失败）、written 恰一次守卫、`reconstruct_region`、逐 core 居中 256
   canvas + transmission sum、result NPZ + 完整候选 GDS。第一轮 +2 nm、
   第二轮 −2 nm 精确回零。
4. **阶段 3** `merge_final`：ownership 权威覆盖选择 →
   `PatchWriter.write_macro_results`（`single_cell` 全局 merge 无 seam /
   `macro_cells` 调试用）→ 面积不变回读验证 → `run` 做回零 XOR 守卫并写
   `summary.json`。

## 4. 关键接口速查

| 需求 | 位置 |
|---|---|
| 两级网格规划 | `opc.input.plan_macros` → `MacroSpec`（`core(i)` 即时构造 `CoreSpec`） |
| 单 macro 参考问题 | `opc.input.edge.prepare_macro_problem` / `MacroProblem.save/load` |
| 位移重建 | `reconstruct_contours/reconstruct_region(problem, displacements)` |
| 居中光刻画布 | `rasterize_mask_canvas` / `ownership_canvas` |
| DBU 点→画布坐标 | `opc.input.points_to_canvas`（含 padding 项，禁手写公式） |
| 最终双模式写出 | `geometry.PatchWriter.write_macro_results` |
| 统一配置体系 | `main.configuration`：9 个业务 Config + `load_config(path, *types)`（单次读、声明式映射、未知段/字段严格）；段：layout/partition/lithography/edge/mbopc/gradient/single_pass/iteration/output |
| 通用辅助函数 | `common.io`（atomic_write_json/npz）、`common.units`（exact_dbu）、`common.runtime`（resolve_device）、`common.arrays`（as_vector/as_matrix/as_points） |
| 跨段契约与派生值 | 各 workflow 装配处（步长/探针≤上限、lr 超限 warning；nm→DBU 内联换算） |

## 5. 光刻模型 lithography（ICCAD13，2026-08-16 迁移）

- 唯一具体模型 `ICCAD13Lithography(torch.nn.Module)`（`lithography/iccad13.py`）；
  `lithography/contracts.py` 的 `LithographyModel` 薄 Protocol 已随 simple
  MB-OPC 建立（首个求解器调用方），无注册器或抽象基类。
- 输入永远是**透光率 tensor**：`1.0=透光，0.0=不透光`，单张 `[H,W]` 或批量
  `[B,H,W]`，H/W ≤ 256；连续 0~1 值合法（模型不强制二值化）。
- 版图、极性、DBU、居中由 layout/geometry/opc.input 在模型之前完成；
  模型不导入这些模块。`opc.input.rasterize_mask_canvas()` 的 256 输出可
  直传（padding 契约逐位一致，模型不二次移动）。
- 坐标方向与输入一致（行 0 = 最低 Y，不翻转 Y）；输出与输入同 shape、
  范围 (0,1)。
- 工艺条件互相独立：`model.condition(name)` 返回 nominal→focus+1.00、
  dose_max→focus+1.02、defocus_min→defocus+0.98；自定义条件直接构造
  `ProcessCondition("focus_101", "focus", 1.01)`，同一次调用名称必须唯一。
- `forward_many(mask, conditions)`：一次 mask FFT + 每 bank 一次传播 +
  `dose²` 缩放 + `sigmoid(steepness×(I−target))`；`forward` 是单条件便捷入口。
- MB-OPC 推理用 `torch.no_grad()`；梯度 OPC/ILT 直接 `loss.backward()`——
  前向全原生可微算子（pad→fft2→乘→ifft2→|·|²→加权→dose²→sigmoid→crop），
  **无手写 backward**，有限差分已验证。
- kernel/scale（35×35×24 complex64 + 24 float32 ×2 bank）注册为 buffer，
  不是 parameter；`model.device` 报告 buffer 设备；`device=None/"auto"`
  = 有 CUDA 用 CUDA。`.to(device)` 会同时移动四个 buffer。
- batch size 由调用方按显存决定（模型不拆 batch）：单个复数场中间量约
  `B × 24 × 256² × complex64 ≈ B × 12 MiB`，反向峰值更高。
- **torch 安装**：默认 PyPI 为 CPU 构建；需 CUDA 时从 PyTorch 官方索引安装
  对应 cuXXX 轮子（实测 2.5.1+cu124 / GTX 1650）。
- **Windows 注意**：直接用环境 python.exe 启动（非 conda run）时，
  `lithography/iccad13.py` 在导入 torch 前把 `<env>/bin` 注册进 DLL 搜索
  目录（NVRTC JIT 运行时位于该目录）；该顺序不可调换。

## 6. 注释规则（2026-08-15 新增）

- `main/` 下文件**每一行**都要有中文短注释；
- 其他目录：文件级 docstring 一句话、函数 docstring 一句话、每个紧凑逻辑块
  前分段注释（解释 why：坐标方向、不变量、性能路径、内存上界、边界归属、
  异常原因）。

## 7. 最简 MB-OPC（opc/iteration/mbopc，2026-08-16 迁移）

固定步长、EPE 驱动的离散边移动求解器（设计文档
`archive/reports/mbopc_migration_design.md`，报告 `archive/reports/mbopc_{development,test}_report.md`；
2026-08-17 起单/多入口合并为 `run_mbopc.py`，不再强制 macro 数量约束）。

```bash
# 单入口，macro 数量由 config 网格决定（单/多通用）
D:/app/miniforge/envs/myopc/python.exe main/run_mbopc.py config/mbopc_multi_macro.toml
```

- **算法**：`evaluate_and_propose()` 评价一个状态（target/current/ownership
  三画布 → no_grad 三条件一次 forward_many → L2/PVBand 只在 ownership 像素
  → owner 探针 `edge_probe_points` + `points_to_canvas` 批量 EPE →
  next = current + {-1,0,+1}×step，批后释放张量再报进度；参考几何由
  `reference` 参数在整迭代内复用一份）；
  `optimize_macro()` baseline（records[0]）起每轮一次评价同时产生下轮提案
  （末轮纯评价不提案、无变化提案直接停止），步长按 `decay_every` 减半，
  EPE 严格更小才更新 best（平局保留早轮）。
- **停止状态**：`zero_epe`（无违规）/ `no_update`（提案与当前一致）/
  `invalid_geometry`（候选重建守卫拦截，含 KLayout ValueError 退化形态）/
  `insufficient_probes`（有 owner 段但有效探针为 0——"无法评价"不是
  "零违规"，保留 baseline）/ `iteration_limit`。
- **坐标契约**：探针 DBU→canvas 必须经 `opc.input.points_to_canvas`（含
  居中 padding 项）；不要手写 `(x-left)/pixel-0.5`。
- **独立 macro 语义**：macro 间不交换中间状态，边界 core 的 context 固定为
  邻区参考几何；全部 macro 完成后只调用一次 `merge_macro_results`
  （显式 macro_id→GDS 映射）。这不是全局同步最优，差异需量化（gcd_45nm：
  single 比 multi 之和小 236 段 EPE，覆盖 XOR 34650860 DBU²）。
- **内存**：target 用有界 uint8 LRU（`TargetCanvasCache`，key 含 macro id）；
  GPU 每 batch 只保留当前张量；不保存整张 reticle tensor；最终光刻 PNG 与
  merge 回读验证均为逐窗口物化（不常驻全量 Region；merge 的 patches 列表
  持有全部 clipped——PatchWriter 接口属 geometry/，为已知上界）。
- **产物**：`work_dir/macros/<id>/{result.npz,best.gds,metrics.json}` +
  `final.gds` + 可选 `final_lithography/`（逐 tile nominal/binary PNG +
  manifest）；`[mbopc]` 段 `show_progress` 控制 tqdm（自动测试一律 false）。

## 8. 梯度 MB-OPC（opc/iteration/mbopc/gradient.py，2026-08-17 迁移）

基于梯度的边段优化（设计 `changes/completed/CHG-20260816-gradient-mbopc/implementation_spec.md`，报告
`changes/completed/CHG-20260816-gradient-mbopc 两报告`）。

```bash
# 单入口、任意 macro 数（config 即 gcd_45nm 2×2 smoke）
D:/app/miniforge/envs/myopc/python.exe main/run_gradient_mbopc.py config/gradient_mbopc.toml
```

- **算法**：`optimize_gradient_macro()` —— KLayout 精确面积覆盖率 hard 前向 +
  DiffOPC Algorithm 4 midpoint STE 反向（`_EdgeGradientMask`：forward 逐位直通、
  backward 在段当前中点双线性采样 dL/dMask 后 ×2——标量位移同时驱动两端点，
  两端链式求和恰为 2·g_mid）；owner-only Adam（批 backward 累积梯度、全部 tile
  完成的屏障后恰一次 step，随后 clamp ±max_displacement）；候选经
  `reconstruct_region` 守卫通过才发布为下一状态。
- **loss**：三项连续（nominal/process/pvband，只在 ownership 像素累计、除以
  全 macro 计分像素总数 P、显式权重至少一正）；离散 L2/PVBand/EPE 只作同状态
  诊断，不参与训练与 best 选择。
- **状态语义**：records[0]=baseline；state N=第 N 次更新后已评价状态；末状态
  纯评价不建图；best 按已评价 total_loss 严格更小更新（平局保留较早状态）。
- **停止状态**：`zero_loss` / `no_update`（梯度全零步长为零）/ `invalid_geometry`
  （宽捕获 ReconstructionError+ValueError——KLayout 几何退化以 ValueError 冒出，
  simple 同款实测证据）/ `no_owned_segments`（空或纯 context macro，不建
  optimizer）/ `iteration_limit`。
- **学习率**：连续 optimizer 步长，Decimal nm 相除转 float DBU（不走 exact_dbu
  整数契约）；epe_distance 等其余 nm 参数仍走精确整数换算。
- **产物**：`work_dir/macros/<id>/{gradient_result.npz,best.gds,
  gradient_metrics.json}`（文件名独立于 simple）+ summary（含 RSS 三采样与
  CUDA 峰值）+ final.gds + 可选 final_lithography/。
- **simple 兼容**：`TargetCanvasCache` 移至 `_cache.py`（两方法共享），包级与
  simple 模块级导入路径不变。

## 9. 迁移状态

layout / geometry / opc.input(+edge) / lithography / evaluation / opc.iteration.mbopc
（simple + gradient）已完成；ilt 与 main 旧入口待评审。历史架构参照
`00_PAST/doc/`（只读归档）。
