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
layout -> geometry -> opc.input -> opc.input.edge -> main
lithography -> torch + 标准库（不导入 layout/geometry/opc/evaluation/main）
evaluation -> torch + 标准库（不导入 layout/geometry/opc/lithography/main）
opc.iteration.mbopc -> opc.input(.edge) + lithography + evaluation + torch/numpy/kdb
main -> 上述全部（应用编排）
```

基础层不得反向依赖；`opc.iteration.<method>` 可依赖输入层、`lithography` 与
`evaluation`（消费 `LithographyModel` 契约而非 ICCAD13 具体类型）；
`main/_macro_pipeline.py` 是两个真实流程（验证管线与 MB-OPC）共用的 macro
生命周期，`main/_mbopc_workflow.py` 是 MB-OPC 两入口的共享工作流。

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
| 居中光刻画布 | `rasterize_mask_canvas` / `ownership_canvas`（映射公式见其注释） |
| 最终双模式写出 | `geometry.PatchWriter.write_macro_results` |
| 配置结构 | `main.run_macro_pipeline.PipelineConfig` / `load_config` |

## 5. 光刻模型 lithography（ICCAD13，2026-08-16 迁移）

- 唯一具体模型 `ICCAD13Lithography(torch.nn.Module)`（`lithography/iccad13.py`）；
  不建 Protocol、注册器或抽象基类（等第二个真实模型再抽契约）。
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
`doc/opc/mbopc_migration_design.md`，报告 `doc/opc/mbopc_{development,test}_report.md`）。

```bash
# 单 macro（全 ROI 一个 macro、内部多 tile）/ 多 macro（2×2，每 macro 多 tile）
D:/app/miniforge/envs/myopc/python.exe main/run_mbopc_single_macro.py config/mbopc_single_macro.toml
D:/app/miniforge/envs/myopc/python.exe main/run_mbopc_multi_macro.py config/mbopc_multi_macro.toml
```

- **算法**：`evaluate_and_propose()` 评价一个状态（target/current/ownership
  三画布 → no_grad 三条件一次 forward_many → L2/PVBand 只在 ownership 像素
  → owner 探针 `edge_probe_points` + `points_to_canvas` 批量 EPE →
  next = current + {-1,0,+1}×step，批后释放张量再报进度）；
  `optimize_macro()` baseline（records[0]）起每轮一次评价同时产生下轮提案，
  步长按 `decay_every` 减半，EPE 严格更小才更新 best（平局保留早轮）。
- **坐标契约**：探针 DBU→canvas 必须经 `opc.input.points_to_canvas`（含
  居中 padding 项）；不要手写 `(x-left)/pixel-0.5`。
- **独立 macro 语义**：macro 间不交换中间状态，边界 core 的 context 固定为
  邻区参考几何；全部 macro 完成后只调用一次 `merge_macro_results`
  （显式 macro_id→GDS 映射）。这不是全局同步最优，差异需量化（gcd_45nm：
  single 比 multi 之和小 236 段 EPE，覆盖 XOR 34650860 DBU²）。
- **内存**：target 用有界 uint8 LRU（`TargetCanvasCache`，key 含 macro id）；
  GPU 每 batch 只保留当前张量；不保存整张 reticle tensor。
- **产物**：`work_dir/macros/<id>/{result.npz,best.gds,metrics.json}` +
  `final.gds` + 可选 `final_lithography/`（逐 tile nominal/binary PNG +
  manifest）；`[mbopc]` 段 `show_progress` 控制 tqdm（自动测试一律 false）。

## 8. 迁移状态

layout / geometry / opc.input(+edge) / lithography / evaluation / opc.iteration.mbopc
已完成；diffopc、ilt 与 main 旧入口待迁移。历史架构参照 `00_PAST/doc/`（只读归档）。
