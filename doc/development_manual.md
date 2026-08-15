# MyOPC 开发手册（迁移期）

面向项目所有者与后续开发的操作手册。规则以仓库根 `AGENTS.md` 为最高准绳。

## 1. 环境与门禁

- 解释器固定：`D:/app/miniforge/envs/myopc/python.exe`（Python ≥ 3.12，
  依赖 klayout / numpy / pillow / psutil）。
- 门禁（范围必须显式，绝不 `ruff check .`）：

```bash
python -m pytest -q tests
python -m ruff check layout geometry opc main tests
python -m compileall -q layout geometry opc main tests
```

- Bash on Windows：路径用正斜杠。

## 2. 依赖方向

```text
layout -> geometry -> opc.input -> opc.input.edge -> main
```

基础层不得反向依赖；`opc.iteration.<method>`（未迁移）将来可依赖输入层、
`lithography` 与 `evaluation`。

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

## 5. 注释规则（2026-08-15 新增）

- `main/` 下文件**每一行**都要有中文短注释；
- 其他目录：文件级 docstring 一句话、函数 docstring 一句话、每个紧凑逻辑块
  前分段注释（解释 why：坐标方向、不变量、性能路径、内存上界、边界归属、
  异常原因）。

## 6. 迁移状态

layout / geometry / opc.input(+edge) 已完成；lithography、evaluation、
opc.iteration、main 旧入口待迁移。历史架构参照 `00_PAST/doc/`（只读归档）。
