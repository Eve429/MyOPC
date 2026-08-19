# 开发报告 — CHG-20260818-simple-ilt

## 实施概览

五阶段本地 commit（基线 540a012）：

| 阶段 | Commit | 内容 |
|---|---|---|
| 0（规格） | 5ad8ac0 | Rev 0.3：§14 补录 `main/run_single_pass.py`（事实核对发现的清单遗漏）；status → approved |
| A | 54ab866 | GridRuntime/resolve_grid_config 抽出；write_macro_gds 显式 LayerSpec；四消费文件迁移；单遍 smoke XOR==0 零回归 |
| B | 1539b6f | `opc/input/pixel`：PixelMacroProblem/NPZ/三种 core 画布映射/游程回写（20 测试） |
| C | fefaea8 | `opc/iteration/ilt`：_common（record/result/loss/曲率）+ simple 优化器（30 测试） |
| D | bdf86ac | get_type_hints 解析 + [simple_ilt] 注册 + `_ilt_workflow`/适配器/入口/smoke 配置（12 测试） |

## 与规格的偏差与裁决记录

1. **merge_macro_results 空 macro 候选容忍（规格未预见的必要修复）**：无材料
   区域（如稀疏版图角落）的宏 binary 全空 → best.gds 不含目标层（GDS 无空层
   表示）→ 原实现在 merge 回读抛 LayerNotFoundError。修复为两处回读
   （候选 + 最终验证）把层缺失按零覆盖处理；端到端测试（mr1c1 全空宏）为
   其回归。这是共享层（edge 管线同路径）的行为扩展，语义上"空候选 = 合法
   零覆盖"。
2. **smoke 配置 pixel 选择**：corners_unit_clear bbox 1900×1900 非 pixel 8
   整数倍（%8=4）而是 pixel 4 整数倍；smoke 采用 pixel_nm=4 + core 512 +
   context 256（恰满 256px 画布）。整像素契约因此对 smoke 成立。
3. **ILTStateRecord 的 stage_index/stage_state_index/scale 三字段保留**
   （用户裁定）：持久化 metrics 格式先行冻结，后三方法复用免升版。
4. **ilt_plan.json 键集**：规格 §8.2 只写概述；实施显式落
   merge_macro_results/save_final_lithography 消费的精确键（layer/dbu_um/
   macros[].ownership_box/polarity/pixel_dbu/canvas_pixels/core_size_dbu/
   context_dbu），"复用现有 merge"的必然推论。
5. **观察性质（非缺陷）**：logit 覆盖保持初始化下，严格 0/1 像素的 sigmoid
   斜率约 β·eps（≈5e-7，饱和），优化梯度经分数覆盖格（几何边界）进入并向
   内传播——测试几何须含非整像素边界；纯对齐几何的一轮更新在 float64
   记录精度下不可见（TEST-009 用例说明已记）。若后续方法需要全域活跃梯度，
   须另立 change 评估初始化策略（OpenILT ±1 初始化不满足 1e-6 恢复契约）。
6. **曲率 context=0 边缘**：valid 卷积使 canvas 边缘一圈 ownership 像素
   不计曲率；实践 context ≥ 64 像素无影响，不加分支（按规格 10.4）。

## 性能基线（PERF-006，只记录不设阈值）

corners_unit_clear 单宏 16 core / 225,625 宏像素 / iterations=1：
device cuda:0，总 1.90s，峰值 RSS 938 MiB，CUDA 峰值 503 MiB；
best_state=1（一轮更新有效），binary L2=2896，最终光刻 16 tile PNG。

## 清理与审计

- 无未调用函数/孤儿导入（ruff F401/F841 全绿）；无重复原子写/栅格化
  （atomic_write_* 与 rasterize_region_window 均复用）。
- `ILTMethod` 仅一个当前调用方（simple 适配器），无基类/注册器/空模块。
- 测试辅助 `_TARGET_LAYER` 未用常量已清理；pytest 弃用警告（类内 fixture）清零。
- 未修改 layout/geometry/00_PAST/用户 GDS；`git diff --check` 干净。
- 环境注记：全程在 WSL `~/miniconda3/envs/myopc312` 自跑（CUDA 可见，
  TEST-009 parity 实测）；push 因本会话网络/凭据不可用延后，本地提交完整。
