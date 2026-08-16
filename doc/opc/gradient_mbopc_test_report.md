# 梯度 MB-OPC 迁移测试报告（CHG-20260816-gradient-mbopc）

日期：2026-08-17。环境：Windows、myopc conda env（Python 3.12、KLayout
0.30.10、NumPy 2.5.1、PyTorch 2.5.1+cu124、GTX 1650 4GB）。

## 命令与结果

```text
pytest -q tests/opc/iteration/test_gradient_mbopc.py      → 44 passed
pytest -q tests/main/test_gradient_mbopc_runner.py        → 25 passed
pytest -q tests/opc/iteration/test_simple_mbopc.py        → 54 passed（REQ-016 回归）
pytest -q tests（全量）                                    → 410 passed（78.95s）
ruff check layout opc lithography evaluation main tests   → 全绿
compileall layout geometry opc lithography evaluation main tests → 全绿
python main/run_gradient_mbopc.py config/gradient_mbopc.toml → 退出码 0（41.61s）
git diff --check                                          → 通过
```

全量 341 → **410 passed**（+69：solver 44 + runner 25）。

## TEST-001..016 覆盖对照

| 规格 TEST | 实现用例（摘要） | 结果 |
|---|---|---|
| TEST-001 forward 零差异 | test_forward_preserves_exact_raster（含 0/部分/1 值逐位直通） | 通过 |
| TEST-002 backward 公式 | test_backward_is_two_times_bilinear_midpoint（半像素/边界整点/越界/重复索引求和）+ 源码无 .item()/循环断言 | 通过 |
| TEST-003 方向有限差分 | test_surrogate_direction_matches_integer_geometry_difference（真实 ICCAD13，clear+opaque 参数化，每边最长段 ±1 DBU 中心差分同号，≥2 段验证） | 通过 |
| TEST-004 loss 公式 | test_continuous_losses_match_independent_recompute（独立 numpy 路径复算三分量与加权 total）+ test_halo_geometry_does_not_score（线性逐像素模型 halo 不计分） | 通过 |
| TEST-005 batch/屏障 | test_batch_size_preserves_gradient_and_published_state（批 1 vs 4，rtol=1e-5/atol=1e-7）+ test_first_optimizer_step_after_all_batch_backward（事件序 [f,f,s]×2+[f,f]） | 通过 |
| TEST-006 状态/best | test_records_and_best_use_same_evaluated_snapshots（三状态、best=1、快照一致）+ test_zero_loss_stops_immediately | 通过 |
| TEST-007 几何拒绝 | test_invalid_reconstruction_keeps_last_legal_best（共线退化真构造：印刷过量驱动内移 clamp 到恰 ±20 → ValueError）+ test_program_runtime_error_is_not_converted（RuntimeError 传播，裁决 1 连带调整）+ test_nonfinite_loss_raises_floating_point_error | 通过 |
| TEST-008 owner/空问题 | test_context_segments_have_no_parameter（SpyAdam numel==O）+ 空 macro / 纯 context macro 返回 baseline | 通过 |
| TEST-009 几何矩阵 | 矩形/2DBU 窄壁+8DBU 探针（部分探针不可用且不触发 zero_loss）/多 polygon+hole/凹形/45° 斜边/跨 core/跨 macro/opaque 八形态 | 通过 |
| TEST-010 调用计数 | forward 每批每 state 恰一次（6 次）、candidate 重建 1+iterations 次、target cache 命中免重栅格（16 vs 24 对照） | 通过 |
| TEST-011 真实集成 | CPU：有限 loss、非零更新、best 不劣于 baseline；CUDA：baseline loss rtol=2e-4 一致、更新段方向一致（无 CUDA 精确 skip） | 通过 |
| TEST-012 simple 兼容 | 全量 54 例 + 三条导入路径同一对象断言 | 通过 |
| TEST-013 配置/直跑 | 未知键/缺键/12 组非法值参数化/epe 非整数 DBU 运行期拒绝 + 仓库外子进程直跑退出 0 + 无参数用法退出 2 | 通过 |
| TEST-014 产物 | summary §8.2 键全集（含 RSS/CUDA）、NPZ 键/dtype、JSON records 字段、不覆盖 simple 文件名、final GDS 与光刻 PNG/manifest | 通过 |
| TEST-015 多 macro | 2×2 独立产物、merge 恰一次（monkeypatch 计数）、正逆序覆盖 XOR==0 | 通过 |
| TEST-016 进度/资源 | 更新总数==(iterations+1)×core、异常时进度条 close 且无半份 summary、资源字段断言 | 通过 |

## smoke 与先例对照

- gcd_45nm 2×2、iterations=1、CUDA：四 macro 全部 best_state=1
  （loss −10.1%）、stop=iteration_limit；数值见开发报告表格。
- state0 EPE（mr0c0 37743）与 simple 轮 baseline 逐位一致——评价路径零漂移。
- simple 真实 smoke 本轮未重跑：批 A 只改 cache 的 import 绑定（数值路径
  零变化），simple 全量 54 例 + 全量回归通过即满足 AC-006。

## 已知口径

- 完整 `ruff check .` 在 `geometry/contour.py` 有一个既存导入空行告警
  （用户 tmp 提交 78ec257 带入，上轮已记录）；专项范围排除该文件后全绿。
- 24 GiB GPU / 64 GiB RAM 未实测（本机 4GB）；不宣称整张 reticle 可跑。
- "跨 3 core"场景由"跨全部 4 core"用例覆盖（2×2 网格下无 3-core 切分）。
- no_update 平局保留语义：连续 loss 浮点下自然平局概率≈0，未构造专门
  平局用例；语义由严格小于比较符保证（simple 整数 EPE 有平局用例）。
