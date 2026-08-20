# 测试报告 — CHG-20260819-gradient-mbopc-epe-loss

## 环境

WSL `~/miniconda3/envs/myopc312/bin/python`（3.12.0；torch 2.13.0+cu130，
CUDA 可见，无 skip）。基线 08f4866 时全量 529 passed；本 change 交付时
全量 **545 passed**（阶段 A/B 新增 16 例：公式/坐标/STE 反传/owner/batch/
关闭兼容/校验/Q 不变/切段不变/EPE-only/真实 CPU·CUDA/forward 守卫等）。

## 命令与结果（§15.3）

```bash
python -m pytest -q tests/opc/iteration/test_gradient_mbopc.py \
                   tests/main/test_configuration.py \
                   tests/main/test_gradient_mbopc_runner.py   # 123 passed
python -m pytest -q tests                                    # 545 passed
python -m ruff check layout geometry opc lithography main tests   # 通过
python -m compileall -q layout geometry opc lithography main tests
python main/run_gradient_mbopc.py config/gradient_mbopc.toml      # smoke
git diff --check
```

定向 123 passed；全量 545 passed；ruff/compileall 无输出；smoke 见
development_report 对照表（EPE ON 183.2s / best_state=10）。

## 规格测试 → 实际用例映射

| 规格 | 用例 |
|---|---|
| TEST-001 | `test_epe_profile_formula_and_zero_baseline`（手算小 tensor + 全零误差 epe_loss==0） |
| TEST-002 | `test_epe_profile_coordinates_all_directions`（H/V/45°/hole/clear/opaque 统一公式） |
| TEST-003 | `test_epe_loss_backpropagates_through_midpoint_ste` |
| TEST-004 | `test_epe_owner_scores_once_membership_gradients_sum` |
| TEST-005 | `test_epe_batch_size_invariant`（L_sum 分母） |
| TEST-006 | `test_epe_disabled_is_exactly_compatible`（旧 TOML/显式 0/fixture 三路逐值） |
| TEST-007 | config 参数化 + `test_epe_training_validation_fails_before_device_allocation` |
| TEST-008 | `test_epe_record_summary_and_npz_contract` + runner additive schema |
| TEST-009 | `test_epe_only_update_improves_evaluated_loss` |
| TEST-010 | 真实 ICCAD13 CPU 有限反传 + CUDA parity（1e-4 容差） |
| TEST-011 | forward_many 计数 spy + profile 单次构造 + graph 释放 |
| TEST-012 | 本报告 + development_report 对照 smoke（连续/离散/资源全记录） |
| TEST-013 | `test_epe_profile_width_invariant_d_s`（R=2/4/8，断 pre-sigmoid d_s；mean 版判别） |
| TEST-014 | `test_epe_loss_invariant_to_segmentation`（1/2/非等长切段；等权版判别） |

## 已知口径

- TEST-013 的 stub 误差支撑区落在最小 R 的 profile 内（理想阶跃偏移 1px），
  断言对象为 pre-sigmoid `d_s`；真实 ICCAD13 过渡区数 pixel，Q 不变性只在
  R 覆盖支撑区时近似成立（规格已声明，非缺陷）。
- 对照 smoke 的 OFF 组与 ON 组 state0 逐值相同是兼容性证据；逐 state 的
  关闭兼容由 TEST-006 在测试层锁定（smoke 不重复断言）。
