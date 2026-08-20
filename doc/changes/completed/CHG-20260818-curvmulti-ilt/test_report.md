# CHG-20260818-curvmulti-ilt 测试报告

## 1. 用例清单（43 = 33 求解器 + 10 runner）

`tests/opc/iteration/test_curvmulti_ilt.py`（33）：

- TestHelpers（3）：resize nearest=块复制 / area=块均值（numpy 参照
  逐位）、smooth_sigmoid_mask 手算逐位（零补边池化 + σ(β(x−offset))）。
- TestConfigValidation（16）：scales 空/非递减/未以 1 结尾/list/bool、
  iterations/batch 非正、kernel 偶/零、step/陡度/offset/权重/阈值越界、
  NaN；合法 (4,2,1)+kernel7 通过。
- TestSolverSemantics（9）：scales=[1] 单尺度退化（records 全 stage 0）、
  records stage 坐标逐条（(0,0,0,2)(1,0,1,2)(2,1,0,1)(3,1,1,1)）、
  float64 镜像逐 state（单 core 整链+SGD 更新，rel 1e-5）、batch 1↔4
  不变、每 stage SGD 独立实例且控制张量不共享、warm-start 调用解剖
  （area 恰 1 次、粗 nearest 计数 7 = 批上采样 4+warm 1+物化 2、
  原始控制网格输入值 == area 参考）、常数模型全平局 best=0、曲率
  =curvature(printed nominal) 且 ≠ curvature(mask)（低通模型判别）、
  curvature=0 不构建卷积、入口四类拒绝（整除/最粗 kernel/画布不一致/
  曲率 context<pixel）。
- TestContextHelpers（2）：终评 helper 与 Simple 逐位一致（同 β 全 core）、
  state0 三值语义逐槽位（padding 0 / context σ(β(2T−1)) / trainable
  =平滑 sigmoid 上采样，手算对照）。
- TestRealModel（3）：真 ICCAD13 CPU 两尺度有限性与物化契约、CUDA
  parity（skipif 无 CUDA；本机 GTX 1650 实跑通过）。

`tests/main/test_curvmulti_ilt_runner.py`（10）：合法完成（method/
macro 数/每宏 4 态）、缺必填键先于 prepare、未知键、未知段、scales
浮点拒绝、scale 不整除 ownership 拒绝、仓库外 cwd subprocess 直跑、
result NPZ schema（format_version/float32 形状 10×20/uint8 binary/
best_state_index 一致）+ metrics stage 坐标、merge 恰一次（spy 重跑）、
evaluated_states=Σ(N+1)=9。

## 2. 门禁

| 批次 | compileall | ruff | pytest | git diff --check |
|---|---|---|---|---|
| A（`721be5a`） | ✓ | ✓ | 650 passed + 1 skipped | ✓ |
| B（`8eebc3c`） | ✓ | ✓ | 660 passed + 1 skipped | ✓ |
| C（本提交） | ✓ | ✓ | 660 passed + 1 skipped | ✓ |

## 3. 测试侧事实

- 复用 test_simple_ilt 基建（跨文件导入先例：test_levelset_ilt）：
  `_problem`/`_macro`/`_DoseModel`/`_IdentityModel`/`_ConstantModel`/
  `_StubConfig`；新增 `_LowpassCaptureModel`（5×5 低通 + 捕获，用于
  wafer-vs-mask 曲率判别）。
- runner 几何教训：三角形顶点曾把 bbox 顶到 y=96，底行 macro 仅 2px
  触发"最粗网格 < kernel"——几何设计需复核每个 macro 的 ownership 像素
  高宽对全部 scale 的整除性，不能只看整幅 bbox。
- warm-start 测试为白盒（resize 调用计数 + 输入值解剖），与项目
  monkeypatch 计数先例一致；若 solver 重构 resize 调用次序需同步维护。
