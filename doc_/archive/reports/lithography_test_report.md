# lithography 迁移测试报告（ICCAD13）

> 实施日期：2026-08-16。测试文件：`tests/lithography/test_iccad13.py`
> （单文件按类组织）。

## 1. 命令与结果

```bash
D:/app/miniforge/envs/myopc/python.exe -m pytest -q tests              # 224 passed
D:/app/miniforge/envs/myopc/python.exe -m pytest -q tests/lithography  # 81 passed
D:/app/miniforge/envs/myopc/python.exe -m ruff check lithography tests/lithography main/main_test_lithography.py   # All checks passed
D:/app/miniforge/envs/myopc/python.exe -m compileall -q layout geometry opc lithography main tests               # 通过
D:/app/miniforge/envs/myopc/python.exe main/main_test_lithography.py   # 退出码 0
D:/app/miniforge/envs/myopc/python.exe -m coverage run --source=lithography -m pytest -q tests/lithography
D:/app/miniforge/envs/myopc/python.exe -m coverage report -m
```

基线：实施前全量 143 passed（设计文档写 115，系文档编写时点早于
run_single_pass 批次；按设计要求以新鲜基线为准）。

## 2. coverage

```text
lithography\__init__.py       2      0   100%
lithography\iccad13.py      202      0   100%
TOTAL                       204      0   100%
```

无未覆盖分支。防御分支（配置行格式、数值转换失败、资产不足、scale
缺失/非一维、类型拦截、auto 设备）全部有注入式用例覆盖，未使用
“报告说明豁免”路径。

## 3. 用例矩阵（设计 §11 对照）

| 设计条目 | 用例（类·方法） | 结果 |
|---|---|---|
| §11.1 配置 1–8 | TestConfigParsing（标准解析/缺失/未知/重复/核数区间/冻结 256/非有限/契约组；含行格式与数值转换补充） | ✅ |
| §11.2 资产 1–7 | TestAssets（四哈希/布局与 dtype/buffer 非 parameter/缺资产/坏布局/数量不符/.to 一致；含核数不足、scale 缺失/非一维补充） | ✅ |
| §11.3 形状 1–9 | TestShapeAndPadding（单张/批量/padding 布局/奇数高侧/满 canvas/超限/四维/raster 直传/方向一致） | ✅ |
| §11.4 数值 | TestCpuNumerics（三 sums 基线/batch 一致/连续值/自定义条件/重名/空条件；含类型拦截补充） | ✅ |
| §11.5 性能 | TestSharedComputation（fft2==1、传播 [focus,defocus] 各一、共享 vs 独立一致） | ✅ |
| §11.6 backward 1–7 | TestBackward（单条件/联合/batch/有限差分/buffer 无 grad/no_grad 无图） | ✅ |
| §11.7 CUDA 1–5 | TestCuda（parity 1e-4、forward+backward、子进程直跑、peak 记录） | ✅ GTX 1650 |
| §11.8 main 1–5 | TestMainEntry（仓库内/仓库外直跑、关键输出逐项、git status 零产物） | ✅ |

## 4. 关键数值记录

**CPU 基线（逐位复现，验收容差 0.05 未用尽）**：

```text
mask = zeros[2,200,150]；[0] 矩形 [40:160,40:110]=1；
[1] 外框 [20:180,20:130]=1 减孔 [70:130,60:90]=0
nominal sum      = 25802.533203125（期望 25802.533203125，差 0.0）
dose_max sum     = 26009.16796875（期望同，差 0.0）
defocus_min sum  = 25675.23828125（期望同，差 0.0）
```

**有限差分**（seed 20260811、[20,18]、权重 linspace(−0.7,1.3)、点 (9,8)、
ε=1e-3）：autograd 与中心差分 rtol=2e-2 / atol=2e-2 通过。

**CUDA**（torch 2.5.1+cu124 / GTX 1650）：CPU/GPU 三条件 rtol=atol=1e-4
通过；main 入口三条件前向 172.4 ms、peak allocated 32.0 MiB；
子进程直跑 `{'shape': [1, 64, 64], 'device': 'cuda:0'}`（依赖模块内
DLL 注册，见开发报告 §9）。

**main 入口 raster 画布**：context 1824/pixel 8/canvas 256 → 228×228
居中 + 14px padding；canvas sum=19375.0 与几何面积
(1200×1100−400×200)/64 精确相等。

## 5. 测试环境

- Python 3.12（myopc conda env，`D:/app/miniforge/envs/myopc/python.exe`）；
- klayout 0.30.10 / numpy 2.5.1 / pillow 12.3.0 / psutil 7.2.2 /
  torch 2.5.1+cu124；
- CUDA 子进程与 main 直跑测试均用 `sys.executable` 直接启动，不依赖
  `conda run` 与项目安装。
