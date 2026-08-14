# 当前规则符合性修正测试报告

## 1. 环境

- 日期：2026-08-14
- Python：`D:\app\miniforge\envs\myopc\python.exe`
- GPU：NVIDIA GeForce GTX 1650
- 基线：250 项全仓测试通过
- 修改后：255 项全仓测试通过

## 2. 新增回归

| 回归 | 覆盖内容 | 结果 |
|---|---|---|
| `test_epe_probe_coordinates_match_raster_pixel_centers` | MB-OPC DBU probe 的 `-0.5` 像素中心映射 | 新公式 10 个有效探针；旧公式 0 个 |
| `test_soft_raster_hot_path_does_not_synchronize_tensor_values` | 用 monkeypatch 禁止 `Tensor.item()`，防止软栅格热路径恢复同步 | 通过 |
| `test_fragmentation_conversion_keeps_continuous_displacement` | 整数分段长度与小数 DBU 位移共存 | 通过 |
| 两组非格点参数化用例 | corner/segment 非整数 DBU 必须拒绝 | 通过 |

Macro 既有矩阵继续覆盖跨边界矩形、斜边、孔洞、窄环、重叠图形、层级 occurrence、owner 和 membership；删除重复 `__post_init__` 后结果不变。

## 3. 执行结果

| 门禁 | 命令/范围 | 结果 |
|---|---|---|
| 定向回归 | configuration、DiffOPC、simple MB-OPC、Macro、offline workbench | 70 passed，28.68 s |
| 全仓测试 | `python -m pytest -q` | 255 passed，74.70 s |
| 基础层覆盖率 | `--cov=layout --cov=geometry --cov=opc --cov=lithography --cov=evaluation --cov-branch` | 255 passed，83.57 s，总覆盖率 90% |
| Ruff | `python -m ruff check .` | 通过 |
| 编译 | `python -m compileall -q layout geometry opc lithography evaluation main tests` | 通过 |
| CUDA 同进程对照 | 256² 空 tile，200 次，预生成像素中心 | 1.844 ms → 0.288 ms，约 6.4× |

静态审计扫描 104 个第一方 Python 文件：中文模块/函数/测试 docstring 缺失 0、完全重复函数体 0、私有函数单引用候选 0。58 份 Markdown 的奇数代码围栏和缺失本地链接均为 0。生产代码中的 `OwnershipError`、DiffOPC rasterizer 的 `.item()`、Macro 错误全局 segment 描述和 runner 重复注释调用点均为 0；`layout/`、`geometry/` 相对核心提交零差异，`git diff --check` 通过。

工作树中的 `.vscode/launch.json` 与 `config/mbopc.toml` 是用户既有修改，内容保持原样且未进入两个功能提交。

## 4. 已知边界

性能数字只测量软栅格入口的同步校验开销，不等价于完整光刻或整轮 DiffOPC 的加速比。软栅格器现在信任 `MBOPCProblem` 和求解器内部状态提供有限、正长度、单位法向数据；直接绕过输入层传入非法数值不属于当前公共调用契约。
