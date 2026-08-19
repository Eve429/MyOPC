# 项目工具错误记录

## [ERR-20260819-001] pytest

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary

WSL shell 的 PATH 中没有 pytest，直接执行 `pytest -q` 失败。

### Error

```text
/bin/bash: line 1: pytest: command not found
/bin/bash: /mnt/d/app/miniforge/envs/myopc/python.exe: cannot execute binary file: Exec format error
```

### Context

- 在 `/home/wzh/workspace/MyOPC` 复核全量测试时直接调用 `pytest -q`。
- 项目手册声明测试解释器为 `D:/app/miniforge/envs/myopc/python.exe`；其 WSL
  路径 `/mnt/d/app/miniforge/envs/myopc/python.exe` 已确认存在。

### Suggested Fix

在当前 Linux 工具会话中使用仓库现有环境
`/home/wzh/miniconda3/envs/myopc312/bin/python -m pytest`；宿主机 Windows
终端仍按项目手册使用 `D:/app/miniforge/envs/myopc/python.exe`。

### Metadata

- Reproducible: yes
- Related Files: doc/development_manual.md, doc/test_manual.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 找到 Linux conda 环境 `myopc312`；全量测试已通过（450 passed，
  8 CUDA tests skipped）。

---

## [ERR-20260819-002] apply_patch

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

更新测试记录时，补丁中的预期英文文本与文件内中文原文不匹配。

### Error

```text
apply_patch verification failed: Failed to find expected lines
```

### Context

- 测试已成功；失败仅发生在工作记录更新。

### Suggested Fix

先读取目标段落，再用实际原文生成最小补丁。

### Metadata

- Reproducible: yes
- Related Files: .learnings/ERRORS.md, findings.md, progress.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 已读取实际内容并精确更新。

---

## [ERR-20260819-003] apply_patch

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

终审补丁误把终端相邻输出看成规格中的重复测试矩阵行，导致整块上下文校验失败。

### Error

```text
apply_patch verification failed: Failed to find expected lines
```

### Context

- 目标是收紧 Simple ILT 的最终 state 语义并删除疑似重复矩阵行。
- 精确读取后确认规格只有一条 Boundary 矩阵行，补丁失败前未应用任何修改。

### Suggested Fix

先用 `rg -n -C` 核对每个目标片段，再只包含真实存在上下文的最小补丁。

### Metadata

- Reproducible: yes
- Related Files: doc/changes/active/CHG-20260818-simple-ilt/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 已确认无重复行，后续补丁只修改最终 state 与 transmission 归一化语义。

---

## [ERR-20260819-004] git-network

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: docs

### Summary

工作区受限网络无法只读访问 NVlabs/DiffOPC 官方仓库。

### Error

```text
fatal: unable to access 'https://github.com/NVlabs/DiffOPC.git/': Couldn't connect to server
```

### Context

- 为 Gradient MB-OPC EPE loss 设计核对官方 DiffOPC 源码。
- Web 搜索已确认官方仓库和论文，但页面提取不足以获得具体 loss 实现。

### Suggested Fix

经用户批准后把官方仓库浅克隆到 `/tmp`，只读分析，不写入项目。

### Metadata

- Reproducible: yes
- Related Files: doc/changes/completed/CHG-20260816-gradient-mbopc/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 获得网络权限后已浅克隆官方仓库到 `/tmp`，只读核对完成。

---

## [ERR-20260819-005] pdftotext

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

环境缺少 Poppler `pdftotext`，官方 DiffOPC 论文文本转换未执行。

### Error

```text
/bin/bash: line 2: pdftotext: command not found
rg: /tmp/C240-ICCAD2024-DiffOPC.txt: No such file or directory
```

### Context

- PDF 已从作者主页成功下载到 `/tmp/C240-ICCAD2024-DiffOPC.pdf`。
- 失败仅是系统转换工具缺失，下载文件仍可由 Python PDF 库读取。

### Suggested Fix

使用项目 Linux 环境内现有 `pdfplumber` 或 `pypdf` 提取相关页；如需布局截图再检查
`pdftoppm`/PyMuPDF 可用性，不为本次只读设计安装系统包。

### Metadata

- Reproducible: yes
- Related Files: /tmp/C240-ICCAD2024-DiffOPC.pdf

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 在 `/tmp` 独立安装 pypdf 6.16.1，成功提取论文相关公式；未修改项目依赖。

---

## [ERR-20260819-006] apply_patch

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

Gradient EPE 规格终审补丁的一行表格上下文与实际 Markdown 不一致。

### Error

```text
apply_patch verification failed: Failed to find expected lines
```

### Context

- 目标是补充四权重校验、边界 pixel center 采样与 summary 字段。
- 整组补丁校验失败，未产生部分修改。

### Suggested Fix

先读取目标区间，按实际表格中未加反引号的 symbol 文本生成精确补丁。

### Metadata

- Reproducible: yes
- Related Files: doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 已读取实际原文并完成最小补丁。

---

## [ERR-20260819-007] apply_patch

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

为公开 state record 增加默认值的补丁包含了一行并不存在的列表文本。

### Error

```text
apply_patch verification failed: Failed to find expected lines
```

### Context

- 目标是保持 `GradientMBOPCIterationRecord` 旧构造方式兼容并明确 summary key。
- 失败源于把表格行误写成列表行；补丁未部分应用。

### Suggested Fix

只包含已通过 `sed` 读取确认的精确段落，不附带无关上下文。

### Metadata

- Reproducible: yes
- Related Files: doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 已拆成精确补丁并成功应用。

---

## [ERR-20260819-008] documentation-path

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: docs

### Summary

新规格的报告计划沿用了不存在的 `doc/opc/` 路径。

### Error

```text
rg: doc/opc: IO error for operation on doc/opc: No such file or directory
```

### Context

- 当前报告归档在 `doc/archive/reports/`，新 change 体系的完成报告应随 change 目录保存。
- 旧归档报告不应为本增量功能直接改写。

### Suggested Fix

完成时把 active change 移入同 ID 的 completed 目录，并在其中新增
`development_report.md` 与 `test_report.md`。

### Metadata

- Reproducible: yes
- Related Files: doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: File-Level Change Plan 已改为真实 completed change 报告路径。

---

## [ERR-20260819-009] evidence-symbol

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary

规格证据表最初写入了不存在的配置解析 helper 名称 `_build_dataclass`。

### Error

```text
rg 未找到 _build_dataclass；实际符号为 main.configuration::_parse_config
```

### Context

- 行为判断正确：parser 确实使用 dataclass default。
- 错误只在证据符号名称，生产代码未改。

### Suggested Fix

规格中的稳定符号必须用 `rg` 对当前 baseline 逐个确认。

### Metadata

- Reproducible: yes
- Related Files: main/configuration.py, doc/changes/active/CHG-20260819-gradient-mbopc-epe-loss/implementation_spec.md

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: 已更正为 `_parse_config` 并核对其他核心符号。

---
