# 项目功能请求记录

## [FEAT-20260819-001] gradient-mbopc-epe-loss

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: high
**Status**: design_complete
**Area**: backend

### Requested Capability

参考 DiffOPC，为现有 gradient MB-OPC 增加可微 EPE loss，并先形成可评审的更新设计。

### User Context

当前 gradient MB-OPC 只有 nominal L2、process L2 与 PVBand 连续训练项；EPE 仅作为
离散诊断指标。用户希望引入更直接约束轮廓误差的训练目标。

### Complexity Estimate

complex

### Suggested Implementation

先核对 DiffOPC 的 EPE 采样、阈值、符号和归一化，再映射到当前 owner 参数、实际重构
midpoint、跨 core membership 梯度与 macro 同步状态，避免复制独立参数或重复计 loss。

### Resolution

- **Designed**: 2026-08-19T00:00:00+08:00
- **Notes**: 独立 draft implementation spec 已完成；实现等待用户明确批准 approval gate。

### Metadata

- Frequency: first_time
- Related Features: gradient_mbopc, evaluate_epe, midpoint_ste

---
