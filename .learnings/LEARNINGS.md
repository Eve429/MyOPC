# 项目持续学习记录

## [LRN-20260819-001] correction

**Logged**: 2026-08-19T00:00:00+08:00
**Priority**: high
**Status**: resolved
**Area**: docs

### Summary

像素 ILT 的 core 应是光刻计算与 loss ownership 分块，不能被建模为相互独立的优化问题。

### Details

Simple ILT 草案原先让每个 core 独立完成全部状态、固定其 context、独立选择
best，因而遗漏 ownership loss 对 context 内同 macro 可训练像素的梯度，并可
拼出一个从未整体评价过的 macro mask。正确语义是 macro 唯一参数快照与 macro
级同步迭代；core 仅唯一统计自己的 ownership loss，梯度可落入 simulation
context 内全部 macro-trainable pixel，并跨 core 累加。

### Suggested Action

修订 Simple ILT 规格的 requirements、invariants、data flow、algorithm、state、
performance、tests 与 acceptance criteria，并让后续 ILT 规格继承同一公共语义。

### Metadata

- Source: user_feedback
- Related Files: doc/changes/active/CHG-20260818-simple-ilt/implementation_spec.md
- Tags: ilt, gradient, ownership, context, synchronization

### Resolution

- **Resolved**: 2026-08-19T00:00:00+08:00
- **Notes**: Simple ILT 规格 Revision 0.2 已改为 macro 唯一参数、跨 core
  梯度求和、同步 step 与 macro best；依赖规格的旧语义列为后续修订项。

---
