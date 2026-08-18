# Contract — ilt

**当前无实现**。`opc/iteration/ilt/` 目录尚未创建，ILT 求解器未迁移，
本文件不定义任何当前接口、数据结构或算法行为（不得当作契约事实引用）。

## 状态

- 旧库参考实现归档于 `00_PAST/opc/iteration/ilt/`（只读参照，含
  simple ILT 与 level-set 变体）；
- 基础层可复用面见 `architecture/system.md` 与其他 contracts（网格、
  光刻、评价、进度、最终合并生命周期）；
- ILT 的优化变量是像素/水平集，**不经过** SegmentBatch/owner/EPE 重建
  （与边段型方法的边界，见 mbopc 变更的设计文档 §22）。

## 新 ILT 工作的入口

按 `implementation_spec_template.md` 创建
`changes/active/CHG-xxx-ilt-<name>/implementation_spec.md`；批准后实施。
在此之前，任何文档不得声称系统具备 ILT 能力。
