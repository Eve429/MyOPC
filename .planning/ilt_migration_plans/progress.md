# ILT 多方法迁移规格进度

## 2026-08-20

- 完成 SimpleILT 开发后重新审查 ILT 迁移规划。
- 确认后续 ILT 开发目标从“建立公共框架”调整为“扩展优化后端”。
- SimpleILT 已冻结 ILT 基础生命周期：target、process condition、loss、evaluation、result persistence。
- 调整迁移顺序：SimpleILT（完成）→ LevelSetILT → CurvMultiILT → MultilevelILT。
- 明确 LevelSet 不重新设计 workflow，不复用 MB-OPC 边段模型，仅增加 phi/SDF 优化状态。
- 增加 LevelSet 重点风险：SDF 重复初始化、Result 类型耦合、core boundary seam。
- 删除后续方法重复设计公共接口的计划内容。
