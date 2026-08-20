# ILT 多方法迁移规格制定计划

## Goal

基于已完成的 SimpleILT 实现，扩展 ILT 方法族。后续方法不重新设计 ILT 框架，而是在已有 ILT runtime、problem、result、evaluation 契约基础上增加新的优化状态和算法后端。

当前迁移顺序：SimpleILT（已完成） → LevelSetILT → CurvMultiILT → MultilevelILT。

## Phases

### Phase 1：ILT 公共契约冻结 — Status: complete

- 已由 SimpleILT 冻结 ILT 生命周期、输入输出、loss、evaluation、result 基础接口。
- 公共层只负责数据生命周期和 workflow，不管理具体优化变量。
- 禁止复用 MB-OPC 的 SegmentBatch、segment owner、edge reconstruction 等边段优化接口。

### Phase 2：LevelSetILT 后端迁移 — Status: pending

- 基于 SimpleILT 现有框架新增 LevelSet 优化状态。
- 重点实现：
  - phi(SDF) 初始化。
  - phi 作为优化变量的生命周期管理。
  - phi→binary mask 转换。
  - surrogate backward：`-|grad(phi)| * upstream`。
  - Adam 优化流程。
- 不重新设计 ILT workflow、loss、evaluation 和结果管理。

### Phase 3：LevelSet 工程化验证 — Status: pending

- 验证 LevelSet 与 SimpleILT 使用相同 target 时的数据契约一致性。
- 验证 SDF 初始化只执行一次。
- 验证 batch size 不改变单样本优化结果。
- 验证 core/macro 边界 seam 风险。

### Phase 4：后续 ILT 方法扩展 — Status: pending

- CurvMultiILT：增加多尺度控制网格和曲率约束。
- MultilevelILT：增加多级参数化和尺度调度。
- 后续方法只能复用已有真实公共接口，不提前抽象不存在的需求。

## Result

- SimpleILT 已作为 ILT 基准实现完成。
- 后续开发重点由“建立 ILT 框架”转为“增加 ILT 优化后端”。

## Constraints

- ILT 不复用 MB-OPC MacroProblem、SegmentBatch、edge owner 语义。
- LevelSet 不允许每次 iteration 重新计算 SDF。
- LevelSet 不直接加入 EPE 等边段指标，首版保持像素级评价体系。
- 不假设 core 独立优化天然无 seam，需要明确边界处理策略。
- 无法确认的算法选择写入 Blocking Open Questions，不自行决定。

## Implementation Rules

- 优化变量必须由算法模块自身定义：
  - SimpleILT：mask parameter。
  - LevelSetILT：phi field。
  - Multi-scale ILT：对应控制网格参数。
- 公共接口只管理：target、process condition、loss、iteration record、result persistence。
- Result 类型保持中性，避免所有方法复用 SimpleILTResult 导致耦合。

## Verification

LevelSet implementation spec 必须包含：

- 算法正确性测试。
- backward 稳定性测试。
- SDF 初始化测试。
- batch consistency 测试。
- boundary/seam 测试。
- 与 SimpleILT 收敛趋势对比测试。
