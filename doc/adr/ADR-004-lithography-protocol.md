# ADR-004 — LithographyModel 薄 Protocol 的引入时机

- Date：2026-08-16
- Status：accepted

## Decision

lithography 迁移期**不建**抽象（单一模型、无消费者）；simple MB-OPC 的
`evaluate_and_propose()` 成为第一个真实求解器调用方时，才建立
`lithography/contracts.py` 的 `LithographyModel`/`LithographyConfigView`
（runtime_checkable 薄 Protocol，只描述 device/config/condition/
forward_many），并从 `__init__` 导出。

## Reason

"新接口必须有当前调用方"纪律：Protocol 在获得首个消费者的同一变更中
落地，不早不晚。测试用假模型（`_PhaseModel`）验证了结构化消费不依赖
ICCAD13 具体类型。

## Rejected alternatives

- lithography 迁移时就建 Protocol：零消费者的投机抽象；
- 建注册器/工厂/统一求解器接口：为多方法预留空壳，违反最简原则。

## Consequences

- `ProcessCondition` 类型直接复用 ICCAD13 定义（focus/defocus bank）——
  对真正不同的未来光刻模型不完全可替换，属已知限制（等真实第二模型
  再演进，独立审查已记录）；
- 不建 CT/combo/TorchLitho 等泛化层（旧库教训：为假想模型付出的复杂度）。
