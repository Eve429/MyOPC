# ILT 多方法迁移规格进度

## 2026-08-18

- 完成会话恢复与 Git 基线检查。
- 确认当前 HEAD 为 `2fa75ea`，工作树干净。
- 建立本任务独立规划记录；尚未修改业务代码。
- 接受用户补充：每个规格同时参照 OpenILT 原实现与 `00_PAST` 首次迁移；已确认四个旧 ILT 候选及当前 ILT 无实现状态。
- 初步确定以 simple ILT 为第一份兼容性基础规格，后续逐项核对 level-set、multilevel、curvmulti；尚未最终冻结迁移顺序。
- 完成当前模型协议、统一配置、极性与栅格接口核对；发现 ILT 输出到 GDS 尚无反向路径，以及可变长度 scale 配置需要显式扩展 tuple 解析契约。
- 完成四个 OpenILT 核心算法与 `00_PAST` 适配差异的第一轮核对；确认旧统一 runner 不能恢复，且 CurvMulti 上游含已知 nominal 误用需要在计划中明确修正。
- 核对当前 MacroSpec、raster、evaluation 与 merge 能力，形成 tile 独立/ownership 回写的首版建议；全量测试首次因工具 timeout 参数过短未得到结果，待重跑。
- 完整重跑当前测试基线：446 passed / 98.28s。
- 冻结建议顺序：Simple（建立最小公共像素管线）→ LevelSet → CurvMulti（首次加入多尺度公共操作）→ Multilevel；开始分别撰写规格。
- 已创建第一份 `CHG-20260818-simple-ilt` draft：完整定义 pixel problem、tile-independent workflow、Simple 数学、N+1 state、逐样本 best、GDS 回写、现有接口迁移和测试矩阵。
- 已创建第二份 `CHG-20260818-levelset-ilt` draft：依赖 Simple 完成基线，冻结精确 SDF、hard forward/STE backward、Adam，并明确不得修改共享 workflow/problem/result。
- 已创建第三份 `CHG-20260818-curvmulti-ilt` draft：核实上游最低级实际用 target 初值；冻结 full-grid optics、uniform SGD、wafer curvature、具名 nominal bug 修复与 variadic tuple parser。
- 已创建第四份 `CHG-20260818-multilevel-ilt` draft：冻结 full optics→stage area supervision、fractional ownership 权重、独立 Adam、显式 per-stage tuples 和 final-stage-only best。
- 第一轮交叉审查修正：ILTMethod 字段计数、LevelSet 不可达 duplicate-condition 描述、四份 File-Level Change Plan 的模糊文档路径，现已全部改为精确路径与报告目标。
- 发现并隔离并发/外部工作树差异 `opc/input/grid.py`（仅注释）；四份规格已如实标注 dirty baseline，本任务不修改或提交该文件。
- 完成第二轮交叉审查：补齐外部 dataclass postponed-annotation 解析契约，避免重复用户/solver Config；四份规格章节、占位符、路径、共享接口和依赖顺序检查通过。
