# P1/P2 架构收敛开发报告

## 1. 范围与结果

本轮实施 `current_architecture_review.md` 的 P1/P2，目标是修正真实契约错误、删除确定重复并保持数值数据面不变。未实施 P3 的类型/归档改名、项目身份修改、异常删除或历史产物清理；`geometry/` 未修改，`layout/` 仅新增一个经用户授权的只读扫描入口。

最终净变化是把跨 runner 的事实公共能力改为正式公共接口，并删除两套 SimpleILT 流程。没有增加 runner 基类、算法注册器、模型工厂、配置继承或 ownership 包装结构。

## 2. P1 修改

- `psutil` 从开发依赖移到运行依赖，匹配 `opc.input.preflight` 的顶层导入。
- `LayoutDB.recursive_polygon_shapes()` 批量返回受数据库生命周期约束的 Polygon 类递归迭代器；preflight 不再读取 `_native_layout/_native_cell/_native_layer_index`。
- `LithographyModel/LithographyConfigView` Protocol 描述 solver 当前实际使用的设备、画布、阈值、工艺条件和批量 forward 能力；runner 仍显式构造 ICCAD13。
- preflight 按预算计算 `max_memberships`；真实构造入口将其传给 `prepare_problem`，`_build_ownership` 在 `np.repeat/argsort` 前按 int32 与用户上限拒绝。

## 3. P2 修改

- `main/artifacts.py` 集中公开原子 JSON/NPZ/PNG、完整最终光刻结果和 ownership-only tile 保存；`offline_inputs.py` 缩减约 240 行，只负责输入物化、归档和 CLI。
- `main.configuration.exact_dbu/parse_layer_spec` 成为唯一 DBU 转换和 Layer 参数解析；删除三个复制/转发函数。
- `opc.iteration.ilt._common` 保存四种现有 ILT 共用的 batch、曲率、缩放和平滑 sigmoid 操作；不建立求解器继承层。
- `MBOPCProblem.owner_segments_for_core()` 固化唯一写入查询，Simple MB-OPC 与 DiffOPC 只在启动时各构造一次 tuple。
- `opc.input._fragmentation.count_edge_fragments()` 是预检和生产切分共用的 O(edge) 纯数组公式，不物化 SegmentBatch。
- `run_simpleilt()` 只映射历史默认值并调用 `run_ilt(method="simple", return_result=True)`；保留 `(SimpleILTResult, summary)`，统一产物为 `ilt_result.npz`。输入、优化、评价、资源统计和保存不再有第二份实现。

## 4. 性能与内存影响

Protocol 仅用于静态类型，不产生运行期适配对象。owner 查询、边段计数与原子写入没有增加热路径复制；共享计数公式反而消除预检/生产漂移风险。SimpleILT 去重减少一整套维护代码，统一入口保留已有 GPU/进程内存统计。

归属容量检查在大数组分配前完成；最终光刻 tile 仍逐 batch 写自身 ownership 后立即释放 GPU 输出。没有把整张 reticle tensor 常驻 GPU，也没有改变 CPU 常驻完整 `MBOPCProblem` 的既有能力边界。

## 5. Bug 后清理与过度设计审计

首轮测试发现共享计数函数名与生产函数局部数组同名，已直接改成动词式 `count_edge_fragments`，没有保留别名、包装或特判。统一 ILT 缺少返回注解导入和一个残留未使用 Layer 导入均已删除。

最终搜索确认：runner 不再跨模块导入 `_atomic_*`、`_exact_dbu` 或复制 `parse_layer`；solver 不再含 `_owner_indices/_owner_segments`；扩展 ILT 不再从 `simple.py` 导入私有 helper。新增三个文件都至少有两个现实调用方或一组明确产物职责，不属于空目录/假想接口。
