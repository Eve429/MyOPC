# MB-OPC 前端开发报告

## 1. 交付结论

本次已完成 MB-OPC 所需的物理边界提取、边段归属、跨 core context、稳定身份、采样、owner-only 位移更新、轮廓重建和调试产物。根目录 `run_mbopc_frontend.py` 可直接运行，不需要安装项目包。

开发过程未修改 `layout/` 和 `geometry/`，也未修改用户文件 `gcd_45nm.png` 或源 GDS。

## 2. 架构决策

### 2.1 可与 ILT 共用的能力

`opc.input` 保存物理 mask、core/context 描述和方法无关输入契约；`opc.input.edge` 保存边界采样模板和诊断可视化。ILT 可直接复用前者，只有采用边界表示时才依赖后者。

### 2.2 边段输入能力

`opc.input.edge` 包含控制段数据契约、分段策略、owner 策略、更新载体与重建。具体 MB-OPC 优化迭代将独立放入 `opc.iteration.mbopc`；当前目录为空，不用占位实现制造无调用抽象。

### 2.3 性能设计

- 物理合并、轮廓和边只构建一次。
- segment 常驻数据使用 NumPy 数组，不创建 Python Segment 对象列表。
- 稳定 128-bit key 在构建阶段生成，排序 token 索引在多轮更新中复用。
- 规则 core 归属按切线定位，halo 仅展开实际相邻 core，不构建 segment×core 布尔矩阵。
- 端点、法向、长度和采样点在评估轮次按需批量物化。

## 3. 边界与分段语义

输入 Shape 属性不参与物理边界判定。重叠图形的内部 cut-line 被消除，GDS 孔洞为编码而引入的零宽桥不会成为可移动边。仅角点接触的两个分量保持独立。

每条数学边优先保留两端拐角短段，中间部分均衡切分，任意正交或斜边长度都按欧氏距离限制。外轮廓和孔洞法向均指向“从材料离开”的方向。

## 4. 归属与跨 core 协调

默认策略用 segment 中点确定唯一 owner。内部 core 共享线采用半开区间，稳定归右侧/上侧 core；整体最大边界仍归最后一列/行。一个段即使跨过 core 分界线也不被额外切断，但可同时出现在两个 halo context 中。只有 owner 能提交该 key 的绝对位移，其他 core 从全局位移向量读到同一结果。

## 5. 重建与精确性

同一数学边上位移相等的相邻段不输出内部分割点，避免斜边浮点参数点在 DBU 取整后产生毛刺。位移不同时输出两个端点形成 jog。数学拐角使用直线解析交点，平行或 miter 过长时退化为 bevel。输出前删除相邻重复点和闭合重复点，并经过轮廓与原生 Polygon 有效性检查。

## 6. 产物与主程序

`run_mbopc_frontend.py` 支持无参数合成验证或 GDS/OASIS 真实文件。真实文件在 `LayoutDB` 打开期间完成物理规范化和紧凑问题构建，然后关闭源数据库，后续更新与输出不依赖源文件。

主程序会自动检查零位移 XOR、core ownership 覆盖/重叠和重建 Region 有效性，再以原子替换方式输出 JSON、NPZ、PNG 和 GDS。core 只划分计算责任，最终矢量结果保持全局重建，避免斜边与整数 DBU 切线相交时引入量化顶点。

## 7. 简化与 bug 审计

开发期间的两个关键修正均已留下回归：

1. 非网格对齐斜边的零位移 XOR：通过删除无意义内部输出点修复，没有增加舍入补偿函数。
2. 真实文件主程序的数据库生命周期：把准备放入现有上下文，没有添加隐式复制 wrapper。

最终数据审计又删除了只重复 `t0/t1` 和 key 信息的 `fragment_indices` / `fragment_counts` 常驻数组。保留排序 key 顺序和 token，因为它们直接提高多轮更新速度。

## 8. 里程碑

- `9f99ffd feat(opc): add shared physical mask and core foundation`
- `bc409f9 feat(mbopc): add compact boundary frontend`
- 本报告、基准与最终审计将作为后续独立本地里程碑提交。
