# MB-OPC 多图形几何测试报告

## 1. 目的

验证物理 mask 规范化、边段切分、法向、owner、inner/outer probe 和全局重建在常见与极端图形上的一致性。测试直接使用当前进程全局 segment 下标，不依赖稳定 key。

## 2. 图形矩阵

| 案例 | 关注点 | 结果图 |
|---|---|---|
| 重叠与角接触 | 重叠内边移除、角接触不错误粘连 | `images/mbopc/overlap_corner_touch.png` |
| 正交凹多边形 | 凹角外法向、角段与重建 | `images/mbopc/orthogonal_concave.png` |
| 负坐标跨 core | 半开归属、外边界闭合、halo | `images/mbopc/negative_cross_core.png` |
| 孔洞与重叠 | hull/hole 法向和材料语义 | `images/mbopc/hole_overlap.png` |
| 多角度斜边 | 欧氏段长、跨 core 连续重建 | `images/mbopc/diagonal_angles.png` |

所有案例检查：零位移 XOR=0、segment 长度不超过配置、法向有限且为单位向量、owner 唯一、Region 合法。

## 3. 极端移动场景

### 3.1 2 nm 中空壁、8 nm probe

inner probe 可能跨过窄壁进入另一侧空区。评价必须先验证 target 的 inner/outer 语义；无效长边 probe 不产生更新。角段可能因局部法向穿过相邻边而呈现不同有效性，测试不使用“一律无效”的错误假设。

### 3.2 外线移动到内线里面

候选外环进入 hole 后，hull/hole 包含关系失效；求解器在轮次屏障前拒绝候选并回滚整轮。

### 3.3 矩形左线移动到右线右边

相对边交叉会改变 ring 的有向面积符号；拓扑保护拒绝发布。

### 3.4 斜边跨多个 core

core 不裁最终矢量；同一参考斜边只按全局位移重建一次。因此不会出现相邻 tile 各自整数裁剪导致的端点不一致、断线或细小 XOR 面积。

## 4. 孔洞规范化

GDS keyhole 桥案例的原始 10 点 hull 会规范为一个 4 边 hull 与一个 4 边 hole，零宽桥不进入物理控制边。外环和内环分别计算指向材料外部的法向。

## 5. 结论

多图形、跨 core 和非法移动回归全部通过；图集与 `geometry_suite.json` 已保存于 `doc/images/mbopc/`。当前逻辑使用固定参考分段，正常迭代不会重新切边；只有显式 remesh 才建立全新的 segment 下标和 owner。
