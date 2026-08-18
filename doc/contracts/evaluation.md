# Contract — evaluation（指标）

锚点：`evaluation/metrics.py`（66 语句，coverage 100%）。

## 公开接口

```python
@dataclass(frozen=True, slots=True)
class EPEEvaluation:
    valid / inner_violations / outer_violations / ambiguous / directions: Tensor
    violation_count -> int                        # inner|outer 至少一个违规的有效段数

def evaluate_binary_l2(target, nominal, threshold=0.5,
                        ownership_mask=None) -> int
def evaluate_pvband(maximum, minimum, threshold=0.5,
                    ownership_mask=None) -> int
def evaluate_edge_probes(target, nominal, batch_indices, inner_xy, outer_xy,
                         threshold=0.499) -> EPEEvaluation
```

## 契约

- **图形状**：[H,W] 或 [B,H,W]（自动升维）；形状/设备不一致 ValueError。
- **ownership 屏蔽**：2D ownership 只配单图 batch，B>1 必须显式 [B,H,W]；
  None=全图计分（跨 tile 会重复计分，调用方负责传掩码）。
- **二值语义**：`>= threshold` 判打印（L2/PVBand 默认 0.5）。
- **EPE 探针**：坐标 (x,y) 连续 float64（x=列、y=行），round（half-to-even）
  后采样；`threshold=0.499` 是与 L2/PVBand 不同的**保留默认**（迁移裁决，
  0.4995 级边界灰度两阈值判定相反，有测试固化）。
- **有效性四条件**：in_bounds（batch/x/y）且 inner≠outer 且 target 上
  inner≥thr 且 outer<thr，同时成立才 valid。
- **方向表**：inner 违规且 outer 不违规 → +1（印刷不足外移）；outer 违规
  且 inner 不违规 → −1（印刷过量内移）；双向冲突 → 0 并计 ambiguous；
  无违规 → 0。
- **violation_count 含 ambiguous 段**（"无法单边解决"仍属违规计数）。

## 消费方式（simple MB-OPC）

每批一次 `evaluate_edge_probes`（batch_indices 指向各自 core 图）；
L2/PVBand 传 ownership_tensor 只统计计分像素；指标只诊断，EPE 驱动方向。

## 事实核对锚点

`tests/evaluation/test_metrics.py`（25 例：方向四情形、invalid 路径、
阈值边界、契约 isinstance）。
