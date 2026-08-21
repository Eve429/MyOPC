# 50nm OPC Test Pattern Design Development Specification

## 1. Objective

设计一套用于 MyOPC 回归测试的标准测试版图。第一阶段针对 50nm 节点，采用固定 `1.024um × 1.024um` clip，每个 clip 只包含一种类型图形，用于独立评估 MB OPC、ILT、EPE 测试、边界处理以及后续算法优化效果。

## 2. Basic Specification

- Reticle clip size: 1024nm × 1024nm
- Coordinate unit: nm
- Target node: 50nm
- Target CD: 50nm
- Minimum distance from clip boundary: >=100nm
- Each pattern is an independent GDS cell
- Target layout and mask polarity cases are generated separately

## 3. Cell List

| Cell Name | Pattern Type | Purpose |
|---|---|---|
| L50S50_DENSE | Dense Line/Space | Dense OPC correction |
| L50S50_ISO | Isolated Line | Proximity effect |
| L50S50_SEMIDENSE | Semi Dense Line | Context dependency |
| LINE_END_50 | Line End | End shortening correction |
| CORNER_50 | Corner | Corner rounding |
| CONTACT_50 | Contact Hole | 2D imaging |
| HOLE_ARRAY_50 | Dense Contact Array | Contact proximity |
| 2D_COMPLEX_50 | 2D Cross Pattern | ILT topology optimization |

## 4. Pattern Design

### 4.1 Dense Line/Space

Purpose: evaluate dense pitch correction.

Parameters:

- Line width: 50nm
- Space: 50nm
- Pitch: 100nm
- Line length: 600nm
- Number of lines: 6

Line X positions:

```
212, 312, 412, 512, 612, 712 nm
```

Y range:

```
212 ~ 812 nm
```

---

### 4.2 Isolated Line

Purpose: evaluate isolated feature bias.

Parameters:

- CD: 50nm
- Length: 600nm

Rectangle:

```
x: 487 ~ 537 nm
y: 212 ~ 812 nm
```

---

### 4.3 Semi Dense Line

Purpose: evaluate neighborhood influence.

Parameters:

- Line width: 50nm
- Dense pitch: 100nm
- Isolated spacing: 150nm

Line positions:

```
237, 337, 437, 587, 687, 787 nm
```

---

### 4.4 Line End

Purpose: evaluate line-end pullback and hammerhead correction.

Horizontal feature:

```
x: 262 ~ 762 nm
y: 487 ~ 537 nm
```

Vertical extension:

```
x: 712 ~ 762 nm
y: 537 ~ 687 nm
```

CD:

```
50nm
```

---

### 4.5 Corner

Purpose: evaluate corner rounding.

L-shape parameters:

- Outer size: 500nm
- Width: 50nm

Horizontal:

```
x: 262 ~ 762 nm
y: 712 ~ 762 nm
```

Vertical:

```
x: 262 ~ 312 nm
y: 212 ~ 762 nm
```

---

### 4.6 Contact Hole

Purpose: evaluate 2D OPC capability.

Parameters:

- Hole size: 100nm × 100nm
- Array: 3 × 3
- Pitch: 200nm

Centers:

```
312, 512, 712 nm
```

---

### 4.7 Dense Contact Array

Purpose: evaluate contact proximity effect.

Parameters:

- Hole size: 100nm × 100nm
- Pitch: 150nm
- Array: 5 × 5

---

### 4.8 2D Complex Pattern

Purpose: evaluate ILT and 2D optimization.

Pattern:

```
    |
 ---+---
    |
    |
```

Parameters:

- Line width: 50nm
- Overall size: 500nm × 500nm
- Center: (512nm,512nm)

## 5. Polarity Support

The testbench shall support both positive tone and negative tone.

Requirements:

- Positive tone: generate original target polygon.
- Negative tone: generate complement within 1024nm × 1024nm clip.
- Keep clip boundary handling independent from feature polarity.

Output:

```
opc_test_50nm_pos.gds
opc_test_50nm_neg.gds
```

## 6. Development Plan

### Phase 1: GDS Generator

Implement parameterized generator:

- Create each pattern cell.
- Generate polygon geometry.
- Support positive/negative polarity.
- Export GDS.

### Phase 2: Verification

Check:

- Polygon CD correctness.
- Pitch correctness.
- Boundary margin.
- GDS readability.

### Phase 3: OPC Regression Integration

Integrate generated patterns into MyOPC regression tests:

- MB OPC convergence.
- ILT optimization.
- EPE measurement.
- Boundary/context test.

## 7. Future Extension

Based on this framework, extend to:

- 20nm node
- 100nm node
- 150nm node
- Cross macro boundary patterns
- Negative tone boundary stress patterns
