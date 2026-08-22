# Contract — lithography（ICCAD13 Hopkins 模型）

锚点：`lithography/iccad13.py`、`lithography/contracts.py`。

## 具体模型

```python
class ICCAD13Config:                              # 九字段 frozen；from_file 严格解析
    kernel_count: int                             # 1..24（资产上限）
    canvas / resolution: int                      # 均冻结 256
    target_density / print_threshold / print_steepness: float
    dose_min / dose_nominal / dose_max: float     # 0<min≤nom≤max

@dataclass(frozen=True, slots=True)
class ProcessCondition:
    name: str; kernel: "focus" | "defocus"; dose: float

class ICCAD13Lithography(torch.nn.Module):        # 无 parameter，kernel/scale 是 buffer
    ICCAD13Lithography(config_path=None, asset_dir=None, device=None)
    device -> torch.device                        # buffer 所在设备（.to 同步移动）
    config -> ICCAD13Config
    def condition(self, name) -> ProcessCondition  # nominal/dose_max/defocus_min
    def forward(self, mask, condition) -> Tensor   # [H,W]/[B,H,W] -> 同形 float32
    def forward_many(self, mask, conditions) -> dict[str, Tensor]
```

## 输入/输出契约

- **透光率 tensor**：1=透光、0=不透光；[H,W] 或 [B,H,W]，H/W ≤ 256；
  连续 0~1 合法（不强制二值）；行 0 = 最低 Y（与输入同向）。
- **256 直传**：`opc.input.rasterize_mask_canvas` 输出可直接作为输入
  （padding 契约逐位一致，模型不二次移动）；小图自动居中补零。
- **输出**：与输入同形 float32，范围 (0,1) 开区间（sigmoid 连续，不是二值）。
- **前向公式链**：pad → fft2(norm="forward") → 四象限 kernel 相乘（象限块
  尺寸由 kernel 35→18/17 决定，不是频谱尺寸）→ ifft2 → scale 加权 |field|²
  → dose² 缩放 → sigmoid(steepness×(I−target)) → crop。全原生可微算子，
  无手写 backward。
- **性能不变量**：一次 forward_many = 1 次 mask fft2 + 每 bank 各 1 次传播；
  dose 只经 dose² 因子复用同一 unit 强度（monkeypatch 计数测试固化）。

## 求解器消费契约（contracts.py）

```python
@runtime_checkable
class LithographyConfigView(Protocol):            # canvas: int; print_threshold: float
@runtime_checkable
class LithographyModel(Protocol):                 # device/config/condition/forward_many
```

只描述求解器消费的能力；`ProcessCondition` 类型当前直接复用 ICCAD13 定义
（设计选择，见 ADR-004）。`__init__` 导出 `LithographyModel`。

## 资产与设备

- 四个 `.pt`（focus/defocus 核 + scale）SHA-256 是模型身份硬断言；
  布局只接受 [H,W,K] 方阵 + [K]，不猜 shape。
- device="auto"=有 CUDA 用 CUDA；Windows 直跑的 DLL 目录注册在模块头
  （必须先于 import torch）。
- 显存：三条件前向约 B×12MiB（含中间场）；B 由调用方决定。

## 事实核对锚点

`tests/lithography/`（81 例，coverage 100%；CPU sums 与 OpenILT 基线逐位
一致；CUDA parity 1e-4）。

## TorchLitho 物理参数化模型（2026-08-23 迁移，CHG-20260823-torchlitho）

锚点：`lithography/torchlitho/`（model/source/tcc 三模块）。满足同一
`LithographyModel` 协议，与 ICCAD13 并列；算法详解见
`doc/algorithms/{abbe,hopkins}.md`，一致性证明见
`doc/changes/completed/CHG-20260823-torchlitho/test_report.md`。

```python
class TorchLithoConfig:                # [torchlitho] 段，全默认
    method: "abbe" | "hopkins"
    source_shape: "point" | "disk" | "dipole" | "quadrupole"   # dipole/quadrupole 要求 pole_center>0
    sigma / pole_center / wavelength_nm / na / refractive_index / defocus_min_nm
    dose 三值 + 胶模型三参数            # 默认对齐 iccad13.txt

@dataclass(frozen=True, slots=True)
class TorchLithoCondition:             # 与 ProcessCondition 鸭子类型并存
    name: str; defocus_nm: float; dose: float

class TorchLithoLithography(torch.nn.Module):
    TorchLithoLithography(config, canvas, pixel_nm, device=None)
    device -> torch.device             # buffer 所在设备
    config -> 视图(canvas, print_threshold)   # 满足 LithographyConfigView
    def condition(self, name) -> TorchLithoCondition   # nominal/dose_max/defocus_min
    def forward / forward_many(...)     # 同 ICCAD13 语义：透光率→printed，同形
```

要点：

- **条件令牌不透明**：求解器只经 `model.condition(name)` 产生并传回
  `forward_many`，从不访问字段（这是两种条件类型并存的依据）；
  `TorchLithoCondition.defocus_nm` 是连续值（瞳/TCC 维度），dose 语义同
  ICCAD13（强度按 dose² 缩放）。
- **画布/物理像素**：canvas 沿用 `[lithography].canvas_pixels`（256 冻结），
  物理视场由 `[lithography].pixel_nm` 表达；居中 padding 与 ICCAD13 逐位一致
  （测试锁定）。
- **模型选择**：`[lithography].model = "iccad13" | "torchlitho"`（默认
  iccad13），分派在 `main/configuration.build_lithography_model`。
- **非点源 Hopkins 的幅度语义**：忠实原库（谱 J(f₁+f₂) 进入，比逐源点平均
  小 ≈1/S²）；物理正确的部分相干请用 `method="abbe"`——详见 hopkins.md §5。
