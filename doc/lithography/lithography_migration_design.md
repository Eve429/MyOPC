# ICCAD13 光刻模型迁移计划

> 状态：**已批准实施，2026-08-16 完成**（提交 6338710 → 8773e37 → 5f0747a →
> 报告批次；实测数据与偏差见 development_report.md / test_report.md）
> 日期：2026-08-16
> 本文件是本次光刻模型迁移的唯一实施依据。重开上下文后，只需读取本文件、仓库根
> `AGENTS.md` 和实际代码即可实施。若实际代码、用户工作树或依赖环境与本文记录不一致，
> 必须先报告差异，不得自行扩大范围或改变接口。

## 1. 本次目标

当前新树已经迁移 `layout`、`geometry` 和 `opc.input`，但根目录尚无 `lithography/`。
只读归档 `00_PAST/lithography/` 中已有一个基于 OpenILT ICCAD13 Hopkins kernel 的
PyTorch 实现。本次对它做“理解后过滤迁移”，目标是：

1. 在新树建立一个独立一级目录 `lithography/`；
2. 迁移固定 ICCAD13 Hopkins 模型所需的配置、四个 kernel/scale 资产和许可证；
3. 提供单张与 batch mask 的 forward；
4. 使用 PyTorch 原生 autograd 支持 backward，供以后梯度 OPC 和 ILT 使用；
5. 支持标称、最大剂量、离焦最小剂量等互相独立的工艺条件；
6. 同一次多工艺条件调用只做一次 mask FFT，相同 kernel bank 只传播一次；
7. 与当前 `opc.input.rasterize_mask_canvas()` 的 256×256 透光率画布直接衔接；
8. 在 `main/` 新增一个专门的直接运行验证文件，不要求安装项目包；
9. 用生成式单元测试、数值参考、有限差分和可选 CUDA 测试证明前向与反向正确；
10. 完成开发报告、测试报告、手册和规划文件同步，并做简化审计。

本次只迁移**光刻模型**。`evaluation` 仍保持待迁移；不因为 `task_plan.md` 当前把两者写在
同一个 Phase 5 就顺带迁移 evaluation。实施时应把 Phase 5 拆成：

```text
Phase 5A: lithography
Phase 5B: evaluation
```

## 2. 本轮明确不做的内容

以下内容没有当前调用方，或者属于后续独立任务，本轮禁止实现：

- 不迁移 `evaluation/`；
- 不迁移任何 MB-OPC、DiffOPC 或 ILT 求解器；
- 不建立模型注册器、工厂、抽象基类或插件系统；
- 不迁移 `00_PAST/lithography/contracts.py` 中的 Protocol；
- 不迁移 OpenILT `exact.py` 的自定义 `torch.autograd.Function` 和手写 backward；
- 不复制 CT、combo、combo CT kernel；
- 不迁移 TorchLitho 的 Abbe、动态 TCC 生成、任意 NA/波长/光源参数系统；
- 不实现全 reticle tensor 常驻 GPU；
- 不在光刻模型内部切 macro、core 或 batch；
- 不加入自动 batch-size 搜索、OOM 重试、混合精度、`torch.compile` 或多 GPU；
- 不实现 NPZ/PNG/GDS 产物管理框架；
- 不修改 `layout/`、`geometry/`、`opc/input/` 或 `00_PAST/`。

以后只有出现真实的第二个光刻模型和当前生产调用方时，才评审统一 Protocol 或模型选择器。

## 3. 已核对事实与来源边界

### 3.1 当前新树事实

- 当前没有 `lithography/`；
- 当前没有任何非归档 Python 文件导入 `torch`；
- 当前 `opc.input.rasterize_mask_canvas()` 输出 `float32[256,256]`；
- 当前 mask 数值定义固定为 `1.0=透光，0.0=不透光`；
- 当前画布数组保持左下原点语义，光刻模型不得翻转 Y；
- 当前项目通过把仓库根加入 `sys.path` 支持直接运行 `main/*.py`；
- 当前环境只读确认：PyTorch `2.5.1`、CUDA `12.4`、CUDA 可用；
- 上一轮已完成基线为 `115 passed`，但计划编写时工作树正在被用户修改，实施者必须先运行
  新鲜基线，不能把 115 当作实施时固定数量。

### 3.2 `00_PAST/lithography` 可迁移内容

```text
00_PAST/lithography/
├─ __init__.py
├─ iccad13.py
├─ config/iccad13.txt
├─ assets/iccad13/
│  ├─ focus.pt
│  ├─ defocus.pt
│  ├─ focus_scale.pt
│  └─ defocus_scale.pt
└─ OPENILT_LICENSE.txt
```

资产事实：

```text
focus.pt          complex64[35,35,24]
defocus.pt        complex64[35,35,24]
focus_scale.pt    float32[24]
defocus_scale.pt  float32[24]
```

SHA-256：

```text
focus.pt          204bee39d8225c6d3cda52ea2d13b7c6f6cf4e4244de2ce960576d1bc741438f
defocus.pt        df624de9e17485d819e488ccada7edff133690cfa01370bfabec8f9e7cb8d532
focus_scale.pt    4e6f6136d419bdf0b56e9b461471c72d97ab3ba582fa19fe65ffaf25d188dab6
defocus_scale.pt  4ce70debf23593594c2fda1bd0cadf427abb0c132b16a047f6589f51148c8dc8
```

### 3.3 OpenILT 与 TorchLitho 的参考范围

只读比较后的结论：

- OpenILT `simple.py` 与 `exact.py` 的前向都采用 FFT、Hopkins kernel bank 和
  `sum(scale×|field|²)`；
- OpenILT `exact.py` 通过 CT kernel 手写 backward，目的是绕开/替代普通 autograd 路径；
- `00_PAST` 已证明同一前向公式可直接由 PyTorch 原生算子构图，并通过有限差分；
- TorchLitho 的价值在于任意物理参数、Abbe/Hopkins 和动态 TCC，但它会引入更多目录、
  插值、TCC 生成和循环，当前固定 ICCAD13 资产不需要这些能力。

因此本次只借鉴数值公式和资产布局，不复制 TorchLitho 代码；OpenILT MIT 许可证随资产和
派生实现一起保留。

## 4. 核心设计结论

### 4.1 只保留一个具体模型

本轮只实现：

```python
ICCAD13Lithography(torch.nn.Module)
```

不建立 `LithographyModel` Protocol。后续求解器可以直接依赖下面稳定的最小公共能力：

```text
model.device
model.config.canvas
model.config.print_threshold
model.condition(name)
model.forward(mask, condition)
model.forward_many(mask, conditions)
```

等第二个真实光刻模型进入新树时，再根据两个真实实现和真实求解器调用点抽取共同契约。

### 4.2 原生 autograd，而不是手写 backward

前向全部使用 PyTorch 原生可微算子：

```text
pad -> fft2 -> complex multiply -> ifft2 -> abs² -> weighted sum
    -> dose² -> sigmoid -> crop
```

PyTorch 自动生成 backward。这样：

- forward 与 backward 使用同一份公式；
- 不需要 CT/combo kernel；
- 不需要自定义 `autograd.Function` 保存第二套导数实现；
- MB-OPC 可在 `torch.no_grad()` 下使用相同模型；
- 梯度 OPC/ILT 不使用 `no_grad()`，损失可直接回传到 mask。

有限差分测试是 backward 正确性的权威验收，不以“梯度非空”作为唯一证据。

### 4.3 工艺条件互相独立

不把三个工艺条件绑定成固定返回 tuple。调用方显式传入任意条件序列，模型按条件名称返回
字典。默认名称只是便利入口：

```text
nominal      -> focus kernel   + DoseNom
dose_max     -> focus kernel   + DoseMax
defocus_min  -> defocus kernel + DoseMin
```

同一次调用中条件名称必须唯一。调用方可以只算 nominal，也可以传自定义
`ProcessCondition("focus_101", "focus", 1.01)`。

### 4.4 模型不负责版图和极性

光刻模型输入永远是透光率 tensor：

```text
1.0 = 透光
0.0 = 不透光
```

GDS/OASIS/GLP、clear/opaque、DBU、context、ownership 和 canvas 居中都由
`layout/geometry/opc.input` 在模型之前完成。模型不导入这些模块，不重复 raster，也不再次
解释极性。

### 4.5 模型不负责 batch 切分

模型接受 `[B,H,W]`，但不决定 B。以后：

- MB-OPC 由 macro/core 调度器形成 batch；
- ILT 可以把一个或多个 tile mask 组成 batch；
- 调用方根据 GPU 显存选择 batch size。

在模型内部再切 batch 会隐藏同步、改变 autograd 图并重复实现调度，因此本轮不做。

## 5. 数值公式与张量契约

### 5.1 输入

```text
单张：mask[H,W]
批量：mask[B,H,W]
H,W <= 256
```

- 输入可以位于 CPU 或 GPU；模型把它转换到自身 device；
- 计算 dtype 固定为 `float32/complex64`；
- 连续优化 mask 可以取 0～1 间值；模型不强制二值化；
- 模型不在入口重复检查来自内部求解器的数值范围；
- 非二维/三维输入或超过 canvas 的输入在频域大数组分配前失败。

### 5.2 居中补零

若输入小于 256：

```text
low_y  = floor((256-H)/2)
high_y = 256-H-low_y
low_x  = floor((256-W)/2)
high_x = 256-W-low_x
```

奇数余量放在高坐标侧，与 `opc.input.raster` 一致。输出在返回前裁回原 H×W。

若输入已经是 `opc.input.rasterize_mask_canvas()` 产生的 256×256 canvas，四侧 padding 都是
0，模型不得再次移动或裁剪内容。

### 5.3 Hopkins 前向

资产从 `[35,35,24]` 一次转换为连续 `[24,35,35]` buffer。对准备后的 mask：

```text
M = fft2(mask, norm="forward")
F_k = ifft2(place_quadrants(M × K_k), norm="forward")
I_bank = Σ_k scale_k × |F_k|²
I_condition = dose² × I_bank
P_condition = sigmoid(PrintSteepness × (I_condition - TargetDensity))
```

`PrintThresh` 不参与连续胶图计算；它是 evaluation 或求解器需要二值化 printed image 时使用的
阈值。

### 5.4 输出

```text
forward([H,W])      -> float32[H,W]
forward([B,H,W])    -> float32[B,H,W]
forward_many(...)   -> dict[str, Tensor]，每个值与输入 shape 相同
```

- 输出与模型位于同一 device；
- 连续输出范围为 0～1；
- 输出坐标方向与输入一致，不翻转 Y；
- 输入 `requires_grad=True` 时输出保留 autograd 图；
- kernel/scale 是 buffer，不是可训练 parameter。

## 6. 性能与内存设计

### 6.1 必须保留的性能路径

一次 `forward_many(mask, conditions)`：

1. mask 只转换和 padding 一次；
2. mask FFT 只计算一次；
3. 每种实际出现的 kernel bank 只传播一次；
4. `nominal` 与 `dose_max` 共用 focus 单位剂量强度；
5. `defocus_min` 单独传播 defocus bank；
6. dose 只通过 `dose²` 缩放单位剂量强度；
7. kernel 四象限通过四次批量赋值完成，不逐 kernel、逐 batch 或逐像素 Python 循环；
8. 缓存只存在于当前调用和当前 autograd 图，不跨 mask 保存结果。

### 6.2 内存上界说明

单个复数场中间量的主体尺寸为：

```text
B × 24 × 256 × 256 × sizeof(complex64)
≈ B × 12 MiB
```

推理时 focus 和 defocus 依次传播，不把两组 complex field 作为跨调用缓存。反向时 PyTorch
会保留求导所需中间量，实际峰值高于上述单数组估算，因此 batch size 必须由上层决定。

本轮只在验证入口报告 elapsed time 和可用时的 CUDA peak allocated；不引入 batch 自动调参。

## 7. 配置与资产

### 7.1 `ICCAD13Config`

```python
@dataclass(frozen=True, slots=True)
class ICCAD13Config:
    """保存 ICCAD13 Hopkins 模型的固定数值配置。"""

    kernel_count: int       # 使用的 Hopkins 核数量，当前默认 24，不能超过资产数量
    target_density: float   # 连续胶 sigmoid 的强度阈值 0.225
    print_threshold: float  # 后续二值化阈值 0.5，不直接参与 forward
    print_steepness: float  # 连续胶 sigmoid 陡峭度 50.0
    dose_max: float         # 最大剂量 1.02
    dose_min: float         # 最小剂量 0.98
    dose_nominal: float     # 标称剂量 1.00
    canvas: int             # 固定 256
    resolution: int         # 固定 256

    @classmethod
    def from_file(cls, path: str | Path) -> ICCAD13Config:
        """读取“字段 值”配置并返回完成数值校验的冻结配置。"""
        ...
```

校验：

```text
1 <= kernel_count <= 24
canvas == 256
resolution == 256
0 < target_density
0 < print_threshold < 1
print_steepness > 0
0 < dose_min <= dose_nominal <= dose_max
全部浮点字段有限
必需字段不缺失
重复字段和未知字段失败
```

冻结 canvas/resolution 后，不迁移 `00_PAST` 中目前没有实际配置使用的 resize 分支。

### 7.2 `ProcessCondition`

```python
@dataclass(frozen=True, slots=True)
class ProcessCondition:
    """描述一次独立工艺条件的名称、kernel bank 和剂量。"""

    name: str                         # 同一次 forward_many 内唯一的结果名称
    kernel: Literal["focus", "defocus"]  # 选择已经加载的 kernel bank
    dose: float                       # 正有限振幅剂量，强度按 dose² 缩放
```

### 7.3 资产处理

- 从 `00_PAST` **复制**四个 `.pt` 文件，不移动、不修改归档；
- 复制 `OPENILT_LICENSE.txt`；
- 单元测试验证四个文件 SHA-256；
- 加载后严格验证 kernel/scale 维度、dtype 和数量；
- kernel 和 scale 注册为 `nn.Module` buffer；
- 不支持根据 shape 猜测多种任意资产布局：本模型只接受已声明的 OpenILT
  `[35,35,K]` + `[K]` 布局。

## 8. 公共类与函数定义

### 8.1 `lithography/iccad13.py`

只定义三个公开类型：

```python
@dataclass(frozen=True, slots=True)
class ICCAD13Config:
    ...

@dataclass(frozen=True, slots=True)
class ProcessCondition:
    ...

class ICCAD13Lithography(torch.nn.Module):
    """使用固定 ICCAD13 Hopkins 资产执行可微批量光刻仿真。"""

    def __init__(
        self,
        config_path: str | Path | None = None,
        asset_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """加载配置与 focus/defocus kernel bank，并移动到指定设备。"""
        ...

    @property
    def device(self) -> torch.device:
        """返回模型 buffer 当前所在设备。"""
        ...

    def condition(self, name: str) -> ProcessCondition:
        """按稳定名称返回一个默认且独立的 ICCAD13 工艺条件。"""
        ...

    @staticmethod
    def _load_kernel_bank(
        asset_dir: Path,
        name: Literal["focus", "defocus"],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """加载并校验一组 `[35,35,K]` kernel 与 `[K]` scale。"""
        ...

    def _prepare_mask(
        self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int], bool]:
        """规范化单张/批量 mask，并居中补零到固定 canvas。"""
        ...

    @staticmethod
    def _kernel_multiply(
        kernels: torch.Tensor,
        spectrum: torch.Tensor,
        kernel_count: int,
    ) -> torch.Tensor:
        """把中心原点 kernel 的四个象限批量映射到 FFT 频谱四角。"""
        ...

    def _propagate(
        self,
        spectrum: torch.Tensor,
        kernels: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """从共享 mask 频谱计算一个 kernel bank 的单位剂量强度。"""
        ...

    def forward_many(
        self,
        mask: torch.Tensor,
        conditions: Sequence[ProcessCondition],
    ) -> dict[str, torch.Tensor]:
        """一次计算多个独立工艺条件，并保留 mask 的 autograd 图。"""
        ...

    def forward(
        self,
        mask: torch.Tensor,
        condition: ProcessCondition,
    ) -> torch.Tensor:
        """执行单工艺条件的 mask 到连续 printed image 前向。"""
        ...
```

不再保留 `_restore_size()` 和 `_kernel_bank()` 两个单调用点包装函数；crop 与 bank 选择直接
写在 `forward_many()` 的对应紧凑逻辑块中。

### 8.2 `lithography/__init__.py`

只导出：

```python
__all__ = ["ICCAD13Config", "ICCAD13Lithography", "ProcessCondition"]
```

不得导出私有数值函数或尚不存在的通用模型契约。

## 9. 文件级迁移方案

### 9.1 新增文件

| 文件                                            | 核心内容                                                  |
| ----------------------------------------------- | --------------------------------------------------------- |
| `lithography/__init__.py`                     | 导出三个真实公共类型。                                    |
| `lithography/iccad13.py`                      | 配置、工艺条件、固定 ICCAD13 Hopkins forward/backward。   |
| `lithography/config/iccad13.txt`              | 固定 24 kernel、256 canvas/resolution 和剂量参数。        |
| `lithography/assets/iccad13/focus.pt`         | OpenILT focus kernel。                                    |
| `lithography/assets/iccad13/defocus.pt`       | OpenILT defocus kernel。                                  |
| `lithography/assets/iccad13/focus_scale.pt`   | focus 权重。                                              |
| `lithography/assets/iccad13/defocus_scale.pt` | defocus 权重。                                            |
| `lithography/OPENILT_LICENSE.txt`             | OpenILT MIT 许可证。                                      |
| `tests/lithography/__init__.py`               | 中文模块 docstring。                                      |
| `tests/lithography/test_iccad13.py`           | 配置、资产、前向、性能路径、backward、CUDA 和 main 测试。 |
| `main/main_test_lithography.py`               | 面向项目所有者的独立正常流程验证入口。                    |
| `doc/lithography/development_report.md`       | 实施后记录实际迁移、偏差和简化审计。                      |
| `doc/lithography/test_report.md`              | 实施后记录命令、数值、CPU/GPU 和 coverage。               |

### 9.2 修改文件

| 文件                          | 核心改动                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt`          | 若实施开始时仍不存在，则创建完整项目依赖清单并加入`torch`；不要求 `pip install -e .`。CUDA/CPU 构建选择写入手册，不在文件中硬编码私有下载源。 |
| `doc/development_manual.md` | 增加模型接口、tensor 方向、设备、no_grad/backward 和资产说明。                                                                                    |
| `doc/test_manual.md`        | 增加 lithography 测试与直接运行命令。                                                                                                             |
| `task_plan.md`              | 把 Phase 5 拆成 lithography/evaluation；只有验收通过后才标记 5A complete。                                                                        |
| `findings.md`               | 记录公式、资产哈希、接口和性能事实。                                                                                                              |
| `progress.md`               | 记录提交、测试、CPU/GPU 数值和耗时。                                                                                                              |

### 9.3 明确不修改

```text
00_PAST/**
layout/**
geometry/**
opc/**
config/macro_pipeline.toml
main/run_macro_pipeline.py
tests/layout/**
tests/geometry/**
tests/opc/**
用户 GDS、output/、.vscode/
```

如果 lithography 与当前 raster 的接口无法按本文完成，必须停下来说明需要修改 `opc.input`
的具体原因、影响和最小改法，未经用户确认不得修改。

## 10. `main/main_test_lithography.py` 设计

### 10.1 定位

与 `main/main_test_layout.py`、`main/main_test_geometry.py` 相同：

- 是迁移期首读入口；
- 只演示正常流程，不在 main 内写 pytest 断言；
- 注释逐阶段解释输入、输出、shape、device、性能和 backward；
- 直接运行，不要求安装项目：

```powershell
D:\app\miniforge\envs\myopc\python.exe main\main_test_lithography.py
```

### 10.2 函数定义

```python
def _build_demo_canvas() -> np.ndarray:
    """用当前 raster 公共接口生成非对称矩形加孔洞的 256×256 透光率画布。"""
    ...

def run_demo(device: str = "auto") -> None:
    """依次演示模型加载、三条件前向、batch、二值阈值和真实 backward。"""
    ...

def main() -> int:
    """在自动选择的设备上运行完整正常流程，成功返回 0。"""
    ...
```

只保留这三个函数。不得为打印统计新增结果 dataclass、runner 或 artifact helper。

### 10.3 执行流程

#### 阶段 1：生成真实模型输入

1. 构造一个非对称 `kdb.Region`，含实心矩形和孔洞；
2. 调用 `rasterize_mask_canvas()`；
3. 使用 `context_box=1824×1824 DBU`、`pixel=8 DBU`、canvas=256；
4. 得到 228×228 local 居中后四边 14 pixel 零 padding 的 `float32[256,256]`；
5. 打印 shape、dtype、min/max/sum，明确 row 0 对应最低 Y，未做图片翻转。

#### 阶段 2：加载模型和资产

1. `ICCAD13Lithography(device="auto")`；
2. 打印 device；
3. 打印 config 的 kernel count/canvas/resolution/threshold；
4. 打印四个 buffer 的 shape/dtype；
5. 不访问或打印整个 tensor。

#### 阶段 3：三工艺条件推理

1. 创建 nominal、dose_max、defocus_min；
2. `torch.no_grad()` 下调用一次 `forward_many()`；
3. 打印每个输出的 shape、范围和 sum；
4. 使用 `print_threshold` 统计二值曝光像素数，但不把二值化放回模型；
5. 可用 CUDA 时同步并打印 elapsed 与 peak allocated。

#### 阶段 4：batch

把原 mask 和一个平移/连续值变体组成 `[2,256,256]`，执行 nominal batch，打印输出 shape，
说明模型不在内部拆 batch。

#### 阶段 5：backward

1. 复制一张 mask 并设置 `requires_grad=True`；
2. 构造非均匀权重，避免对称损失掩盖梯度错误；
3. 只计算 nominal 并调用 `loss.backward()`；
4. 打印梯度是否有限、非零元素数和范数；
5. 不在 main 内修改 mask 或实现优化器。

## 11. 自动测试矩阵

### 11.1 配置

1. 标准配置解析为 `24/256/256`；
2. 缺字段失败；
3. 未知字段失败；
4. 重复字段失败；
5. kernel count 为 0 或超过 24 失败；
6. canvas/resolution 不是 256 失败；
7. 非有限参数失败；
8. threshold、steepness、dose 顺序失败。

### 11.2 资产

1. 四个 SHA-256 与 §3.2 一致；
2. 加载后 kernel 是连续 `complex64[24,35,35]`；
3. scale 是连续 `float32[24]`；
4. kernel/scale 是 buffer，`named_parameters()` 为空；
5. 缺资产失败；
6. kernel/scale 维度或数量不符失败；
7. `.to(device)` 后四个 buffer 和 `model.device` 一致。

### 11.3 输入、padding 与 shape

1. `[H,W]` 返回 `[H,W]`；
2. `[B,H,W]` 返回 `[B,H,W]`；
3. 200×150 的低/高 padding 正确；
4. 奇数 padding 余量落高侧；
5. 满 256 输入不移动、不裁错；
6. 257×256 在频域数组分配前失败；
7. 四维输入失败；
8. 当前 raster 产生的 256 canvas 不被二次移动；
9. 输出方向与输入一致，不发生 Y 翻转。

### 11.4 CPU 数值前向

使用 `00_PAST` 已验证的确定性两张 mask：

```text
shape = [2,200,150]
nominal sum      = 25802.533203125
dose_max sum     = 26009.16796875
defocus_min sum  = 25675.23828125
```

验收：

- batch 与逐张运行逐像素一致，`atol<=1e-6`；
- 上述 sum 使用 `atol<=0.05`；
- 满 canvas 输出不是原 mask，并存在 0～1 之间的连续值；
- 自定义 condition 的结果名称、shape、kernel/dose 生效；
- 重复名称、空 conditions、未知默认名失败。

### 11.5 共享计算性能不变量

通过 monkeypatch 调用计数证明一次默认三条件 `forward_many()`：

- `torch.fft.fft2` 对 mask 只调用一次；
- focus bank 传播一次；
- defocus bank 传播一次；
- `forward_many()` 与三个条件分别独立运行的结果数值一致；
- 代码审查确认没有逐 kernel、逐 batch、逐 pixel Python 循环。

性能测试只验证调用次数和记录耗时，不设置机器相关的强制速度阈值。

### 11.6 backward

1. nominal 单条件对 mask 产生有限非零梯度；
2. nominal+dose_max-defocus_min 联合损失产生有限非零梯度；
3. batch 每张 mask 都能获得梯度；
4. 非均匀上游权重下，选定像素的 autograd 与中心有限差分一致；
5. 建议 `epsilon=1e-3`、`rtol=2e-2`、`atol=2e-2`；
6. backward 后 kernel/scale 仍无 `.grad` parameter；
7. `torch.no_grad()` 推理输出不保留 grad graph。

### 11.7 CUDA（有 GPU 时）

1. `pytest.mark.skipif(not torch.cuda.is_available())`；
2. CPU/GPU 同输入同条件输出在明确容差内一致；
3. CUDA forward/backward 均完成且梯度有限；
4. 使用环境内 `python.exe` 直接启动子进程，验证不依赖 `conda run` 和项目安装；
5. 记录 CUDA elapsed 和 peak allocated，不设置显存绝对阈值。

如果直接环境 Python 已能完成 CUDA FFT，就不迁移 `00_PAST` 的全局 PATH/DLL 修改；只有
实际复现缺 DLL，才能提出最小 Windows 修复，并添加对应回归测试。

### 11.8 main 入口

子进程直接执行：

```powershell
python main/main_test_lithography.py
```

验证：

- 退出码 0；
- 从仓库外工作目录执行也成功；
- 输出包含模型 device、三个工艺条件 shape/range、batch shape 和 backward 梯度摘要；
- 不生成仓库内临时产物；
- 不需要 `pip install -e .`。

## 12. 阶段边界与依赖方向

迁移后的依赖：

```text
lithography -> torch + pathlib/标准库

main_test_lithography
├─ opc.input.rasterize_mask_canvas
└─ lithography.ICCAD13Lithography

未来 opc.iteration.<method>
├─ opc.input / opc.input.edge
├─ lithography
└─ evaluation
```

`lithography` 不得反向导入：

```text
layout
geometry
opc
evaluation
main
```

这样模型可被 MB-OPC、梯度 OPC 和 ILT 共用，但不包含任何具体求解方法。

## 13. 实施阶段与本地提交

只有用户明确批准本计划后才开始。每阶段开始前检查并保留用户工作树；不得把正在进行的
macro 文档、`opc/input/grid.py` 或 `.vscode` 修改纳入提交。

### 实施 A：配置、资产与最小包结构

#### 要做什么

建立可追溯、可校验的 ICCAD13 配置和资产基础，不实现求解器。

#### 执行

1. 建立 `lithography/`；
2. 从 `00_PAST` 复制四个资产、配置和 MIT 许可证；
3. 实现 `ICCAD13Config`、`ProcessCondition` 和严格资产加载；
4. 完成配置、哈希、shape/dtype/buffer 测试；
5. 本地提交：`feat(lithography): 迁移 ICCAD13 配置与 Hopkins 资产`。

### 实施 B：可微批量前向

#### 要做什么

实现一次 FFT、多条件复用和原生 autograd 前向。

#### 执行

1. 实现 padding、四象限 kernel multiply、传播、forward/forward_many；
2. 完成 CPU 固定数值、batch/single、满 canvas 和共享调用次数测试；
3. 完成有限差分与联合条件 backward；
4. 可用时完成 CUDA parity 和直接环境测试；
5. 本地提交：`feat(lithography): 完成可微批量 ICCAD13 前向`。

### 实施 C：独立 main 验证入口

#### 要做什么

提供项目所有者可以直接阅读和运行的模型首读入口。

#### 执行

1. 新增 `main/main_test_lithography.py`；
2. 使用当前 raster 公共接口生成真实 256 canvas；
3. 演示三条件、batch、阈值和 backward；
4. 添加仓库内/仓库外直接运行测试；
5. 本地提交：`feat(main): 添加光刻模型独立验证入口`。

### 实施 D：报告与简化审计

#### 要做什么

证明本次只迁移当前光刻闭环所需代码，并记录可复现证据。

#### 执行

1. 更新依赖清单与开发/测试手册；
2. 更新 `task_plan.md`、`findings.md`、`progress.md`；
3. 写开发报告和测试报告；
4. 审计未调用函数、重复前向、异常入口、资产副本和注释；
5. 确认没有 Protocol、注册器、手写 backward、第二套 raster 或求解器代码；
6. 运行 §14 全部命令；
7. 本地提交：`docs: 完成光刻模型迁移报告`；
8. 不推送远端。

## 14. 验收命令

实施者先记录实际基线，再运行：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests
D:\app\miniforge\envs\myopc\python.exe -m pytest -q tests\lithography
D:\app\miniforge\envs\myopc\python.exe -m ruff check lithography tests\lithography main\main_test_lithography.py
D:\app\miniforge\envs\myopc\python.exe -m compileall -q layout geometry opc lithography main tests
D:\app\miniforge\envs\myopc\python.exe main\main_test_lithography.py
```

Coverage：

```powershell
D:\app\miniforge\envs\myopc\python.exe -m coverage run --source=lithography -m pytest -q tests\lithography
D:\app\miniforge\envs\myopc\python.exe -m coverage report -m
```

不为了追求 100% 添加无意义防御测试；所有未覆盖分支必须在测试报告逐项说明。核心数值、
配置、资产、shape、forward_many、backward 和直接 main 路径必须覆盖。

## 15. 开发报告必须记录

- 实际复制与重写的文件；
- OpenILT 许可证和资产 SHA-256；
- 为什么没有迁移手写 backward、CT/combo 和 TorchLitho 框架；
- 公共接口与 tensor shape/dtype/device；
- 与当前 raster 的 0/1、方向和 padding 对齐；
- 一次 FFT/每 bank 一次传播的调用计数证据；
- 原生 autograd 与有限差分误差；
- CPU 固定数值；
- CUDA 是否可用、实际版本、数值差和显存/耗时；
- 直接环境 Python 是否需要 DLL 修复；
- 测试数量、coverage 和未覆盖分支；
- 简化审计删除或拒绝迁移的内容；
- 与本计划的全部偏差及原因。

## 16. 最终调用关系

```text
main_test_lithography.main()
└─ run_demo(device="auto")
   ├─ _build_demo_canvas()
   │  └─ opc.input.rasterize_mask_canvas(...)
   ├─ ICCAD13Lithography(...)
   │  ├─ ICCAD13Config.from_file(...)
   │  └─ _load_kernel_bank(focus/defocus)
   ├─ model.condition(nominal/dose_max/defocus_min)
   ├─ torch.no_grad()
   │  └─ model.forward_many(mask, conditions)
   │     ├─ _prepare_mask(...)
   │     ├─ fft2(...)                         一次
   │     ├─ _propagate(focus)                 一次
   │     │  └─ _kernel_multiply(...)
   │     ├─ _propagate(defocus)               一次
   │     │  └─ _kernel_multiply(...)
   │     └─ dose² + sigmoid + crop
   ├─ model(batch, nominal)
   └─ model(mask.requires_grad_(), nominal)
      └─ loss.backward()                      PyTorch 原生 autograd
```

## 17. 已知限制

### 17.1 不是完整物理参数模型

当前使用已经分解好的 ICCAD13 Hopkins kernel，不能通过配置任意修改 NA、波长、光源形状
或光刻胶物理参数。TorchLitho 路线以后作为独立模型评审，不在本次伪装支持。

### 17.2 Context 充分性不由模型证明

35×35 是频域 kernel 尺寸，不是 17 pixel 空间影响半径。本模型只消费调用方给出的 canvas；
context 是否足够仍需后续收敛实验。

### 17.3 固定 256 canvas

本次资产、配置和当前 macro-core 输入都冻结为 256。支持其他 canvas 需要新的 kernel/采样
契约，不通过简单插值冒充支持。

### 17.4 不负责 full-reticle 调度

模型只计算当前 batch 的 core canvas。整张 reticle 由 macro/core 管线逐任务加载、批处理和
释放；本模型不会把整张 reticle 放入 GPU。

### 17.5 连续 printed image 不是最终二值版图

forward 返回 sigmoid 连续胶图。以 `print_threshold` 二值化、计算 L2/PVBand/EPE 或生成
几何属于 evaluation/求解器任务，本轮不实现。

## 18. 最终完成标准

- [ ] 新树存在独立 `lithography/`，只有一个具体 ICCAD13 模型；
- [ ] `00_PAST/` 未修改；
- [ ] 四个资产、配置和 MIT 许可证已复制且哈希正确；
- [ ] 不存在 `contracts.py`、Protocol、注册器或抽象基类；
- [ ] 不存在手写 backward、CT/combo 资产或 TorchLitho 通用框架；
- [ ] `[H,W]`、`[B,H,W]` 前向 shape 正确；
- [ ] 小输入居中补零，256 输入不二次移动；
- [ ] 输入输出方向一致，1 始终表示透光；
- [ ] nominal/dose_max/defocus_min 可独立选择；
- [ ] 默认三条件一次 mask FFT，focus/defocus 各传播一次；
- [ ] CPU 固定数值与参考一致；
- [ ] native autograd 与有限差分一致；
- [ ] CUDA 可用时 forward/backward 和 CPU parity 通过；
- [ ] `main/main_test_lithography.py` 可从仓库内外直接运行；
- [ ] 不需要安装 MyOPC 包；
- [ ] 没有修改 layout/geometry/opc 和用户工作树；
- [ ] 全量测试、专项 Ruff、compileall、coverage 通过并记录；
- [ ] 开发报告、测试报告、手册和规划三文件同步；
- [ ] 完成未调用函数、重复实现、异常入口和 bug 遗留审计；
- [ ] 关键阶段已本地 commit，未推送远端。
