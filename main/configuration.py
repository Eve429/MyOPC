"""统一配置体系：按业务划分的 Config 与单一 load_config 入口（每 Config 一个 TOML section）。"""

import re
import sys
import types
import warnings
from dataclasses import MISSING, dataclass, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

from tomllib import loads as toml_loads

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 opc 可导入

from common.units import exact_dbu
from opc.input import MaskPolarity
from opc.input.edge.fragmentation import FragmentationConfig

# 求解器 DBU 输入包（resolve 构造目标）
from opc.iteration.ilt import (
    CurvMultiConfig,
    LevelSetILTConfig,
    SimpleILTConfig,
)
from opc.iteration.mbopc import (
    GradientMBOPCConfig,
    SimpleMBOPCConfig,
)

# device 只接受 auto / cpu / cuda / cuda:N（N 为非负整数）。
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(:[0-9]+)?)$")


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """输入版图与唯一目标层（[layout] 段）。"""

    layout: Path  # 输入 GDS/OASIS/GLP 路径
    layer: int  # 目标层号（严格 int）
    datatype: int  # 目标 datatype（严格 int）
    polarity: MaskPolarity  # clear=图形透光 / opaque=图形材料
    top_cell: str | None = None  # 显式顶层；缺省要求版图唯一顶层
    field_box_nm: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
    # 精确处理框 [left, bottom, right, top]（绝对 GDS 坐标，nm）——
    # 00_PAST --box 的迁移等价；省略时用 layer bbox
    field_size_nm: tuple[Decimal, Decimal] | None = None
    # 处理框尺寸 [width, height]（nm），layer bbox 居中放置——精确坐标
    # 难以获取时的省心写法

    def __post_init__(self) -> None:
        """处理框两种写法至多一个：双空=layer bbox 现行行为，双填即意图不明。"""
        if self.field_box_nm is not None and self.field_size_nm is not None:
            raise ValueError("field_box_nm 与 field_size_nm 至多填写一个")


@dataclass(frozen=True, slots=True)
class PartitionConfig:
    """大版图的空间划分参数（[partition] 段），算法无关。"""

    core_size_nm: Decimal  # 名义 core 边长
    context_nm: Decimal  # core 每侧只读上下文宽度
    macro_grid: tuple[int, int] | None = None  # 数量模式 [列,行]
    macro_size_nm: Decimal | None = None  # 尺寸模式；与 macro_grid 互斥

    def __post_init__(self) -> None:
        """互斥与正数契约：macro_grid 与 macro_size_nm 恰好一个非空。"""
        if (self.macro_grid is None) == (self.macro_size_nm is None):  # 同空同非空
            raise ValueError("macro_grid 与 macro_size_nm 必须恰好填写一个")  # 报互斥
        if self.core_size_nm <= 0 or self.context_nm < 0:  # 尺寸非法
            raise ValueError("core_size_nm 必须为正，context_nm 必须非负")  # 报范围
        # 数量模式元素必须为正
        if self.macro_grid is not None and (self.macro_grid[0] <= 0 or self.macro_grid[1] <= 0):
            raise ValueError("macro_grid 必须是两项正整数 [列, 行]")  # 报格式


@dataclass(frozen=True, slots=True)
class LithographyConfig:
    """光刻采样与执行环境（[lithography] 段），算法无关。"""

    pixel_nm: Decimal  # 采样像素尺寸（网格对齐粒度共享）
    canvas_pixels: int = 256  # 冻结为 ICCAD13 画布 256
    device: str = "auto"  # 全 run 执行环境 auto/cpu/cuda[:N]

    def __post_init__(self) -> None:
        """画布冻结、像素正数与设备枚举契约。"""
        if self.canvas_pixels != 256:  # 模型资产契约
            raise ValueError("canvas_pixels 当前固定为 256")  # 报冻结
        if self.pixel_nm <= 0:  # 像素非法
            raise ValueError("pixel_nm 必须为正")  # 报范围
        if not _DEVICE_PATTERN.match(self.device):  # 设备枚举
            # 报设备
            raise ValueError(f"未知 device：{self.device}（只接受 auto/cpu/cuda[:N]）")


@dataclass(frozen=True, slots=True)
class EdgeConfig:
    """边段化共享参数（[edge] 段），算法无关。"""

    corner_nm: Decimal  # 拐角控制段长度
    segment_nm: Decimal  # 普通控制段最大长度
    max_displacement_nm: Decimal  # 允许的绝对位移上限
    miter_limit: float  # 拐角重建 miter 上限

    def __post_init__(self) -> None:
        """边段几何契约：正长度与位移上限、miter 正数。"""
        # 段长
        if self.corner_nm <= 0 or self.segment_nm <= 0 or self.max_displacement_nm <= 0:
            raise ValueError("corner_nm/segment_nm/max_displacement_nm 必须为正")  # 报
        if self.miter_limit <= 0.0:  # miter
            raise ValueError("miter_limit 必须为正数")  # 报


@dataclass(frozen=True, slots=True)
class MBOPCConfig:
    """simple MB-OPC 优化器用户参数（[mbopc] 段）。"""

    iterations: int  # 最多发布更新次数
    initial_step_nm: Decimal  # 初始步长（nm）
    decay_every: int  # 步长减半周期
    epe_distance_nm: Decimal  # EPE 探针距离（nm）
    batch_size: int  # 一次 forward 的 core 数
    target_cache_mb: int  # target uint8 LRU 上限（MiB）

    def __post_init__(self) -> None:
        """迭代与批处理正数契约。"""
        # 迭代类
        if self.iterations < 1 or self.decay_every < 1 or self.batch_size < 1 or self.target_cache_mb < 0:
            raise ValueError("iterations/decay_every/batch_size 必须为正，cache 为非负")  # 报
        if self.initial_step_nm <= 0 or self.epe_distance_nm <= 0:  # 物理量
            raise ValueError("initial_step_nm 与 epe_distance_nm 必须为正")  # 报


@dataclass(frozen=True, slots=True)
class GradientConfig:
    """梯度 MB-OPC 优化器用户参数（[gradient] 段）。"""

    iterations: int  # 最多发布更新次数
    learning_rate_nm: Decimal  # Adam 学习率（连续 DBU 步长）
    weight_nominal_l2: float  # nominal 连续 loss 权重
    weight_process_l2: float  # max/min 对 target 权重
    weight_pvband: float  # max-min 连续差权重
    epe_distance_nm: Decimal  # EPE 探针距离（诊断+训练共用，nm）
    batch_size: int  # 一次 forward 的 core 数
    target_cache_mb: int  # target uint8 LRU 上限（MiB）
    weight_epe: float = 0.0  # 可微 EPE loss 权重（0=关闭）
    epe_steepness: float = 4.0  # EPE penalty sigmoid 陡度

    def __post_init__(self) -> None:
        """迭代/学习率正数与四权重非负且至少一正。"""
        if self.iterations < 1 or self.batch_size < 1:  # 迭代与批
            raise ValueError("iterations/batch_size 必须为正")  # 报
        if self.target_cache_mb < 0:  # 缓存
            raise ValueError("target_cache_mb 必须为非负")  # 报
        if self.learning_rate_nm <= 0:  # 学习率
            raise ValueError("learning_rate_nm 必须为正")  # 报
        # 三权重
        weights = (self.weight_nominal_l2, self.weight_process_l2, self.weight_pvband, self.weight_epe)
        if any(weight < 0.0 for weight in weights):  # 负权重
            raise ValueError("loss 权重必须非负")  # 报
        if not any(weight > 0.0 for weight in weights):  # 全零
            raise ValueError("四个 loss 权重至少一个为正")  # 报
        if self.epe_distance_nm <= 0:  # 探针
            raise ValueError("epe_distance_nm 必须为正")  # 报
        if self.epe_steepness <= 0.0:  # EPE 陡度
            raise ValueError("epe_steepness 必须为正数")  # 报


@dataclass(frozen=True, slots=True)
class SinglePassConfig:
    """单遍偏置扩张参数（[single_pass] 段）。"""

    displacement_nm: Decimal  # 单遍位移；正=沿外法向


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """验证管线双轮位移（[iteration] 段），冻结 [+2,-2] nm 精确回零。"""

    round_deltas_nm: tuple[Decimal, Decimal]  # 双轮位移（nm）

    def __post_init__(self) -> None:
        """双轮位移冻结为 [+2,-2]：只查和为零会放行 [3,-3]，回零失去约束力。"""
        if self.round_deltas_nm != (Decimal(2), Decimal(-2)):  # 值不符
            # 报冻结要求与实际值
            raise ValueError(f"round_deltas_nm 当前冻结为 [+2nm, -2nm]，实际为 {list(self.round_deltas_nm)}")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """输出行为与工作目录（[output] 段），算法无关。"""

    final_layout: Path  # 最终版图路径
    final_cell_mode: Literal["single_cell", "macro_cells"]  # 写出 Cell 模式
    work_dir: Path | None = None  # 工作产物根目录；流程消费方查 None
    save_final_lithography: bool = False  # 是否保存最终光刻 PNG
    show_progress: bool = False  # 是否显示 tqdm 进度


# Config 与 TOML section 的声明式映射；load_config 不含任何算法分支。
CONFIG_SECTIONS: dict[type, str] = {
    LayoutConfig: "layout",
    PartitionConfig: "partition",
    LithographyConfig: "lithography",
    EdgeConfig: "edge",
    MBOPCConfig: "mbopc",
    GradientConfig: "gradient",
    SimpleILTConfig: "simple_ilt",
    CurvMultiConfig: "curvmulti_ilt",
    LevelSetILTConfig: "levelset_ilt",
    SinglePassConfig: "single_pass",
    ValidationConfig: "iteration",
    OutputConfig: "output",
}
_SECTION_TO_TYPE = {name: cls for cls, name in CONFIG_SECTIONS.items()}  # 反查表


def _parse_scalar(annotation: object, value: object, name: str, base_dir: Path) -> object:
    """按字段注解把 TOML 原始值解析为目标类型（拒绝 bool 冒充数值等）。"""
    origin = get_origin(annotation)  # 泛型起源
    if origin is types.UnionType:  # X | None：TOML 无显式 null，仅解非 None 分支
        inner = [a for a in get_args(annotation) if a is not type(None)]  # 去None
        return _parse_scalar(inner[0], value, name, base_dir)  # 递归单分支
    if annotation is bool:  # 布尔先于 int 判定（bool 是 int 子类）
        if not isinstance(value, bool):  # 非布尔
            raise ValueError(f"{name} 必须是布尔值")  # 报类型
        return value  # 原样
    if annotation is int:  # 严格整数（拒 bool/float/str）
        if not isinstance(value, int) or isinstance(value, bool):  # 非纯int
            raise ValueError(f"{name} 必须是整数，不接受 {value!r}")  # 报类型
        return value  # 原样
    if annotation is float:  # 数值（拒 bool/str；int 可升 float）
        if not isinstance(value, (int, float)) or isinstance(value, bool):  # 非数值
            raise ValueError(f"{name} 必须是数值，不接受 {value!r}")  # 报类型
        return float(value)  # 统一 float
    if annotation is Decimal:  # 十进制精确数值（nm 参数换算链的事实源）
        if not isinstance(value, (int, float)) or isinstance(value, bool):  # 非数值
            raise ValueError(f"{name} 必须是数值，不接受 {value!r}")  # 报类型
        try:  # str(int/float) 是十进制短串，Decimal 无二进制误差
            return Decimal(str(value))  # 精确转换
        except InvalidOperation as exc:  # 理论不可达，防御
            raise ValueError(f"{name} 无法解析为十进制数") from exc  # 报解析
    if annotation is str:  # 字符串
        if not isinstance(value, str):  # 非字符串
            raise ValueError(f"{name} 必须是字符串")  # 报类型
        return value  # 原样
    if annotation is Path:  # 路径三态：绝对/相对 TOML 目录/~/expanduser
        if not isinstance(value, str):  # 非字符串
            raise ValueError(f"{name} 必须是字符串路径")  # 报类型
        expanded = Path(value).expanduser()  # 展开 ~
        return expanded if expanded.is_absolute() else (base_dir / expanded).resolve()  # 归一
    if annotation is MaskPolarity:  # 极性枚举（值集唯一事实源）
        try:  # 枚举构造自身校验合法值
            return MaskPolarity(str(value))  # 转枚举
        except ValueError as exc:  # 未知极性
            raise ValueError(f"不支持的极性：{value!r}") from exc  # 报极性
    if origin is tuple:  # 元组注解：定长（macro_grid [列,行]）或变长（scales）
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:  # 变长 tuple[X, ...]
            if not isinstance(value, list):  # TOML 数组形态
                raise ValueError(f"{name} 必须是列表")  # 报类型
            return tuple(_parse_scalar(args[0], item, name, base_dir) for item in value)
        if not isinstance(value, list) or len(value) != len(args):  # 定长形状
            raise ValueError(f"{name} 必须是列表 [列, 行]")  # 报形状
        # 逐元素
        return tuple(_parse_scalar(args[0], item, name, base_dir) for item in value)
    if origin is Literal:  # 字面量枚举（final_cell_mode 等）
        choices = get_args(annotation)  # 合法值集
        if value not in choices:  # 越界
            # 报枚举
            raise ValueError(f"{name} 必须是 {list(choices)} 之一，不接受 {value!r}")
        return value  # 原样
    raise ValueError(f"{name} 的类型注解 {annotation!r} 不受通用解析器支持")  # 兜底


def _parse_config(raw: dict, config_path: Path, config_type: type) -> object:
    """从已解析的 TOML 字典构造一个 Config（必填/默认/类型/未知字段）。"""
    section_name = CONFIG_SECTIONS[config_type]  # 段名查表
    section = raw.get(section_name, {})  # 本段原始字典
    base_dir = config_path.parent  # 相对路径基准
    known = {field.name for field in fields(config_type)}  # 字段名全集
    unknown = set(section) - known  # 段内未知键
    if unknown:  # 拼写错误必须在加载期暴露
        raise ValueError(f"[{section_name}] 含未知键：{sorted(unknown)}")  # 报键
    # 经 get_type_hints 解析真实注解：外部 dataclass（如 ILT 求解器 Config）
    # 启用 postponed annotations 时 field.type 只是字符串，直接交给
    # _parse_scalar 会落到"不受支持"兜底；解析失败的原异常必须传播。
    hints = get_type_hints(config_type)
    kwargs = {}  # 构造参数
    for field in fields(config_type):  # 逐字段
        if field.name in section:  # TOML 显式给出
            # 类型解析
            kwargs[field.name] = _parse_scalar(
                hints[field.name], section[field.name], f"[{section_name}].{field.name}", base_dir
            )
        # 无默认值
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"[{section_name}] 缺少必填键：['{field.name}']")  # 报必填
    return config_type(**kwargs)  # dataclass 默认值兜底 + __post_init__ 业务校验


def load_config(config_path: str | Path, *config_types: type) -> tuple:
    """统一加载入口：TOML 只读一次，返回与请求顺序一致的 Config 元组。

    未请求的合法 section 只做未知字段检查（不查必填）；未知 section
    一律报错——两者保证任何拼写错误都在加载期暴露，与请求无关。
    """
    path = Path(config_path).expanduser().resolve()  # 配置绝对路径
    raw = toml_loads(path.read_text(encoding="utf-8"))  # 唯一一次读盘解析
    unknown_sections = set(raw) - set(_SECTION_TO_TYPE)  # 未注册段
    if unknown_sections:  # 拼错段名
        raise ValueError(f"{path}：未知配置段 {sorted(unknown_sections)}")  # 报段
    for section_name in raw:  # 出现的每个段（含未请求）都查未知字段
        config_type = _SECTION_TO_TYPE[section_name]  # 段归属的 Config
        known = {field.name for field in fields(config_type)}  # 字段全集
        unknown = set(raw[section_name]) - known  # 段内未知键
        if unknown:  # 未请求段的拼写错误同样致命
            raise ValueError(f"[{section_name}] 含未知键：{sorted(unknown)}")  # 报
    # 按请求顺序解析
    return tuple(_parse_config(raw, path, config_type) for config_type in config_types)


@dataclass(frozen=True, slots=True)
class GridRuntime:
    """阶段 0 网格规划的 nm→DBU 解析结果（算法无关，edge 与像素 ILT 共用）。"""

    core_dbu: int  # core 边长
    context_dbu: int  # 只读上下文宽度
    pixel_dbu: int  # 采样像素
    macro_size_dbu: int | None  # 尺寸模式（数量模式为 None）


def resolve_grid_config(partition: PartitionConfig, litho: LithographyConfig, dbu_nm: Decimal) -> GridRuntime:
    """把划分/光刻配置换算为算法无关的 DBU 级网格参数。

    不读取 EdgeConfig：像素 ILT 无边段参数，却需要同一套网格换算与
    像素整除契约，抽出到独立入口避免它伪造边段配置才可用。
    """
    # 全部 nm 参数精确换算：不能整除直接失败，不四舍五入吸收误差。
    core_dbu = exact_dbu(partition.core_size_nm, dbu_nm, "core_size_nm")  # core
    context_dbu = exact_dbu(partition.context_nm, dbu_nm, "context_nm")  # context
    pixel_dbu = exact_dbu(litho.pixel_nm, dbu_nm, "pixel_nm")  # pixel
    # 尺寸模式才换算；数量模式保持 None
    macro_size_dbu = (
        exact_dbu(partition.macro_size_nm, dbu_nm, "macro_size_nm") if partition.macro_size_nm is not None else None
    )
    return GridRuntime(core_dbu=core_dbu, context_dbu=context_dbu, pixel_dbu=pixel_dbu, macro_size_dbu=macro_size_dbu)


@dataclass(frozen=True, slots=True)
class PrepareRuntime:
    """阶段 0/1 消费的 nm→DBU 解析结果（resolve_prepare_config 的打包）。"""

    grid: GridRuntime  # 算法无关网格参数
    fragmentation: FragmentationConfig  # DBU 级边段配置


def resolve_prepare_config(
    partition: PartitionConfig, litho: LithographyConfig, edge: EdgeConfig, dbu_nm: Decimal
) -> PrepareRuntime:
    """把划分/光刻/边段配置换算为 DBU 级准备参数并构造边段配置。"""
    # 网格换算与像素契约复用算法无关入口，保证 edge 与像素流程同源。
    grid = resolve_grid_config(partition, litho, dbu_nm)
    corner_dbu = exact_dbu(edge.corner_nm, dbu_nm, "corner_nm")  # 拐角段
    segment_dbu = exact_dbu(edge.segment_nm, dbu_nm, "segment_nm")  # 中段
    # 位移上限
    max_displacement_dbu = exact_dbu(edge.max_displacement_nm, dbu_nm, "max_displacement_nm")
    if max_displacement_dbu > grid.context_dbu:  # context 必须覆盖最大位移
        raise ValueError("context_nm 必须不小于 max_displacement_nm")
    # 边段数值约束（正长度、segment≥2×corner、非负位移）由 FragmentationConfig
    # 构造统一校验，这里不重复检查。
    fragmentation = FragmentationConfig(
        corner_length_dbu=float(corner_dbu),
        max_segment_length_dbu=float(segment_dbu),
        max_displacement_dbu=float(max_displacement_dbu),
        miter_limit=edge.miter_limit,
    )
    # 打包 problem 准备消费的解析结果
    return PrepareRuntime(grid=grid, fragmentation=fragmentation)


def resolve_mbopc_config(
    mbopc: MBOPCConfig, partition: PartitionConfig, edge: EdgeConfig, dbu_nm: Decimal
) -> SimpleMBOPCConfig:
    """校验 simple 跨段契约并构造 DBU 级求解器配置。"""
    if mbopc.initial_step_nm > edge.max_displacement_nm:  # 步长超位移上限
        raise ValueError("initial_step_nm 不得超过 max_displacement_nm")
    if mbopc.epe_distance_nm > partition.context_nm:  # 探针越上下文
        raise ValueError("epe_distance_nm 不得超过 context_nm")
    # nm→DBU 运行时派生（solver 输入包）
    return SimpleMBOPCConfig(
        iterations=mbopc.iterations,
        initial_step_dbu=float(exact_dbu(mbopc.initial_step_nm, dbu_nm, "initial_step_nm")),
        decay_every=mbopc.decay_every,
        epe_distance_dbu=float(exact_dbu(mbopc.epe_distance_nm, dbu_nm, "epe_distance_nm")),
        batch_size=mbopc.batch_size,
        target_cache_bytes=mbopc.target_cache_mb * 1024 * 1024,
    )


def resolve_gradient_config(
    gradient: GradientConfig, partition: PartitionConfig, edge: EdgeConfig, dbu_nm: Decimal
) -> GradientMBOPCConfig:
    """校验 gradient 跨段契约（含 lr 超限提示）并构造 DBU 级求解器配置。"""
    if gradient.learning_rate_nm > edge.max_displacement_nm:  # 超限仍合法只提示
        # Adam 首步更新尺度与 lr 同量级，超限会让大量段一步打到 ±上限 被
        # clamp，抬高 invalid_geometry/优化停滞风险；不改参数、不硬拒绝。
        warnings.warn(
            f"learning_rate_nm={gradient.learning_rate_nm} 超过 "
            f"max_displacement_nm={edge.max_displacement_nm}；"
            "Adam 更新可能在早期大量触发位移 clamp，"
            "增加 invalid_geometry 或优化停滞风险",
            UserWarning,
            stacklevel=2,
        )
    if gradient.epe_distance_nm > partition.context_nm:  # 探针越上下文
        raise ValueError("epe_distance_nm 不得超过 context_nm")
    # nm→DBU 运行时派生（solver 输入包）
    return GradientMBOPCConfig(
        iterations=gradient.iterations,
        # 学习率是连续 optimizer 步长：Decimal 相除后转 float，不走
        # exact_dbu 整数契约（其余参数仍走精确整数换算）。
        learning_rate_dbu=float(gradient.learning_rate_nm / dbu_nm),
        weight_nominal_l2=gradient.weight_nominal_l2,
        weight_process_l2=gradient.weight_process_l2,
        weight_pvband=gradient.weight_pvband,
        weight_epe=gradient.weight_epe,
        epe_steepness=gradient.epe_steepness,
        epe_distance_dbu=float(exact_dbu(gradient.epe_distance_nm, dbu_nm, "epe_distance_nm")),
        batch_size=gradient.batch_size,
        target_cache_bytes=gradient.target_cache_mb * 1024 * 1024,
    )
