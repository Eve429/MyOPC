"""统一配置体系：按业务划分的 Config 与单一 load_config 入口（每 Config 一个 TOML section）。"""

import sys  # 仓库根加入模块路径，保证免安装直接运行
import types  # X | None 联合类型的运行时判定
from dataclasses import MISSING, dataclass, fields  # 字段元数据驱动解析
from decimal import Decimal, InvalidOperation  # nm 参数的十进制精确载体
from pathlib import Path  # 全部路径统一使用 Path 对象
from typing import Literal, get_args, get_origin  # Literal 与联合类型解包

from opc.input import MaskPolarity  # 极性枚举（合法值集的唯一事实源）

# 仓库根 = main/ 的上一级；直接运行脚本时把它加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 opc 可导入

from tomllib import loads as toml_loads  # Python 3.12 标准库 TOML 解析


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """输入版图与唯一目标层（[layout] 段）。"""

    layout: Path                                  # 输入 GDS/OASIS/GLP 路径
    layer: int                                    # 目标层号（严格 int）
    datatype: int                                 # 目标 datatype（严格 int）
    polarity: MaskPolarity                        # clear=图形透光 / opaque=图形材料
    top_cell: str | None = None                   # 显式顶层；缺省要求版图唯一顶层


@dataclass(frozen=True, slots=True)
class PartitionConfig:
    """大版图的空间划分参数（[partition] 段），算法无关。"""

    core_size_nm: Decimal                         # 名义 core 边长
    context_nm: Decimal                           # core 每侧只读上下文宽度
    macro_grid: tuple[int, int] | None = None     # 数量模式 [列,行]
    macro_size_nm: Decimal | None = None          # 尺寸模式；与 macro_grid 互斥

    def __post_init__(self) -> None:
        """互斥与正数契约：macro_grid 与 macro_size_nm 恰好一个非空。"""
        if (self.macro_grid is None) == (self.macro_size_nm is None):  # 同空同非空
            raise ValueError("macro_grid 与 macro_size_nm 必须恰好填写一个")  # 报互斥
        if self.core_size_nm <= 0 or self.context_nm < 0:  # 尺寸非法
            raise ValueError("core_size_nm 必须为正，context_nm 必须非负")  # 报范围
        if self.macro_grid is not None and (  # 数量模式元素必须为正
                self.macro_grid[0] <= 0 or self.macro_grid[1] <= 0):  # 列/行
            raise ValueError("macro_grid 必须是两项正整数 [列, 行]")  # 报格式


@dataclass(frozen=True, slots=True)
class LithographyConfig:
    """光刻采样与执行环境（[lithography] 段），算法无关。"""

    pixel_nm: Decimal                             # 采样像素尺寸（网格对齐粒度共享）
    canvas_pixels: int = 256                      # 冻结为 ICCAD13 画布 256
    device: str = "auto"                          # 全 run 执行环境 auto/cpu/cuda[:N]

    def __post_init__(self) -> None:
        """画布冻结与像素正数契约。"""
        if self.canvas_pixels != 256:  # 模型资产契约
            raise ValueError("canvas_pixels 当前固定为 256")  # 报冻结
        if self.pixel_nm <= 0:  # 像素非法
            raise ValueError("pixel_nm 必须为正")  # 报范围


@dataclass(frozen=True, slots=True)
class MBOPCConfig:
    """simple MB-OPC 优化器用户参数（[mbopc] 段）。"""

    iterations: int                               # 最多发布更新次数
    initial_step_nm: Decimal                      # 初始步长（nm）
    decay_every: int                              # 步长减半周期
    epe_distance_nm: Decimal                      # EPE 探针距离（nm）
    batch_size: int                               # 一次 forward 的 core 数
    target_cache_mb: int                          # target uint8 LRU 上限（MiB）

    def __post_init__(self) -> None:
        """迭代与批处理正数契约。"""
        if (self.iterations < 1 or self.decay_every < 1  # 迭代类
                or self.batch_size < 1  # 批
                or self.target_cache_mb < 0):  # 缓存
            raise ValueError("iterations/decay_every/batch_size 必须为正，cache 为非负")  # 报
        if self.initial_step_nm <= 0 or self.epe_distance_nm <= 0:  # 物理量
            raise ValueError("initial_step_nm 与 epe_distance_nm 必须为正")  # 报


@dataclass(frozen=True, slots=True)
class GradientConfig:
    """梯度 MB-OPC 优化器用户参数（[gradient] 段）。"""

    iterations: int                               # 最多发布更新次数
    learning_rate_nm: Decimal                     # Adam 学习率（连续 DBU 步长）
    weight_nominal_l2: float                      # nominal 连续 loss 权重
    weight_process_l2: float                      # max/min 对 target 权重
    weight_pvband: float                          # max-min 连续差权重
    epe_distance_nm: Decimal                      # EPE 探针距离（仅诊断，nm）
    batch_size: int                               # 一次 forward 的 core 数
    target_cache_mb: int                          # target uint8 LRU 上限（MiB）

    def __post_init__(self) -> None:
        """迭代/学习率正数与三权重非负且至少一正。"""
        if self.iterations < 1 or self.batch_size < 1:  # 迭代与批
            raise ValueError("iterations/batch_size 必须为正")  # 报
        if self.target_cache_mb < 0:  # 缓存
            raise ValueError("target_cache_mb 必须为非负")  # 报
        if self.learning_rate_nm <= 0:  # 学习率
            raise ValueError("learning_rate_nm 必须为正")  # 报
        weights = (self.weight_nominal_l2, self.weight_process_l2,  # 三权重
                   self.weight_pvband)  # 收集
        if any(weight < 0.0 for weight in weights):  # 负权重
            raise ValueError("loss 权重必须非负")  # 报
        if not any(weight > 0.0 for weight in weights):  # 全零
            raise ValueError("三个 loss 权重至少一个为正")  # 报
        if self.epe_distance_nm <= 0:  # 探针
            raise ValueError("epe_distance_nm 必须为正")  # 报


@dataclass(frozen=True, slots=True)
class SinglePassConfig:
    """单遍偏置扩张参数（[single_pass] 段）。"""

    displacement_nm: Decimal                      # 单遍位移；正=沿外法向


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """输出行为与工作目录（[output] 段），算法无关。"""

    final_layout: Path                            # 最终版图路径
    final_cell_mode: Literal["single_cell", "macro_cells"]  # 写出 Cell 模式
    work_dir: Path | None = None                  # 工作产物根目录；流程消费方查 None
    save_final_lithography: bool = False          # 是否保存最终光刻 PNG
    show_progress: bool = False                   # 是否显示 tqdm 进度


# Config 与 TOML section 的声明式映射；load_config 不含任何算法分支。
CONFIG_SECTIONS: dict[type, str] = {
    LayoutConfig: "layout",                       # 输入版图段
    PartitionConfig: "partition",                  # 空间划分段
    LithographyConfig: "lithography",             # 光刻与环境段
    MBOPCConfig: "mbopc",                         # simple 算法段
    GradientConfig: "gradient",                   # gradient 算法段
    SinglePassConfig: "single_pass",              # 单遍专属段
    OutputConfig: "output",                       # 输出段
}
_SECTION_TO_TYPE = {name: cls for cls, name in CONFIG_SECTIONS.items()}  # 反查表


def _parse_scalar(annotation: object, value: object, name: str,
                  base_dir: Path) -> object:
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
    if origin is tuple:  # 定长整数元组（macro_grid [列,行]）
        inner = get_args(annotation)[0]  # 元素注解
        if (not isinstance(value, list) or len(value) != len(get_args(annotation))):  # 形状
            raise ValueError(f"{name} 必须是列表 [列, 行]")  # 报形状
        return tuple(_parse_scalar(inner, item, name, base_dir)  # 逐元素
                     for item in value)  # 组元组
    if origin is Literal:  # 字面量枚举（final_cell_mode 等）
        choices = get_args(annotation)  # 合法值集
        if value not in choices:  # 越界
            raise ValueError(f"{name} 必须是 {list(choices)} 之一，"
                             f"不接受 {value!r}")  # 报枚举
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
    kwargs = {}  # 构造参数
    for field in fields(config_type):  # 逐字段
        if field.name in section:  # TOML 显式给出
            kwargs[field.name] = _parse_scalar(  # 类型解析
                field.type, section[field.name],  # 注解与原值
                f"[{section_name}].{field.name}", base_dir)  # 报错定位名
        elif (field.default is MISSING  # 无默认值
                and field.default_factory is MISSING):  # 也无工厂
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
    return tuple(_parse_config(raw, path, config_type)  # 按请求顺序解析
                 for config_type in config_types)  # 组元组
