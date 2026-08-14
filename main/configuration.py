"""为直接运行入口提供严格、一次读取的 TOML 默认值合并。"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from layout import LayerSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def exact_dbu(value_nm: float, dbu_nm: float, name: str,
              allow_zero: bool = False) -> int:
    """把必须落在版图格点上的纳米配置严格换算为整数 DBU。"""
    value = float(value_nm)
    if (not np.isfinite(value) or value < 0.0 or
            (value == 0.0 and not allow_zero)):
        requirement = "有限非负数" if allow_zero else "有限正数"
        raise ValueError(f"{name} 必须是{requirement}")
    raw = value / dbu_nm
    rounded = round(raw)
    # tile、halo、像素和边段配置共同决定运行坐标契约。静默取整会让输入边界、
    # 光刻像素和后续迭代配置不一致，因此只接受当前 DBU 精确可表达的值。
    if (rounded < 0 or (rounded == 0 and not allow_zero) or
            not np.isclose(raw, rounded, atol=1e-9, rtol=0.0)):
        raise ValueError(f"{name}={value_nm} nm 不能由当前 {dbu_nm} nm/DBU 精确表达")
    return int(rounded)


def fragmentation_dbu(
        corner_nm: float, segment_nm: float, max_displacement_nm: float,
        dbu_nm: float) -> tuple[float, float, float]:
    """换算分段参数，并强制几何分段长度落在整数 DBU 格点上。"""
    # corner 与 segment 决定实际切分端点，非整数 DBU 会引入不可表示的版图坐标；
    # 最大位移只参与连续优化和重建上限，可以保留小数 DBU，避免无意义量化步长。
    corner = exact_dbu(corner_nm, dbu_nm, "corner_nm")
    segment = exact_dbu(segment_nm, dbu_nm, "segment_nm")
    displacement = float(max_displacement_nm) / dbu_nm
    return float(corner), float(segment), displacement


def _config_path(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    """从任意参数位置提取唯一 `--config`，其余参数原样交回 argparse。"""
    cleaned: list[str] = []
    custom: Path | None = None
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--config":
            if custom is not None or index + 1 >= len(argv):
                raise ValueError("--config 必须且只能提供一次有效路径")
            custom = Path(argv[index + 1]).expanduser().resolve()
            index += 2
            continue
        if value.startswith("--config="):
            if custom is not None:
                raise ValueError("--config 只能提供一次")
            custom = Path(value.split("=", 1)[1]).expanduser().resolve()
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned, custom


def _read(path: Path, entries: set[str]) -> dict[str, dict[str, Any]]:
    """读取并校验配置顶层结构，不容忍拼错的 section。"""
    if not path.is_file():
        raise ValueError(f"配置文件不存在：{path}")
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"无法读取 TOML 配置 {path}: {exc}") from exc
    unknown = set(data) - {"common", "entry"}
    if unknown:
        raise ValueError(f"配置包含未知 section：{', '.join(sorted(unknown))}")
    common, entry = data.get("common", {}), data.get("entry", {})
    if not isinstance(common, dict) or not isinstance(entry, dict):
        raise ValueError("[common] 和 [entry.*] 必须是 TOML table")
    unknown_entries = set(entry) - entries
    if unknown_entries:
        raise ValueError(f"配置包含未知 entry：{', '.join(sorted(unknown_entries))}")
    if any(not isinstance(value, dict) for value in entry.values()):
        raise ValueError("每个 [entry.*] 必须是 TOML table")
    return {"common": common, **{f"entry.{name}": value for name, value in entry.items()}}


def _selected_parser(parser: argparse.ArgumentParser, argv: Sequence[str]) -> argparse.ArgumentParser:
    """定位离线子命令解析器；普通入口直接返回根解析器。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for value in argv:
                if value in action.choices:
                    return action.choices[value]
            return parser
    return parser


def _actions(parser: argparse.ArgumentParser, selected: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """汇总当前入口可配置的可选参数，位置参数保持由 CLI 提供。"""
    result: dict[str, argparse.Action] = {}
    for current in (parser, selected) if selected is not parser else (parser,):
        for action in current._actions:
            if action.option_strings and action.dest not in {"help", "config"}:
                result[action.dest] = action
    return result


def _convert(action: argparse.Action, value: Any, origin: Path) -> Any:
    """按 argparse action 严格转换 TOML 值，并解析配置相对路径。"""
    if isinstance(action, argparse.BooleanOptionalAction) or action.const in (True, False):
        if not isinstance(value, bool):
            raise TypeError(f"配置 {action.dest} 必须是布尔值")
        return value
    if action.dest == "glp_layers":
        if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                              for k, v in value.items()):
            raise TypeError("glp_layers 必须是 NAME='LAYER/DATATYPE' 映射")
        return [(name, parse_layer_spec(spec)) for name, spec in value.items()]
    if isinstance(action, argparse._AppendAction):
        if not isinstance(value, list):
            raise TypeError(f"配置 {action.dest} 必须是数组")
        return [_scalar(action, item, origin) for item in value]
    values = value if action.nargs not in (None, "?") else None
    if values is not None:
        if not isinstance(value, list):
            raise TypeError(f"配置 {action.dest} 必须是数组")
        return [_scalar(action, item, origin) for item in value]
    return _scalar(action, value, origin)


def _scalar(action: argparse.Action, value: Any, origin: Path) -> Any:
    """转换一个标量，同时拒绝 bool 冒充 int/float。"""
    converter = action.type
    if converter is Path:
        if not isinstance(value, str):
            raise TypeError(f"配置 {action.dest} 必须是路径字符串")
        path = Path(value).expanduser()
        return (origin / path).resolve() if not path.is_absolute() else path.resolve()
    if converter is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"配置 {action.dest} 必须是整数")
    if converter is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise TypeError(f"配置 {action.dest} 必须是数值")
    if converter in (str, None) and not isinstance(value, str):
        raise TypeError(f"配置 {action.dest} 必须是字符串")
    try:
        converted = value if converter is None else converter(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise TypeError(f"配置 {action.dest} 的值无效：{value!r}") from exc
    if action.choices is not None and converted not in action.choices:
        raise ValueError(f"配置 {action.dest} 不在允许值中：{converted!r}")
    return converted


def _json_value(value: Any) -> Any:
    """把最终 Namespace 值转换为 JSON 可记录形式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, LayerSpec):
        return f"{value.layer}/{value.datatype}"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class ConfiguredArgumentParser(argparse.ArgumentParser):
    """在正常 argparse 解析前叠加默认和自定义 TOML。"""

    def __init__(self, *args: Any, workflow: str, entry: str,
                 valid_entries: Sequence[str] | None = None, **kwargs: Any) -> None:
        """绑定一个 workflow 默认文件和当前入口名称。"""
        super().__init__(*args, **kwargs)
        self._workflow = workflow
        self._entry = entry
        self._entries = set(valid_entries or (entry,))
        self.add_argument("--config", type=Path,
                          help="自定义 TOML；优先级低于显式 CLI 参数")

    def parse_args(self, args: Sequence[str] | None = None,
                   namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        """严格合并配置后解析 CLI，并在 Namespace 记录来源及最终参数。"""
        try:
            return self._parse_configured(args, namespace)
        except (TypeError, ValueError) as exc:
            self.error(str(exc))

    def _parse_configured(self, args: Sequence[str] | None,
                          namespace: argparse.Namespace | None) -> argparse.Namespace:
        """执行配置读取和 argparse 合并，异常由公共入口转换为标准 CLI 错误。"""
        argv, custom = _config_path(list(sys.argv[1:] if args is None else args))
        selected = _selected_parser(self, argv)
        if selected is self and any(value in ("-h", "--help") for value in argv):
            return super().parse_args(argv, namespace)
        # 只有真正的 subcommand 才能从 argv 决定 entry；普通入口的位置输入即使
        # 恰好名为另一个 entry，也不得改变配置 section。
        entry = self._entry
        if selected is not self:
            entry = next(value for value in argv if value in self._entries)
        actions = _actions(self, selected)
        default_path = PROJECT_ROOT / "config" / f"{self._workflow}.toml"
        sources = [(default_path, _read(default_path, self._entries))]
        if custom is not None:
            sources.append((custom, _read(custom, self._entries)))
        merged: dict[str, Any] = {}
        for path, tables in sources:
            for table_name in ("common", f"entry.{entry}"):
                for key, value in tables.get(table_name, {}).items():
                    if key not in actions:
                        raise ValueError(f"配置 {path} 的 {table_name}.{key} 不是有效参数")
                    merged[key] = _convert(actions[key], value, path.parent)
        selected.set_defaults(**merged)
        parsed = super().parse_args(argv, namespace)
        setattr(parsed, "config", custom)
        effective = {key: _json_value(getattr(parsed, key)) for key in actions
                     if hasattr(parsed, key)}
        setattr(parsed, "_configuration", {
            "default": str(default_path),
            "custom": None if custom is None else str(custom),
            "precedence": "default.common < default.entry < custom.common < custom.entry < CLI",
            "effective": effective,
        })
        return parsed


def parse_layer_spec(value: str) -> LayerSpec:
    """解析配置和 GLP 映射共用的 `layer[/datatype]`。"""
    parts = value.replace(":", "/").split("/")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("Layer 格式应为 layer 或 layer/datatype")
    try:
        return LayerSpec(int(parts[0]), int(parts[1]) if len(parts) == 2 else 0)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"非法 Layer：{value}") from exc


def parse_glp_layer(value: str) -> tuple[str, LayerSpec]:
    """解析重复 CLI 参数 `NAME=LAYER/DATATYPE`。"""
    if "=" not in value:
        raise argparse.ArgumentTypeError("GLP 层映射格式应为 NAME=LAYER/DATATYPE")
    name, spec = value.split("=", 1)
    if not name or not name.strip():
        raise argparse.ArgumentTypeError("GLP 符号层名称不能为空")
    return name.strip(), parse_layer_spec(spec)


def glp_layer_map(values: Sequence[tuple[str, LayerSpec]] | None) -> dict[str, LayerSpec]:
    """按出现顺序合并 GLP 层映射，显式 CLI 的同名项覆盖配置默认。"""
    return {} if values is None else {name: layer for name, layer in values}
