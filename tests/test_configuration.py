"""验证 TOML 配置优先级、严格校验和相对路径语义。"""

from pathlib import Path

import pytest

from main.run_lithography import build_parser


def _write(path: Path, text: str) -> Path:
    """写出一个自定义 TOML 配置。"""
    path.write_text(text, encoding="utf-8")
    return path


def test_custom_config_and_cli_precedence(tmp_path: Path) -> None:
    """自定义 entry 应覆盖默认值，显式 CLI 再覆盖自定义值。"""
    config = _write(tmp_path / "custom.toml", """
[common]
device = "cpu"
output_dir = "result"
[entry.lithography]
pixel_nm = 12.0
glp_layers = { METAL = "7/2" }
""")
    args = build_parser().parse_args([
        "input.glp", "--config", str(config), "--pixel-nm", "10"])
    assert args.device == "cpu" and args.pixel_nm == 10.0
    assert args.output_dir == (tmp_path / "result").resolve()
    assert args.glp_layers[0][0] == "METAL"
    assert args._configuration["custom"] == str(config.resolve())
    assert args._configuration["effective"]["pixel_nm"] == 10.0


@pytest.mark.parametrize("content,pattern", [
    ("[common]\nunknown = 1\n", "不是有效参数"),
    ("[common]\ncanvas = 'large'\n", "必须是整数"),
    ("[wrong]\ncanvas = 1\n", "未知 section"),
    ("[entry.ghost]\ncanvas = 1\n", "未知 entry"),
])
def test_invalid_config_fails_as_cli_error(
        tmp_path: Path, content: str, pattern: str, capsys: pytest.CaptureFixture[str]) -> None:
    """未知键、错误类型和错误 section 必须由 argparse 以退出码 2 拒绝。"""
    config = _write(tmp_path / "invalid.toml", content)
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["input.gds", "--config", str(config)])
    assert raised.value.code == 2
    assert pattern in capsys.readouterr().err


def test_missing_custom_config_fails_before_input_open(tmp_path: Path) -> None:
    """缺失配置应在版图或模型加载前稳定失败。"""
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args([
            "input.gds", "--config", str(tmp_path / "missing.toml")])
    assert raised.value.code == 2
