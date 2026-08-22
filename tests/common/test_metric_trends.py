"""公共指标趋势图组件的通用输入、布局和异常契约测试。"""

import json
from pathlib import Path

import pytest

from common.metric_trends import save_metric_trends


def _write_metrics(tmp_path: Path, series_id: str, records: list[dict]) -> Path:
    """写出一个与 MB-OPC/ILT 均兼容的指标 JSON。"""
    path = tmp_path / f"{series_id}.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def test_generic_ilt_like_records_save_series_pngs(tmp_path):
    """不依赖算法名称即可绘制 ILT 风格 loss 字段。"""
    records = [
        {
            "state_index": 0,
            "total_loss": 4.0,
            "nominal_l2": 2.0,
            "process_l2": 1.0,
            "pvband_loss": 0.5,
            "curvature_loss": 0.25,
        },
        {
            "state_index": 1,
            "total_loss": 3.0,
            "nominal_l2": 1.5,
            "process_l2": 0.75,
            "pvband_loss": 0.4,
            "curvature_loss": 0.2,
        },
    ]
    metrics = {"ilt_stage0": _write_metrics(tmp_path, "ilt_stage0", records)}
    result = save_metric_trends(
        metrics,
        tmp_path / "trends",
        ("total_loss", "nominal_l2", "process_l2", "pvband_loss", "curvature_loss"),
        best_state_indices={"ilt_stage0": 1},
    )
    assert set(result["series_pngs"]) == {"ilt_stage0"}
    assert result["fields"][-1] == "curvature_loss"
    assert Path(result["overview_png"]).is_file()
    assert Path(result["series_pngs"]["ilt_stage0"]).name == "series_ilt_stage0.png"


def test_metric_trends_lines_support_different_state_counts(tmp_path):
    """lines 总览不要求不同结果序列拥有相同状态数量。"""
    first = _write_metrics(tmp_path, "a", [{"state_index": 0, "epe": 2}, {"state_index": 1, "epe": 1}])
    second = _write_metrics(tmp_path, "b", [{"state_index": 0, "epe": 3}])
    result = save_metric_trends({"a": first, "b": second}, tmp_path / "trends", ("epe",), overview_mode="lines")
    assert result["overview_mode"] == "lines"
    assert Path(result["overview_png"]).name == "overview_lines.png"


@pytest.mark.parametrize(
    "fields, message",
    [
        ((), "不能为空"),
        (("epe", "epe"), "不允许重复"),
        (("missing",), "缺少指标"),
    ],
)
def test_metric_trends_reject_invalid_fields(tmp_path, fields, message):
    """字段配置错误必须直接失败，不生成成功 metadata。"""
    metrics = {"series": _write_metrics(tmp_path, "series", [{"state_index": 0, "epe": 1}])}
    with pytest.raises(ValueError, match=message):
        save_metric_trends(metrics, tmp_path / "trends", fields)
