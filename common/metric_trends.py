"""算法无关的迭代指标趋势图输出组件。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

_TITLE_OVERRIDES = {
    "epe": "EPE",
    "l2": "L2",
    "pvband": "PVBand",
    "pvband_loss": "PVBand Loss",
    "epe_loss": "EPE Loss",
    "total_loss": "Total Loss",
    "nominal_l2": "Nominal L2",
    "nominal_l2_loss": "Nominal L2 Loss",
    "process_l2": "Process L2",
    "process_l2_loss": "Process L2 Loss",
    "curvature_loss": "Curvature Loss",
    "moved_segments": "Moved Segments",
    "displaced_segments": "Displaced Segments",
}


def _field_title(field: str) -> str:
    """把指标字段转换为图标题；未知字段使用可读化名称。"""
    return _TITLE_OVERRIDES.get(field, field.replace("_", " ").title())


def _read_records(metrics_files: Mapping[str, Path]) -> dict[str, list[dict]]:
    """读取各结果序列的 JSON 记录，不解释其算法来源。"""
    if not metrics_files:
        raise ValueError("metrics_files 不能为空")
    records_by_series: dict[str, list[dict]] = {}
    for series_id, metrics_path in metrics_files.items():
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        records = payload["records"]
        if not records:
            raise ValueError(f"指标文件没有 records：{metrics_path}")
        records_by_series[str(series_id)] = records
    return records_by_series


def _validate_fields(
    records_by_series: Mapping[str, Sequence[Mapping[str, object]]], fields: Sequence[str]
) -> tuple[str, ...]:
    """校验字段列表，并确认每个记录都提供这些指标。"""
    metric_fields = tuple(fields)
    if not metric_fields:
        raise ValueError("metric_fields 不能为空")
    if len(set(metric_fields)) != len(metric_fields):
        raise ValueError("metric_fields 不允许重复")
    for series_id, records in records_by_series.items():
        for record in records:
            if "state_index" not in record:
                raise ValueError(f"结果序列 {series_id} 的记录缺少 state_index")
            for field in metric_fields:
                if field not in record:
                    raise ValueError(f"结果序列 {series_id} 的记录缺少指标：{field}")
    return metric_fields


def _plot_panels(
    plt,
    records_by_series: Mapping[str, Sequence[Mapping[str, object]]],
    metric_fields: Sequence[str],
    output_path: Path,
    *,
    overview_mode: Literal["mean", "lines", "series"],
    best_state_index: int | None = None,
) -> None:
    """绘制一张指标面板图；每个面板只读取已保存的状态记录。"""
    columns = 2
    rows = (len(metric_fields) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, max(4, 4 * rows)), constrained_layout=True)
    axes_flat = np.asarray(axes, dtype=object).reshape(-1)
    for axis, field in zip(axes_flat, metric_fields):
        if overview_mode == "mean":
            state_values: dict[int, list[float]] = {}
            for records in records_by_series.values():
                for record in records:
                    state = int(record["state_index"])
                    state_values.setdefault(state, []).append(float(record[field]))
            states = sorted(state_values)
            values = [float(np.mean(state_values[state])) for state in states]
            axis.plot(states, values, marker="o", label="series mean")
        else:
            for series_id, records in records_by_series.items():
                states = [int(record["state_index"]) for record in records]
                values = [float(record[field]) for record in records]
                axis.plot(states, values, marker="o", label=series_id)
            if overview_mode == "series" and best_state_index is not None:
                axis.axvline(best_state_index, color="tab:red", linestyle="--", alpha=0.6)
        axis.set_title(_field_title(field))
        axis.set_xlabel("state_index")
        axis.grid(True, alpha=0.3)
    for axis in axes_flat[len(metric_fields) :]:
        axis.set_visible(False)
    if overview_mode in ("lines", "series") and len(records_by_series) > 1:
        axes_flat[0].legend(loc="best")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metric_trends(
    metrics_files: Mapping[str, Path],
    output_dir: Path,
    metric_fields: Sequence[str],
    *,
    best_state_indices: Mapping[str, int] | None = None,
    overview_mode: Literal["mean", "lines"] = "mean",
) -> dict:
    """保存通用迭代指标趋势图，输入可来自 MB-OPC、ILT 或其他迭代方法。"""
    if overview_mode not in ("mean", "lines"):
        raise ValueError("overview_mode 必须是 'mean' 或 'lines'")
    records_by_series = _read_records(metrics_files)
    fields = _validate_fields(records_by_series, metric_fields)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_pngs: dict[str, str] = {}
    for series_id, records in records_by_series.items():
        output_path = output_dir / f"series_{series_id}.png"
        _plot_panels(
            plt,
            {series_id: records},
            fields,
            output_path,
            overview_mode="series",
            best_state_index=None if best_state_indices is None else best_state_indices.get(series_id),
        )
        series_pngs[series_id] = str(output_path)
    overview_path = output_dir / f"overview_{overview_mode}.png"
    _plot_panels(plt, records_by_series, fields, overview_path, overview_mode=overview_mode)
    return {
        "directory": str(output_dir),
        "overview_mode": overview_mode,
        "overview_png": str(overview_path),
        "series_pngs": series_pngs,
        "fields": list(fields),
    }
