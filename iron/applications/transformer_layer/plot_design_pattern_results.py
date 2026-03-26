#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

SERIES_COLORS = {
    "encoder_pipeline": "#0b6e4f",
    "gemm_only": "#c84c09",
    "operator_runlist": "#1d4ed8",
    "amd_gpu_reference": "#7c3aed",
    "best_npu": "#111827",
}


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _unique_sorted_seq_lens(rows: list[dict[str, str]]) -> list[int]:
    return sorted(
        {
            int(seq_len)
            for seq_len in (_optional_int(row.get("seq_len")) for row in rows)
            if seq_len is not None
        }
    )


def _npu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("backend") == "npu"]


def _gpu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("backend") == "gpu"]


def _series_by_execution_mode(
    rows: list[dict[str, str]],
    value_key: str,
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        execution_mode = str(row.get("execution_mode"))
        seq_len = _optional_int(row.get("seq_len"))
        value = _optional_float(row.get(value_key))
        if execution_mode in ("", "None") or seq_len is None or value is None:
            continue
        series.setdefault(execution_mode, []).append((seq_len, value))
    for execution_mode in series:
        series[execution_mode].sort(key=lambda item: item[0])
    return series


def _efficiency_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    derived = []
    for row in rows:
        energy_j = _optional_float(row.get("energy_j"))
        flops = _optional_float(row.get("estimated_flops_per_inference"))
        if energy_j in (None, 0.0) or flops is None:
            continue
        derived_row = dict(row)
        derived_row["flops_per_joule"] = str(flops / energy_j)
        derived.append(derived_row)
    return derived


def _best_npu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_seq: dict[int, dict[str, str]] = {}
    for row in _npu_rows(rows):
        seq_len = _optional_int(row.get("seq_len"))
        latency = _optional_float(row.get("avg_latency_ms"))
        if seq_len is None or latency is None:
            continue
        current = best_by_seq.get(seq_len)
        if current is None or latency < _optional_float(current.get("avg_latency_ms")):
            best_by_seq[seq_len] = dict(row)
    output = []
    for seq_len in sorted(best_by_seq):
        row = dict(best_by_seq[seq_len])
        row["execution_mode"] = "best_npu"
        row["pattern_label"] = f"best_npu({row.get('pattern_label')})"
        output.append(row)
    return output


def _bottleneck_series(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[tuple[int, float]]], dict[tuple[str, int], str]]:
    series: dict[str, list[tuple[int, float]]] = {}
    labels: dict[tuple[str, int], str] = {}
    for row in rows:
        execution_mode = str(row.get("execution_mode"))
        seq_len = _optional_int(row.get("seq_len"))
        fraction = _optional_float(row.get("dominant_component_fraction"))
        component = row.get("dominant_component")
        if execution_mode in ("", "None") or seq_len is None or fraction is None:
            continue
        series.setdefault(execution_mode, []).append((seq_len, fraction))
        labels[(execution_mode, seq_len)] = str(component)
    for execution_mode in series:
        series[execution_mode].sort(key=lambda item: item[0])
    return series, labels


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="30" y="35" font-size="20" font-family="sans-serif" fill="#111827">{html.escape(title)}</text>',
    ]


def _nice_upper_bound(values: list[float]) -> float:
    maximum = max(values) if values else 1.0
    if maximum <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(maximum))
    normalized = maximum / magnitude
    if normalized <= 1:
        return magnitude
    if normalized <= 2:
        return 2 * magnitude
    if normalized <= 5:
        return 5 * magnitude
    return 10 * magnitude


def _line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    percent_axis: bool = False,
) -> str:
    width, height = 960, 520
    left, right, top, bottom = 80, 40, 70, 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    svg = _svg_header(width, height, title)

    all_x = sorted({point[0] for points in series.values() for point in points})
    all_y = [point[1] for points in series.values() for point in points]
    if not all_x:
        svg.append(
            '<text x="30" y="80" font-size="14" font-family="sans-serif" fill="#6b7280">No data available.</text>'
        )
        svg.append("</svg>")
        return "\n".join(svg)

    y_max = 1.0 if percent_axis else _nice_upper_bound(all_y)
    if percent_axis:
        all_y = [min(max(value, 0.0), 1.0) for value in all_y]
    x_min, x_max = min(all_x), max(all_x)
    x_span = max(1, x_max - x_min)

    def x_pos(x_value: int) -> float:
        return left + ((x_value - x_min) / x_span) * chart_width

    def y_pos(y_value: float) -> float:
        clamped = min(max(y_value, 0.0), y_max)
        return top + chart_height - (clamped / y_max) * chart_height

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5.0
        y = y_pos(tick_value)
        label = (
            f"{tick_value:.0%}"
            if percent_axis
            else f"{tick_value:.2f}".rstrip("0").rstrip(".")
        )
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#4b5563">{html.escape(label)}</text>'
        )

    for seq_len in all_x:
        x = x_pos(seq_len)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#f1f5f9" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#4b5563">{seq_len}</text>'
        )

    svg.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg.append(
        f'<text x="{width/2:.1f}" y="{height-25}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#111827">{html.escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {height/2:.1f})" font-size="14" font-family="sans-serif" fill="#111827">{html.escape(y_label)}</text>'
    )

    legend_y = 55
    legend_x = width - right - 250
    for legend_index, series_name in enumerate(sorted(series)):
        color = SERIES_COLORS.get(series_name, "#374151")
        x = legend_x + legend_index * 120
        svg.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x+30}" y="{legend_y+5}" font-size="12" font-family="sans-serif" fill="#111827">{html.escape(series_name)}</text>'
        )

    for series_name, points in sorted(series.items()):
        color = SERIES_COLORS.get(series_name, "#374151")
        path_points = []
        for seq_len, value in points:
            y_value = min(max(value, 0.0), y_max) if percent_axis else value
            path_points.append(f"{x_pos(seq_len):.1f},{y_pos(y_value):.1f}")
        if len(path_points) >= 2:
            svg.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(path_points)}"/>'
            )
        for seq_len, value in points:
            y_value = min(max(value, 0.0), y_max) if percent_axis else value
            svg.append(
                f'<circle cx="{x_pos(seq_len):.1f}" cy="{y_pos(y_value):.1f}" r="4" fill="{color}"/>'
            )
    svg.append("</svg>")
    return "\n".join(svg)


def _grouped_bar_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    label_lookup: dict[tuple[str, int], str] | None = None,
    percent_axis: bool = False,
) -> str:
    width, height = 960, 520
    left, right, top, bottom = 80, 40, 70, 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    svg = _svg_header(width, height, title)

    categories = sorted({point[0] for points in series.values() for point in points})
    if not categories:
        svg.append(
            '<text x="30" y="80" font-size="14" font-family="sans-serif" fill="#6b7280">No data available.</text>'
        )
        svg.append("</svg>")
        return "\n".join(svg)

    y_values = [point[1] for points in series.values() for point in points]
    y_max = 1.0 if percent_axis else _nice_upper_bound(y_values)

    def y_pos(y_value: float) -> float:
        clamped = min(max(y_value, 0.0), y_max)
        return top + chart_height - (clamped / y_max) * chart_height

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5.0
        y = y_pos(tick_value)
        label = (
            f"{tick_value:.0%}"
            if percent_axis
            else f"{tick_value:.2f}".rstrip("0").rstrip(".")
        )
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#4b5563">{html.escape(label)}</text>'
        )

    group_width = chart_width / max(1, len(categories))
    bar_width = max(12.0, (group_width * 0.7) / max(1, len(series)))

    svg.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg.append(
        f'<text x="{width/2:.1f}" y="{height-25}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#111827">{html.escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {height/2:.1f})" font-size="14" font-family="sans-serif" fill="#111827">{html.escape(y_label)}</text>'
    )

    legend_y = 55
    legend_x = width - right - 250
    for legend_index, series_name in enumerate(sorted(series)):
        color = SERIES_COLORS.get(series_name, "#374151")
        x = legend_x + legend_index * 120
        svg.append(
            f'<rect x="{x}" y="{legend_y-10}" width="16" height="16" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{x+22}" y="{legend_y+3}" font-size="12" font-family="sans-serif" fill="#111827">{html.escape(series_name)}</text>'
        )

    for category_index, seq_len in enumerate(categories):
        group_start_x = left + category_index * group_width + (group_width * 0.15)
        svg.append(
            f'<text x="{group_start_x + group_width*0.35:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#4b5563">{seq_len}</text>'
        )
        for series_index, series_name in enumerate(sorted(series)):
            value = next((v for x, v in series[series_name] if x == seq_len), None)
            if value is None:
                continue
            color = SERIES_COLORS.get(series_name, "#374151")
            bar_height = chart_height - (y_pos(value) - top)
            x = group_start_x + series_index * bar_width
            y = y_pos(value)
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-4:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.9"/>'
            )
            if label_lookup is not None:
                label = label_lookup.get((series_name, seq_len))
                if label:
                    svg.append(
                        f'<text x="{x + (bar_width-4)/2:.1f}" y="{max(top+12, y-6):.1f}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#111827">{html.escape(label)}</text>'
                    )

    svg.append("</svg>")
    return "\n".join(svg)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _chart_output_name(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_") + ".svg"


def _render_index_html(study_title: str, sections: list[tuple[str, list[str]]]) -> str:
    lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8' />",
        f"  <title>{html.escape(study_title)}</title>",
        "  <style>body{font-family:sans-serif;margin:32px;background:#f8fafc;color:#111827} h1,h2{margin-bottom:12px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px} figure{margin:0;padding:16px;background:white;border:1px solid #e5e7eb;border-radius:12px} img{max-width:100%;height:auto;display:block} figcaption{margin-top:8px;font-size:14px}</style>",
        "</head>",
        "<body>",
        f"  <h1>{html.escape(study_title)}</h1>",
    ]
    for section_title, image_names in sections:
        if not image_names:
            continue
        lines.append(f"  <h2>{html.escape(section_title)}</h2>")
        lines.append("  <div class='grid'>")
        for image_name in image_names:
            lines.append("    <figure>")
            lines.append(
                f"      <img src='{html.escape(image_name)}' alt='{html.escape(image_name)}' />"
            )
            lines.append(f"      <figcaption>{html.escape(image_name)}</figcaption>")
            lines.append("    </figure>")
        lines.append("  </div>")
    lines.extend(["</body>", "</html>"])
    return "\n".join(lines)


def generate_plots(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    bottleneck_csv: str | Path | None = None,
    gpu_compare_csv: str | Path | None = None,
) -> dict[str, list[str]]:
    suite_rows = _load_csv_rows(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npu_rows = _npu_rows(suite_rows)
    sections: dict[str, list[str]] = {"main": [], "gpu_compare": []}

    main_specs = [
        ("Latency By Sequence Length", "avg_latency_ms", "Latency (ms)"),
        (
            "Throughput By Sequence Length",
            "throughput_flops_per_sec",
            "Throughput (FLOP/s)",
        ),
        ("Power By Sequence Length", "avg_power_w", "Power (W)"),
    ]
    for title, field_name, y_label in main_specs:
        series = _series_by_execution_mode(npu_rows, field_name)
        if not series:
            continue
        file_name = _chart_output_name(title)
        _write_text(
            output_path / file_name,
            _line_chart_svg(
                title=title,
                x_label="Sequence Length",
                y_label=y_label,
                series=series,
            ),
        )
        sections["main"].append(file_name)

    efficiency_series = _series_by_execution_mode(
        _efficiency_rows(npu_rows),
        "flops_per_joule",
    )
    if efficiency_series:
        file_name = _chart_output_name("Energy Efficiency By Sequence Length")
        _write_text(
            output_path / file_name,
            _line_chart_svg(
                title="Energy Efficiency By Sequence Length",
                x_label="Sequence Length",
                y_label="Efficiency (FLOP/J)",
                series=efficiency_series,
            ),
        )
        sections["main"].append(file_name)

    for title, field_name in [
        ("Percent Of Peak By Sequence Length", "backend_pct_of_peak"),
        ("Percent Of Roofline By Sequence Length", "roofline_pct"),
    ]:
        series = _series_by_execution_mode(npu_rows, field_name)
        if not series:
            continue
        file_name = _chart_output_name(title)
        _write_text(
            output_path / file_name,
            _line_chart_svg(
                title=title,
                x_label="Sequence Length",
                y_label="Fraction",
                series=series,
                percent_axis=True,
            ),
        )
        sections["main"].append(file_name)

    if bottleneck_csv is not None and Path(bottleneck_csv).exists():
        bottleneck_rows = _load_csv_rows(bottleneck_csv)
        series, labels = _bottleneck_series(bottleneck_rows)
        if series:
            file_name = _chart_output_name("Bottleneck Breakdown")
            _write_text(
                output_path / file_name,
                _grouped_bar_svg(
                    title="Bottleneck Breakdown",
                    x_label="Sequence Length",
                    y_label="Dominant Component Fraction",
                    series=series,
                    label_lookup=labels,
                    percent_axis=True,
                ),
            )
            sections["main"].append(file_name)

    compare_rows = _best_npu_rows(suite_rows)
    if gpu_compare_csv is not None and Path(gpu_compare_csv).exists():
        compare_rows.extend(_gpu_rows(_load_csv_rows(gpu_compare_csv)))
    else:
        compare_rows.extend(_gpu_rows(suite_rows))

    for title, field_name, y_label in [
        ("Best NPU Vs AMD GPU Latency", "avg_latency_ms", "Latency (ms)"),
        (
            "Best NPU Vs AMD GPU Throughput",
            "throughput_flops_per_sec",
            "Throughput (FLOP/s)",
        ),
    ]:
        series = _series_by_execution_mode(compare_rows, field_name)
        if not series or "amd_gpu_reference" not in series or "best_npu" not in series:
            continue
        file_name = _chart_output_name(title)
        _write_text(
            output_path / file_name,
            _line_chart_svg(
                title=title,
                x_label="Sequence Length",
                y_label=y_label,
                series={key: series[key] for key in ("best_npu", "amd_gpu_reference")},
            ),
        )
        sections["gpu_compare"].append(file_name)

    index_html = _render_index_html(
        study_title=f"Transformer Layer Thesis Plots: {Path(input_csv).stem}",
        sections=[
            ("Main NPU Study", sections["main"]),
            ("Best NPU vs AMD GPU", sections["gpu_compare"]),
        ],
    )
    _write_text(output_path / "index.html", index_html)
    return sections


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis-style SVG/HTML plots for transformer_layer study results."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bottleneck-csv", default=None)
    parser.add_argument("--gpu-compare-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        bottleneck_csv=args.bottleneck_csv,
        gpu_compare_csv=args.gpu_compare_csv,
    )


if __name__ == "__main__":
    main()
