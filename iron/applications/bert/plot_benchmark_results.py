#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "automated_benchmark_all_studies.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
MODE_ORDER = ("cpu", "npu", "igpu")
MODE_LABELS = {
    "cpu": "CPU",
    "npu": "NPU",
    "igpu": "iGPU",
}
MODE_COLORS = {
    "cpu": "#264653",
    "npu": "#E76F51",
    "igpu": "#2A9D8F",
}
MODEL_COLORS = [
    "#C1121F",
    "#003049",
    "#669BBC",
    "#9C6644",
    "#6A994E",
    "#7B2CBF",
]
BACKGROUND_COLOR = "#F5F0E8"
PANEL_COLOR = "#FFFDF8"
GRID_COLOR = "#D8CCBF"
TEXT_COLOR = "#302821"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render publication-style benchmark plots from an automated benchmark CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the suite CSV produced by automated_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plot images and the HTML index into.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Raster DPI for generated PNG files.",
    )
    return parser.parse_args()


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": PANEL_COLOR,
            "savefig.facecolor": BACKGROUND_COLOR,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def parse_numeric(raw_value, cast):
    if raw_value in ("", None):
        return None
    return cast(raw_value)


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw in reader:
            row = dict(raw)
            row["seq_len"] = int(raw["seq_len"])
            row["num_threads"] = parse_numeric(raw.get("num_threads"), int)
            for key in (
                "avg_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "avg_pkg_watt",
                "pseudo_device_avg_pkg_watt",
                "throughput_flops_per_sec",
                "estimated_gflops_per_watt_sec",
                "pseudo_device_estimated_gflops_per_watt_sec",
                "estimated_flops_per_joule",
                "pseudo_device_estimated_flops_per_joule",
                "backend_pct_of_peak",
                "operational_intensity_flops_per_byte",
                "roofline_pct",
                "backend_peak_ops_per_sec",
                "ddr_peak_bytes_per_sec",
                "roofline_bound_ops_per_sec",
            ):
                row[key] = parse_numeric(raw.get(key), float)
            rows.append(row)
    return rows


def sort_by_seq(rows):
    return sorted(rows, key=lambda row: row["seq_len"])


def group_by(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def best_cpu_rows(rows):
    best = {}
    for row in rows:
        if row["mode"] != "cpu":
            continue
        key = (row["study_id"], row["seq_len"])
        current = best.get(key)
        if current is None or row["avg_latency_ms"] < current["avg_latency_ms"]:
            best[key] = row
    return best


def representative_rows_for_study(rows, study_id):
    cpu_best = best_cpu_rows(rows)
    selected = []
    for mode in MODE_ORDER:
        study_rows = [
            row for row in rows if row["study_id"] == study_id and row["mode"] == mode
        ]
        if mode == "cpu":
            study_rows = [
                cpu_best[(study_id, seq_len)]
                for seq_len in sorted({row["seq_len"] for row in study_rows})
            ]
        selected.extend(sort_by_seq(study_rows))
    return selected


def representative_rows(rows):
    cpu_best = best_cpu_rows(rows)
    selected = []
    for row in rows:
        if row["mode"] == "cpu":
            key = (row["study_id"], row["seq_len"])
            if cpu_best.get(key) is row:
                selected.append(row)
            continue
        selected.append(row)
    return selected


def cpu_thread_rows_for_study(rows, study_id):
    cpu_rows = [
        row for row in rows if row["study_id"] == study_id and row["mode"] == "cpu"
    ]
    grouped = group_by(cpu_rows, key_fn=lambda row: row["num_threads"])
    return {thread_count: sort_by_seq(group) for thread_count, group in grouped.items()}


def throughput_gflops_per_sec(row):
    if row["throughput_flops_per_sec"] is None:
        return None
    return row["throughput_flops_per_sec"] * 1e-9


def effective_power_watts(row):
    if row["mode"] == "cpu":
        return row["avg_pkg_watt"]
    return row["pseudo_device_avg_pkg_watt"] or row["avg_pkg_watt"]


def effective_efficiency_gflops_per_watt_sec(row):
    if row["mode"] == "cpu":
        if row["estimated_gflops_per_watt_sec"] is not None:
            return row["estimated_gflops_per_watt_sec"]
        if row["estimated_flops_per_joule"] is not None:
            return row["estimated_flops_per_joule"] * 1e-9
        return None
    if row["pseudo_device_estimated_gflops_per_watt_sec"] is not None:
        return row["pseudo_device_estimated_gflops_per_watt_sec"]
    if row["pseudo_device_estimated_flops_per_joule"] is not None:
        return row["pseudo_device_estimated_flops_per_joule"] * 1e-9
    return row["estimated_gflops_per_watt_sec"]


def backend_pct_of_peak_percent(row):
    if row.get("backend_pct_of_peak") is None:
        return None
    return row["backend_pct_of_peak"] * 100.0


def roofline_pct_percent(row):
    if row.get("roofline_pct") is None:
        return None
    return row["roofline_pct"] * 100.0


def finite_series(rows, value_fn):
    xs = []
    ys = []
    for row in rows:
        value = value_fn(row)
        if value is None:
            continue
        xs.append(row["seq_len"])
        ys.append(value)
    return xs, ys


def apply_seq_axis(ax, seq_lens):
    if not seq_lens:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(
            lambda value, _pos: f"{int(value)}" if value in seq_lens else ""
        )
    )


def beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both")


def model_color_map(rows):
    studies = sorted({row["study_id"] for row in rows})
    return {
        study_id: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, study_id in enumerate(studies)
    }


def plot_mode_lines(ax, study_rows, value_fn, ylabel, title, *, legend=True):
    seq_lens = sorted({row["seq_len"] for row in study_rows})
    for mode in MODE_ORDER:
        mode_rows = [row for row in study_rows if row["mode"] == mode]
        if not mode_rows:
            continue
        xs, ys = finite_series(mode_rows, value_fn)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2.6,
            marker="o",
            markersize=5.5,
        )
        ax.fill_between(xs, ys, [0] * len(ys), color=MODE_COLORS[mode], alpha=0.08)
    apply_seq_axis(ax, seq_lens)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    beautify_axes(ax)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")


def geometric_mean(values):
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def has_metric(rows, value_fn):
    for row in rows:
        value = value_fn(row)
        if value is not None:
            return True
    return False


def plot_cross_model_metric(rows, out_dir, dpi, *, filename, title, value_fn, ylabel):
    rep_rows = representative_rows(rows)
    if not has_metric(rep_rows, value_fn):
        return None
    study_colors = model_color_map(rep_rows)
    seq_lens = sorted({row["seq_len"] for row in rep_rows})

    fig, axes = plt.subplots(1, len(MODE_ORDER), figsize=(18, 6.2), sharex=True)
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.06, ha="left")

    for ax, mode in zip(axes, MODE_ORDER):
        mode_rows = [row for row in rep_rows if row["mode"] == mode]
        studies = sorted({row["study_id"] for row in mode_rows})
        for study_id in studies:
            study_rows = sort_by_seq(
                [row for row in mode_rows if row["study_id"] == study_id]
            )
            xs, ys = finite_series(study_rows, value_fn)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                linewidth=2.4,
                marker="o",
                markersize=5,
                color=study_colors[study_id],
                label=study_id,
            )
        apply_seq_axis(ax, seq_lens)
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Sequence Length")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        beautify_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(labels)),
            bbox_to_anchor=(0.5, 0.98),
        )

    output_path = out_dir / filename
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_study_dashboard(study_id, rows, out_dir, dpi):
    study_rows = representative_rows_for_study(rows, study_id)
    if not study_rows:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{study_id} Benchmark Overview",
        fontsize=18,
        fontweight="bold",
        x=0.07,
        ha="left",
    )

    plot_mode_lines(
        axes[0, 0],
        study_rows,
        lambda row: row["avg_latency_ms"],
        "Latency (ms)",
        "Average Latency",
    )
    plot_mode_lines(
        axes[0, 1],
        study_rows,
        throughput_gflops_per_sec,
        "Throughput (GFLOPs/s)",
        "Throughput",
    )
    plot_mode_lines(
        axes[1, 0],
        study_rows,
        effective_power_watts,
        "Power (W)",
        "Effective Power",
    )
    plot_mode_lines(
        axes[1, 1],
        study_rows,
        effective_efficiency_gflops_per_watt_sec,
        "Efficiency (GFLOPs/W*s)",
        "Effective Efficiency",
    )

    fig.text(
        0.07,
        0.03,
        "CPU uses package power/efficiency. NPU and iGPU use idle-subtracted pseudo-device values when available.",
        fontsize=10,
        color="#6A5C50",
    )

    output_path = out_dir / f"{study_id}_overview.png"
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_cpu_thread_comparison(study_id, rows, out_dir, dpi):
    thread_groups = cpu_thread_rows_for_study(rows, study_id)
    if len(thread_groups) <= 1:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    seq_lens = sorted(
        {
            row["seq_len"]
            for rows_by_thread in thread_groups.values()
            for row in rows_by_thread
        }
    )
    thread_palette = ["#3D405B", "#81B29A", "#F2CC8F", "#E07A5F"]
    for color, (thread_count, thread_rows) in zip(
        thread_palette, sorted(thread_groups.items())
    ):
        xs, ys = finite_series(thread_rows, lambda row: row["avg_latency_ms"])
        ax.plot(
            xs,
            ys,
            label=f"CPU {thread_count} threads",
            linewidth=2.4,
            marker="o",
            markersize=5,
            color=color,
        )

    apply_seq_axis(ax, seq_lens)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{study_id} CPU Thread Comparison")
    beautify_axes(ax)
    ax.legend(loc="best")

    output_path = out_dir / f"{study_id}_cpu_threads.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_summary(rows, out_dir, dpi):
    studies = sorted({row["study_id"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Benchmark Summary", fontsize=18, fontweight="bold", x=0.06, ha="left")

    latency_ax, efficiency_ax = axes
    group_width = 0.24
    x_positions = list(range(len(studies)))

    for idx, mode in enumerate(MODE_ORDER):
        latency_values = []
        efficiency_values = []
        for study_id in studies:
            study_rows = representative_rows_for_study(rows, study_id)
            mode_rows = [row for row in study_rows if row["mode"] == mode]
            latency_values.append(
                geometric_mean([row["avg_latency_ms"] for row in mode_rows])
                if mode_rows
                else None
            )
            efficiency_values.append(
                geometric_mean(
                    [
                        value
                        for row in mode_rows
                        if (value := effective_efficiency_gflops_per_watt_sec(row))
                        is not None
                    ]
                )
                if mode_rows
                else None
            )

        bar_x = [position + (idx - 1) * group_width for position in x_positions]
        latency_ax.bar(
            bar_x,
            [value or 0.0 for value in latency_values],
            width=group_width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            alpha=0.92,
        )
        efficiency_ax.bar(
            bar_x,
            [value or 0.0 for value in efficiency_values],
            width=group_width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            alpha=0.92,
        )

    for ax, ylabel, title in (
        (latency_ax, "Geometric Mean Latency (ms)", "Latency by Study"),
        (
            efficiency_ax,
            "Geometric Mean Efficiency (GFLOPs/W*s)",
            "Efficiency by Study",
        ),
    ):
        ax.set_xticks(x_positions)
        ax.set_xticklabels(studies, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        beautify_axes(ax)
        ax.legend(loc="upper left")

    summary_path = out_dir / "summary_overview.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(summary_path, dpi=dpi)
    plt.close(fig)
    return summary_path


def plot_grouped_comparisons(rows, out_dir, dpi):
    image_paths = [
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_latency_by_backend.png",
            title="Cross-Model Latency Comparison",
            value_fn=lambda row: row["avg_latency_ms"],
            ylabel="Latency (ms)",
        ),
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_throughput_by_backend.png",
            title="Cross-Model Throughput Comparison",
            value_fn=throughput_gflops_per_sec,
            ylabel="Throughput (GFLOPs/s)",
        ),
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_power_by_backend.png",
            title="Cross-Model Power Comparison",
            value_fn=effective_power_watts,
            ylabel="Power (W)",
        ),
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_efficiency_by_backend.png",
            title="Cross-Model Efficiency Comparison",
            value_fn=effective_efficiency_gflops_per_watt_sec,
            ylabel="Efficiency (GFLOPs/W*s)",
        ),
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_pct_of_peak_by_backend.png",
            title="Cross-Model Percent-of-Peak Comparison",
            value_fn=backend_pct_of_peak_percent,
            ylabel="Percent of Peak (%)",
        ),
        plot_cross_model_metric(
            rows,
            out_dir,
            dpi,
            filename="grouped_roofline_pct_by_backend.png",
            title="Cross-Model Roofline Utilization Comparison",
            value_fn=roofline_pct_percent,
            ylabel="Roofline Utilization (%)",
        ),
    ]
    return [path for path in image_paths if path is not None]


def plot_roofline_overview(rows, out_dir, dpi):
    rep_rows = representative_rows(rows)
    eligible_rows = [
        row
        for row in rep_rows
        if row.get("operational_intensity_flops_per_byte") is not None
        and row.get("throughput_flops_per_sec") is not None
    ]
    if not eligible_rows:
        return None

    study_colors = model_color_map(rep_rows)
    fig, axes = plt.subplots(1, len(MODE_ORDER), figsize=(18, 6.2), sharey=True)
    fig.suptitle("Roofline Overview", fontsize=18, fontweight="bold", x=0.06, ha="left")

    for ax, mode in zip(axes, MODE_ORDER):
        mode_rows = [row for row in eligible_rows if row["mode"] == mode]
        if not mode_rows:
            continue
        for row in mode_rows:
            ax.scatter(
                row["operational_intensity_flops_per_byte"],
                throughput_gflops_per_sec(row),
                color=study_colors[row["study_id"]],
                s=48,
                alpha=0.9,
                edgecolors="none",
            )

        peak_ops = max(
            (
                row["backend_peak_ops_per_sec"]
                for row in mode_rows
                if row.get("backend_peak_ops_per_sec") is not None
            ),
            default=None,
        )
        ddr_bytes = max(
            (
                row["ddr_peak_bytes_per_sec"]
                for row in mode_rows
                if row.get("ddr_peak_bytes_per_sec") is not None
            ),
            default=None,
        )
        if peak_ops is not None and ddr_bytes is not None:
            oi_values = [
                row["operational_intensity_flops_per_byte"]
                for row in mode_rows
                if row.get("operational_intensity_flops_per_byte") is not None
                and row["operational_intensity_flops_per_byte"] > 0.0
            ]
            if oi_values:
                min_oi = min(oi_values) / 2.0
                max_oi = max(oi_values) * 2.0
                if min_oi <= 0.0:
                    min_oi = min(oi_values)
                roof_xs = [
                    min_oi * ((max_oi / min_oi) ** (index / 63.0))
                    for index in range(64)
                ]
                peak_gflops = peak_ops * 1e-9
                ddr_gbytes = ddr_bytes * 1e-9
                roof_ys = [min(peak_gflops, ddr_gbytes * value) for value in roof_xs]
                ax.plot(
                    roof_xs,
                    roof_ys,
                    linestyle="--",
                    linewidth=2.0,
                    color=MODE_COLORS[mode],
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Operational Intensity (FLOPs/byte)")
        if ax is axes[0]:
            ax.set_ylabel("Throughput (GFLOPs/s)")
        beautify_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=study_id)
        for study_id, color in study_colors.items()
    ]
    if handles:
        fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="upper center",
            ncol=min(3, len(handles)),
            bbox_to_anchor=(0.5, 0.98),
        )

    output_path = out_dir / "roofline_overview.png"
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_html_index(out_dir, image_paths):
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>BERT Benchmark Plots</title>",
        "  <style>",
        "    body { font-family: 'DejaVu Sans', sans-serif; margin: 2rem; background: #f5f0e8; color: #302821; }",
        "    h1, h2 { margin-bottom: 0.4rem; }",
        "    p { max-width: 60rem; }",
        "    .plot { margin: 1.5rem 0 2.5rem; padding: 1rem; background: #fffdf8; border-radius: 14px; box-shadow: 0 8px 24px rgba(48, 40, 33, 0.08); }",
        "    img { width: 100%; max-width: 1200px; border-radius: 10px; display: block; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>BERT Benchmark Plot Report</h1>",
        "  <p>Rendered from the automated benchmark suite CSV. CPU uses package power/efficiency. "
        "NPU and iGPU use idle-subtracted pseudo-device values when available. Percent-of-peak and "
        "roofline views appear when the suite CSV includes peak-reference annotations.</p>",
    ]
    for image_path in image_paths:
        lines.extend(
            [
                '  <div class="plot">',
                f"    <h2>{image_path.stem.replace('_', ' ').title()}</h2>",
                f'    <img src="{image_path.name}" alt="{image_path.stem}">',
                "  </div>",
            ]
        )
    lines.extend(["</body>", "</html>"])
    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main():
    args = parse_args()
    configure_matplotlib()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    if args.output_dir is None:
        output_dir = (DEFAULT_OUTPUT_DIR / input_csv.stem).resolve()
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    image_paths = []
    summary_path = plot_summary(rows, output_dir, args.dpi)
    image_paths.append(summary_path)
    image_paths.extend(plot_grouped_comparisons(rows, output_dir, args.dpi))
    roofline_path = plot_roofline_overview(rows, output_dir, args.dpi)
    if roofline_path is not None:
        image_paths.append(roofline_path)

    for study_id in sorted({row["study_id"] for row in rows}):
        overview_path = plot_study_dashboard(study_id, rows, output_dir, args.dpi)
        if overview_path is not None:
            image_paths.append(overview_path)
        cpu_threads_path = plot_cpu_thread_comparison(
            study_id, rows, output_dir, args.dpi
        )
        if cpu_threads_path is not None:
            image_paths.append(cpu_threads_path)

    index_path = write_html_index(output_dir, image_paths)

    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"HTML index: {index_path}")
    print("Generated plots:")
    for image_path in image_paths:
        print(f"  - {image_path}")


if __name__ == "__main__":
    main()
