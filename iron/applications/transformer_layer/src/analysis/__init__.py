# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .peak_reference import (
    PEAK_REFERENCE_ARTIFACT_VERSION,
    BackendPeakReference,
    load_peak_reference,
    load_peak_references,
    save_peak_reference,
    save_peak_references,
    upsert_peak_reference,
)
from .bottlenecks import (
    COMPONENT_FIELDS,
    SUMMARY_FIELD_ORDER,
    analyze_results,
    build_execution_mode_summary,
    build_row_summary,
    render_execution_mode_summaries,
)
from .gpu_compare_best_npu import (
    benchmark_best_npu_vs_gpu,
    load_compare_config,
    select_best_npu_rows,
)
from .plot_design_pattern_results import SERIES_COLORS, generate_plots
from .roofline import (
    annotate_result_row_with_peak,
    annotate_results_csv,
    estimate_layer_bytes,
    estimate_layer_flops,
    flops_per_joule,
    gflops_per_joule,
    operational_intensity,
    roofline_bound_from_metrics,
    roofline_bound_ops_per_sec,
)
from .support_matrix import (
    SUPPORT_MATRIX_FIELD_ORDER,
    render_support_summary_text,
    summarize_support_rows,
)

__all__ = [
    "PEAK_REFERENCE_ARTIFACT_VERSION",
    "BackendPeakReference",
    "save_peak_reference",
    "load_peak_reference",
    "save_peak_references",
    "load_peak_references",
    "upsert_peak_reference",
    "COMPONENT_FIELDS",
    "SUMMARY_FIELD_ORDER",
    "build_row_summary",
    "build_execution_mode_summary",
    "render_execution_mode_summaries",
    "analyze_results",
    "load_compare_config",
    "select_best_npu_rows",
    "benchmark_best_npu_vs_gpu",
    "SERIES_COLORS",
    "generate_plots",
    "estimate_layer_flops",
    "estimate_layer_bytes",
    "operational_intensity",
    "roofline_bound_ops_per_sec",
    "roofline_bound_from_metrics",
    "flops_per_joule",
    "gflops_per_joule",
    "annotate_result_row_with_peak",
    "annotate_results_csv",
    "SUPPORT_MATRIX_FIELD_ORDER",
    "summarize_support_rows",
    "render_support_summary_text",
]
