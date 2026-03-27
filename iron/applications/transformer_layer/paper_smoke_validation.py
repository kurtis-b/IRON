#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iron.applications.transformer_layer.analyze_design_pattern_bottlenecks import (
    analyze_results,
)
from iron.applications.transformer_layer.benchmark_common import (
    load_study_manifest,
    write_dict_rows_csv,
    write_results_csv,
)
from iron.applications.transformer_layer.peak_reference import (
    BackendPeakReference,
    save_peak_references,
)
from iron.applications.transformer_layer.plot_design_pattern_results import (
    generate_plots,
)
from iron.applications.transformer_layer.roofline import (
    annotate_results_csv,
    flops_per_joule,
    gflops_per_joule,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def _latency_row(
    *,
    study_id: str,
    backend: str,
    execution_mode: str,
    seq_len: int,
    avg_latency_ms: float,
    estimated_flops: float,
    estimated_bytes: float,
    avg_power_w: float | None,
    **extra: object,
) -> dict[str, object]:
    measured_inference_count = 10
    timed_total_sec = (avg_latency_ms / 1000.0) * measured_inference_count
    row: dict[str, object] = {
        "study_id": study_id,
        "backend": backend,
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": seq_len,
        "batch_size": 1,
        "dtype": "bfloat16",
        "use_bias": False,
        "weights_source": "synthetic",
        "source_model_name": None,
        "source_layer_index": None,
        "warmup_runs": 2,
        "runs_per_sample": measured_inference_count,
        "measured_inference_count": measured_inference_count,
        "timed_total_sec": timed_total_sec,
        "avg_latency_ms": avg_latency_ms,
        "throughput_flops_per_sec": estimated_flops / (avg_latency_ms / 1000.0),
        "estimated_flops_per_inference": estimated_flops,
        "estimated_bytes_per_inference": estimated_bytes,
        "operational_intensity_flops_per_byte": estimated_flops / estimated_bytes,
        "avg_power_w": avg_power_w,
        "max_power_w": avg_power_w,
        "energy_j": None if avg_power_w is None else avg_power_w * timed_total_sec,
        "flops_per_joule": flops_per_joule(
            throughput_flops_per_sec=estimated_flops / (avg_latency_ms / 1000.0),
            avg_power_w=avg_power_w,
        ),
        "gflops_per_joule": gflops_per_joule(
            throughput_flops_per_sec=estimated_flops / (avg_latency_ms / 1000.0),
            avg_power_w=avg_power_w,
        ),
        "power_sample_count": 4 if avg_power_w is not None else None,
    }
    row.update(extra)
    return row


def _fixture_suite_rows(study_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seq_len, scale in ((64, 1.0), (128, 2.0)):
        estimated_flops = 1.0e8 * scale
        estimated_bytes = 5.0e7 * scale
        rows.append(
            _latency_row(
                study_id=study_id,
                backend="npu",
                execution_mode="encoder_pipeline",
                seq_len=seq_len,
                avg_latency_ms=4.2 * scale,
                estimated_flops=estimated_flops,
                estimated_bytes=estimated_bytes,
                avg_power_w=14.0,
                compile_setup_time_ms=35.0,
                avg_encoder_pipeline_latency_ms=3.7 * scale,
                npu_dispatch_count=1,
                npu_unique_instruction_binary_count=1,
                topology_id="seq32_kv64__ps1_ph1_pffn1",
                topology_family="seq32_kv64",
                parallel_seq=1,
                parallel_heads=1,
                parallel_ffn=1,
                compute_tile_count=16,
                compute_tile_utilization_fraction=0.5,
                process_model="in_process",
            )
        )
        rows.append(
            _latency_row(
                study_id=study_id,
                backend="npu",
                execution_mode="gemm_only",
                seq_len=seq_len,
                avg_latency_ms=18.0 * scale,
                estimated_flops=estimated_flops,
                estimated_bytes=estimated_bytes,
                avg_power_w=13.0,
                compile_setup_time_ms=55.0,
                avg_npu_gemm_latency_ms=10.0 * scale,
                avg_host_preprocess_latency_ms=1.1 * scale,
                avg_host_postprocess_latency_ms=6.1 * scale,
                avg_device_sync_latency_ms=0.3 * scale,
                npu_dispatch_count=27,
                npu_unique_instruction_binary_count=4,
                process_model="in_process",
            )
        )
        rows.append(
            _latency_row(
                study_id=study_id,
                backend="npu",
                execution_mode="operator_runlist",
                seq_len=seq_len,
                avg_latency_ms=20.0 * scale,
                estimated_flops=estimated_flops,
                estimated_bytes=estimated_bytes,
                avg_power_w=14.5,
                compile_setup_time_ms=40.0,
                avg_operator_runlist_latency_ms=17.8 * scale,
                avg_host_preprocess_latency_ms=0.8 * scale,
                avg_host_postprocess_latency_ms=0.0,
                avg_device_sync_latency_ms=0.2 * scale,
                npu_dispatch_count=12,
                npu_unique_instruction_binary_count=1,
                process_model="child_process",
            )
        )
    return rows


def _fixture_gpu_rows(study_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seq_len, scale in ((64, 1.0), (128, 2.0)):
        estimated_flops = 1.0e8 * scale
        estimated_bytes = 5.0e7 * scale
        rows.append(
            _latency_row(
                study_id=study_id,
                backend="gpu",
                execution_mode="amd_gpu_reference",
                seq_len=seq_len,
                avg_latency_ms=1.2 * scale,
                estimated_flops=estimated_flops,
                estimated_bytes=estimated_bytes,
                avg_power_w=28.0,
                gpu_device="cuda:0",
                gpu_power_backend="rocm-smi",
            )
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_paper_smoke_validation(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "paper_smoke_manifest.json"
    peak_reference_path = root / "peak_references.json"
    suite_csv = root / "paper_smoke_suite.csv"
    annotated_csv = root / "paper_smoke_suite_annotated.csv"
    bottleneck_csv = root / "paper_smoke_bottlenecks.csv"
    bottleneck_json = root / "paper_smoke_bottlenecks.json"
    bottleneck_text = root / "paper_smoke_bottlenecks.txt"
    gpu_compare_csv = root / "paper_smoke_gpu_compare.csv"
    plots_dir = root / "plots"
    summary_json = root / "paper_smoke_summary.json"

    manifest_payload = {
        "study_id": "paper_smoke",
        "layer_spec": TransformerLayerSpec(
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            seq_len=64,
            use_bias=False,
            weights_source="synthetic",
        ).to_dict(),
        "execution_modes": "encoder_pipeline,gemm_only,operator_runlist",
        "seq_lens": "64,128",
        "warmup_runs": 2,
        "runs_per_sample": 10,
        "seed": 0,
        "output_csv": "paper_smoke_suite.csv",
        "peak_reference": "peak_references.json",
        "annotated_output_csv": "paper_smoke_suite_annotated.csv",
    }
    _write_json(manifest_path, manifest_payload)
    manifest = load_study_manifest(manifest_path)

    save_peak_references(
        peak_reference_path,
        [
            BackendPeakReference(
                backend="gpu",
                peak_ops_per_sec=2.0e11,
                peak_bytes_per_sec=6.0e10,
                source_note="paper_smoke_fixture",
            ),
            BackendPeakReference(
                backend="npu",
                peak_ops_per_sec=8.0e10,
                peak_bytes_per_sec=2.0e10,
                source_note="paper_smoke_fixture",
            ),
        ],
    )

    study_id = str(manifest["study_id"])
    write_results_csv(suite_csv, _fixture_suite_rows(study_id))
    write_results_csv(gpu_compare_csv, _fixture_gpu_rows(study_id))
    annotated_rows = annotate_results_csv(suite_csv, peak_reference_path, annotated_csv)
    summary_rows, aggregates, text_summary = analyze_results(annotated_csv)
    write_dict_rows_csv(
        bottleneck_csv,
        summary_rows,
        fieldnames=[
            "study_id",
            "execution_mode",
            "seq_len",
            "avg_latency_ms",
            "dominant_component",
            "dominant_component_latency_ms",
            "dominant_component_fraction",
            "secondary_component",
            "secondary_component_latency_ms",
            "npu_dispatch_count",
            "npu_unique_instruction_binary_count",
            "topology_id",
            "topology_family",
            "parallel_seq",
            "parallel_heads",
            "parallel_ffn",
            "compute_tile_count",
            "compute_tile_utilization_fraction",
            "process_model",
            "bottleneck_summary",
        ],
    )
    _write_json(bottleneck_json, aggregates)
    bottleneck_text.write_text(text_summary, encoding="utf-8")
    plot_sections = generate_plots(
        input_csv=annotated_csv,
        output_dir=plots_dir,
        bottleneck_csv=bottleneck_csv,
        gpu_compare_csv=gpu_compare_csv,
    )

    summary = {
        "study_id": study_id,
        "manifest_path": str(manifest_path),
        "suite_csv": str(suite_csv),
        "annotated_csv": str(annotated_csv),
        "bottleneck_csv": str(bottleneck_csv),
        "gpu_compare_csv": str(gpu_compare_csv),
        "plots_dir": str(plots_dir),
        "annotated_row_count": len(annotated_rows),
        "bottleneck_row_count": len(summary_rows),
        "plot_sections": plot_sections,
        "execution_modes": list(manifest["execution_modes"]),
        "seq_lens": list(manifest["seq_lens"]),
    }
    _write_json(summary_json, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a pure-Python paper-smoke validation for the transformer-layer "
            "study workflow."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="iron/applications/transformer_layer/results/paper_smoke_validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_paper_smoke_validation(args.output_dir)
    print(f"paper_smoke_validation: wrote outputs under {summary['plots_dir']}")
    print(
        "paper_smoke_validation: "
        f"{summary['annotated_row_count']} annotated rows, "
        f"{summary['bottleneck_row_count']} bottleneck rows"
    )


if __name__ == "__main__":
    main()
