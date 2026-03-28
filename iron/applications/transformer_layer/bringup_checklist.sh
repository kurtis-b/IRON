#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "1. Source XRT and ironenv."
echo "   source /opt/xilinx/xrt/setup.sh && source ./ironenv/bin/activate"
echo
echo "2. Verify focused transformer_layer tests pass."
echo "   python -m pytest -q iron/applications/transformer_layer/test_*.py"
echo
echo "3. Run a one-shot synthetic smoke for one DesignPats design pattern."
echo "   python iron/applications/transformer_layer/npu_inference.py --execution-mode dataflow --seq-len 64 --warmup-runs 1 --runs-per-sample 1"
echo
echo "4. Run the block study."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/dataflow_blocks.json"
echo
echo "5. Run the manifest-driven end-to-end study."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_end_to_end.json"
echo
echo "6. Generate or update the peak-reference artifact before roofline annotation."
echo "   python iron/applications/transformer_layer/calibrate_backend_peaks.py --backend npu --peak-ops-per-sec 1.0 --peak-bytes-per-sec 1.0 --output iron/applications/transformer_layer/config/peak_references.json --append"
echo
echo "7. Capture structured programmability/debug events during a study run."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_end_to_end.json --debug-log-csv iron/applications/transformer_layer/results/design_patterns_end_to_end_debug_log.csv"
echo
echo "8. Generate thesis plots from the annotated suite, bottlenecks, and iGPU comparison."
echo "   python -m iron.applications.transformer_layer.plot_design_pattern_results --input-csv iron/applications/transformer_layer/results/design_patterns_end_to_end_annotated.csv --bottleneck-csv iron/applications/transformer_layer/results/design_patterns_end_to_end_bottlenecks.csv --gpu-compare-csv iron/applications/transformer_layer/results/gpu_compare_end_to_end_igpu.csv --output-dir iron/applications/transformer_layer/results/plots/design_patterns_end_to_end"
echo
echo "9. For unattended execution, resolve the systemd job command first."
echo "   python iron/applications/transformer_layer/run_automated_benchmark_job.py iron/applications/transformer_layer/systemd/benchmark_job.example.json --print-command"
