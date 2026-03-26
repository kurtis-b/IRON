#!/usr/bin/env bash
set -euo pipefail

echo "1. Source XRT and ironenv."
echo "   source /opt/xilinx/xrt/setup.sh && source ./ironenv/bin/activate"
echo
echo "2. Verify focused transformer_layer tests pass."
echo "   python -m pytest -q iron/applications/transformer_layer"
echo
echo "3. Run the pure-Python paper-smoke validation target."
echo "   python -m iron.applications.transformer_layer.paper_smoke_validation --output-dir iron/applications/transformer_layer/results/paper_smoke_validation"
echo
echo "4. Run a one-shot synthetic smoke for one design pattern."
echo "   python iron/applications/transformer_layer/npu_inference.py --execution-mode encoder_pipeline --seq-len 64 --warmup-runs 1 --runs-per-sample 1"
echo
echo "5. Run the manifest-driven main study."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json"
echo
echo "6. Generate or update the peak-reference artifact before roofline annotation."
echo "   python iron/applications/transformer_layer/calibrate_backend_peaks.py --backend npu --peak-ops-per-sec 1.0 --peak-bytes-per-sec 1.0 --output iron/applications/transformer_layer/config/peak_references.json --append"
echo
echo "7. Capture structured programmability/debug events during a study run."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json --debug-log-csv iron/applications/transformer_layer/results/design_patterns_main_debug_log.csv"
echo
echo "8. Generate thesis plots from the annotated suite, bottlenecks, and AMD GPU comparison."
echo "   python -m iron.applications.transformer_layer.plot_design_pattern_results --input-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv --bottleneck-csv iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.csv --gpu-compare-csv iron/applications/transformer_layer/results/gpu_compare_amd.csv --output-dir iron/applications/transformer_layer/results/plots/design_patterns_main"
echo
echo "9. For unattended execution, resolve the systemd job command first."
echo "   python iron/applications/transformer_layer/run_automated_benchmark_job.py iron/applications/transformer_layer/systemd/benchmark_job.example.json --print-command"
