#!/usr/bin/env bash
set -euo pipefail

echo "1. Source XRT and ironenv."
echo "   source /opt/xilinx/xrt/setup.sh && source ./ironenv/bin/activate"
echo
echo "2. Verify focused transformer_layer tests pass."
echo "   python -m pytest -q iron/applications/transformer_layer"
echo
echo "3. Run a one-shot synthetic smoke for one design pattern."
echo "   python iron/applications/transformer_layer/npu_inference.py --execution-mode encoder_pipeline --seq-len 64 --warmup-runs 1 --runs-per-sample 1"
echo
echo "4. Run the manifest-driven main study."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json"
echo
echo "5. Generate or update the peak-reference artifact before roofline annotation."
echo "   python iron/applications/transformer_layer/calibrate_backend_peaks.py --backend npu --peak-ops-per-sec 1.0 --peak-bytes-per-sec 1.0 --output iron/applications/transformer_layer/config/peak_references.json --append"
echo
echo "6. Capture structured programmability/debug events during a study run."
echo "   python iron/applications/transformer_layer/automated_benchmark.py --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json --debug-log-csv iron/applications/transformer_layer/results/design_patterns_main_debug_log.csv"
echo
echo "7. For unattended execution, resolve the systemd job command first."
echo "   python iron/applications/transformer_layer/run_automated_benchmark_job.py iron/applications/transformer_layer/systemd/benchmark_job.example.json --print-command"
