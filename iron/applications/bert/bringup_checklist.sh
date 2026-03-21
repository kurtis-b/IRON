#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
APP_DIR="$SCRIPT_DIR"
DEFAULT_LOGS_DIR="$APP_DIR/logs/preflight"
DEFAULT_TOPOLOGY_CACHE="$APP_DIR/npu_topology_cache_latest.json"
DEFAULT_JOB_CONFIG="$APP_DIR/systemd/benchmark_job.json"
EXPECTED_MLIR_AIE_VERSION_DEFAULT="0.0.1.2026031811+71fb44f147"
POWERCAP_HELPER="$APP_DIR/read_powercap_rapl.py"

PHASES=(
  "env"
  "download"
  "cpu-smoke"
  "npu-smoke"
  "autotune"
  "power"
  "suite"
  "job-print"
)

STUDY_ID="bert-base-uncased"
STUDY_MANIFEST=""
MODELS_ROOT=""
WEIGHTS_FILE_PATH=""
CONFIG_FILE_PATH=""
XRT_SETUP_PATH="/opt/xilinx/xrt/setup.sh"
IRONENV_PATH="$REPO_ROOT/ironenv"
EXPECTED_MLIR_AIE_VERSION="$EXPECTED_MLIR_AIE_VERSION_DEFAULT"
CPU_SMOKE_SEQ_LENS="64"
NPU_SMOKE_SEQ_LENS="64,512"
AUTOTUNE_SEQ_LENS="64,128,256,512,1024,2048,4096,8192"
SUITE_SEQ_LENS="64,512"
SMOKE_NUM_SAMPLES="1"
SMOKE_WARMUP_RUNS="0"
SMOKE_RUNS_PER_SAMPLE="1"
SUITE_NUM_SAMPLES="1"
SUITE_WARMUP_RUNS="10"
SUITE_RUNS_PER_SAMPLE="100"
CPU_THREAD_COUNTS=""
NPU_NUM_THREADS=""
SUITE_POWER_BACKEND="auto"
POWER_INTERVAL_SEC="0.5"
NPU_IDLE_BASELINE_SEC="5.0"
TOPOLOGY_CACHE="$DEFAULT_TOPOLOGY_CACHE"
LOGS_DIR="$DEFAULT_LOGS_DIR"
JOB_CONFIG="$DEFAULT_JOB_CONFIG"
START_AT="env"
STOP_AFTER="job-print"
SKIP_DOWNLOAD="0"
SKIP_AUTOTUNE="0"
SKIP_POWER_CHECK="0"
SKIP_SUITE="0"
SKIP_JOB_PRINT="0"
DRY_RUN="0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the BERT automated-benchmark bring-up checklist on a target machine.

Model selection:
  --study-id ID                  Study manifest id to use. Default: $STUDY_ID
  --study-manifest PATH          Optional study manifest JSON.
  --models-root PATH             Optional models root for downloaded artifacts.
  --weights-file-path PATH       Explicit weights path. Mutually exclusive with --study-id.
  --config-file-path PATH        Explicit app config path. Mutually exclusive with --study-id.

Environment:
  --xrt-setup PATH               XRT setup script. Default: $XRT_SETUP_PATH
  --ironenv PATH                 Virtualenv root. Default: $IRONENV_PATH
  --expected-mlir-aie VERSION    Required mlir_aie wheel version.

Phases:
  --start-at PHASE               Start phase. One of: ${PHASES[*]}
  --stop-after PHASE             Stop after phase. One of: ${PHASES[*]}
  --skip-download                Skip model download/cache check phase.
  --skip-autotune                Skip topology-cache warmup phase.
  --skip-power-check             Skip CPU/NPU power-backend preflight.
  --skip-suite                   Skip short supervised suite phase.
  --skip-job-print               Skip resolved job command print phase.
  --dry-run                      Print commands without executing them.

Benchmark knobs:
  --cpu-smoke-seq-lens CSV       Default: $CPU_SMOKE_SEQ_LENS
  --npu-smoke-seq-lens CSV       Default: $NPU_SMOKE_SEQ_LENS
  --autotune-seq-lens CSV        Default: $AUTOTUNE_SEQ_LENS
  --suite-seq-lens CSV           Default: $SUITE_SEQ_LENS
  --smoke-num-samples N          Default: $SMOKE_NUM_SAMPLES
  --smoke-warmup-runs N          Default: $SMOKE_WARMUP_RUNS
  --smoke-runs-per-sample N      Default: $SMOKE_RUNS_PER_SAMPLE
  --suite-num-samples N          Default: $SUITE_NUM_SAMPLES
  --suite-warmup-runs N          Default: $SUITE_WARMUP_RUNS
  --suite-runs-per-sample N      Default: $SUITE_RUNS_PER_SAMPLE
  --cpu-thread-counts CSV        Optional override for automated CPU suite.
  --npu-num-threads N            Optional override for NPU smokes and suite.
  --suite-power-backend MODE     none|auto. Default: $SUITE_POWER_BACKEND
  --power-interval-sec SEC       Default: $POWER_INTERVAL_SEC
  --npu-idle-baseline-sec SEC    Default: $NPU_IDLE_BASELINE_SEC
  --topology-cache PATH          Default: $TOPOLOGY_CACHE
  --logs-dir PATH                Default: $LOGS_DIR
  --job-config PATH              Default: $JOB_CONFIG

Examples:
  $(basename "$0")
  $(basename "$0") --dry-run --skip-power-check
  $(basename "$0") --start-at npu-smoke --stop-after suite
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

phase_banner() {
  log "Phase: $1"
}

command_string() {
  local rendered=()
  local token
  for token in "$@"; do
    rendered+=("$(printf '%q' "$token")")
  done
  printf '%s' "${rendered[*]}"
}

run_cmd() {
  log "+ $(command_string "$@")"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

phase_index() {
  local target="$1"
  local i
  for i in "${!PHASES[@]}"; do
    if [[ "${PHASES[$i]}" == "$target" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

should_run_phase() {
  local phase="$1"
  local phase_i start_i stop_i
  phase_i=$(phase_index "$phase") || die "Unknown phase: $phase"
  start_i=$(phase_index "$START_AT") || die "Unknown --start-at phase: $START_AT"
  stop_i=$(phase_index "$STOP_AFTER") || die "Unknown --stop-after phase: $STOP_AFTER"
  [[ "$start_i" -le "$stop_i" ]] || die "--start-at must not come after --stop-after"
  [[ "$phase_i" -ge "$start_i" && "$phase_i" -le "$stop_i" ]]
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --study-id)
        STUDY_ID="$2"
        shift 2
        ;;
      --study-manifest)
        STUDY_MANIFEST="$2"
        shift 2
        ;;
      --models-root)
        MODELS_ROOT="$2"
        shift 2
        ;;
      --weights-file-path)
        WEIGHTS_FILE_PATH="$2"
        shift 2
        ;;
      --config-file-path)
        CONFIG_FILE_PATH="$2"
        shift 2
        ;;
      --xrt-setup)
        XRT_SETUP_PATH="$2"
        shift 2
        ;;
      --ironenv)
        IRONENV_PATH="$2"
        shift 2
        ;;
      --expected-mlir-aie)
        EXPECTED_MLIR_AIE_VERSION="$2"
        shift 2
        ;;
      --cpu-smoke-seq-lens)
        CPU_SMOKE_SEQ_LENS="$2"
        shift 2
        ;;
      --npu-smoke-seq-lens)
        NPU_SMOKE_SEQ_LENS="$2"
        shift 2
        ;;
      --autotune-seq-lens)
        AUTOTUNE_SEQ_LENS="$2"
        shift 2
        ;;
      --suite-seq-lens)
        SUITE_SEQ_LENS="$2"
        shift 2
        ;;
      --smoke-num-samples)
        SMOKE_NUM_SAMPLES="$2"
        shift 2
        ;;
      --smoke-warmup-runs)
        SMOKE_WARMUP_RUNS="$2"
        shift 2
        ;;
      --smoke-runs-per-sample)
        SMOKE_RUNS_PER_SAMPLE="$2"
        shift 2
        ;;
      --suite-num-samples)
        SUITE_NUM_SAMPLES="$2"
        shift 2
        ;;
      --suite-warmup-runs)
        SUITE_WARMUP_RUNS="$2"
        shift 2
        ;;
      --suite-runs-per-sample)
        SUITE_RUNS_PER_SAMPLE="$2"
        shift 2
        ;;
      --cpu-thread-counts)
        CPU_THREAD_COUNTS="$2"
        shift 2
        ;;
      --npu-num-threads)
        NPU_NUM_THREADS="$2"
        shift 2
        ;;
      --suite-power-backend)
        SUITE_POWER_BACKEND="$2"
        shift 2
        ;;
      --power-interval-sec)
        POWER_INTERVAL_SEC="$2"
        shift 2
        ;;
      --npu-idle-baseline-sec)
        NPU_IDLE_BASELINE_SEC="$2"
        shift 2
        ;;
      --topology-cache)
        TOPOLOGY_CACHE="$2"
        shift 2
        ;;
      --logs-dir)
        LOGS_DIR="$2"
        shift 2
        ;;
      --job-config)
        JOB_CONFIG="$2"
        shift 2
        ;;
      --start-at)
        START_AT="$2"
        shift 2
        ;;
      --stop-after)
        STOP_AFTER="$2"
        shift 2
        ;;
      --skip-download)
        SKIP_DOWNLOAD="1"
        shift
        ;;
      --skip-autotune)
        SKIP_AUTOTUNE="1"
        shift
        ;;
      --skip-power-check)
        SKIP_POWER_CHECK="1"
        shift
        ;;
      --skip-suite)
        SKIP_SUITE="1"
        shift
        ;;
      --skip-job-print)
        SKIP_JOB_PRINT="1"
        shift
        ;;
      --dry-run)
        DRY_RUN="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  if [[ -n "$WEIGHTS_FILE_PATH" || -n "$CONFIG_FILE_PATH" ]]; then
    [[ -n "$WEIGHTS_FILE_PATH" && -n "$CONFIG_FILE_PATH" ]] || \
      die "Use both --weights-file-path and --config-file-path together"
    [[ -z "$STUDY_ID" || "$STUDY_ID" == "bert-base-uncased" ]] || \
      die "Use either explicit weights/config paths or --study-id, not both"
  fi

  [[ "$SUITE_POWER_BACKEND" == "none" || "$SUITE_POWER_BACKEND" == "auto" ]] || \
    die "--suite-power-backend must be one of: none, auto"
}

build_model_args() {
  MODEL_ARGS=()
  if [[ -n "$WEIGHTS_FILE_PATH" ]]; then
    MODEL_ARGS+=("$WEIGHTS_FILE_PATH" "$CONFIG_FILE_PATH")
    return
  fi

  MODEL_ARGS+=(--study-id "$STUDY_ID")
  if [[ -n "$STUDY_MANIFEST" ]]; then
    MODEL_ARGS+=(--study-manifest "$STUDY_MANIFEST")
  fi
  if [[ -n "$MODELS_ROOT" ]]; then
    MODEL_ARGS+=(--models-root "$MODELS_ROOT")
  fi
}

activate_env() {
  require_file "$XRT_SETUP_PATH"
  require_file "$IRONENV_PATH/bin/activate"
  # shellcheck source=/dev/null
  source "$XRT_SETUP_PATH"
  # shellcheck source=/dev/null
  source "$IRONENV_PATH/bin/activate"
}

phase_env() {
  phase_banner "env"
  require_dir "$APP_DIR"
  require_file "$APP_DIR/cpu_inference.py"
  require_file "$APP_DIR/npu_inference.py"
  require_file "$APP_DIR/automated_benchmark.py"
  require_file "$APP_DIR/download_model.py"
  require_file "$APP_DIR/run_automated_benchmark_job.py"
  mkdir -p "$LOGS_DIR"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run: environment activation, imports, and device checks skipped"
    return
  fi

  activate_env
  run_cmd python3 -c "import importlib.metadata as m; print(m.version('mlir_aie'))"
  local mlir_aie_version
  mlir_aie_version=$(python3 -c "import importlib.metadata as m; print(m.version('mlir_aie'))")
  if [[ "$mlir_aie_version" != "$EXPECTED_MLIR_AIE_VERSION" ]]; then
    die "Expected mlir_aie $EXPECTED_MLIR_AIE_VERSION but found $mlir_aie_version"
  fi

  run_cmd python3 -c "import pyxrt, torch, transformers, safetensors, huggingface_hub"
  run_cmd xrt-smi examine
  run_cmd ls -l /dev/accel/accel0
  [[ -r /dev/accel/accel0 && -w /dev/accel/accel0 ]] || \
    die "Current user does not have read/write access to /dev/accel/accel0"
}

phase_download() {
  phase_banner "download"
  local cmd=(python3 "$APP_DIR/download_model.py")
  if [[ -n "$WEIGHTS_FILE_PATH" ]]; then
    log "Explicit weights/config paths selected; skipping model download"
    return
  fi
  cmd+=(--study-id "$STUDY_ID")
  if [[ -n "$STUDY_MANIFEST" ]]; then
    cmd+=(--study-manifest "$STUDY_MANIFEST")
  fi
  if [[ -n "$MODELS_ROOT" ]]; then
    cmd+=(--models-root "$MODELS_ROOT")
  fi
  run_cmd "${cmd[@]}"
}

phase_cpu_smoke() {
  phase_banner "cpu-smoke"
  local output_csv="$LOGS_DIR/cpu_smoke.csv"
  local cmd=(
    python3 "$APP_DIR/cpu_inference.py"
    "${MODEL_ARGS[@]}"
    --seq-lens "$CPU_SMOKE_SEQ_LENS"
    --num-samples "$SMOKE_NUM_SAMPLES"
    --warmup-runs "$SMOKE_WARMUP_RUNS"
    --runs-per-sample "$SMOKE_RUNS_PER_SAMPLE"
    --output-csv "$output_csv"
  )
  run_cmd "${cmd[@]}"
  [[ "$DRY_RUN" == "1" ]] || require_file "$output_csv"
}

phase_npu_smoke() {
  phase_banner "npu-smoke"
  local seq_len
  IFS=',' read -r -a seq_len_list <<< "$NPU_SMOKE_SEQ_LENS"
  for seq_len in "${seq_len_list[@]}"; do
    local trimmed_seq_len="${seq_len// /}"
    local output_csv="$LOGS_DIR/npu_smoke_${trimmed_seq_len}.csv"
    local cmd=(
      python3 "$APP_DIR/npu_inference.py"
      "${MODEL_ARGS[@]}"
      --seq-lens "$trimmed_seq_len"
      --num-samples "$SMOKE_NUM_SAMPLES"
      --warmup-runs "$SMOKE_WARMUP_RUNS"
      --runs-per-sample "$SMOKE_RUNS_PER_SAMPLE"
      --topology-policy fixed
      --topology-cache "$TOPOLOGY_CACHE"
      --output-csv "$output_csv"
    )
    if [[ -n "$NPU_NUM_THREADS" ]]; then
      cmd+=(--num-threads "$NPU_NUM_THREADS")
    fi
    run_cmd "${cmd[@]}"
    [[ "$DRY_RUN" == "1" ]] || require_file "$output_csv"
  done
}

phase_autotune() {
  phase_banner "autotune"
  local output_csv="$LOGS_DIR/npu_autotune.csv"
  local cmd=(
    python3 "$APP_DIR/automated_benchmark.py"
    "${MODEL_ARGS[@]}"
    --modes npu
    --seq-lens "$AUTOTUNE_SEQ_LENS"
    --num-samples "$SUITE_NUM_SAMPLES"
    --warmup-runs "$SUITE_WARMUP_RUNS"
    --runs-per-sample "$SUITE_RUNS_PER_SAMPLE"
    --npu-topology-policy autotune
    --npu-topology-cache "$TOPOLOGY_CACHE"
    --npu-autotune-warmup-runs 2
    --npu-autotune-runs 5
    --power-backend none
    --state-json "$LOGS_DIR/autotune_state.json"
    --output-csv "$output_csv"
    --logs-dir "$LOGS_DIR/autotune_logs"
  )
  if [[ -n "$NPU_NUM_THREADS" ]]; then
    cmd+=(--npu-num-threads "$NPU_NUM_THREADS")
  fi
  run_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$output_csv"
    require_file "$TOPOLOGY_CACHE"
  fi
}

phase_power() {
  phase_banner "power"
  if [[ "$SUITE_POWER_BACKEND" == "none" ]]; then
    log "Skipping power preflight because --suite-power-backend is none"
    return
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd "$POWERCAP_HELPER" --check
  elif ! "$POWERCAP_HELPER" --check; then
    run_cmd sudo -n "$POWERCAP_HELPER" --check
  fi

  local turbostat_cmd=(
    sudo -n turbostat --Summary --quiet
    --show PkgWatt,CorWatt,GFXWatt,RAMWatt
    --interval "$POWER_INTERVAL_SEC"
    --num_iterations 2
    sleep 1
  )
  run_cmd "${turbostat_cmd[@]}"
}

phase_suite() {
  phase_banner "suite"
  local output_csv="$LOGS_DIR/short_suite.csv"
  local state_json="$LOGS_DIR/short_suite_state.json"
  local suite_logs_dir="$LOGS_DIR/short_suite_logs"
  local cmd=(
    python3 "$APP_DIR/automated_benchmark.py"
    "${MODEL_ARGS[@]}"
    --modes cpu,npu
    --seq-lens "$SUITE_SEQ_LENS"
    --num-samples "$SUITE_NUM_SAMPLES"
    --warmup-runs "$SUITE_WARMUP_RUNS"
    --runs-per-sample "$SUITE_RUNS_PER_SAMPLE"
    --npu-topology-policy cache
    --npu-topology-cache "$TOPOLOGY_CACHE"
    --power-backend "$SUITE_POWER_BACKEND"
    --power-interval-sec "$POWER_INTERVAL_SEC"
    --npu-idle-baseline-sec "$NPU_IDLE_BASELINE_SEC"
    --state-json "$state_json"
    --output-csv "$output_csv"
    --logs-dir "$suite_logs_dir"
  )
  if [[ -n "$CPU_THREAD_COUNTS" ]]; then
    cmd+=(--cpu-thread-counts "$CPU_THREAD_COUNTS")
  fi
  if [[ -n "$NPU_NUM_THREADS" ]]; then
    cmd+=(--npu-num-threads "$NPU_NUM_THREADS")
  fi
  run_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$output_csv"
    require_file "$state_json"
    require_dir "$suite_logs_dir"
    run_cmd python3 - "$output_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("Expected at least one short-suite row")
npu_rows = [row for row in rows if row.get("mode") == "npu"]
if not npu_rows:
    raise SystemExit("Expected at least one NPU row in short suite CSV")
required = ("topology_id", "idle_pkg_watt", "avg_pkg_watt", "pseudo_npu_avg_pkg_watt")
missing = [name for name in required if name not in npu_rows[0]]
if missing:
    raise SystemExit(f"Missing expected NPU columns: {missing}")
print(f"validated_short_suite_rows={len(rows)}")
PY
  fi
}

phase_job_print() {
  phase_banner "job-print"
  if [[ ! -f "$JOB_CONFIG" ]]; then
    log "Skipping job-print: missing job config at $JOB_CONFIG"
    return
  fi
  run_cmd python3 "$APP_DIR/run_automated_benchmark_job.py" "$JOB_CONFIG" --print-command
}

main() {
  parse_args "$@"
  validate_args
  build_model_args

  if should_run_phase "env"; then
    phase_env
  fi

  if should_run_phase "download" && [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    phase_download
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    activate_env
  fi

  if should_run_phase "cpu-smoke"; then
    phase_cpu_smoke
  fi

  if should_run_phase "npu-smoke"; then
    phase_npu_smoke
  fi

  if should_run_phase "autotune" && [[ "$SKIP_AUTOTUNE" != "1" ]]; then
    phase_autotune
  fi

  if should_run_phase "power" && [[ "$SKIP_POWER_CHECK" != "1" ]]; then
    phase_power
  fi

  if should_run_phase "suite" && [[ "$SKIP_SUITE" != "1" ]]; then
    phase_suite
  fi

  if should_run_phase "job-print" && [[ "$SKIP_JOB_PRINT" != "1" ]]; then
    phase_job_print
  fi

  log "Bring-up checklist completed"
}

main "$@"
