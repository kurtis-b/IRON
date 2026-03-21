<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Encoder Benchmarks

`applications/bert` now has three explicit benchmark entrypoints:
- [cpu_inference.py](cpu_inference.py): Hugging Face CPU baseline for supported encoder-only families
- [igpu_inference.py](igpu_inference.py): Hugging Face iGPU baseline through a GPU-enabled PyTorch runtime
- [npu_inference.py](npu_inference.py): local NPU benchmark using the `encoder_pipeline` operator
- [download_model.py](download_model.py): manifest-driven Hugging Face downloader for supported study models
- [automated_benchmark.py](automated_benchmark.py): resumable case runner with optional topology autotune, power logging, and power-cycle hooks
- [bringup_checklist.sh](bringup_checklist.sh): guided device preflight for CPU/NPU smokes, topology-cache warmup, and short supervised suite validation
- [run_automated_benchmark_job.py](run_automated_benchmark_job.py): wrapper that runs the automation harness from a JSON job file
- [plot_benchmark_results.py](plot_benchmark_results.py): headless plot renderer for suite CSVs
- [validate_npu_parity.py](validate_npu_parity.py): CPU-vs-NPU hidden-state parity validator for canonical synthetic sequence lengths
- [peak_reference.py](peak_reference.py): host fingerprint / hardware-profile artifact generator for peak-normalization and roofline groundwork
- [calibrate_backend_peaks.py](calibrate_backend_peaks.py): peak-reference artifact builder that combines the host profile with measured calibration inputs

The shared code under `src/` is now NPU-only. It exists to build the encoder-pipeline-backed local backbone used by `npu_inference.py`. The CPU benchmark uses Hugging Face directly and does not go through `src/`.

`igpu_inference.py` uses the `torch.cuda` API, which means it requires a GPU-enabled PyTorch build. On AMD systems that usually means a ROCm-enabled wheel rather than the CPU-only wheel used by the CPU/NPU flow.

In shell examples below:
- replace `<repo_root>` with the root of your local checkout
- replace `<benchmark_user>` with the Linux account that will run the benchmarks

Currently supported encoder families:
- `bert`
- `roberta`
- `distilbert`

The benchmark family is chosen from `model_config.model_type` in the config JSON.

## Model Files

Download `model.safetensors` and `config.json` for one of the supported model configs:

| Local config | Hugging Face model | Weights | Config |
|---|---|---|---|
| [config.json](config/config.json) | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | [`model.safetensors`](https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors) | [`config.json`](https://huggingface.co/bert-base-uncased/resolve/main/config.json) |
| [config_bert_large.json](config/config_bert_large.json) | [`bert-large-uncased`](https://huggingface.co/bert-large-uncased) | [`model.safetensors`](https://huggingface.co/bert-large-uncased/resolve/main/model.safetensors) | [`config.json`](https://huggingface.co/bert-large-uncased/resolve/main/config.json) |
| [config_roberta_base.json](config/config_roberta_base.json) | [`roberta-base`](https://huggingface.co/roberta-base) | [`model.safetensors`](https://huggingface.co/roberta-base/resolve/main/model.safetensors) | [`config.json`](https://huggingface.co/roberta-base/resolve/main/config.json) |
| [config_roberta_large.json](config/config_roberta_large.json) | [`roberta-large`](https://huggingface.co/roberta-large) | [`model.safetensors`](https://huggingface.co/roberta-large/resolve/main/model.safetensors) | [`config.json`](https://huggingface.co/roberta-large/resolve/main/config.json) |
| [config_distilbert_base.json](config/config_distilbert_base.json) | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) | [`model.safetensors`](https://huggingface.co/distilbert-base-uncased/resolve/main/model.safetensors) | [`config.json`](https://huggingface.co/distilbert-base-uncased/resolve/main/config.json) |

The scripts expect the safetensors file on the command line. The JSON config passed to the scripts is the local app config under `iron/applications/bert/config/`.

To download supported models into the local benchmark layout automatically:

```bash
cd iron/applications/bert
python3 download_model.py --study-id bert-base-uncased
```

This writes:
- `iron/applications/bert/models/<study_id>/model.safetensors`
- `iron/applications/bert/models/<study_id>/config.json`
- local tokenizer assets such as `tokenizer.json`, `tokenizer_config.json`, and vocab files

To download every study target in the manifest:

```bash
cd iron/applications/bert
python3 download_model.py --all
```

Useful options:
- `--print-plan`: resolve destinations without downloading
- `--models-root <dir>`: override the local models directory
- `--force`: overwrite existing local files

Useful config files:
- [config.json](config/config.json): `bert-base` baseline
- [config_bert_large.json](config/config_bert_large.json): `bert-large`
- [config_roberta_base.json](config/config_roberta_base.json): `roberta-base`
- [config_roberta_large.json](config/config_roberta_large.json): `roberta-large`
- [config_distilbert_base.json](config/config_distilbert_base.json): `distilbert-base`

Suggested study targets are listed in:
- [encoder_only_models.json](study/encoder_only_models.json)

## Hardware Profile

Generate a machine-readable host profile artifact before adding backend peak or roofline analysis:

```bash
cd iron/applications/bert
python3 peak_reference.py --output host_profile.json
```

If firmware inventory does not expose installed memory type or speed, provide a small override JSON:

```json
{
  "memory": {
    "type": "LPDDR5X",
    "speed_mt_s": 8000,
    "interface_bits": 128,
    "source": "manual"
  }
}
```

Then run:

```bash
python3 peak_reference.py \
  --memory-override memory_override.json \
  --output host_profile.json
```

When the host cannot be matched confidently to one of the seeded HX-class candidates, the artifact keeps `chip_sku` as `unknown` rather than guessing.

Build the peak-reference artifact as a separate step so official, derived, and measured ceilings are auditable:

```bash
python3 calibrate_backend_peaks.py \
  --host-profile host_profile.json \
  --output peak_references.json
```

Optional measured calibration inputs can be supplied as JSON:

```json
{
  "tool_versions": {
    "benchdnn": "3.5.0",
    "rocblas-bench": "6.5.0",
    "stream": "5.10"
  },
  "peaks": {
    "cpu": {
      "bfloat16": {"measured_peak_ops_per_sec": 1.2e12}
    },
    "igpu": {
      "float32": {"measured_peak_ops_per_sec": 4.9e12}
    },
    "npu": {
      "int8": {"measured_peak_ops_per_sec": 4.7e13},
      "bfloat16": {"measured_peak_ops_per_sec": 1.7e13}
    }
  },
  "memory_bandwidth": {
    "copy_bytes_per_sec": 9.8e10,
    "triad_bytes_per_sec": 8.9e10,
    "measured_peak_bytes_per_sec": 8.9e10,
    "note": "STREAM triad"
  }
}
```

Then run:

```bash
python3 calibrate_backend_peaks.py \
  --host-profile host_profile.json \
  --measured-input measured_peaks.json \
  --output peak_references.json
```

For a first direct local calibration pass without external benchmark clients:

```bash
python3 calibrate_backend_peaks.py \
  --host-profile host_profile.json \
  --collect-measured \
  --measured-output measured_peaks.json \
  --output peak_references.json
```

The built-in collector currently does the following:
- CPU: measured `torch.mm` GEMM peaks for `bfloat16` and `float32`
- iGPU: measured ROCm `torch.mm` GEMM peaks for `bfloat16`, `float16`, and `float32`
- NPU: measured `xrt-smi validate --run gemm` INT8 reference when the tool can supply it
- NPU: measured first-pass IRON `bfloat16` peak from the existing deterministic `encoder_pipeline` synthetic-dense path in `npu_inference.py`
- Memory: built-in STREAM-style copy/triad bandwidth benchmark on shared CPU memory

Current limitation:
- the first-pass NPU `bfloat16` denominator is an `encoder_pipeline` synthetic-dense calibration, not a pure bf16 GEMM microbenchmark yet. It is still the correct primary denominator for bf16 BERT results, and it is intentionally kept separate from the INT8 `xrt-smi` TOPS reference.

## Installation

1. Follow the repository root setup instructions so `ironenv` is available.
2. Install the extra Python packages:

```bash
python3 -m pip install -r iron/applications/bert/requirements_bert.txt
```

## CPU Benchmark

Run the Hugging Face baseline:

```bash
cd iron/applications/bert
python3 cpu_inference.py /path/to/model.safetensors config/config.json
```

Useful options:

```bash
python3 cpu_inference.py <weights> <config> \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --num-samples 1 \
  --warmup-runs 10 \
  --runs-per-sample 100 \
  --dtype bfloat16 \
  --output-csv cpu_benchmark_latest.csv
```

Behavior:
- uses the Hugging Face encoder model implied by `model_config.model_type`
- excludes pooler and classifier head
- uses a built-in local text corpus
- generates deterministic local token ids from that corpus, so no tokenizer download is required
- `--benchmark-mode synthetic_dense` is the default stress path
- `--benchmark-mode model_valid` switches to real local tokenizer outputs plus attention masks and currently requires `seq_len <= 512`
- `--disable-all-biases` zeros every model bias tensor after load for apples-to-apples CPU/iGPU comparisons against the current fused NPU encoder path
- runs unmasked inference (`attention_mask=None`) to match the NPU path
- defaults to `10` warmup runs and `100` timed runs per sample
- by default runs two CPU thread configurations for comparison:
  - maximum physical cores
  - full logical thread count
- `--num-threads` overrides that and forces a single thread-count run
- writes results to `cpu_benchmark_latest.csv`
- checkpoints the CSV after each completed sequence length

You can also run by study id after downloading model artifacts:

```bash
cd iron/applications/bert
python3 cpu_inference.py --study-id bert-base-uncased
```

## iGPU Benchmark

Run the Hugging Face iGPU baseline:

```bash
cd iron/applications/bert
python3 igpu_inference.py /path/to/model.safetensors config/config.json
```

Useful options:

```bash
python3 igpu_inference.py <weights> <config> \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --num-samples 1 \
  --warmup-runs 10 \
  --runs-per-sample 100 \
  --dtype bfloat16 \
  --output-csv igpu_benchmark_latest.csv
```

Behavior:
- uses the Hugging Face encoder model implied by `model_config.model_type`
- excludes pooler and classifier head
- uses the same built-in local text corpus as the CPU benchmark
- defaults to `bfloat16`, while still allowing explicit `float16`, `float32`, or `bfloat16`
- `--benchmark-mode synthetic_dense` is the default stress path
- `--benchmark-mode model_valid` switches to real local tokenizer outputs plus attention masks and currently requires `seq_len <= 512`
- `--disable-all-biases` zeros every model bias tensor after load for apples-to-apples iGPU comparisons against the current fused NPU encoder path
- writes results to `igpu_benchmark_latest.csv`
- checkpoints the CSV after each completed sequence length

## NPU Benchmark

Run the NPU benchmark with `encoder_pipeline`:

```bash
cd iron/applications/bert
python3 npu_inference.py /path/to/model.safetensors config/config.json
```

Useful options:

```bash
python3 npu_inference.py <weights> <config> \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --num-samples 1 \
  --warmup-runs 10 \
  --runs-per-sample 100 \
  --topology-policy cache \
  --topology-cache npu_topology_cache_latest.json \
  --output-csv npu_benchmark_latest.csv
```

Behavior:
- builds a minimal local backbone: embeddings + `encoder_pipeline`
- excludes pooler and classifier head
- uses the same built-in local text corpus as the CPU benchmark
- generates the same deterministic local token ids as the CPU benchmark
- only supports `--benchmark-mode synthetic_dense` today because the current `encoder_pipeline` path still requires `attention_mask=None`
- uses the same CSV schema as the CPU benchmark
- runs unmasked inference (`attention_mask=None`)
- child CSV rows now also include:
  - `compile_setup_time_ms`
  - `topology_selection_time_ms`
  - `topology_cache_status`
  - `cached_steady_state_avg_latency_ms`
  - `avg_embedding_latency_ms`
  - `avg_qkv_projection_latency_ms`
  - `avg_encoder_pipeline_latency_ms`
  so the NPU path can be read as topology selection + compile/setup + embeddings + host QKV + encoder-pipeline + end-to-end latency
- defaults to `10` warmup runs and `100` timed runs per sample
- can select encoder-pipeline topology per sequence length:
  - `fixed`: use the topology encoded in the config
  - `cache`: use a cached topology per sequence length, autotuning cache misses across both supported tile families when legal
  - `autotune`: retune every sequence length on each run across both supported tile families when legal
  - autotune keeps latency as the primary objective; candidates within a 1% latency band are broken by higher compute-tile utilization and then by the config family
- the NPU path currently expects BERT-like encoder structure; `bert`, `roberta`, and `distilbert` are supported
- writes results to `npu_benchmark_latest.csv`
- checkpoints the CSV after each completed sequence length

You can also run by study id after downloading model artifacts:

```bash
cd iron/applications/bert
python3 npu_inference.py --study-id bert-base-uncased
```

## CPU-vs-NPU Parity Check

Validate final hidden-state parity on the deterministic synthetic path used by the timing harness:

```bash
cd iron/applications/bert
python3 validate_npu_parity.py \
  --study-id bert-base-uncased \
  --seq-lens 64,128,512 \
  --output-csv npu_parity_latest.csv
```

Behavior:
- uses the same synthetic token generation as the benchmark harness
- compares CPU final hidden states against the NPU `encoder_pipeline` path
- records per-sequence parity metrics:
  - `cosine_similarity`
  - `max_abs_error`
  - `mean_abs_error`
- also records `error_count`, `error_fraction`, `max_acceptable_errors`, and the selected `topology_id`, `topology_cache_status`, `topology_selection_time_ms`, and `compile_setup_time_ms`
- defaults to a quick `fixed` topology policy; pass `--topology-policy cache` if you want to validate the cached benchmark path instead
- uses the same default tolerance model as the `encoder_pipeline` operator tests: `--rel-tol 0.04`, `--abs-tol 0.15`, `--max-error-fraction 0.005`
- exits nonzero if any row exceeds the allowed elementwise error budget after applying the rel/abs tolerance window
- current branch status: embeddings and host QKV projection parity are close, but divergence still starts in the fused encoder layer. The residual-add / LayerNorm ordering now matches the operator reference, but the fused path still omits encoder dense and LayerNorm biases, so HF-equivalent parity is not expected until those bias terms are carried through the operator interface and kernels.

## Automated Benchmarking

Use the automation harness when you want:
- a case-by-case CPU/NPU sweep
- resumable state across reboots
- optional per-case power logging
- optional pre-case cooldown by fixed interval or temperature threshold
- optional NPU topology autotuning
- optional `igpu` comparison mode
- optional multi-study runs through `--study-ids`

Example:

```bash
cd iron/applications/bert
python3 automated_benchmark.py /path/to/model.safetensors config/config.json \
  --modes cpu,npu,igpu \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --runs-per-sample 100 \
  --warmup-runs 10 \
  --cooldown-until-temp-c 50 \
  --npu-topology-policy cache \
  --power-backend auto \
  --power-cycle-cmd "sudo reboot"
```

Or resolve the model paths from the study manifest:

```bash
cd iron/applications/bert
python3 automated_benchmark.py \
  --study-ids all \
  --modes cpu,npu,igpu \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --benchmark-mode synthetic_dense \
  --runs-per-sample 100 \
  --warmup-runs 10 \
  --cooldown-until-temp-c 50 \
  --npu-topology-policy cache \
  --peak-reference peak_references.json \
  --power-backend auto \
  --power-cycle-cmd "sudo reboot"
```

Behavior:
- writes a resumable state file
- writes one master CSV summarizing all completed cases
- writes per-case child benchmark CSVs and power logs under `logs/automated_benchmark`
- every child row and suite row now records `benchmark_mode` and `execution_mode`
- `--benchmark-mode synthetic_dense` is the current default dense stress path
- `--benchmark-mode model_valid` uses real local tokenizer outputs plus attention masks and currently applies only to CPU and iGPU with `seq_len <= 512`
- `--benchmark-mode task_eval` is reserved for later task-level evaluation scripts and is not implemented in the timing harness yet
- `--power-backend auto` uses `powercap-rapl` for CPU, `turbostat` for NPU, and `rocm-smi` for iGPU
- when `--peak-reference` is set, suite rows also include dtype-aware percent-of-peak and roofline fields derived from the supplied artifact
- `--bytes-model-version` controls the analytical bytes model used for roofline fields; `v1` is the current encoder-only model
- `--bytes-model-weights-policy resident` treats model weights as already resident, while `streamed` adds a per-inference DDR weight-read term
- active power logging now covers only the timed inference region, not model load or warmup
- CPU child benchmarks log start/end package/core energy snapshots for the timed region and derive average watts from the energy delta
- NPU child benchmarks log periodic `turbostat` package-power samples for the timed region
- iGPU child benchmarks log periodic `rocm-smi` graphics-package power samples for the timed region without requiring `sudo`
- when `--cooldown-until-temp-c` is set, the suite accepts temperatures within the default 5% band above the target
- when that temperature target is still not met after the max cooldown wait, the suite continues instead of failing
- if `--power-cycle-cmd` is set, it runs one case, records results, invokes the hook, and exits
- after the machine comes back, rerun the same command to continue from the saved state
- for non-CPU accelerator cases with power logging enabled, it also measures an idle baseline before each case and writes:
  - `idle_pkg_watt`
  - `pseudo_device_avg_pkg_watt`
  - `pseudo_device_max_pkg_watt`
  - `pseudo_npu_avg_pkg_watt`
  - `pseudo_npu_max_pkg_watt`
- all child CSVs and suite rows also include:
  - `study_id`
  - `measured_inference_count`
  - `timed_total_sec`
  - `throughput_flops_per_sec`
  - `estimated_flops_per_inference`
- suite rows additionally include:
  - `cooldown_wait_sec`
  - `power_window_sec`
  - `estimated_gflops_per_watt_sec`
  - `pseudo_device_estimated_gflops_per_watt_sec`
  - `compile_setup_time_ms`
  - `topology_selection_time_ms`
  - `topology_cache_status`
  - `cached_steady_state_avg_latency_ms`
  - `avg_embedding_latency_ms`
  - `avg_qkv_projection_latency_ms`
  - `avg_encoder_pipeline_latency_ms`
  - `estimated_bytes_per_inference`
  - `backend_pct_of_peak`
  - `operational_intensity_flops_per_byte`
  - `roofline_bound_ops_per_sec`
  - `roofline_pct`

Cooldown notes:
- default temperature tolerance is `5%`, so a `50 C` target is treated as acceptable once the selected sensor reaches `52.5 C` or lower
- default max cooldown wait is `300` seconds
- set `--cooldown-temp-tolerance-frac 0` if you want an exact threshold
- set `--cooldown-timeout-sec` higher if you want stricter thermal settling

## Plotting Results

Render summary plots and per-study dashboards from a completed suite CSV:

```bash
cd iron/applications/bert
python3 plot_benchmark_results.py automated_benchmark_all_studies.csv
```

Useful options:

```bash
python3 plot_benchmark_results.py <suite_csv> \
  --output-dir plots/<name> \
  --dpi 180
```

Behavior:
- writes a summary overview plot across studies/backends
- writes grouped cross-model comparison plots for latency, throughput, power, and efficiency
- when peak-reference annotations are present in the suite CSV, also writes grouped percent-of-peak and grouped roofline-utilization plots
- when peak-reference annotations are present in the suite CSV, also writes a backend-split roofline overview scatter plot
- writes one dashboard per study with latency, throughput, effective power, and effective efficiency
- writes CPU thread-comparison plots when multiple CPU thread counts are present
- writes an `index.html` file so the plots are easy to browse

## Systemd Resume Flow

For unattended reboot/resume:

1. Copy the example job file and edit it:

```bash
cd iron/applications/bert/systemd
cp benchmark_job.example.json benchmark_job.json
```

2. Review [benchmark_job.example.json](systemd/benchmark_job.example.json) and set:
- exactly one of: weights/config paths, `study_id`, or `study_ids`
- `modes`
- sequence lengths
- benchmark mode
- CPU thread counts
- optional iGPU thread count / dtype
- NPU topology policy
- optional `peak_reference` / `bytes_model_version` / `bytes_model_weights_policy`
- output/state/log paths
- `power_cycle_cmd`

3. Test the resolved command without systemd:

```bash
cd iron/applications/bert
python3 run_automated_benchmark_job.py systemd/benchmark_job.json --print-command
```

4. Edit the example service file:

- replace `<benchmark_user>` with the Linux user that will run the benchmark
- replace `<repo_root>` with the root of your checkout

5. Install the service:

```bash
sudo cp <repo_root>/iron/applications/bert/systemd/bert-automated-benchmark.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bert-automated-benchmark.service
sudo systemctl start bert-automated-benchmark.service
```

The service file sets `LimitMEMLOCK=65536K`. Keep that limit in place for unattended NPU runs; the default systemd `memlock` limit is too low for reliable `pyxrt.device(0)` startup on this host, while `64 MiB` was sufficient in service-context validation.

6. Watch the service log:

```bash
tail -f iron/applications/bert/logs/systemd/bert-automated-benchmark.log
```

Notes:
- the service expects the live job file at:
  - `iron/applications/bert/systemd/benchmark_job.json`
- the service runs [run_automated_benchmark_job.py](run_automated_benchmark_job.py), which in turn invokes [automated_benchmark.py](automated_benchmark.py)
- each completed case is persisted before any reboot is requested
- after reboot, systemd starts the service again and the suite resumes from the saved state

Power notes:
- `powertop` is not used here because it is weaker for scripted benchmarking
- `none` and `auto` are the only supported user-facing `--power-backend` values
- `auto` is the fixed mixed backend: `powercap-rapl` for CPU, `turbostat` for NPU, and `rocm-smi` for iGPU
- `powercap-rapl` is the preferred local backend for timed-region CPU package/core energy logging
- `turbostat` remains the preferred local backend for timed-region NPU package logging
- `rocm-smi` is the preferred local backend for timed-region iGPU graphics-package logging
- `powercap-rapl` reads `energy_uj` counters under `/sys/class/powercap` and may require either direct read access or a narrowly whitelisted helper
- `turbostat` needs privileged access to MSRs on this machine
- `rocm-smi` does not need `sudo`, but it only applies to iGPU cases
- for NPU runs, the suite treats pseudo-NPU power as:
  - `measured package power - idle package power`
- for iGPU runs under `rocm-smi`, the suite treats pseudo-device power as:
  - `measured graphics-package power - idle graphics-package power`
- under `rocm-smi`, the direct iGPU reading is reported in `avg_gfx_watt` / `max_gfx_watt`
- the idle package power is measured immediately before each NPU case
- the idle graphics-package power is measured immediately before each iGPU case
- for total wall-power comparison across CPU and NPU runs, an external meter or smart PDU is still better than host-local telemetry

Suggested sudoers entries for unattended runs:

```text
<benchmark_user> ALL=(root) NOPASSWD: <repo_root>/iron/applications/bert/read_powercap_rapl.py
<benchmark_user> ALL=(root) NOPASSWD: /usr/bin/turbostat
<benchmark_user> ALL=(root) NOPASSWD: /usr/sbin/reboot, /usr/bin/systemctl reboot
```

## Preflight

Before the first unattended run, do this in order.

Fast path:

```bash
cd <repo_root>/iron/applications/bert
./bringup_checklist.sh
```

Useful variants:

```bash
./bringup_checklist.sh --dry-run
./bringup_checklist.sh --start-at npu-smoke --stop-after suite
./bringup_checklist.sh --study-id bert-base-uncased --models-root ./models
```

The script runs the same checklist below and writes preflight artifacts under:
- `iron/applications/bert/logs/preflight`

1. Verify privileged helpers work non-interactively

```bash
sudo -n /absolute/path/to/iron/applications/bert/read_powercap_rapl.py --check
sudo -n turbostat --Summary --quiet --show PkgWatt,CorWatt,GFXWatt,RAMWatt --interval 0.5 --num_iterations 2 sleep 1
```

Do not run reboot until you are ready for it, but make sure the sudoers rule exists for the reboot command you plan to use.

2. Warm the NPU topology cache under supervision

This generates a fresh best-topology entry per sequence length for the current machine and software state.

```bash
cd <repo_root>/iron/applications/bert
source /opt/xilinx/xrt/setup.sh
source <repo_root>/ironenv/bin/activate
python3 automated_benchmark.py \
  --study-id bert-base-uncased \
  --modes npu \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --num-samples 1 \
  --warmup-runs 10 \
  --runs-per-sample 100 \
  --npu-topology-policy autotune \
  --npu-autotune-warmup-runs 2 \
  --npu-autotune-runs 5 \
  --power-backend none
```

After that, unattended runs should use:
- `--npu-topology-policy cache`

3. Run one short supervised suite without reboot

Use a trimmed sequence-length set first and leave out `--power-cycle-cmd`.

```bash
cd <repo_root>/iron/applications/bert
python3 automated_benchmark.py \
  --study-id bert-base-uncased \
  --modes cpu,npu \
  --seq-lens 64,512 \
  --num-samples 1 \
  --warmup-runs 10 \
  --runs-per-sample 100 \
  --npu-topology-policy cache \
  --power-backend auto
```

4. Verify outputs from the supervised run

Check that these files exist and look sane:
- [automated_benchmark_latest.csv](automated_benchmark_latest.csv)
- `automated_benchmark_state.json`
- `logs/automated_benchmark/*`

Before trusting CPU-vs-NPU speedup claims, also run:

```bash
python3 validate_npu_parity.py \
  --study-id bert-base-uncased \
  --seq-lens 64,128,512
```

For NPU rows in the suite CSV, inspect:
- `topology_id`
  This is now the canonical family-aware id, for example `seq32_kv64__ps2_ph1_pffn4`.
  Legacy shorthand filters like `2ps_4pffn` and `4ps` are still accepted on the CLI.
- `compute_tile_count`
- `compute_tile_utilization_fraction`
- `idle_pkg_watt`
- `avg_pkg_watt`
- `pseudo_npu_avg_pkg_watt`

For iGPU rows in the suite CSV, inspect:
- `avg_gfx_watt`
- `idle_gfx_watt`
- `pseudo_device_avg_pkg_watt`

For cross-device comparisons, also inspect:
- `study_id`
- `throughput_flops_per_sec`
- `power_sample_count`
- `power_window_sec`
- `estimated_gflops_per_watt_sec`
- `pseudo_device_estimated_gflops_per_watt_sec`

5. Enable reboot only after the short supervised run looks correct

At that point:
- set `power_cycle_cmd` in `benchmark_job.json`
- install/enable the systemd service
- start the unattended run

## Second Device Quickstart

Use this exact sequence on a second machine before attempting an unattended run.

1. Inspect the bring-up plan without touching hardware state:

```bash
cd <repo_root>/iron/applications/bert
./bringup_checklist.sh --dry-run
```

2. Verify everything up to the automated benchmark path:

```bash
./bringup_checklist.sh --stop-after power
```

This checks:
- environment and expected `mlir_aie` wheel
- XRT and `/dev/accel/accel0` access
- model download/cache
- CPU smoke
- NPU smokes at `64` and `512`
- topology-cache warmup
- CPU `powercap-rapl` helper access and non-interactive `turbostat`

3. Verify the automated benchmark path itself with the short supervised suite:

```bash
./bringup_checklist.sh
```

This additionally checks:
- short `cpu,npu` automated suite
- suite CSV/state/log outputs
- resolved job command, if `systemd/benchmark_job.json` exists

4. Create the real job config:

```bash
cd <repo_root>/iron/applications/bert/systemd
cp benchmark_job.example.json benchmark_job.json
```

5. Print the resolved automated-benchmark command:

```bash
cd <repo_root>/iron/applications/bert
python3 run_automated_benchmark_job.py systemd/benchmark_job.json --print-command
```

6. Run the job directly once under supervision:

```bash
python3 run_automated_benchmark_job.py systemd/benchmark_job.json
```

7. Only after that, enable unattended reboot/resume:
- configure sudoers for `read_powercap_rapl.py`, `turbostat`, and reboot
- install/enable the systemd service
- set `power_cycle_cmd` in `benchmark_job.json`

## Sequence Lengths

All benchmark entrypoints default to:
- `64,128,256,512,1024,2048,4096,8192`

For lengths above `512`, the scripts extend the learned position embeddings by repeating them so CPU and NPU benchmark the same effective backbone shape.

## Legacy Entry Point

[inference.py](inference.py) is now a legacy stub. Use the CPU and NPU benchmark scripts directly.
