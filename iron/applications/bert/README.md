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
  --dtype float32 \
  --output-csv cpu_benchmark_latest.csv
```

Behavior:
- uses the Hugging Face encoder model implied by `model_config.model_type`
- excludes pooler and classifier head
- uses a built-in local text corpus
- generates deterministic local token ids from that corpus, so no tokenizer download is required
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
- uses the same CSV schema as the CPU benchmark
- runs unmasked inference (`attention_mask=None`)
- defaults to `10` warmup runs and `100` timed runs per sample
- can select encoder-pipeline topology per sequence length:
  - `fixed`: use the topology encoded in the config
  - `cache`: use a cached topology per sequence length, autotuning cache misses
  - `autotune`: retune every sequence length on each run
- the NPU path currently expects BERT-like encoder structure; `bert`, `roberta`, and `distilbert` are supported
- writes results to `npu_benchmark_latest.csv`
- checkpoints the CSV after each completed sequence length

You can also run by study id after downloading model artifacts:

```bash
cd iron/applications/bert
python3 npu_inference.py --study-id bert-base-uncased
```

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
  --power-backend turbostat \
  --power-cycle-cmd "sudo reboot"
```

Or resolve the model paths from the study manifest:

```bash
cd iron/applications/bert
python3 automated_benchmark.py \
  --study-ids all \
  --modes cpu,npu,igpu \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --runs-per-sample 100 \
  --warmup-runs 10 \
  --cooldown-until-temp-c 50 \
  --npu-topology-policy cache \
  --power-backend turbostat \
  --power-cycle-cmd "sudo reboot"
```

Behavior:
- writes a resumable state file
- writes one master CSV summarizing all completed cases
- writes per-case child benchmark CSVs and power logs under `logs/automated_benchmark`
- when `power_backend=turbostat`, logs periodic package-power samples for the full case window instead of a single summary-only row
- if `--power-cycle-cmd` is set, it runs one case, records results, invokes the hook, and exits
- after the machine comes back, rerun the same command to continue from the saved state
- for non-CPU accelerator cases with `turbostat`, it also measures an idle baseline before each case and writes:
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
- CPU thread counts
- optional iGPU thread count / dtype
- NPU topology policy
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
- `turbostat` is the preferred local backend for package-power logging
- `turbostat` needs privileged access to MSRs on this machine
- for NPU runs, the suite treats pseudo-NPU power as:
  - `measured package power - idle package power`
- the idle package power is measured immediately before each NPU case
- for total wall-power comparison across CPU and NPU runs, an external meter or smart PDU is still better than host-local telemetry

Suggested sudoers entries for unattended runs:

```text
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
  --power-backend turbostat
```

4. Verify outputs from the supervised run

Check that these files exist and look sane:
- [automated_benchmark_latest.csv](automated_benchmark_latest.csv)
- `automated_benchmark_state.json`
- `logs/automated_benchmark/*`

For NPU rows in the suite CSV, inspect:
- `topology_id`
- `idle_pkg_watt`
- `avg_pkg_watt`
- `pseudo_npu_avg_pkg_watt`

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
- non-interactive `turbostat`

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
- configure sudoers for `turbostat` and reboot
- install/enable the systemd service
- set `power_cycle_cmd` in `benchmark_job.json`

## Sequence Lengths

Both benchmarks default to:
- `64,128,256,512,1024,2048,4096,8192`

For lengths above `512`, the scripts extend the learned position embeddings by repeating them so CPU and NPU benchmark the same effective backbone shape.

## Legacy Entry Point

[inference.py](inference.py) is now a legacy stub. Use the CPU and NPU benchmark scripts directly.
