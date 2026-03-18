<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BERT Encoder Benchmarks

`applications/bert` now has two explicit benchmark entrypoints:
- [cpu_inference.py](/home/agi-demo/iron/applications/bert/cpu_inference.py): Hugging Face CPU baseline using `BertModel(add_pooling_layer=False)`
- [npu_inference.py](/home/agi-demo/iron/applications/bert/npu_inference.py): local NPU benchmark using the `encoder_pipeline` operator
- [automated_benchmark.py](/home/agi-demo/iron/applications/bert/automated_benchmark.py): resumable case runner with optional topology autotune, power logging, and power-cycle hooks
- [run_automated_benchmark_job.py](/home/agi-demo/iron/applications/bert/run_automated_benchmark_job.py): wrapper that runs the automation harness from a JSON job file

The shared code under `src/` is now NPU-only. It exists to build the encoder-pipeline-backed BERT backbone used by `npu_inference.py`. The CPU benchmark uses Hugging Face directly and does not go through `src/`.

## Model Files

Download the base BERT weights and config from Hugging Face:
- `model.safetensors`
- `config.json`

The scripts expect the safetensors file on the command line. The JSON config passed to the scripts is the local app config under `applications/bert/config/`.

## Installation

1. Follow the repository root setup instructions so `ironenv` is available.
2. Install the extra Python packages:

```bash
python3 -m pip install -r applications/bert/requirements_bert.txt
```

## CPU Benchmark

Run the Hugging Face baseline:

```bash
cd applications/bert
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
- uses Hugging Face `BertModel(add_pooling_layer=False)`
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

## NPU Benchmark

Run the NPU benchmark with `encoder_pipeline`:

```bash
cd applications/bert
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
- builds a minimal local BERT backbone: embeddings + `encoder_pipeline`
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
- writes results to `npu_benchmark_latest.csv`
- checkpoints the CSV after each completed sequence length

## Automated Benchmarking

Use the automation harness when you want:
- a case-by-case CPU/NPU sweep
- resumable state across reboots
- optional per-case power logging
- optional NPU topology autotuning

Example:

```bash
cd applications/bert
python3 automated_benchmark.py /path/to/model.safetensors config/config.json \
  --modes cpu,npu \
  --seq-lens 64,128,256,512,1024,2048,4096,8192 \
  --runs-per-sample 100 \
  --warmup-runs 10 \
  --npu-topology-policy cache \
  --power-backend turbostat \
  --power-cycle-cmd "sudo reboot"
```

Behavior:
- writes a resumable state file
- writes one master CSV summarizing all completed cases
- writes per-case child benchmark CSVs and power logs under `logs/automated_benchmark`
- if `--power-cycle-cmd` is set, it runs one case, records results, invokes the hook, and exits
- after the machine comes back, rerun the same command to continue from the saved state
- for NPU cases with `turbostat`, it also measures an idle baseline before each case and writes:
  - `idle_pkg_watt`
  - `pseudo_npu_avg_pkg_watt`
  - `pseudo_npu_max_pkg_watt`

## Systemd Resume Flow

For unattended reboot/resume:

1. Copy the example job file and edit it:

```bash
cd applications/bert/systemd
cp benchmark_job.example.json benchmark_job.json
```

2. Review [benchmark_job.example.json](/home/agi-demo/iron/applications/bert/systemd/benchmark_job.example.json) and set:
- weights/config paths
- `modes`
- sequence lengths
- CPU thread counts
- NPU topology policy
- output/state/log paths
- `power_cycle_cmd`

3. Test the resolved command without systemd:

```bash
cd applications/bert
python3 run_automated_benchmark_job.py systemd/benchmark_job.json --print-command
```

4. Install the service:

```bash
sudo cp /home/agi-demo/iron/applications/bert/systemd/bert-automated-benchmark.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bert-automated-benchmark.service
sudo systemctl start bert-automated-benchmark.service
```

5. Watch the service log:

```bash
tail -f applications/bert/logs/systemd/bert-automated-benchmark.log
```

Notes:
- the service expects the live job file at:
  - `applications/bert/systemd/benchmark_job.json`
- the service runs [run_automated_benchmark_job.py](/home/agi-demo/iron/applications/bert/run_automated_benchmark_job.py), which in turn invokes [automated_benchmark.py](/home/agi-demo/iron/applications/bert/automated_benchmark.py)
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
agi-demo ALL=(root) NOPASSWD: /usr/bin/turbostat
agi-demo ALL=(root) NOPASSWD: /usr/sbin/reboot, /usr/bin/systemctl reboot
```

## Preflight

Before the first unattended run, do this in order.

1. Verify privileged helpers work non-interactively

```bash
sudo -n turbostat --Summary --quiet --show PkgWatt,CorWatt,GFXWatt,RAMWatt --interval 0.5 --num_iterations 2 sleep 1
```

Do not run reboot until you are ready for it, but make sure the sudoers rule exists for the reboot command you plan to use.

2. Warm the NPU topology cache under supervision

This generates a fresh best-topology entry per sequence length for the current machine and software state.

```bash
cd /home/agi-demo/iron/applications/bert
source /opt/xilinx/xrt/setup.sh
source /home/agi-demo/iron/ironenv/bin/activate
python3 automated_benchmark.py model.safetensors config/config.json \
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
cd /home/agi-demo/iron/applications/bert
python3 automated_benchmark.py model.safetensors config/config.json \
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
- [automated_benchmark_latest.csv](/home/agi-demo/iron/applications/bert/automated_benchmark_latest.csv)
- `automated_benchmark_state.json`
- `logs/automated_benchmark/*`

For NPU rows in the suite CSV, inspect:
- `topology_id`
- `idle_pkg_watt`
- `avg_pkg_watt`
- `pseudo_npu_avg_pkg_watt`

5. Enable reboot only after the short supervised run looks correct

At that point:
- set `power_cycle_cmd` in `benchmark_job.json`
- install/enable the systemd service
- start the unattended run

## Sequence Lengths

Both benchmarks default to:
- `64,128,256,512,1024,2048,4096,8192`

For lengths above `512`, the scripts extend the learned position embeddings by repeating them so CPU and NPU benchmark the same effective backbone shape.

## Legacy Entry Point

[inference.py](/home/agi-demo/iron/applications/bert/inference.py) is now a legacy stub. Use the CPU and NPU benchmark scripts directly.
