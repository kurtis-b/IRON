<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Study

Status: `implemented`

## Goal

Benchmark the three full-layer NPU implementations starting from hidden states
only.

Compared execution modes:

- `dataflow`
- `runlist`
- `offload`

Current offload semantics:

- the study still starts from hidden states at the benchmark boundary
- offload performs `q/k/v` projections on NPU inside the pattern
- offload keeps one shared xclbin across all offloaded GEMMs
- host-side work remains softmax, GeLU, and residual add/layer norm

## Case Matrix

Families:

- `baseline_768`
- `baseline_1024`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

Total planned surface:

- `2 families x 9 seq lengths x 3 modes`

## Input Boundary

The study always starts from hidden states.

- shared hidden-states-only benchmark boundary
- shared synthetic weights from `pattern/reference.py`
- workload-based cases only in this first pass

Topology attribution is deferred in this study runner.

## Tuning Flow

This study tunes internal operators before the final end-to-end benchmark.

Configuration source:

- `dataflow_candidates.json`
- `runlist_candidates.json`
- `offload_candidates.json`

Flow per `(family, seq_len, mode)`:

- benchmark each internal operator candidate in isolation
- select the fastest passing candidate for that internal operator
- assemble the selected operator configs
- run the final end-to-end benchmark once with the selected config set

For `offload`, the current tuning unit remains one shared runtime GEMM config:

- `shared_gemm`

That shared config is applied across the offloaded `q/k/v`, attention, output,
and FFN GEMMs.

The candidate file is loaded by default. There is no CLI override for it in
this first pass.

## Owned Files

This study will own:

- `dataflow_candidates.json`
- `runlist_candidates.json`
- `offload_candidates.json`
- `cases.py`
- `modes.py`
- `power.py`
- `run.py`
- `test.py`

## Outputs

Canonical output:

- `results/end_to_end/tuning.csv`
- `results/end_to_end/results.csv`

`results.csv` is the planned input to the `igpu` study.

`tuning.csv` keeps one row per internal-operator candidate and marks the
selected isolated winner with `is_operator_best=True`.

Required metrics in each row include:

- `avg_latency_ms`
- `effective_gflops_per_sec`
- `avg_power_w`
- `effective_gflops_per_sec_per_watt`

Compatibility/runtime metrics:

- `host_qkv_precompute_ms`
  retained in the schema for study continuity; currently `0.0` for offload
  because the offload pattern now starts from hidden states and computes
  `q/k/v` on NPU

`effective_gflops_per_sec` is computed from the dominant dense transformer-layer
matmuls per forward pass:

- `q/k/v` projections
- attention-score and attention-output GEMMs
- output projection
- FFN up/down GEMMs

The study reports `effective_gflops_per_sec = flop_count / avg_latency_sec / 1e9`.
Small elementwise ops such as softmax, GeLU, residual adds, and layer norms are
excluded from the effective FLOP count.

Power measurement is best-effort:

- `--power-backend auto`
  tries `turbostat_pkgwatt` and falls back to no power measurement
- `--power-backend none`
  disables power measurement
- `--power-backend turbostat_pkgwatt`
  requires `turbostat`

Validation policy:

- `seq_len <= 512`
  compare against the shared golden reference
- `seq_len > 512`
  check that the output is finite without materializing the full attention
  reference

Entry point:

- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`

Convenience wrapper for powered ladder prefixes:

- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep`

Example:

- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep --max-seq-len 256`

The power-sweep helper defaults to:

- `--family all`
- `--mode all`
- `--power-backend turbostat_pkgwatt`
- `--warmup-iters 1`
- `--timed-iters 1`

Power sampling uses a conservative default cadence for `turbostat`:

- requested interval: `0.1 s`
- minimum interval floor: `0.1 s`

It writes merged outputs into the app-level results directory:

- `results/end_to_end/results_upto<max_seq_len>_power.csv`
- `results/end_to_end/tuning_upto<max_seq_len>_power.csv`

For a full ladder sweep through `16384`, the default filenames are:

- `results/end_to_end/results_all_power.csv`
- `results/end_to_end/tuning_all_power.csv`

Environment:

- `source /opt/xilinx/xrt/setup.sh`
- `source ironenv/bin/activate`
