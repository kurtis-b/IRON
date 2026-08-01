<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer

`transformer_layer` contains the paper-facing transformer-layer studies and
the retained internal NPU execution modes:

- `pattern/hybrid`
- `pattern/runlist`
- `pattern/offload`

The paper-facing taxonomy uses offload and runlist boundaries:

| Paper label | Repo mode | Role in this study |
| --- | --- | --- |
| `offload` | `offload` | Host-controlled transformer layer that offloads GEMM work to the NPU. |
| `runlist` | `runlist` | Fine-grained NPU operator sequence with explicit intermediate movement. |
| `coarse runlist` | `hybrid` | Runlist orchestration over coarse staged kernels; `hybrid` remains the internal CSV/schema key. |

Retained studies:

- `study/block`
- `study/end_to_end`
- `study/memory_tile_staging`
- `study/resource_usage`
- `study/host_comparison`
- `study/memcpy_bandwidth`
- `study/roofline`

The shared case matrix is:

- workloads: `encoder_bert`, `decoder_gpt2`
- families: `tinybert_512`, `baseline_768`, `baseline_1024`,
  `gpt2_512`, `gpt2_small_768`, `gpt2_medium_1024`
- sequence lengths: `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

If you are starting from a fresh clone, read `## Prerequisites` and then
`## First Run on a New Machine`. Nothing in this directory ships measured
results, so the smoke tests cannot run until you have generated some.

## Prerequisites

### Hardware

An AMD NPU (XDNA) and an AMD iGPU on the same host. `study/host_comparison`
benchmarks the iGPU as the host baseline, so a machine without one can run
every other study but not the full suite.

### System packages

The unattended runner shells out to all of these. Each must be on `PATH` (or
at the fallback location noted) for the user that runs the suite:

| Tool | Package | Used for |
| --- | --- | --- |
| `xrt-smi` | XRT | NPU power mode; also probed at `/opt/xilinx/xrt/bin`, `/usr/local/bin`, `/usr/bin` |
| `amd-ttm` | `amd-debug-tools` | TTM page-limit transitions; also probed at `~/.local/bin` |
| `turbostat` | `linux-tools-$(uname -r)` | package power sampling for every end-to-end row |
| `sensors` | `lm-sensors` | thermal gate between jobs |
| `rocm-smi` | ROCm | iGPU power sampling, and the thermal-gate fallback |
| `crontab` | `cron` | the `@reboot` hook that carries a run across reboots |

A missing `sensors` **and** `rocm-smi` fails at `start`, not mid-run:
`Unable to determine PC temperature from sensors or rocm-smi`.

> **NOTE:** `turbostat` is tied to the running kernel. `/usr/bin/turbostat` is
> only a dispatcher; the real binary ships in `linux-tools-$(uname -r)`. If you
> change or upgrade kernels — including the reboots this suite performs — the
> package for the new kernel may not be installed, and every end-to-end row
> then fails power sampling. Check with `sudo -n turbostat --version`; if it
> warns `turbostat not found for kernel ...`, install the package it names.

### Passwordless sudo

The runner executes from your crontab, so it uses `sudo -n` and cannot answer
a password prompt. Grant `NOPASSWD` for `xrt-smi`, `amd-ttm`, `turbostat`, and
`reboot`:

```bash
sudo visudo -f /etc/sudoers.d/transformer-layer-unattended
```

```text
<your-user> ALL=(root) NOPASSWD: /usr/bin/xrt-smi, /opt/xilinx/xrt/bin/xrt-smi, \
                                 /usr/local/bin/amd-ttm, /usr/bin/turbostat, \
                                 /usr/sbin/reboot, /usr/bin/systemctl reboot
```

> **NOTE:** list every `xrt-smi` path that exists on your machine, not just
> one. The runner resolves the binary with `shutil.which` first and only then
> falls back to the well-known locations, so a rule naming `/usr/bin/xrt-smi`
> alone still fails when `PATH` resolves `/opt/xilinx/xrt/bin/xrt-smi`. The
> symptom is `sudo: a password is required` followed by
> `Privileged setup failed for set_turbo with exit code 1` on the very first
> job.

Verify before starting a long run:

```bash
sudo -n xrt-smi examine -r all
sudo -n amd-ttm --help
sudo -n turbostat --version   # must not warn about the running kernel
```

### Python environment

Follow the repository `README.md` `## Installation (Linux)` first, then note
one deviation that matters here:

`requirements.txt` sets `--index-url https://download.pytorch.org/whl/cpu`,
so a plain install gives you a CPU-only torch. That wheel cannot run any
`study/host_comparison` job. Reinstall torch from a ROCm index instead —
pick the one matching your ROCm release from
[pytorch.org](https://pytorch.org/get-started/locally/):

```bash
pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/rocm<version> torch
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True` — ROCm exposes AMD GPUs through
the `cuda` device type. The suite was last validated with
`torch 2.9.1+rocm7.2.1`. This is only checked when the first `host_comparison`
job runs, job 821 of 888, so verify it up front rather than losing a day.

### Paths the runner assumes

Cron-launched jobs re-derive their environment from two hardcoded paths. There
is no override flag for either:

- `/opt/xilinx/xrt/setup.sh`
- `<repo>/ironenv/bin/activate` — the virtualenv must be named `ironenv` and
  live at the repository root

The repository root itself is derived from the module location, so any
checkout path works.

Finally, `/etc/modprobe.d/ttm.conf` must not exist when you call `start`; the
runner refuses otherwise, because it captures the normal TTM state at that
moment.

## First Run on a New Machine

The smoke tests replay a previous run's outputs as input fixtures, and results
trees are gitignored. So on a fresh clone you must generate those fixtures
first. These three studies depend on nothing but the NPU and each other, and
take minutes rather than hours:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all          # confirm "Power Mode : Turbo"

python3 -m iron.applications.transformer_layer.study.block.run \
  --family baseline_768 --seq-len 512
python3 -m iron.applications.transformer_layer.study.memory_tile_staging.run \
  --family baseline_768 --seq-len 512
python3 -m iron.applications.transformer_layer.study.memcpy_bandwidth.run
```

Run `block` before `memory_tile_staging`; the latter reads the former's CSV
through `--reference-input`.

These write to `results/block/results.csv`,
`results/memory_tile_staging/results.csv`, and
`results/memcpy_bandwidth/results.csv`. That is exactly the tree the smoke
tests look for first, so you can now run one without `--source-results-root`:

```bash
python3 -m iron.applications.transformer_layer.study.unattended_reboot \
  execution-smoke-test --log-level INFO
```

## Checking a Setup First

A full suite can run for a day or more, so validate the environment before
starting one. Both commands below install no boot hook and default
`--reboot-command` to a no-op, so they are safe to run interactively.

`execution-smoke-test` is the one to run first. It exercises a reduced 21-job
plan — `baseline_768` / `encoder_bert` at `seq_len=512` across all three
execution modes, plus the downstream exports and the manifest — so it touches
the NPU, the runner, and the output contract end to end:

```bash
python3 -m iron.applications.transformer_layer.study.unattended_reboot \
  execution-smoke-test --log-level INFO
```

`smoke-test` is a 3-job plan covering only the plot/regeneration path. It
measures nothing:

```bash
python3 -m iron.applications.transformer_layer.study.unattended_reboot \
  smoke-test --log-level INFO
```

Both need an existing results tree to use as an input fixture, and both
discover one automatically — `results/` first, then the most recent
`results_unattended_*` that has the files they need. Pass
`--source-results-root /path/to/results_unattended_<run_id>` to choose
explicitly. Their fixture requirements differ:

- `execution-smoke-test`: `block/results.csv`,
  `memory_tile_staging/results.csv`, `memcpy_bandwidth/results.csv`
- `smoke-test`: `block/results.csv`, `memory_tile_staging/results.csv`,
  `end_to_end/results_all_power.csv`, `end_to_end/tuning_all_power.csv`,
  `host_comparison/results.csv`

Only `execution-smoke-test` is reachable from the bootstrap above; `smoke-test`
additionally needs end-to-end and host-comparison output, which means a real
suite run has to have happened first.

Neither smoke plan carries privileged setup steps, so **set turbo yourself
first** — otherwise every measured row fails the turbo check.

The last job of `execution-smoke-test` checks more than that the expected files
appeared: it requires at least one measured row per CSV to have
`run_status=passed`. A broken environment still writes complete, well-formed
CSVs full of failed rows, so file existence alone would report success. If it
fails you get the first `failure_message` verbatim, which is usually enough to
identify the cause:

```text
Execution smoke produced outputs but measured nothing successfully:
.../end_to_end/results_all_power.csv: 3 row(s), none with run_status=passed.
First failure: CalledProcessError: Command '['sudo', '-n', 'turbostat', ...
```

## Running the Full Suite

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

sudo xrt-smi configure --pmode turbo

python3 -m iron.applications.transformer_layer.study.unattended_reboot start \
  --run-id full_suite_$(date +%Y%m%d_%H%M%S) \
  --run-user "$USER" \
  --log-level INFO
```

What to expect:

| Plan | Jobs | Observed wall clock |
| --- | --- | --- |
| `--suite-profile full` (default) | 888 | 11 h to 2 days |
| `--suite-profile paper` | 834 | ~20 h; helper studies restricted to `seq_len` 512/2048/8192 |
| `execution-smoke-test` | 21 | minutes |
| `smoke-test` | 3 | seconds; no measurement |

Wall clock swings by a factor of four depending on how much of `build/` is
already warm — compilation, not measurement, dominates a cold run. Preserve
`build/` between runs. A completed results root is about 2.4 GB.

Output lands in `results_unattended_<run_id>/`, which is gitignored. Run state
lives at `<results_root>/automation/state.json` and per-job logs at
`<results_root>/automation/logs/`.

`start` installs an `@reboot` entry in the **invoking user's** crontab, tagged
with a `# transformer-layer-unattended:<state path>` marker line. It is removed
when the run completes, when you call `stop`, and when a job fails
permanently. If you `sudo` the `start`, the entry lands in root's crontab
instead — don't.

A healthy run reboots **exactly twice**: once into a 26 GB TTM page limit
before the six `host_comparison_igpu_*_16384` jobs, and once back to the
normal limit before the fairness, plot-regeneration, and manifest tail. Both
are driven by the crontab hook, so the run continues unattended.

`--suite-profile paper` refuses to start into a results root that already
holds state or outputs. `full` does not, so use a fresh `--run-id`.

## Recovering a Stopped Run

```bash
STATE=/path/to/results_unattended_<run_id>/automation/state.json

python3 -m iron.applications.transformer_layer.study.unattended_reboot status --state "$STATE"
python3 -m iron.applications.transformer_layer.study.unattended_reboot stop   --state "$STATE"
python3 -m iron.applications.transformer_layer.study.unattended_reboot resume --state "$STATE" --log-level INFO
```

`resume` re-arms the crontab hook, retries the job that failed, and continues.
Per-study `--resume-input` means completed work is not repeated.

Two failure modes worth recognising:

- **`Privileged setup failed for set_turbo with exit code 1`**, with
  `sudo: a password is required` in the job log. The sudoers rule does not
  cover the `xrt-smi` path that `PATH` resolves — see `## Prerequisites`.
- **`TTM state still requires the same reboot-triggering action after
  reboot`**. The runner stopped itself rather than risk a reboot loop: after
  rebooting to restore the normal TTM state, the state still did not look
  normal. It compares `/sys/module/ttm/parameters/pages_limit` against the
  value recorded at `start`, within 1%, and also requires
  `/etc/modprobe.d/ttm.conf` to be gone. If `ttm.conf` is absent and the limit
  is nowhere near the 26 GB override (6815744 pages), the machine really is
  normal and the recorded baseline is stale — most often because the reboot
  landed on a different kernel, which derives a slightly different default.
  Update `normal_ttm_pages_limit` in `state.json` to the current value and
  `resume`.

## Comparing Runs Against a Reference

Two runs of the same code never produce identical numbers, so comparing by
equality is not useful. `study/compare_results_roots` applies per-mode
statistical tolerances instead:

```bash
python3 -m iron.applications.transformer_layer.study.compare_results_roots \
  --baseline /path/to/results_unattended_<reference> \
  --candidate /path/to/results_unattended_<new>
```

It matches rows on `(study_case_id, execution_mode, seq_len)`, requires
identifier columns to match exactly, and gates on the median and p90 of the
drift spread for `avg_latency_ms`, `effective_gflops_per_sec`, and
`avg_power_w`. `min_*` and `max_*` columns are reported but never gate — they
are single samples. `offload` is given a wider tolerance than `hybrid` and
`runlist` because it is roughly ten times noisier between runs. It exits
non-zero on `VERDICT: PROBLEM`.

**Check provenance before believing any comparison.** `results_manifest.json`
records the git commit, whether the tree was dirty, the platform, and the full
`xrt-smi examine` output; the comparator prints the differences. Runtime
changes can dominate code changes: between two runs of this suite an XRT
downgrade from 2.23.0 to 2.21.0 slowed `offload` by 19-39% at every
`seq_len >= 4096` while leaving `hybrid` and `runlist` within 0.6%, with
identical dispatch counts and identical selected configurations.

> **NOTE:** `system.platform` is captured when the manifest is written, which
> is the last job of the run. On a run that rebooted it therefore names the
> kernel at the end, not necessarily the one the measurements ran under.

Some differences are expected and are counted rather than flagged:
`pattern_label` moved from `Hybrid` to `Coarse runlist`, and the autotuner
legitimately picks a different configuration when two candidates are near-tied,
which also regroups the derived `resource_usage` and `roofline` CSVs.

## Outputs

A completed run writes these under its results root — `results/` for a manual
run, `results_unattended_<run_id>/` for a suite run. Neither tree is checked
in; both are produced by running the studies.

- `block/results.csv`
- `end_to_end/results_all_power.csv`
- `end_to_end/tuning_all_power.csv`
- `end_to_end/correctness_spot_checks.csv`
- `end_to_end/latency_variation.csv`
- `end_to_end/staging_ablation.csv`
- `end_to_end/fairness_repeatability.csv`
- `memory_tile_staging/results.csv`
- `resource_usage/dataflow_block_best_configs.csv`
- `resource_usage/hybrid_selected_ops.csv`
- `resource_usage/runlist_selected_ops.csv`
- `resource_usage/offload_selected_ops.csv`
- `host_comparison/results.csv`
- `host_comparison/fairness_repeatability.csv`
- `memcpy_bandwidth/results.csv`
- `roofline/kernel_points.csv`
- `roofline/implementation_points.csv`
- `results_manifest.json`

Check completeness of any results root with:

```bash
python3 -m iron.applications.transformer_layer.study.results_manifest \
  --results-root /path/to/results_unattended_<run_id> --suite-profile full
```

It exits non-zero when files or rows are missing.

## Manual Study Order

The unattended runner handles ordering for you. To drive the studies by hand:

1. `study.block.run`
2. `study.memory_tile_staging.run`
3. `study.end_to_end.run`
4. end-to-end helper studies
5. `study.host_comparison.run`
6. `study.memcpy_bandwidth.run`
7. `study.resource_usage.run`
8. `study.roofline.run`
9. `study.regenerate_plots`

Dependency notes:

- `memory_tile_staging` depends on `block`
- `host_comparison` depends on completed end-to-end outputs
- `resource_usage` depends on end-to-end results plus the matching build tree
- `roofline` depends on end-to-end, tuning, and memcpy-bandwidth outputs

Each study takes a `--output` path, so a manual run can write into whichever
results root you choose.

## Tests

The execution modes and the study logic are covered by two different suites:

```bash
# Study logic: pure-Python, no NPU, seconds.
pytest iron/applications/transformer_layer/study/

# Execution modes: compiles and runs each strategy on the NPU, slow.
pytest iron/applications/transformer_layer/pattern/
```

`pytest.ini` sets `python_files = test.py`, so the runner's own tests are not
collected by a directory-wide run and must be named explicitly:

```bash
pytest iron/applications/transformer_layer/study/test_unattended_reboot.py \
       iron/applications/transformer_layer/study/test_compare_results_roots.py
```

## Documentation

- `pattern/hybrid/README.md`
- `pattern/runlist/README.md`
- `pattern/offload/README.md`
- `study/block/README.md`
- `study/end_to_end/README.md`
- `study/memory_tile_staging/README.md`
- `study/resource_usage/README.md`
- `study/host_comparison/README.md`
- `study/memcpy_bandwidth/README.md`
- `study/roofline/README.md`
