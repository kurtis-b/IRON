<!--
SPDX-FileCopyrightText: Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 🦾 IRON Transformer Layer Study

<a href="https://discord.gg/cW99Ds85e8">
    <img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord" /></a>
<a href="https://github.com/amd/iron/releases/latest" title="Download the latest release">
   <img src="https://img.shields.io/github/v/release/amd/iron?include_prereleases" alt="Latest Release" /></a>
<a href="https://github.com/amd/iron/actions" title="Check out our tests">
   <img src="https://github.com/amd/iron/actions/workflows/small.yml/badge.svg" alt="Iron Tests" /></a>
<a href="https://github.com/amd/iron/blob/main/CONTRIBUTING.md" title="Contribution Guide">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" /></a>
<a href="https://github.com/amd/iron/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache-yellow.svg" alt="license: Apache" /></a>

<p align="center">
   <img src="./images/XDNA2.png" alt="IRON on AMD Ryzen AI" style="max-width: 100%; height: auto;">
</p>

This repository's primary application compares three ways to execute a full
transformer layer on AMD Ryzen™ AI NPUs. It measures how the boundary between
the host and NPU affects latency, effective throughput, power efficiency, and
device-resource use.

**Choose a run:** [validate the complete setup in minutes](#quick-run) or
[launch the full benchmark suite](#full-benchmark-suite). For recovery,
troubleshooting, and every study entrypoint, use the
[detailed transformer-layer guide](./iron/applications/transformer_layer/README.md).

## What the Study Compares

| Paper label | Repo mode | Execution boundary |
| --- | --- | --- |
| `offload` | `offload` | The host executes the layer and offloads its GEMMs to the NPU. |
| `runlist` | `runlist` | A fine-grained NPU operator sequence moves intermediates explicitly. |
| `coarse runlist` | `hybrid` | An NPU runlist sequences a few fused, staged kernels. |

The case matrix covers BERT encoder and GPT-2 decoder layers, six model
families, and sequence lengths from 64 through 16384 tokens. Seven studies
measure block tuning, end-to-end behavior, memory-tile staging, host/iGPU
comparison, memcpy bandwidth, resource use, and roofline placement. The source
tree does not include measured results; each run produces its own result tree.

## Installation (Linux)

The documented and validated path is **Ubuntu 24.04**, **Python 3.12**, and
**ROCm 7.2.1** on a machine with both an AMD XDNA NPU and AMD iGPU. The
21-job quick run uses both devices. The three fixture-generation commands are
NPU-only, but the final execution smoke test also measures the iGPU baseline.

### 1. Install the device stacks

1. Install the XDNA driver and XRT using the
   [MLIR-AIE instructions](https://github.com/Xilinx/mlir-aie?tab=readme-ov-file#install-the-xdna-driver-and-xrt).
   The study expects XRT's setup script at `/opt/xilinx/xrt/setup.sh`.
2. Install ROCm 7.2.1 for the Ryzen iGPU using the
   [AMD Ryzen Linux instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/native_linux/install-ryzen.html).
3. Install the build, Python, power-sampling, temperature, and cron packages:

   ```bash
   sudo apt update
   sudo apt install \
     build-essential clang clang-14 lld lld-14 cmake ninja-build uuid-dev \
     python3-venv python3-pip \
     "linux-tools-$(uname -r)" lm-sensors cron
   ```

`turbostat` is supplied by the `linux-tools` package for the running kernel.
After a kernel change, install the corresponding package before rerunning the
study.

### 2. Create the required Python environment

Run these commands from the repository root. The runner deliberately expects
the virtual environment at `<repo>/ironenv`; a differently named or located
environment will not work.

```bash
python3 -m venv ironenv
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r iron/applications/transformer_layer/requirements.txt
```

The repository requirements initially install CPU-only Torch. The application
requirements replace it with the validated `torch 2.9.1+rocm7.2.1` wheel for
Ubuntu 24.04 and Python 3.12.

### 3. Allow non-interactive power sampling

The execution smoke test invokes `sudo -n turbostat`, so it cannot prompt for a
password. Add the narrow rule below, replacing `<your-user>` with your login:

```bash
sudo visudo -f /etc/sudoers.d/transformer-layer-smoke
```

```text
<your-user> ALL=(root) NOPASSWD: /usr/bin/turbostat
```

The full suite needs additional passwordless commands; use its
[complete prerequisite list](./iron/applications/transformer_layer/README.md#passwordless-sudo)
before starting it.

### 4. Verify the setup

```bash
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh

python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
xrt-smi examine -r all
rocm-smi --showpower --json
sensors
sudo -n turbostat --version
command -v crontab
```

Python must report 3.12, the Torch command must report `True`, `xrt-smi` must
see the NPU, and `rocm-smi`, `turbostat`, and the `crontab` lookup must complete
successfully. It is okay if `sensors` does not recognize the machine because
`rocm-smi` can also supply the runner's temperature reading.

## Quick Run

Start at the repository root with the environment active. First put the NPU in
turbo mode and generate the three input fixtures required by the smoke runner:

```bash
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh

sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all          # Confirm "Power Mode : Turbo".

python3 -m iron.applications.transformer_layer.study.block.run \
  --family baseline_768 --seq-len 512
python3 -m iron.applications.transformer_layer.study.memory_tile_staging.run \
  --family baseline_768 --seq-len 512
python3 -m iron.applications.transformer_layer.study.memcpy_bandwidth.run
```

Keep that order: the memory-tile staging study reads the block-study CSV. The
commands create:

- `iron/applications/transformer_layer/results/block/results.csv`
- `iron/applications/transformer_layer/results/memory_tile_staging/results.csv`
- `iron/applications/transformer_layer/results/memcpy_bandwidth/results.csv`

Now run the reduced end-to-end plan:

```bash
python3 -m iron.applications.transformer_layer.study.unattended_reboot \
  execution-smoke-test --log-level INFO
```

This runs 21 jobs for `baseline_768`, BERT encoder, and sequence length 512
across all three execution modes, then exercises the downstream studies and
output manifest. It takes minutes, installs no boot hook, and does not reboot.
Success ends with a completed state and a new
`iron/applications/transformer_layer/results_unattended_execution_smoke_*`
directory. Per-job logs are under its `automation/logs/` directory.

## Full Benchmark Suite

The full suite covers the entire shared case matrix and is designed to continue
across the two reboots needed by the longest iGPU cases.

| Profile | Jobs | Expected wall clock |
| --- | ---: | --- |
| `full` (default) | 888 | 11 hours to 2 days |
| `paper` | 834 | About 20 hours |

Before launching it, complete the detailed guide's
[full prerequisites](./iron/applications/transformer_layer/README.md#prerequisites),
including `amd-ttm`, passwordless NPU/TTM/reboot commands, and the TTM-state
check. A completed run uses about 2.4 GB. Do not invoke `start` with `sudo`.

```bash
python3 -m iron.applications.transformer_layer.study.unattended_reboot start \
  --run-id full_suite_$(date +%Y%m%d_%H%M%S) \
  --run-user "$USER" \
  --log-level INFO
```

See [Running the Full Suite](./iron/applications/transformer_layer/README.md#running-the-full-suite)
for boot-hook behavior, status and recovery commands, result completeness, and
the paper profile.

## Results and Documentation

Study runs write CSV measurements, SVG/PNG plots, automation logs, and a
provenance manifest under `iron/applications/transformer_layer/`. Results are
gitignored and are not part of the repository checkout.

- [Detailed transformer-layer guide](./iron/applications/transformer_layer/README.md)
- [Output inventory](./iron/applications/transformer_layer/README.md#outputs)
- [Execution-mode and per-study documentation](./iron/applications/transformer_layer/README.md#documentation)
- [Comparing two result trees](./iron/applications/transformer_layer/README.md#comparing-runs-against-a-reference)

## Other IRON Components

IRON is a close-to-metal Python API built on MLIR-AIE for AMD Ryzen AI NPUs.
The transformer application composes the reusable kernels and operators in the
rest of the repository:

- [NPU operators](./iron/operators/)
- [AIE kernels](./aie_kernels/)
- [Llama 3.2 1B reference application](./iron/applications/llama_3.2_1b/)
- [Contribution guide](./CONTRIBUTING.md)
- [License](./LICENSE)

The IRON Python API is described in:

> E. Hunhoff, J. Melber, K. Denolf, A. Bisca, S. Bayliss, S. Neuendorffer,
> J. Fifield, J. Lo, P. Vasireddy, P. James-Roxby, and E. Keller. “Efficiency,
> Expressivity, and Extensibility in a Close-to-Metal NPU Programming
> Interface.” 33rd IEEE International Symposium on Field-Programmable Custom
> Computing Machines, 2025. [arXiv:2504.18430](https://arxiv.org/abs/2504.18430)

---

<p align="center">Copyright&copy; 2025-2026 Advanced Micro Devices, Inc.</p>
