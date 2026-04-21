<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer

`transformer_layer` contains the paper-facing transformer-layer studies and
the retained NPU execution patterns:

- `pattern/hybrid`
- `pattern/runlist`
- `pattern/offload`

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
  `gpt2_small_768`, `gpt2_medium_1024`
- sequence lengths: `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

Canonical outputs live under a results root such as `results/` or
`results_unattended_<run_id>/`, including:

- `artifact_manifest.json`
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

## Recommended Execution

Before NPU benchmark studies, configure turbo mode:

```bash
sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all
```

For unattended full-suite execution:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.unattended_reboot start \
  --run-id full_suite_$(date +%Y%m%d_%H%M%S) \
  --run-user "$USER" \
  --log-level INFO
```

To resume a stopped unattended run:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.unattended_reboot resume \
  --state /path/to/results_unattended_<run_id>/automation/state.json \
  --log-level INFO
```

The unattended runner keeps normal jobs in one session, reboots only for TTM
transitions needed by the iGPU `16384` host-comparison rows, and regenerates
plots after restoring the normal TTM state.

The unattended runner must be able to invoke `xrt-smi`, `amd-ttm`, and
`reboot` without an interactive password prompt.

After a full unattended run completes, publish the validated artifact tree into
the canonical `results/` location:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.publish_results_root \
  --source-root /path/to/results_unattended_<run_id> \
  --force
```

`publish_results_root` validates the source tree, copies the retained
paper-facing study outputs into `results/`, and writes `artifact_manifest.json`
so the published artifact root records the campaign ID, git SHA, tool paths,
and study command lines. Legacy `transformer_layer_new/results` outputs are not
canonical evidence.

Manual order:

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

## Documentation

- `study/block/README.md`
- `study/end_to_end/README.md`
- `study/memory_tile_staging/README.md`
- `study/resource_usage/README.md`
- `study/host_comparison/README.md`
- `study/memcpy_bandwidth/README.md`
- `study/roofline/README.md`
