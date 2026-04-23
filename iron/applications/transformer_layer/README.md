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
- `end_to_end/offload_breakdown.csv`
- `end_to_end/runlist_launch_ablation.csv`
- `end_to_end/staging_ablation.csv`
- `end_to_end/search_validation.csv`
- `end_to_end/search_validation_failure_taxonomy.csv`
- `end_to_end/error_distribution.csv`
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
- `analysis/crossover_report.csv`

Manual-only correctness output:

- `end_to_end/real_weight_sanity.csv`

Paper-profile required outputs are narrower than the full suite. The default
submission-facing `paper_p0` bundle requires:

- `artifact_manifest.json`
- `block/results.csv`
- `memory_tile_staging/results.csv`
- `end_to_end/results_all_power.csv`
- `end_to_end/tuning_all_power.csv`
- `end_to_end/offload_breakdown.csv`
- `end_to_end/runlist_launch_ablation.csv`
- `end_to_end/staging_ablation.csv`
- `end_to_end/search_validation.csv`
- `end_to_end/search_validation_failure_taxonomy.csv`
- `end_to_end/error_distribution.csv`
- `host_comparison/results.csv`
- `memcpy_bandwidth/results.csv`
- `resource_usage/dataflow_block_best_configs.csv`
- `resource_usage/hybrid_selected_ops.csv`
- `resource_usage/runlist_selected_ops.csv`
- `resource_usage/offload_selected_ops.csv`
- `roofline/kernel_points.csv`
- `roofline/implementation_points.csv`
- `analysis/crossover_report.csv`

These appendix/debug outputs are intentionally optional for `paper_p0`:

- `end_to_end/correctness_spot_checks.csv`
- `end_to_end/latency_variation.csv`
- `end_to_end/fairness_repeatability.csv`
- `host_comparison/fairness_repeatability.csv`
- `end_to_end/real_weight_sanity.csv`

## Recommended Execution

Before NPU benchmark studies, configure turbo mode:

```bash
sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all
```

For unattended paper-profile execution:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.unattended_reboot start \
  --plan-kind paper_p0 \
  --run-id paper_p0_$(date +%Y%m%d_%H%M%S) \
  --run-user "$USER" \
  --log-level INFO
```

For the broader appendix-oriented full-suite execution, keep using:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.unattended_reboot start \
  --plan-kind full \
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

`paper_p0` narrows the automated submission-facing matrix to
`baseline_768` and `gpt2_small_768` at `seq_len={256,2048,8192}`. The full
plan keeps the wider family and sequence sweep, including the `16384`
host-comparison rows that require TTM transitions.

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
  --artifact-profile paper \
  --force
```

`publish_results_root` validates the source tree, copies the retained
paper-facing study outputs into `results/`, and writes
`artifact_manifest.json` so the published artifact root records the campaign
ID, git SHA, tool paths, study command lines, evaluation profile, and plan
kind. Use `--artifact-profile p0` or `--artifact-profile canonical` for the
legacy fuller bundles. Legacy
`transformer_layer_new/results` outputs are not canonical evidence.

For a reviewer-oriented reduced smoke:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron
python3 -m iron.applications.transformer_layer.study.reviewer_quickcheck.run \
  --run-id quickcheck_$(date +%Y%m%d_%H%M%S) \
  --run-user "$USER" \
  --reboot-command true \
  --artifact-profile paper \
  --log-level INFO
```

Manual order for `paper_p0`:

1. `study.block.run`
2. `study.memory_tile_staging.run`
3. `study.end_to_end.run`
4. end-to-end helper studies:
   `study.end_to_end.run_staging_ablation`,
   `study.offload_partitioning.run_query_block_sweep`,
   `study.runlist_launch_ablation.run`,
   `study.search_validation.run`,
   `study.correctness.error_distribution`
5. `study.host_comparison.run` with `--host-backends all`
6. `study.memcpy_bandwidth.run`
7. `study.resource_usage.run`
8. `study.roofline.run`
9. `study.analysis.crossover_report`
10. `study.regenerate_plots --artifact-profile paper`

Manual order for `full`:

1. `study.block.run`
2. `study.memory_tile_staging.run`
3. `study.end_to_end.run`
4. end-to-end helper studies:
   `study.end_to_end.run_correctness_spot_checks`,
   `study.end_to_end.run_latency_variation`,
   `study.end_to_end.run_staging_ablation`,
   `study.end_to_end.run_fairness_repeatability`
5. P0 extension studies:
   `study.offload_partitioning.run_query_block_sweep`,
   `study.runlist_launch_ablation.run`,
   `study.search_validation.run`,
   `study.correctness.error_distribution`
6. `study.host_comparison.run` with `--host-backends all`
7. `study.host_comparison.run_fairness_repeatability`
8. `study.memcpy_bandwidth.run`
9. `study.resource_usage.run`
10. `study.roofline.run`
11. `study.analysis.crossover_report`
12. `study.regenerate_plots`

Dependency notes:

- `memory_tile_staging` depends on `block`
- `offload_partitioning`, `runlist_launch_ablation`, `search_validation`, and
  `correctness.error_distribution` depend on completed end-to-end outputs
- `host_comparison` depends on completed end-to-end outputs
- `resource_usage` depends on end-to-end results plus the matching build tree
- `roofline` depends on end-to-end, tuning, and memcpy-bandwidth outputs
- `analysis.crossover_report` depends on end-to-end, roofline, and memcpy
  outputs

Before submission, run `study.correctness.real_weight_sanity` manually on one
encoder payload and one decoder payload at `seq_len=2048`. It remains outside
the default unattended `paper_p0` plan because it depends on local exported
payloads.

## Documentation

- `study/block/README.md`
- `study/end_to_end/README.md`
- `study/memory_tile_staging/README.md`
- `study/resource_usage/README.md`
- `study/host_comparison/README.md`
- `study/memcpy_bandwidth/README.md`
- `study/roofline/README.md`
