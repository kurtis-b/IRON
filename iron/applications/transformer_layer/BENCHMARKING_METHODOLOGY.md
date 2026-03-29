<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking Methodology

## Scope

The DesignPats app evaluates:

- `Dataflow`
- `Runlist`
- `GEMM offload`

The benchmark boundary is hidden states only. There is no public study surface
that begins from supplied `Q/K/V/R`.

The authoritative block-contract and topology-policy note is:

- `docs/dataflow_operator_contracts.md`

The retained implementation policy is to stay as close to `devel` as practical:

- keep shared-kernel and shared-framework edits minimal
- keep `Runlist` and `GEMM offload` close to their imported baseline behavior
- concentrate thesis-specific changes in Dataflow blocks and thesis-app glue

## Study Set

The retained studies are:

1. `dataflow_blocks`
2. `design_patterns_end_to_end`
3. `gpu_compare_end_to_end_igpu`
4. `reconfiguration_overhead`

For the retained Dataflow studies, the checked-in manifests pin the intended
`block1_topology_id`, `block2_topology_id`, and `block3_topology_id` values for
the `768 / 3072 / 12` and `1024 / 4096 / 16` families. Manifest loading
normalizes those `layer_spec` payloads through `TransformerLayerSpec`, so both
checked-in and ad hoc studies are validated against the same retained layer
surface before benchmarking starts. The checked-in long-ladder NPU manifests
also use `continue_on_error=true` so known retained runtime failures are
recorded as failure rows and support-matrix entries instead of aborting the
entire study sweep.

Retained model families:

- `768 / 3072 / 12`
- `1024 / 4096 / 16`

Retained sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

The reconfiguration-overhead study uses representative points:

- `64`
- `512`
- `16384`

## Measurements

Per completed row, the harness records:

- `avg_latency_ms`
- `timed_total_sec`
- `measured_inference_count`
- `compile_setup_time_ms`
- per-stage timing columns when available
- throughput / operational intensity / roofline metrics
- power / energy / efficiency metrics
- dispatch count and artifact-count metadata
- retained block topology IDs/families when the selected execution surface has
  them

Block-study rows report block-local timings.

End-to-end rows report implementation-local timings:

- `Dataflow`: block timing breakdown
- `Runlist`: projection plus stitched runlist timing
- `GEMM offload`: projection plus GEMM timing, with host-side preprocessing and postprocessing

Derived reporting surfaces preserve the same retained topology metadata:

- parity CSV outputs keep stable topology columns
- support-matrix outputs preserve topology IDs/families
- bottleneck summaries retain the topology IDs from the summarized rows
- iGPU compare rows preserve the selected best-NPU topology under
  `reference_npu_*` columns

For the retained Dataflow block contract:

- `block1_qkv_proj` returns `q/k/v [num_heads, seq, head_dim]`
- head-major means each head is laid out contiguously in memory
- `block2_mha_out_proj` is responsible for consuming that layout

## FLOP Model

The roofline model is execution-mode aware.

- `block1_qkv_proj`: Q/K/V GEMMs only
- `block2_mha_out_proj`: attention score GEMM, attention output GEMM, output projection GEMM
- `block3_addnorm_ffn_addnorm`: FFN up and FFN down GEMMs
- `dataflow`, `runlist`, `gemm_offload`: full hidden-states layer GEMM set
- `gemm_offload_gemm_sequence`, `runlist_gemm_sequence`: the same GEMM-only subgraph used for the reconfiguration study

The FLOP estimate is GEMM-centric; it intentionally does not try to model
softmax, GeLU, or add/norm FLOPs in detail.

## Power Method

### NPU

NPU power is recorded as pseudo-NPU power:

- collect package power with `turbostat`
- collect a quiescent baseline before the timed probe
- subtract the baseline from package power
- clamp negative values to zero

Reported fields include:

- `avg_power_w`
- `max_power_w`
- `energy_j`
- `flops_per_joule`
- `gflops_per_joule`
- `raw_package_avg_power_w`
- `raw_package_max_power_w`
- `quiescent_package_power_w`

### iGPU

iGPU power is recorded with `rocm-smi` during the iGPU compare study.

## Unattended Runs

The unattended pipeline is driven by:

- `run_study_pipeline.py`
- `study/designpats_pipeline.json`

Between benchmark-producing steps, the pipeline waits for thermal recovery using
`k10temp` `Tctl` rather than attempting external power cycling.

For the retained Dataflow studies, the checked-in pipeline now facets plot
steps on `block2_topology_id` so generated SVG/HTML outputs stay grouped by the
resolved retained Block 2 topology rather than only by study case. Plot
`facet_key` values are validated on pipeline-config load against the stable
result-schema columns so mistyped retained topology keys fail early.

## Correctness

End-to-end parity is run on the short-sequence surface:

- `seq_len=64`
- `seq_len=128`

for:

- `dataflow`
- `runlist`
- `gemm_offload`

The parity reference is the hidden-states-only CPU reference layer.

Parity uses the same optional retained `block1_topology_id`,
`block2_topology_id`, and `block3_topology_id` override surface as the main
benchmark path, so parity and benchmark rows stay aligned when a study pins a
specific retained topology.

## Operator Test Coverage

The retained operator-test policy is:

- one `generate_test_params()` helper per operator
- one test method per operator
- params built from workload dimensions and topology dimensions
- constructibility owned by the design/operator layer, not prefiltered in the
  test helper
