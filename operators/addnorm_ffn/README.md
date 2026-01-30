<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined Add & Norm -> FFN -> Add & Norm Example for Ryzen NPU

For the large BERT workloads, the expected performance is essentially the maximum between two stages: up projection stage and fused down projection + GeLU stage. Keep in mind that by pipelining these operations, each stage has only half of the total AIE cores to work with compared to having all resources available if each operation is executed sequentially. Therefore, the expected performance should be the maximum across all stages, with the latency of each stage based on the latency of an individual operation executing with the total resources used for that operation, and adding latencies together for fused operations. It's easiest to use the measurements from other tests for these calculations.

To calculate theoretical performance for `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_gelustage1` as an example:

- Get the performance of `gemm_512x768x3072_64x48x96_4cols_prioaccFalse_emubf16True_batch1_stridedims000`
- Get the performance of `gelu_8_cols_2_channels_1572864_tile_6144`
- Get the performance of `gemm_512x3072x768_64x96x48_4cols_prioaccFalse_emubf16True_batch1_stridedims000`

Performance measurements show:
- `gemm_512x768x3072_64x48x96_4cols_prioaccFalse_emubf16True_batch1_stridedims000`: ~782ms
- `gelu_8_cols_2_channels_1572864_tile_6144`: ~348ms
- `gemm_512x3072x768_64x96x48_4cols_prioaccFalse_emubf16True_batch1_stridedims000`: ~1171ms

GeLU fused with up projection gives theoretical performance of `max(782+348,1171)=1171ms`.
GeLU fused with down projection gives theoretical performance of `max(782,1171+348)=1519ms`.
Interestingly enough, GeLU fused with down projection seems to give the better performance. Below are the data for the current design point that gives the best performance:
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_gelustage0`: 2027ms
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_gelustage1`: 1397.9ms
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_stageonly0_gelustage0`: 2074.9ms
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_stageonly1_gelustage0`: 1379.1ms
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_stageonly0_gelustage1`: 1364.3ms
- `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_stageonly1_gelustage1`: 1618.8ms
Based on the performance of the isolated stages, i.e. max of last two vs max of the 3rd and 4th points, GeLU being in the down projection stage is better. This warrants some thought as to why it happens--maybe due to the different data movement pattern, and how tiles across the output columns are stored in MTs for down projection?

NOTE: `design_alternate.py` is a design implementing an alternate data movement pattern for GEMM. Part of the data movement here is what is executed for the down projection stage in `design.py`. It's just kept here for reference as a first step towards the FFN design, and not used in the operator or tests.

## Notes
To generate visualization of the routing, a tool in the mlir-aie repo can be used. It's assumed that the tests have been run and the `build` directory is present in the project's root. 

As an example example, generating the routes for design `ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_gelustage1` can be done as below:
```
mkdir routes && cd routes
aie-opt --aie-create-pathfinder-flows --aie-find-flows <iron-dir>/build/ffn_512x768x3072_64x48x96_8cols_dprojdepth8_nA4_nB4_gelustage1.mlir.prj/input_with_addresses.mlir | aie-translate --aie-flows-to-json > example.json
```
From there, run:
```
python3 <mlir-aie-dir>/tools/aie-routing-command-line/visualize.py -j example.json
```