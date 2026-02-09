<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined FFN Example for Ryzen NPU

With this design, the workloads run at each core are as follow:

Each up projection core: $\frac{MKN}{n_a\times n_b}\times\frac{K}{k\times d_m}$.

Each down projection core: $\frac{MNK}{n_a\times n_b}$.

With $n_a$=`n_a_tiles_distributed`, $n_b$=`n_b_tiles_distributed`, $d_m$=`down_proj_depth` in `design.py`. 

As can be seen, the down projection cores execute less compute when $K\neq k \times d_m$. Thus, in these cases, the up projection becomes the bottleneck. Due to limitations with the DMA BDs, $d_m=8$ seems to be the largest that can be used. The test results show this to be the case, as the configuration where $K\neq k \times d_m$ performs the best. However, some configurations are currently failing, which may be due to the output objfifos at the down projection cores being set to 0. This needs some investigation, but the ones passing can be used for bert inference.

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