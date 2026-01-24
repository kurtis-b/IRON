<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined FFN Example for Ryzen NPU

## Notes
To generate visualization of the routing, a tool in the mlir-aie repo can be used. It's assumed that the tests have been run and the `build` directory is present in the project's root. 

As an example example, generating the routes for design `ffn_512x768x3072_64x48x96_8_4_4_None_0_0` can be done as below:
```
mkdir routes && cd routes
aie-opt --aie-create-pathfinder-flows --aie-find-flows <iron-dir>/build/ffn_512x768x3072_64x48x96_8_4_4_None_0_0.mlir.prj/input_with_addresses.mlir | aie-translate --aie-flows-to-json > example.json
```
From there, run:
```
python3 <mlir-aie-dir>/tools/aie-routing-command-line/visualize.py -j example.json
```