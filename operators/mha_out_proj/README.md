<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined MHA with Output Projection

## ObjectFifo parallelization

### Parallel seq
- inQ (inQ2)->memQ, split
- memO (memO2)->outO, join

### Parallel seq->heads
- memA
- outA
- memP
- outP
- scaleOF
- outOProj

### Parallel heads
- inK (inK2)->memK, split
- inV (inV2)->memV, split
- inOW (inOW2)->memOW, split

TODO: Adjust objectfifo depth and execution with accumulate in MT