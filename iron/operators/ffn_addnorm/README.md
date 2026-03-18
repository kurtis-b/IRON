<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined Add & Norm -> FFN -> Add & Norm Example for Ryzen NPU

Noting that we could try to reduce the number of cores for the Add & Norm blocks to 1 instead of 2, which would free up 1 core each for increasing the parallelism across the N dimension (so that nA=4 and nB=3 is possible instead of nB=2). 

However, by only having one core send two tiles' worth of data, the add & layer norm calculation would have to be repeated to get each tile across the columns of Up Projection's left mtx for each block of input rows with the core executing Add & Norm. This is because there's not enough space in Data Memory to store both blocks of inputs to Up Projection within the core, so the calculation would need to be repeated. E.g. if Up projection expects a 16x96 tile (two 8x96 tiles across the row dimension), and the core running Add & Norm works with 8x768 data to send the two 8x96 tiles over, if two 8x768 output data from the add & norm calculation can't be stored in Data Memory, then the add & norm calculation needs to be repeated on two 8x768 activation and two 8x768 residual data. The current design avoids this by having two cores generate the 8x768 add & norm output in parallel, and streaming the input tiles for Up projection through one core. 

Considering the objfifo depths were set to 1 in the Add & Norm cores in order to get the current best performance (using depth of 1 in those cores allowed for increasing the m tile dimension for the GEMMs), it's likely that changing the design so that it re-executes fused Add & Norm for each block of input rows (instead of just keeping them in Data Memory and copying repeatedly as in the current design) would lead to worse performance. Using the Mem Tile to route store that data and send back to the core executing Add & Norm isn't possible since two DMA channels are already used for the activation data and residual data (for eltwise add).