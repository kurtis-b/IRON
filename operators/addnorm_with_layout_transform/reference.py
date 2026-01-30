# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import torch
from operators.common.utils import torch_dtype_map


def generate_golden_reference(
    M: int, K: int, m: int, k: int, s: int, debug_idxs=False, dtype="bf16", seed=42
):
    torch.manual_seed(seed)
    val_range = 4
    if debug_idxs:
        input1_tensor = torch.ones(
            M, K, dtype=torch_dtype_map[dtype]
        )  # Layer norm of all 1's will result in 0's
        input2_tensor = torch.arange(M * K, dtype=torch_dtype_map[dtype]).reshape(M, K)
        weights = torch.ones(K, dtype=torch_dtype_map[dtype])
    else:
        input1_tensor = torch.rand(M, K, dtype=torch_dtype_map[dtype]) * val_range
        input2_tensor = torch.rand(M, K, dtype=torch_dtype_map[dtype]) * val_range
        weights = torch.rand(K, dtype=torch_dtype_map[dtype]) * val_range

    # Compute layer norm followed by addition
    intermediate_tensor = torch.nn.functional.layer_norm(
        input1_tensor, normalized_shape=(K,), weight=weights, bias=None
    )
    output_tensor = intermediate_tensor + input2_tensor
    output_tensor = output_tensor.flatten()

    # The data layout of the output is based on the microkernel dims for the GEMM mmul API,
    # which is what this design is supposed to feed into. So we need to rearrange the output
    # tensor into contiguous tiles of (m x k), where within each tile the data is laid out in (m x s) subtiles.
    output_rearranged = torch.empty_like(output_tensor)
    num_m_tiles = M // m
    num_k_tiles = K // k
    for m_tile in range(num_m_tiles):
        for k_tile in range(num_k_tiles):
            for k_subtile in range(k // s):
                for m_sub in range(m):
                    for t_idx in range(s):
                        src_row = m_tile * m + m_sub
                        src_col = k_tile * k + k_subtile * s + t_idx
                        dst_index = (
                            m_tile * num_k_tiles * m * k
                            + k_tile * m * k
                            + k_subtile * m * s
                            + m_sub * s
                            + t_idx
                        )
                        output_rearranged[dst_index] = output_tensor[
                            src_row * K + src_col
                        ]

    # Save output to file
    if debug_idxs:
        torch.set_printoptions(threshold=float("inf"), sci_mode=False)
        cwd = Path(__file__).parent
        output_file = cwd / "golden_reference_output.txt"
        with open(output_file, "w") as f:
            f.write(f"M={M}, K={K}, m={m}, k={k}, s={s}, dtype={dtype}\n")
            f.write(f"Weights:\n{weights}\n")
            f.write(f"Output shape: {output_rearranged.shape}\n")
            f.write(f"Output (rearranged):\n{output_rearranged}\n")

    return {
        "input1": input1_tensor,
        "input2": input2_tensor,
        "weight": weights,
        "output": output_rearranged,
    }
