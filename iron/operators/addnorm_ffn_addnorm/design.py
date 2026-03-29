# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

from iron.operators.addnorm_ffn_addnorm.topology import addnorm_ffn_addnorm_design
from iron.operators.ffn_addnorm.design import my_matmul as _ffn_addnorm_matmul


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Block 3 AddNormFFNAddNorm Design",
        description="Resolve retained thesis topology parameters for Block 3",
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--topology-id", type=str, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            addnorm_ffn_addnorm_design(
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                topology_id=args.topology_id,
            ),
            sort_keys=True,
        )
    )


def fused_addnorm_ffn_addnorm(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    down_proj_depth,
    n_aie_cols,
    nA_tiles_distributed,
    nB_tiles_distributed,
    dtype_in_str,
    dtype_out_str,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    gelu_stage,
    ln2_weight_file,
    stage_only=None,
    archive=None,
    generate_taps=False,
):
    return _ffn_addnorm_matmul(
        dev,
        M,
        K,
        N,
        m,
        k,
        n,
        down_proj_depth,
        n_aie_cols,
        nA_tiles_distributed,
        nB_tiles_distributed,
        dtype_in_str,
        dtype_out_str,
        emulate_bf16_mmul_with_bfp16,
        trace_size,
        gelu_stage,
        ln2_weight_file,
        stage_only,
        archive,
        generate_taps,
    )


if __name__ == "__main__":
    main()
