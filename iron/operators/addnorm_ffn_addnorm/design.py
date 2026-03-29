# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

_BLOCK3_TOPOLOGIES = {
    (768, 3072): [
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 1,
        },
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        },
    ],
    (1024, 4096): [
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        }
    ],
}


def addnorm_ffn_addnorm_topologies(
    *,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    try:
        topologies = _BLOCK3_TOPOLOGIES[(hidden_size, intermediate_size)]
    except KeyError as exc:
        raise ValueError(
            "Block 3 currently supports only the retained thesis families 768/3072 and 1024/4096"
        ) from exc
    return [
        {
            **candidate,
            "topology_id": _topology_id(candidate),
            "topology_family": "pipelined_addnorm_ffn_addnorm",
        }
        for candidate in topologies
    ]


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


def addnorm_ffn_addnorm_design(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    topologies = addnorm_ffn_addnorm_topologies(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    if topology_id is None:
        config = max(
            topologies,
            key=lambda topology: (
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
            ),
        )
    else:
        try:
            config = next(
                candidate
                for candidate in topologies
                if str(candidate["topology_id"]) == topology_id
            )
        except StopIteration as exc:
            raise ValueError(
                f"Unknown Block 3 topology_id={topology_id!r} for {hidden_size}/{intermediate_size}"
            ) from exc

    compile_rows = int(config["compile_rows"])
    tile_m = int(config["tile_m"])
    tile_k = int(config["tile_k"])
    tile_n = int(config["tile_n"])
    parallel_seq = int(config["parallel_seq"])
    parallel_int_dim = int(config["parallel_int_dim"])
    down_proj_depth = int(config["down_proj_depth"])
    gelu_stage = int(config["gelu_stage"])
    num_aie_columns = int(config["num_aie_columns"])

    if seq_len <= 0:
        raise ValueError("Block 3 requires seq_len > 0")
    if parallel_seq not in (1, 2, 4, 6, 8):
        raise ValueError("Block 3 requires parallel_seq in {1, 2, 4, 6, 8}")
    if compile_rows <= 0 or compile_rows % (parallel_seq * tile_m) != 0:
        raise ValueError(
            "Block 3 requires compile_rows divisible by parallel_seq * tile_m"
        )
    if hidden_size <= 0 or hidden_size % tile_k != 0:
        raise ValueError("Block 3 requires hidden_size divisible by tile_k")
    if intermediate_size <= 0 or intermediate_size % tile_n != 0:
        raise ValueError("Block 3 requires intermediate_size divisible by tile_n")
    if intermediate_size % parallel_int_dim != 0:
        raise ValueError(
            "Block 3 requires intermediate_size divisible by parallel_int_dim"
        )
    if down_proj_depth <= 0:
        raise ValueError("Block 3 requires down_proj_depth > 0")
    if gelu_stage not in (0, 1):
        raise ValueError("Block 3 requires gelu_stage to be 0 or 1")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 3 requires 1 <= num_aie_columns <= 8")
    if parallel_seq * (1 + 2 * parallel_int_dim) > num_aie_columns * 4:
        raise ValueError(
            "Block 3 topology requires more cores than the selected array columns provide"
        )

    return {
        **config,
        "topology_id": str(config["topology_id"]),
        "topology_family": str(config["topology_family"]),
    }


def _topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


if __name__ == "__main__":
    main()
