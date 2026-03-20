#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.common.test_utils import run_test
from iron.operators.encoder_pipeline_memtile.op import AIEEncoderPipelineMemtile
from iron.operators.encoder_pipeline_memtile.reference import generate_golden_reference

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
TEST_WARMUP_ITERS = 3
TEST_TIMED_ITERS = 20
BASE_TOPOLOGY = (64, 12, 3072, 32, 64, 96, 64)
BENCHMARK_SEQ_LENS = (64, 512)
EXTENSIVE_SEQ_LENS = (1024, 2048, 4096, 8192)
NON_SEQ_TOPOLOGY_CASES = (
    ("", (1, 1, 8, 1, 1, 1)),
    ("_2ph", (1, 2, 8, 1, 1, 1)),
    ("_4ph", (1, 4, 8, 1, 1, 1)),
    ("_2pffn", (1, 1, 8, 1, 1, 2)),
    ("_2ph_2pffn", (1, 2, 8, 1, 1, 2)),
    ("_4pffn", (1, 1, 8, 1, 1, 4)),
    ("_2ph_4pffn", (1, 2, 8, 1, 1, 4)),
)
SEQ_PAR_TOPOLOGY_CASES = (
    ("_2ps", (2, 1, 8, 1, 1, 1)),
    ("_2ps_2ph", (2, 2, 8, 1, 1, 1)),
    ("_2ps_2pffn", (2, 1, 8, 1, 1, 2)),
    ("_2ps_2ph_2pffn", (2, 2, 8, 2, 1, 2)),
)


def _topology_name(suffix: str) -> str:
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        BASE_TOPOLOGY
    )
    return (
        f"{d}d_{num_heads}h_{ffn_intermediate_size}ffn_{seq_tile}q_"
        f"{kv_seq_tile}kv_{emb_tile}emb_{ffn_tile}ffnt{suffix}"
    )


def _generate_params(seq_lens, topology_cases, marks=()):
    params = []
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        BASE_TOPOLOGY
    )
    for seq_len in seq_lens:
        for topology_suffix, runtime_topology in topology_cases:
            (
                parallel_seq,
                parallel_heads,
                proj_acc_depth,
                o_proj_acc_group_size,
                ffn_down_acc_group_size,
                nB_tiles_distributed,
            ) = runtime_topology
            if seq_len % seq_tile != 0 or (seq_len // seq_tile) % parallel_seq != 0:
                continue
            params.append(
                pytest.param(
                    seq_len,
                    d,
                    num_heads,
                    ffn_intermediate_size,
                    seq_tile,
                    kv_seq_tile,
                    emb_tile,
                    ffn_tile,
                    parallel_seq,
                    parallel_heads,
                    proj_acc_depth,
                    o_proj_acc_group_size,
                    ffn_down_acc_group_size,
                    nB_tiles_distributed,
                    id=(
                        f"encoder_pipeline_memtile_{seq_len}seq_"
                        f"{_topology_name(topology_suffix)}"
                    ),
                    marks=marks,
                )
            )
    return params


ALL_PARAMS = [
    *_generate_params(BENCHMARK_SEQ_LENS, NON_SEQ_TOPOLOGY_CASES),
    *_generate_params(BENCHMARK_SEQ_LENS, SEQ_PAR_TOPOLOGY_CASES),
    *_generate_params(
        EXTENSIVE_SEQ_LENS,
        NON_SEQ_TOPOLOGY_CASES,
        marks=(pytest.mark.extensive,),
    ),
    *_generate_params(
        EXTENSIVE_SEQ_LENS,
        SEQ_PAR_TOPOLOGY_CASES,
        marks=(pytest.mark.extensive,),
    ),
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,ffn_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,ffn_down_acc_group_size,nB_tiles_distributed",
    ALL_PARAMS,
)
def test_encoder_pipeline_memtile(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=d,
        intermediate_size=ffn_intermediate_size,
        seq_tile=seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
    )

    operator = AIEEncoderPipelineMemtile(
        num_heads=num_heads,
        seq_len=seq_len,
        d=d,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        ffn_down_acc_group_size=ffn_down_acc_group_size,
        nB_tiles_distributed=nB_tiles_distributed,
        ffn_intermediate_size=ffn_intermediate_size,
        ln1_weight=golden_ref["ln1_weight"],
        ln2_weight=golden_ref["ln2_weight"],
        context=aie_context,
    )

    input_buffers = {
        "QKV": golden_ref["QKV"],
        "OR": golden_ref["OR"],
        "W_O": golden_ref["W_O"],
        "B_Up": golden_ref["B_Up"],
        "B_Down": golden_ref["B_Down"],
    }
    output_buffers = {"O": golden_ref["O"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=TEST_WARMUP_ITERS,
        timed_iters=TEST_TIMED_ITERS,
    )

    max_acceptable_errors = int(seq_len * d * num_heads * ERROR_THRESHOLD)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    print(
        f"({len(errors.get('O', []))} errors out of {max_acceptable_errors} max allowable)"
    )

    assert len(errors.get("O", [])) <= max_acceptable_errors
