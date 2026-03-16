#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.common.test_utils import run_test
from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.reference import generate_golden_reference

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005


def generate_test_params():
    params = [
        (64, 64, 12, 3072, 32, 64, 96, 1, 1, 8, 1, 1),
        (128, 64, 12, 3072, 32, 64, 96, 1, 1, 8, 1, 1),
        (256, 64, 12, 3072, 32, 64, 96, 1, 1, 8, 1, 1),
        (64, 64, 12, 3072, 32, 64, 96, 1, 2, 8, 1, 1),
        (64, 64, 12, 3072, 32, 64, 96, 1, 4, 8, 1, 1),
        (64, 64, 12, 3072, 32, 64, 96, 1, 1, 8, 1, 2),
        (64, 64, 12, 3072, 32, 64, 96, 1, 2, 8, 1, 2),
        (64, 64, 12, 3072, 32, 64, 96, 1, 1, 8, 1, 4),
        (64, 64, 12, 3072, 32, 64, 96, 1, 2, 8, 1, 4),
    ]
    names = [
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb",
        "encoder_pipeline_128seq_64d_12h_3072ffn_32q_64kv_96emb",
        "encoder_pipeline_256seq_64d_12h_3072ffn_32q_64kv_96emb",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_2ph",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_4ph",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_2pffn",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_2ph_2pffn",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_4pffn",
        "encoder_pipeline_64seq_64d_12h_3072ffn_32q_64kv_96emb_2ph_4pffn",
    ]
    return params, names


params, names = generate_test_params()
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,nB_tiles_distributed",
    all_params,
)
def test_encoder_pipeline(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=d,
        intermediate_size=ffn_intermediate_size,
    )

    operator = AIEEncoderPipeline(
        num_heads=num_heads,
        seq_len=seq_len,
        d=d,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
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
    )

    max_acceptable_errors = int(seq_len * d * num_heads * ERROR_THRESHOLD)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    print(
        f"({len(errors.get('O', []))} errors out of {max_acceptable_errors} max allowable)"
    )

    assert len(errors.get("O", [])) <= max_acceptable_errors
