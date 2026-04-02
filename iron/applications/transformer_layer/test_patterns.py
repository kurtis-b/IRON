# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from pathlib import Path
import pytest
import subprocess
import sys
import torch

from iron.applications.transformer_layer.npu_inference import build_pattern
from iron.applications.transformer_layer.src.core import (
    TransformerLayerSpec as CoreSpec,
)
from iron.applications.transformer_layer.src.pattern_dataflow import (
    DataflowPattern as LegacyDataflowPattern,
)
from iron.applications.transformer_layer.src.pattern_gemm_only import (
    GemmOffloadPattern as LegacyGemmOffloadPattern,
    GemmOnlyPattern as LegacyGemmOnlyPattern,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.patterns import (
    DataflowPattern as StructuredDataflowPattern,
)
from iron.applications.transformer_layer.src.patterns import GemmOnlyPattern
from iron.operators.addnorm_ffn_addnorm.topology import addnorm_ffn_addnorm_topologies
from iron.operators.mha_out_proj.topology import mha_out_proj_topologies
from iron.operators.qkv_proj.topology import qkv_proj_topologies
from iron.applications.transformer_layer.src.utils import (
    make_in_process_npu_metadata,
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)
from iron.operators.mha_out_proj.op import _pack_qkv_head_major
from iron.operators.qkv_proj.op import AIEQKVProj

REPO_ROOT = Path(__file__).resolve().parents[3]
_SINGLE_CASE = [pytest.param(None, id="default")]


def _first_dataflow_compatible_block3_topology(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    head_dim: int,
    block2_topology_id: str | None = None,
    reverse: bool = False,
) -> dict[str, int | str]:
    block3_topologies = addnorm_ffn_addnorm_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if reverse:
        block3_topologies = list(reversed(block3_topologies))
    if block2_topology_id is not None:
        assert any(
            str(candidate["topology_id"]) == block2_topology_id
            for candidate in mha_out_proj_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        )
    if not block3_topologies:
        raise AssertionError("expected at least one Block 3 topology")
    return block3_topologies[0]


def _first_dataflow_compatible_block2_topology(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, int | str]:
    block2_topologies = mha_out_proj_topologies(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if not block2_topologies:
        raise AssertionError("expected at least one Block 2 topology")
    return block2_topologies[0]


def _run_dataflow_parity_isolated(
    spec: TransformerLayerSpec,
    *,
    seed: int,
) -> dict[str, float]:
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from iron.applications.transformer_layer.src.layer_spec import "
            "TransformerLayerSpec; "
            "from iron.applications.transformer_layer.src.pipeline.validate_npu_parity "
            "import validate_pattern_parity; "
            f"spec = TransformerLayerSpec.from_dict({spec.to_dict()!r}); "
            f"row = validate_pattern_parity(execution_mode='dataflow', spec=spec, seed={seed}); "
            "print(json.dumps({'max_abs_diff': row['max_abs_diff'], "
            "'mean_abs_diff': row['mean_abs_diff'], "
            "'error_count': row['error_count'], "
            "'max_acceptable_errors': row['max_acceptable_errors']}))"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_restructured_src_packages_preserve_legacy_imports(_case):
    assert TransformerLayerSpec is CoreSpec
    assert LegacyDataflowPattern is StructuredDataflowPattern
    assert LegacyGemmOnlyPattern is GemmOnlyPattern
    assert LegacyGemmOffloadPattern is GemmOnlyPattern


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_build_pattern_supports_thesis_modes(_case):
    spec = TransformerLayerSpec(seq_len=64)
    for execution_mode in (
        "dataflow",
        "runlist",
        "gemm_offload",
        "block1_qkv_proj",
        "block2_mha_out_proj",
        "block3_addnorm_ffn_addnorm",
        "gemm_offload_gemm_sequence",
        "runlist_gemm_sequence",
    ):
        pattern = build_pattern(execution_mode, spec)
        assert getattr(pattern, "pattern_label")


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block_patterns_prepare_hidden_state_inputs(_case):
    spec = TransformerLayerSpec(seq_len=64)
    weights = make_synthetic_layer_weights(spec, seed=1)
    inputs = make_synthetic_layer_inputs(spec, seed=2)
    for execution_mode in (
        "block1_qkv_proj",
        "block2_mha_out_proj",
        "block3_addnorm_ffn_addnorm",
    ):
        pattern = build_pattern(execution_mode, spec)
        pattern.assign_weights(weights)
        pattern.prepare_benchmark_inputs(inputs)


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_make_in_process_npu_metadata_formats_common_fields(_case):
    metadata = make_in_process_npu_metadata(
        compile_setup_time_sec=0.25,
        dispatch_count=7,
        unique_instruction_binary_count=3,
        unique_xclbin_count=2,
    )
    assert metadata == {
        "compile_setup_time_ms": 250.0,
        "npu_dispatch_count": 7,
        "npu_unique_instruction_binary_count": 3,
        "npu_unique_xclbin_count": 2,
        "process_model": "in_process",
    }


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_make_in_process_npu_metadata_includes_extra_fields(_case):
    metadata = make_in_process_npu_metadata(
        compile_setup_time_sec=0.25,
        dispatch_count=7,
        unique_instruction_binary_count=3,
        unique_xclbin_count=2,
        extra_fields={
            "block2_topology_id": "q32_kv64_e128_ps1_ph1_acc1",
            "block2_topology_family": "fused_mha_out_proj",
        },
    )
    assert metadata["block2_topology_id"] == "q32_kv64_e128_ps1_ph1_acc1"
    assert metadata["block2_topology_family"] == "fused_mha_out_proj"


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_dataflow_patterns_report_selected_block_topologies_in_metadata(_case):
    spec = TransformerLayerSpec(seq_len=64)

    dataflow_pattern = build_pattern("dataflow", spec)
    dataflow_metadata = dataflow_pattern.get_benchmark_metadata()
    assert (
        dataflow_metadata["block1_topology_id"] == dataflow_pattern.block1.topology_id
    )
    assert (
        dataflow_metadata["block2_topology_id"] == dataflow_pattern.block2.topology_id
    )
    assert (
        dataflow_metadata["block3_topology_id"] == dataflow_pattern.block3.topology_id
    )

    block2_pattern = build_pattern("block2_mha_out_proj", spec)
    block2_metadata = block2_pattern.get_benchmark_metadata()
    assert block2_metadata["block2_topology_id"] == block2_pattern.block.topology_id
    assert (
        block2_metadata["block2_topology_family"]
        == block2_pattern.block.topology_family
    )


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_build_pattern_honors_block_topology_overrides(_case):
    block2_topology_id = str(
        _first_dataflow_compatible_block2_topology(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            num_heads=12,
            head_dim=64,
        )["topology_id"]
    )
    block3_topology_id = str(
        _first_dataflow_compatible_block3_topology(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            num_heads=12,
            head_dim=64,
            block2_topology_id=block2_topology_id,
        )["topology_id"]
    )
    spec = TransformerLayerSpec(
        seq_len=64,
        block1_topology_id=str(
            qkv_proj_topologies(seq_len=64, hidden_size=768, num_heads=12)[0][
                "topology_id"
            ]
        ),
        block2_topology_id=block2_topology_id,
        block3_topology_id=block3_topology_id,
    )

    dataflow_pattern = build_pattern("dataflow", spec)
    assert dataflow_pattern.block1.topology_id == spec.block1_topology_id
    assert dataflow_pattern.block2.topology_id == spec.block2_topology_id
    assert dataflow_pattern.block3.topology_id == spec.block3_topology_id

    block1_pattern = build_pattern("block1_qkv_proj", spec)
    assert block1_pattern.block.topology_id == spec.block1_topology_id

    block2_pattern = build_pattern("block2_mha_out_proj", spec)
    assert block2_pattern.block.topology_id == spec.block2_topology_id

    block3_pattern = build_pattern("block3_addnorm_ffn_addnorm", spec)
    assert block3_pattern.block.topology_id == spec.block3_topology_id


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_build_pattern_honors_nondefault_block1_topology_override(_case):
    block1_topologies = qkv_proj_topologies(seq_len=64, hidden_size=768, num_heads=12)
    assert len(block1_topologies) > 1

    spec = TransformerLayerSpec(
        seq_len=64,
        block1_topology_id=str(block1_topologies[-1]["topology_id"]),
    )

    dataflow_pattern = build_pattern("dataflow", spec)
    assert dataflow_pattern.block1.topology_id == spec.block1_topology_id

    block1_pattern = build_pattern("block1_qkv_proj", spec)
    assert block1_pattern.block.topology_id == spec.block1_topology_id


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_build_pattern_honors_nondefault_block3_topology_override(_case):
    compatible_block3 = _first_dataflow_compatible_block3_topology(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        head_dim=64,
        reverse=True,
    )

    spec = TransformerLayerSpec(
        seq_len=64,
        block3_topology_id=str(compatible_block3["topology_id"]),
    )

    dataflow_pattern = build_pattern("dataflow", spec)
    assert dataflow_pattern.block3.topology_id == spec.block3_topology_id

    block3_pattern = build_pattern("block3_addnorm_ffn_addnorm", spec)
    assert block3_pattern.block.topology_id == spec.block3_topology_id


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block1_contract_reshapes_projection_outputs_to_head_major(_case):
    matrix = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    head_major = AIEQKVProj._to_head_major(
        matrix,
        seq_len=3,
        hidden_size=8,
        num_heads=2,
    )

    expected = matrix.view(3, 2, 4).permute(1, 0, 2).contiguous()
    assert head_major.shape == (2, 3, 4)
    assert torch.equal(head_major, expected)


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block1_pattern_metadata_reports_fused_qkv_dispatch(_case):
    spec = TransformerLayerSpec(seq_len=64)
    pattern = build_pattern("block1_qkv_proj", spec)

    metadata = pattern.get_benchmark_metadata()

    assert metadata["npu_dispatch_count"] == 1
    assert metadata["npu_unique_instruction_binary_count"] == 1
    assert metadata["npu_unique_xclbin_count"] == 1


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block2_contract_packs_head_major_qkv_into_runtime_qkv_layout(_case):
    q = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    k = torch.arange(24, 48, dtype=torch.bfloat16).reshape(2, 3, 4)
    v = torch.arange(48, 72, dtype=torch.bfloat16).reshape(2, 3, 4)

    packed = _pack_qkv_head_major(q, k, v, seq_len=3, embed_sz=8)

    expected = torch.cat(
        (
            q.permute(1, 0, 2).contiguous().view(3, 8),
            k.permute(1, 0, 2).contiguous().view(3, 8),
            v.permute(1, 0, 2).contiguous().view(3, 8),
        ),
        dim=0,
    )
    assert packed.shape == (9, 8)
    assert np.array_equal(packed.astype(np.float32), expected.float().numpy())


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_dataflow_pattern_uses_plain_block2_block3_handoff(_case):
    spec = TransformerLayerSpec(seq_len=64)
    pattern = build_pattern("dataflow", spec)

    assert pattern.block2.packed_output_parallel_seq is None
    assert pattern.block2.packed_output_rows is None


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_dataflow_pattern_resolves_explicit_block2_and_block3_ids_independently(
    _case,
):
    spec = TransformerLayerSpec(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        block3_topology_id="m16_k96_n96_ps4_pi2_d8_g1",
        block2_topology_id="q32_kv64_e96_ps2_ph2_acc8",
    )

    pattern = build_pattern("dataflow", spec)

    assert pattern.block2.topology_id == "q32_kv64_e96_ps2_ph2_acc8"
    assert pattern.block3.topology_id == "m16_k96_n96_ps4_pi2_d8_g1"


@pytest.mark.parametrize(
    "spec_kwargs",
    (
        pytest.param(
            {
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "block2_topology_id": "q32_kv64_e96_ps2_ph2_acc8",
                "block3_topology_id": "m16_k96_n96_ps4_pi2_d8_g1",
            },
            id="dataflow_64x768x3072_block2_ps2_ph2_acc8_to_block3_ps4_pi2",
        ),
        pytest.param(
            {
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "block3_topology_id": "m16_k96_n96_ps4_pi2_d8_g1",
            },
            id="dataflow_64x768x3072_compatible_ps4_pi2",
        ),
        pytest.param(
            {
                "seq_len": 512,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "block2_topology_id": "q32_kv64_e96_ps4_ph2_acc1",
                "block3_topology_id": "m16_k96_n96_ps4_pi2_d8_g1",
            },
            id="dataflow_512x768x3072_staged_ps4_pi2",
        ),
        pytest.param(
            {
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "block2_topology_id": "q32_kv64_e96_ps2_ph2_acc1",
                "block3_topology_id": "m16_k96_n96_ps4_pi2_d8_g1",
            },
            id="dataflow_64x768x3072_block2_ps2_ph2_to_block3_ps4_pi2",
        ),
    ),
)
def test_dataflow_pattern_runs_plain_block2_block3_handoff_against_reference(
    spec_kwargs,
):
    spec = TransformerLayerSpec(
        use_bias=False,
        weights_source="synthetic",
        **spec_kwargs,
    )
    stats = _run_dataflow_parity_isolated(spec, seed=7)
    assert int(stats["error_count"]) <= int(stats["max_acceptable_errors"])
