import pytest

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pattern_gemm_only import GemmOnlyPattern


def test_gemm_only_uses_direct_path_on_short_sequences():
    spec = TransformerLayerSpec(
        seq_len=8192,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )

    pattern = GemmOnlyPattern(spec)

    assert pattern.query_block_size == 8192
    assert pattern.uses_query_blocking is False
    assert pattern.attn_scores.force_batched_design is True
    assert pattern.attn_output.force_batched_design is True
    assert pattern.get_benchmark_metadata()["npu_unique_xclbin_count"] == 1


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_attention_heads",
    [
        (768, 3072, 12),
        (1024, 4096, 16),
    ],
)
def test_gemm_only_query_blocks_long_sequences(
    hidden_size,
    intermediate_size,
    num_attention_heads,
):
    spec = TransformerLayerSpec(
        seq_len=16384,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )

    pattern = GemmOnlyPattern(spec)

    assert pattern.query_block_size == 256
    assert pattern.uses_query_blocking is True
    assert pattern.attn_scores.force_batched_design is True
    assert pattern.attn_output.force_batched_design is True


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,num_attention_heads",
    [
        (256, 768, 3072, 12),
        (16384, 1024, 4096, 16),
    ],
)
def test_gemm_only_uses_one_runtime_xclbin_per_case(
    seq_len,
    hidden_size,
    intermediate_size,
    num_attention_heads,
):
    spec = TransformerLayerSpec(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )

    pattern = GemmOnlyPattern(spec)
    gemm_ops = [
        pattern.attn_scores,
        pattern.attn_output,
        pattern.out_proj,
        pattern.ffn_up,
        pattern.ffn_down,
    ]
    runtime_xclbins = {
        str((op.runtime_xclbin_artifact or op.xclbin_artifact).path) for op in gemm_ops
    }
    insts = {str(op.insts_artifact.path) for op in gemm_ops}

    assert len(runtime_xclbins) == 1
    assert len(insts) == len(gemm_ops)
    assert pattern.get_benchmark_metadata()["npu_unique_xclbin_count"] == 1


def test_gemm_only_dispatch_count_uses_batched_attention():
    spec = TransformerLayerSpec(
        seq_len=16384,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )

    pattern = GemmOnlyPattern(spec)

    block_count = spec.seq_len // pattern.query_block_size
    assert pattern.get_benchmark_metadata()["npu_dispatch_count"] == 5 * block_count


def test_gemm_only_hidden_state_boundary_adds_projection_gemms():
    spec = TransformerLayerSpec(
        seq_len=256,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        input_boundary="hidden_states",
    )

    pattern = GemmOnlyPattern(spec)

    assert pattern.q_proj is not None
    assert pattern.k_proj is not None
    assert pattern.v_proj is not None
    assert pattern.get_benchmark_metadata()["npu_unique_xclbin_count"] == 1
    assert pattern.get_benchmark_metadata()["npu_dispatch_count"] == 8
