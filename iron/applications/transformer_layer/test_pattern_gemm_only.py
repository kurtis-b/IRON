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
    assert pattern.attn_scores.partition_N == 1
    assert pattern.get_benchmark_metadata()["npu_unique_xclbin_count"] == 1


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_attention_heads",
    [
        (768, 3072, 12),
        (1024, 4096, 16),
    ],
)
def test_gemm_only_partitions_long_attention_scores(
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
    assert pattern.attn_scores.partition_N == 4


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,num_attention_heads",
    [
        (256, 768, 3072, 12),
        (16384, 1024, 4096, 16),
    ],
)
def test_gemm_only_shares_one_runtime_xclbin_per_case(
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
