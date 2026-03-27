import json
from pathlib import Path

import torch

from iron.applications.transformer_layer.src.input_bundle import TransformerLayerInputs
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pattern_encoder_pipeline import (
    EncoderPipelinePattern,
)
from iron.operators.encoder_pipeline.topology import all_supported_topologies


class _FakeContext:
    def __init__(self):
        self.compiled = False
        self.prepared = False
        self.reset_calls = 0

    def compile_all(self):
        self.compiled = True

    def prepare_runtime(self):
        self.prepared = True

    def reset_runtime(self):
        self.reset_calls += 1


class _FakeEncoderPipeline:
    def __init__(self, topology, latency_sec):
        self.num_heads = topology.num_heads
        self.seq_len = topology.seq_len
        self.d = topology.d
        self.seq_tile = topology.seq_tile
        self.kv_seq_tile = topology.kv_seq_tile
        self.emb_tile = topology.emb_tile
        self.ffn_tile = topology.ffn_tile
        self.parallel_seq = topology.parallel_seq
        self.parallel_heads = topology.parallel_heads
        self.proj_acc_depth = topology.proj_acc_depth
        self.o_proj_acc_group_size = topology.o_proj_acc_group_size
        self.nB_tiles_distributed = topology.parallel_ffn
        self.ffn_intermediate_size = topology.ffn_intermediate_size
        self.autotune_latency_sec = latency_sec
        self.runlist = [("kernel", "q", "k", "v", "r", "out")]
        self.kernels = {"kernel": object()}

    def __call__(self, q, k, v, r):
        del q, k, v
        return r


def _make_spec() -> TransformerLayerSpec:
    return TransformerLayerSpec(
        seq_len=128,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )


def _make_weights(spec: TransformerLayerSpec) -> dict[str, torch.Tensor]:
    return {
        "out_proj_weight": torch.ones(
            (spec.hidden_size, spec.hidden_size), dtype=spec.torch_dtype
        ),
        "ffn_up_weight": torch.ones(
            (spec.intermediate_size, spec.hidden_size), dtype=spec.torch_dtype
        ),
        "ffn_down_weight": torch.ones(
            (spec.hidden_size, spec.intermediate_size), dtype=spec.torch_dtype
        ),
        "ln1_weight": torch.ones((spec.hidden_size,), dtype=spec.torch_dtype),
        "ln2_weight": torch.ones((spec.hidden_size,), dtype=spec.torch_dtype),
    }


def _make_inputs(spec: TransformerLayerSpec) -> TransformerLayerInputs:
    return TransformerLayerInputs(
        q=torch.ones(
            (
                spec.batch_size,
                spec.num_attention_heads,
                spec.seq_len,
                spec.attention_head_size,
            ),
            dtype=spec.torch_dtype,
        ),
        k=torch.ones(
            (
                spec.batch_size,
                spec.num_attention_heads,
                spec.seq_len,
                spec.attention_head_size,
            ),
            dtype=spec.torch_dtype,
        ),
        v=torch.ones(
            (
                spec.batch_size,
                spec.num_attention_heads,
                spec.seq_len,
                spec.attention_head_size,
            ),
            dtype=spec.torch_dtype,
        ),
        r=torch.ones(
            (spec.batch_size, spec.seq_len, spec.hidden_size),
            dtype=spec.torch_dtype,
        ),
    )


def _candidate_topologies(spec: TransformerLayerSpec):
    return [
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == spec.num_attention_heads
        and topology.seq_len == spec.seq_len
        and topology.d == spec.attention_head_size
        and topology.ffn_intermediate_size == spec.intermediate_size
    ]


def _two_representative_topologies(spec: TransformerLayerSpec):
    candidates = _candidate_topologies(spec)
    default = next(
        topology
        for topology in candidates
        if topology.family_id == "seq32_kv64"
        and topology.parallel_seq == 1
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
        and topology.ffn_tile == 64
    )
    fastest = max(
        (topology for topology in candidates if topology.key != default.key),
        key=lambda topology: (
            topology.compute_tile_count,
            topology.parallel_seq,
            topology.parallel_heads,
            topology.parallel_ffn,
        ),
    )
    return default, fastest


def test_encoder_pipeline_autotune_enumerates_all_valid_topologies():
    spec = _make_spec()

    pattern = EncoderPipelinePattern(spec)
    candidates = pattern._candidate_topologies()
    expected = _candidate_topologies(spec)

    assert len(candidates) == len(expected)
    assert {candidate.key for candidate in candidates} == {
        candidate.key for candidate in expected
    }
    assert candidates[0].key == pattern._default_topology.key


def test_encoder_pipeline_autotune_selects_fastest_topology(monkeypatch, tmp_path):
    spec = _make_spec()
    slow, fast = _two_representative_topologies(spec)
    latency_by_key = {
        slow.key: 0.004,
        fast.key: 0.001,
    }

    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_candidate_topologies",
        lambda self: [slow, fast],
    )
    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_autotune_cache_path",
        lambda self: tmp_path / "encoder_pipeline_autotune_cache.json",
    )
    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_autotune_cache_key",
        lambda self: "test-cache-key",
    )

    def _fake_instantiate(self, topology):
        return _FakeContext(), _FakeEncoderPipeline(
            topology, latency_by_key[topology.key]
        )

    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_instantiate_candidate",
        _fake_instantiate,
    )
    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_benchmark_candidate",
        lambda self, operator, **kwargs: operator.autotune_latency_sec,
    )

    pattern = EncoderPipelinePattern(spec)
    pattern.assign_weights(_make_weights(spec))

    output, stage_timings = pattern.forward_with_stage_timings(_make_inputs(spec))

    assert tuple(output.shape) == (1, spec.seq_len, spec.hidden_size)
    assert stage_timings["encoder_pipeline_sec"] >= 0.0
    assert pattern.get_benchmark_metadata()["topology_id"] == fast.topology_id
    assert pattern.get_benchmark_metadata()["parallel_ffn"] == fast.parallel_ffn
    cache_payload = json.loads(
        (tmp_path / "encoder_pipeline_autotune_cache.json").read_text(encoding="utf-8")
    )
    assert cache_payload["entries"]["test-cache-key"]["topology_key"] == list(fast.key)


def test_encoder_pipeline_autotune_uses_cached_topology(monkeypatch, tmp_path):
    spec = _make_spec()
    slow, fast = _two_representative_topologies(spec)
    cache_path = tmp_path / "encoder_pipeline_autotune_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "test-cache-key": {
                        "topology_key": list(fast.key),
                        "topology_id": fast.topology_id,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_candidate_topologies",
        lambda self: [slow, fast],
    )
    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_autotune_cache_path",
        lambda self: cache_path,
    )
    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_autotune_cache_key",
        lambda self: "test-cache-key",
    )

    def _fake_instantiate(self, topology):
        return _FakeContext(), _FakeEncoderPipeline(topology, 0.003)

    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_instantiate_candidate",
        _fake_instantiate,
    )

    def _unexpected_benchmark(self, operator, **kwargs):
        del self, operator, kwargs
        raise AssertionError("cache hit should skip encoder_pipeline autotune probes")

    monkeypatch.setattr(
        EncoderPipelinePattern,
        "_benchmark_candidate",
        _unexpected_benchmark,
    )

    pattern = EncoderPipelinePattern(spec)
    pattern.assign_weights(_make_weights(spec))

    output, _ = pattern.forward_with_stage_timings(_make_inputs(spec))

    assert tuple(output.shape) == (1, spec.seq_len, spec.hidden_size)
    assert pattern.get_benchmark_metadata()["topology_id"] == fast.topology_id
    assert pattern.get_benchmark_metadata()["parallel_ffn"] == fast.parallel_ffn
