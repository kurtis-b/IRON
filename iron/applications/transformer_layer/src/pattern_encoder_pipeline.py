# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext, AIEDeviceManager
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline
from iron.operators.encoder_pipeline.topology import (
    EncoderTopology,
    all_supported_topologies,
    load_encoder_pipeline_topology_placements,
    topology_from_fields,
)

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from .utils import require_keys


class EncoderPipelinePattern(nn.Module):
    """Single-layer encoder_pipeline wrapper with layer-centric inputs."""

    pattern_label = "encoder_pipeline"
    _AUTOTUNE_CACHE_VERSION = 2
    _AUTOTUNE_WARMUP_RUNS = 1
    _AUTOTUNE_TIMED_RUNS = 1

    @staticmethod
    def _family_topology_defaults(spec: TransformerLayerSpec) -> EncoderTopology:
        placements = load_encoder_pipeline_topology_placements()
        if (
            spec.hidden_size == 768
            and spec.intermediate_size == 3072
            and spec.num_attention_heads == 12
        ):
            topology = topology_from_fields(
                num_heads=spec.num_attention_heads,
                seq_len=spec.seq_len,
                d=spec.attention_head_size,
                seq_tile=32,
                kv_seq_tile=64,
                emb_tile=96,
                ffn_tile=64,
                parallel_seq=1,
                parallel_heads=1,
                proj_acc_depth=8,
                o_proj_acc_group_size=1,
                parallel_ffn=1,
                ffn_intermediate_size=spec.intermediate_size,
            )
            return topology_from_fields(
                num_heads=topology.num_heads,
                seq_len=topology.seq_len,
                d=topology.d,
                seq_tile=topology.seq_tile,
                kv_seq_tile=topology.kv_seq_tile,
                emb_tile=topology.emb_tile,
                ffn_tile=topology.ffn_tile,
                parallel_seq=topology.parallel_seq,
                parallel_heads=topology.parallel_heads,
                proj_acc_depth=topology.proj_acc_depth,
                o_proj_acc_group_size=topology.o_proj_acc_group_size,
                parallel_ffn=topology.parallel_ffn,
                ffn_intermediate_size=topology.ffn_intermediate_size,
                placement=placements.get(topology.key),
            )
        if (
            spec.hidden_size == 1024
            and spec.intermediate_size == 4096
            and spec.num_attention_heads == 16
        ):
            topology = topology_from_fields(
                num_heads=spec.num_attention_heads,
                seq_len=spec.seq_len,
                d=spec.attention_head_size,
                seq_tile=32,
                kv_seq_tile=64,
                emb_tile=128,
                ffn_tile=64,
                parallel_seq=1,
                parallel_heads=1,
                proj_acc_depth=8,
                o_proj_acc_group_size=1,
                parallel_ffn=1,
                ffn_intermediate_size=spec.intermediate_size,
            )
            return topology_from_fields(
                num_heads=topology.num_heads,
                seq_len=topology.seq_len,
                d=topology.d,
                seq_tile=topology.seq_tile,
                kv_seq_tile=topology.kv_seq_tile,
                emb_tile=topology.emb_tile,
                ffn_tile=topology.ffn_tile,
                parallel_seq=topology.parallel_seq,
                parallel_heads=topology.parallel_heads,
                proj_acc_depth=topology.proj_acc_depth,
                o_proj_acc_group_size=topology.o_proj_acc_group_size,
                parallel_ffn=topology.parallel_ffn,
                ffn_intermediate_size=topology.ffn_intermediate_size,
                placement=placements.get(topology.key),
            )
        raise ValueError(
            "encoder_pipeline thesis pattern currently supports only the "
            "768/3072/12 and 1024/4096/16 families; got "
            f"hidden_size={spec.hidden_size}, intermediate_size={spec.intermediate_size}, "
            f"num_attention_heads={spec.num_attention_heads}"
        )

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "encoder_pipeline thesis pattern currently requires use_bias=False"
            )
        hidden = spec.hidden_size
        dtype = spec.torch_dtype
        self.spec = spec
        self._uses_projection = spec.input_boundary == "hidden_states"
        self._default_topology = self._family_topology_defaults(spec)
        self.context: AIEContext | None = None
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        self._selected_topology: EncoderTopology = self._default_topology
        self.q_proj_weight: torch.Tensor | None = None
        self.k_proj_weight: torch.Tensor | None = None
        self.v_proj_weight: torch.Tensor | None = None
        self.out_proj_weight = torch.zeros((hidden, hidden), dtype=dtype).contiguous()
        self.ffn_up_weight = torch.zeros(
            (spec.intermediate_size, hidden), dtype=dtype
        ).contiguous()
        self.ffn_down_weight = torch.zeros(
            (hidden, spec.intermediate_size), dtype=dtype
        ).contiguous()
        self.ln1_weight = torch.ones(hidden, dtype=dtype).contiguous()
        self.ln2_weight = torch.ones(hidden, dtype=dtype).contiguous()
        self.encoder_pipeline: AIEEncoderPipeline | None = None

    def _prepare_runtime(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        if self._runtime_ready:
            return
        if not self._weights_assigned:
            raise RuntimeError("assign_weights() must be called before execution")
        start = time.perf_counter()
        cached_topology = self._load_cached_topology()
        if cached_topology is not None:
            self.context, self.encoder_pipeline = self._instantiate_candidate(
                cached_topology
            )
            try:
                self.context.compile_all()
                self.context.prepare_runtime()
                self._selected_topology = cached_topology
            except Exception:
                logging.exception(
                    "encoder_pipeline cached topology %s failed; retuning",
                    cached_topology.key,
                )
                self._cleanup_context(self.context)
                self.context = None
                self.encoder_pipeline = None
                self._selected_topology = self._default_topology
                self._drop_cached_topology()
            else:
                self.compile_setup_time_sec = time.perf_counter() - start
                self._runtime_ready = True
                return

        best_context: AIEContext | None = None
        best_operator: AIEEncoderPipeline | None = None
        best_topology: EncoderTopology | None = None
        best_latency_sec: float | None = None
        last_error: Exception | None = None
        for topology in self._candidate_topologies():
            candidate_context, candidate_operator = self._instantiate_candidate(
                topology
            )
            try:
                candidate_context.compile_all()
                candidate_context.prepare_runtime()
                candidate_latency_sec = self._benchmark_candidate(
                    candidate_operator,
                    q=q,
                    k=k,
                    v=v,
                    residual=residual,
                )
            except Exception as exc:
                last_error = exc
                logging.exception(
                    "encoder_pipeline autotune candidate failed for topology %s",
                    topology.key,
                )
                self._cleanup_context(candidate_context)
                continue
            if best_latency_sec is None or candidate_latency_sec < best_latency_sec:
                self._cleanup_context(best_context)
                best_context = candidate_context
                best_operator = candidate_operator
                best_topology = topology
                best_latency_sec = candidate_latency_sec
            else:
                self._cleanup_context(candidate_context)

        if best_context is None or best_operator is None or best_topology is None:
            if last_error is not None:
                raise RuntimeError(
                    "encoder_pipeline autotune could not prepare any supported topology"
                ) from last_error
            raise RuntimeError(
                "encoder_pipeline autotune found no supported topology candidates"
            )

        self.context = best_context
        self.encoder_pipeline = best_operator
        self._selected_topology = best_topology
        self._write_cached_topology(best_topology)
        self.compile_setup_time_sec = time.perf_counter() - start
        self._runtime_ready = True

    def _autotune_cache_path(self) -> Path:
        return Path.cwd() / "build" / "encoder_pipeline_autotune_cache.json"

    def _autotune_cache_key(self) -> str:
        device_str = AIEDeviceManager().device_str()
        return (
            f"v{self._AUTOTUNE_CACHE_VERSION}|"
            f"{device_str}|"
            f"{self.spec.hidden_size}|"
            f"{self.spec.intermediate_size}|"
            f"{self.spec.num_attention_heads}|"
            f"{self.spec.seq_len}|"
            f"{self.spec.dtype}"
        )

    def _load_cached_topology(self) -> EncoderTopology | None:
        cache_path = self._autotune_cache_path()
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entry = payload.get("entries", {}).get(self._autotune_cache_key())
        if not isinstance(entry, dict):
            return None
        raw_key = entry.get("topology_key")
        if not isinstance(raw_key, list) or len(raw_key) != 13:
            return None
        cached_key = tuple(int(value) for value in raw_key)
        for topology in self._candidate_topologies():
            if topology.key == cached_key:
                return topology
        return None

    def _write_cached_topology(self, topology: EncoderTopology) -> None:
        cache_path = self._autotune_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                payload = {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        entries = payload.setdefault("entries", {})
        entries[self._autotune_cache_key()] = {
            "topology_key": list(topology.key),
            "topology_id": topology.topology_id,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload["version"] = self._AUTOTUNE_CACHE_VERSION
        cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _drop_cached_topology(self) -> None:
        cache_path = self._autotune_cache_path()
        if not cache_path.exists():
            return
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return
        if entries.pop(self._autotune_cache_key(), None) is None:
            return
        cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _candidate_topologies(self) -> list[EncoderTopology]:
        candidates = [
            topology
            for topology in all_supported_topologies()
            if topology.num_heads == self.spec.num_attention_heads
            and topology.seq_len == self.spec.seq_len
            and topology.d == self.spec.attention_head_size
            and topology.ffn_intermediate_size == self.spec.intermediate_size
        ]
        candidates.sort(
            key=lambda topology: (
                0 if topology.key == self._default_topology.key else 1,
                -topology.compute_tile_count,
                topology.family_id,
                topology.ffn_tile,
                -topology.parallel_seq,
                -topology.parallel_heads,
                -topology.parallel_ffn,
            )
        )
        return candidates

    def _instantiate_candidate(
        self, topology: EncoderTopology
    ) -> tuple[AIEContext, AIEEncoderPipeline]:
        context = AIEContext(use_runlist=True)
        operator = AIEEncoderPipeline(
            num_heads=self.spec.num_attention_heads,
            seq_len=self.spec.seq_len,
            d=self.spec.attention_head_size,
            seq_tile=topology.seq_tile,
            kv_seq_tile=topology.kv_seq_tile,
            emb_tile=topology.emb_tile,
            ffn_tile=topology.ffn_tile,
            parallel_seq=topology.parallel_seq,
            parallel_heads=topology.parallel_heads,
            proj_acc_depth=topology.proj_acc_depth,
            o_proj_acc_group_size=topology.o_proj_acc_group_size,
            nB_tiles_distributed=topology.parallel_ffn,
            ffn_intermediate_size=self.spec.intermediate_size,
            static_weights=True,
            ln1_weight=self.ln1_weight,
            ln2_weight=self.ln2_weight,
            context=context,
        )
        operator.w_o_proj = self.out_proj_weight
        operator.weight_up_proj = self.ffn_up_weight
        operator.weight_down_proj = self.ffn_down_weight
        operator.ln1_weight = self.ln1_weight
        operator.ln2_weight = self.ln2_weight
        return context, operator

    def _benchmark_candidate(
        self,
        operator: AIEEncoderPipeline,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
    ) -> float:
        for _ in range(self._AUTOTUNE_WARMUP_RUNS):
            operator(q, k, v, r=residual)
        total_sec = 0.0
        for _ in range(self._AUTOTUNE_TIMED_RUNS):
            start = time.perf_counter()
            operator(q, k, v, r=residual)
            total_sec += time.perf_counter() - start
        return total_sec / float(self._AUTOTUNE_TIMED_RUNS)

    def _cleanup_context(self, context: AIEContext | None) -> None:
        if context is None:
            return
        try:
            context.reset_runtime()
        except Exception:
            logging.exception(
                "Failed to reset encoder_pipeline autotune runtime cleanly"
            )

    def _materialize_pipeline_inputs(
        self, layer_inputs: TransformerLayerInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        host_projection_start = time.perf_counter()
        if self._uses_projection:
            hidden_states = layer_inputs.hidden_states.squeeze(0).to(
                self.spec.torch_dtype
            )
            residual = layer_inputs.r.squeeze(0).to(self.spec.torch_dtype)
            q = F.linear(hidden_states, self.q_proj_weight)
            k = F.linear(hidden_states, self.k_proj_weight)
            v = F.linear(hidden_states, self.v_proj_weight)
            q = (
                q.view(
                    self.spec.seq_len,
                    self.spec.num_attention_heads,
                    self.spec.attention_head_size,
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            k = (
                k.view(
                    self.spec.seq_len,
                    self.spec.num_attention_heads,
                    self.spec.attention_head_size,
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            v = (
                v.view(
                    self.spec.seq_len,
                    self.spec.num_attention_heads,
                    self.spec.attention_head_size,
                )
                .permute(1, 0, 2)
                .contiguous()
            )
        else:
            q = layer_inputs.q.squeeze(0)
            k = layer_inputs.k.squeeze(0)
            v = layer_inputs.v.squeeze(0)
            residual = layer_inputs.r.squeeze(0)
        host_projection_end = time.perf_counter()
        return q, k, v, residual, host_projection_end - host_projection_start

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if self._runtime_ready:
            raise RuntimeError(
                "assign_weights() after runtime preparation is not supported"
            )
        require_keys(
            weights,
            [
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln1_weight",
                "ln2_weight",
            ]
            + (
                ["q_proj_weight", "k_proj_weight", "v_proj_weight"]
                if self._uses_projection
                else []
            ),
        )
        if self._uses_projection:
            self.q_proj_weight = weights["q_proj_weight"].contiguous()
            self.k_proj_weight = weights["k_proj_weight"].contiguous()
            self.v_proj_weight = weights["v_proj_weight"].contiguous()
        self.out_proj_weight = weights["out_proj_weight"].contiguous()
        self.ffn_up_weight = weights["ffn_up_weight"].contiguous()
        self.ffn_down_weight = weights["ffn_down_weight"].contiguous()
        self.ln1_weight = weights["ln1_weight"].contiguous()
        self.ln2_weight = weights["ln2_weight"].contiguous()
        self._weights_assigned = True

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "encoder_pipeline thesis pattern currently requires attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        batch_size = (
            layer_inputs.hidden_states.shape[0]
            if self._uses_projection
            else layer_inputs.q.shape[0]
        )
        if batch_size != 1:
            raise RuntimeError(
                "encoder_pipeline thesis pattern currently supports batch_size=1"
            )
        q, k, v, residual, host_projection_sec = self._materialize_pipeline_inputs(
            layer_inputs
        )
        self._prepare_runtime(q, k, v, residual)
        if self.encoder_pipeline is None:
            raise RuntimeError("encoder_pipeline autotune did not select an operator")

        start = time.perf_counter()
        output = self.encoder_pipeline(
            q,
            k,
            v,
            r=residual,
        ).unsqueeze(0)
        end = time.perf_counter()
        return output, {
            "host_projection_sec": host_projection_sec,
            "encoder_pipeline_sec": end - start,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        topology = self._selected_topology
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": (
                None
                if self.encoder_pipeline is None
                else len(self.encoder_pipeline.runlist)
            ),
            "npu_unique_instruction_binary_count": (
                None
                if self.encoder_pipeline is None
                else len(self.encoder_pipeline.kernels)
            ),
            "topology_id": topology.topology_id,
            "topology_family": topology.family_id,
            "parallel_seq": topology.parallel_seq,
            "parallel_heads": topology.parallel_heads,
            "parallel_ffn": topology.parallel_ffn,
            "compute_tile_count": topology.compute_tile_count,
            "compute_tile_utilization_fraction": topology.utilization_fraction,
            "process_model": "in_process",
        }

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            layer_inputs, attention_mask=attention_mask
        )
        return output
