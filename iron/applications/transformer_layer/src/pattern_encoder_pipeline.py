# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline
from iron.operators.encoder_pipeline.topology import (
    load_encoder_pipeline_topology_placements,
    topology_from_fields,
)

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from .utils import require_keys


class EncoderPipelinePattern(nn.Module):
    """Single-layer encoder_pipeline wrapper with layer-centric inputs."""

    pattern_label = "encoder_pipeline"

    @staticmethod
    def _family_topology_defaults(spec: TransformerLayerSpec) -> dict[str, int]:
        if (
            spec.hidden_size == 768
            and spec.intermediate_size == 3072
            and spec.num_attention_heads == 12
        ):
            return {
                "seq_tile": 32,
                "kv_seq_tile": 64,
                "emb_tile": 96,
                "ffn_tile": 64,
                "parallel_seq": 1,
                "parallel_heads": 1,
                "proj_acc_depth": 8,
                "o_proj_acc_group_size": 1,
                "nB_tiles_distributed": 1,
            }
        if (
            spec.hidden_size == 1024
            and spec.intermediate_size == 4096
            and spec.num_attention_heads == 16
        ):
            return {
                "seq_tile": 32,
                "kv_seq_tile": 64,
                "emb_tile": 128,
                "ffn_tile": 64,
                "parallel_seq": 1,
                "parallel_heads": 1,
                "proj_acc_depth": 8,
                "o_proj_acc_group_size": 1,
                "nB_tiles_distributed": 1,
            }
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
        topology = self._family_topology_defaults(spec)
        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        self.encoder_pipeline = AIEEncoderPipeline(
            num_heads=spec.num_attention_heads,
            seq_len=spec.seq_len,
            d=spec.attention_head_size,
            seq_tile=topology["seq_tile"],
            kv_seq_tile=topology["kv_seq_tile"],
            emb_tile=topology["emb_tile"],
            ffn_tile=topology["ffn_tile"],
            parallel_seq=topology["parallel_seq"],
            parallel_heads=topology["parallel_heads"],
            proj_acc_depth=topology["proj_acc_depth"],
            o_proj_acc_group_size=topology["o_proj_acc_group_size"],
            nB_tiles_distributed=topology["nB_tiles_distributed"],
            ffn_intermediate_size=spec.intermediate_size,
            static_weights=True,
            ln1_weight=torch.ones(hidden, dtype=dtype),
            ln2_weight=torch.ones(hidden, dtype=dtype),
            context=self.context,
        )

    def _prepare_runtime(self) -> None:
        if self._runtime_ready:
            return
        if not self._weights_assigned:
            raise RuntimeError("assign_weights() must be called before execution")
        start = time.perf_counter()
        self.context.compile_all()
        self.context.prepare_runtime()
        self.compile_setup_time_sec = time.perf_counter() - start
        self._runtime_ready = True

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
            ],
        )
        self.encoder_pipeline.w_o_proj = weights["out_proj_weight"].contiguous()
        self.encoder_pipeline.weight_up_proj = weights["ffn_up_weight"].contiguous()
        self.encoder_pipeline.weight_down_proj = weights["ffn_down_weight"].contiguous()
        self.encoder_pipeline.ln1_weight = weights["ln1_weight"].contiguous()
        self.encoder_pipeline.ln2_weight = weights["ln2_weight"].contiguous()
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
        if layer_inputs.q.shape[0] != 1:
            raise RuntimeError(
                "encoder_pipeline thesis pattern currently supports batch_size=1"
            )
        self._prepare_runtime()

        start = time.perf_counter()
        output = self.encoder_pipeline(
            layer_inputs.q.squeeze(0),
            layer_inputs.k.squeeze(0),
            layer_inputs.v.squeeze(0),
            r=layer_inputs.r.squeeze(0),
        ).unsqueeze(0)
        end = time.perf_counter()
        return output, {
            "encoder_pipeline_sec": end - start,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        placement_key = topology_from_fields(
            num_heads=self.encoder_pipeline.num_heads,
            seq_len=self.encoder_pipeline.seq_len,
            d=self.encoder_pipeline.d,
            seq_tile=self.encoder_pipeline.seq_tile,
            kv_seq_tile=self.encoder_pipeline.kv_seq_tile,
            emb_tile=self.encoder_pipeline.emb_tile,
            ffn_tile=self.encoder_pipeline.ffn_tile,
            parallel_seq=self.encoder_pipeline.parallel_seq,
            parallel_heads=self.encoder_pipeline.parallel_heads,
            proj_acc_depth=self.encoder_pipeline.proj_acc_depth,
            o_proj_acc_group_size=self.encoder_pipeline.o_proj_acc_group_size,
            parallel_ffn=self.encoder_pipeline.nB_tiles_distributed,
            ffn_intermediate_size=self.encoder_pipeline.ffn_intermediate_size,
        ).key
        placements = load_encoder_pipeline_topology_placements()
        topology = topology_from_fields(
            num_heads=self.encoder_pipeline.num_heads,
            seq_len=self.encoder_pipeline.seq_len,
            d=self.encoder_pipeline.d,
            seq_tile=self.encoder_pipeline.seq_tile,
            kv_seq_tile=self.encoder_pipeline.kv_seq_tile,
            emb_tile=self.encoder_pipeline.emb_tile,
            ffn_tile=self.encoder_pipeline.ffn_tile,
            parallel_seq=self.encoder_pipeline.parallel_seq,
            parallel_heads=self.encoder_pipeline.parallel_heads,
            proj_acc_depth=self.encoder_pipeline.proj_acc_depth,
            o_proj_acc_group_size=self.encoder_pipeline.o_proj_acc_group_size,
            parallel_ffn=self.encoder_pipeline.nB_tiles_distributed,
            ffn_intermediate_size=self.encoder_pipeline.ffn_intermediate_size,
            placement=placements.get(placement_key),
        )
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": len(self.encoder_pipeline.runlist),
            "npu_unique_instruction_binary_count": len(self.encoder_pipeline.kernels),
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
