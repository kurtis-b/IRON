# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from iron.common.utils import numpy_to_torch, torch_to_numpy
from .topology import (
    load_encoder_pipeline_topology_placements,
    load_supported_encoder_pipeline_topology_keys,
    topology_from_fields,
)


def _format_optional_bool_token(value: bool | None) -> str:
    if value is None:
        return "d"
    return str(int(value))


class AIEEncoderPipeline(AIEOperatorBase):
    """Minimal full encoder pipeline operator."""

    _QKV_PROJECTION_MODES = ("packed_input", "staged_hidden_states")

    _OPTIONAL_POSITIVE_LAYOUT_FIELDS = (
        "weight_forward_depth",
        "o_proj_fifo_depth",
        "ffn_replay_fifo_depth",
        "ffn_up_consumer_depth",
        "ffn_up_out_depth",
        "ffn_down_output_producer_depth",
    )
    _OPTIONAL_BOOL_DATA_MOVEMENT_FIELDS = (
        "use_transport_groups",
        "use_unified_qr_split",
        "use_explicit_o_proj_stage_mem_cols",
        "use_explicit_ffn_down_acc_mem_cols",
        "use_explicit_ffn_down_stage_mem_cols",
    )

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        seq_tile: int = 32,
        kv_seq_tile: int = 64,
        emb_tile: int = 96,
        ffn_tile: int | None = None,
        parallel_seq: int = 1,
        parallel_heads: int = 1,
        proj_acc_depth: int = 1,
        o_proj_acc_group_size: int = 1,
        ffn_down_acc_group_size: int = 1,
        nB_tiles_distributed: int = 1,
        ffn_intermediate_size: int | None = None,
        weight_forward_depth: int | None = None,
        o_proj_fifo_depth: int | None = None,
        ffn_replay_fifo_depth: int | None = None,
        ffn_up_consumer_depth: int | None = None,
        ffn_up_out_depth: int | None = None,
        ffn_down_output_producer_depth: int | None = None,
        use_fused_replayed_addnorm: bool | None = None,
        use_transport_groups: bool | None = None,
        use_unified_qr_split: bool | None = None,
        use_explicit_o_proj_stage_mem_cols: bool | None = None,
        use_explicit_ffn_down_acc_mem_cols: bool | None = None,
        use_explicit_ffn_down_stage_mem_cols: bool | None = None,
        qkv_projection_mode: str = "packed_input",
        static_weights: bool = False,
        ln1_weight=None,
        ln2_weight=None,
        ln_weight=None,
        context=None,
        skip_add_to_list: bool = False,
    ):
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.seq_tile = seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.ffn_tile = emb_tile if ffn_tile is None else ffn_tile
        self.parallel_seq = parallel_seq
        self.parallel_heads = parallel_heads
        self.proj_acc_depth = proj_acc_depth
        self.o_proj_acc_group_size = o_proj_acc_group_size
        self.ffn_down_acc_group_size = ffn_down_acc_group_size
        self.nB_tiles_distributed = nB_tiles_distributed
        self.embed_sz = d * num_heads
        self.ffn_intermediate_size = (
            4 * self.embed_sz
            if ffn_intermediate_size is None
            else ffn_intermediate_size
        )
        self.weight_forward_depth = self._normalize_optional_positive_int(
            "weight_forward_depth", weight_forward_depth
        )
        self.o_proj_fifo_depth = self._normalize_optional_positive_int(
            "o_proj_fifo_depth", o_proj_fifo_depth
        )
        self.ffn_replay_fifo_depth = self._normalize_optional_positive_int(
            "ffn_replay_fifo_depth", ffn_replay_fifo_depth
        )
        self.ffn_up_consumer_depth = self._normalize_optional_positive_int(
            "ffn_up_consumer_depth", ffn_up_consumer_depth
        )
        self.ffn_up_out_depth = self._normalize_optional_positive_int(
            "ffn_up_out_depth", ffn_up_out_depth
        )
        self.ffn_down_output_producer_depth = self._normalize_optional_positive_int(
            "ffn_down_output_producer_depth", ffn_down_output_producer_depth
        )
        self.use_fused_replayed_addnorm = self._normalize_optional_bool(
            "use_fused_replayed_addnorm", use_fused_replayed_addnorm
        )
        self.use_transport_groups = self._normalize_optional_bool(
            "use_transport_groups", use_transport_groups
        )
        self.use_unified_qr_split = self._normalize_optional_bool(
            "use_unified_qr_split", use_unified_qr_split
        )
        self.use_explicit_o_proj_stage_mem_cols = self._normalize_optional_bool(
            "use_explicit_o_proj_stage_mem_cols",
            use_explicit_o_proj_stage_mem_cols,
        )
        self.use_explicit_ffn_down_acc_mem_cols = self._normalize_optional_bool(
            "use_explicit_ffn_down_acc_mem_cols",
            use_explicit_ffn_down_acc_mem_cols,
        )
        self.use_explicit_ffn_down_stage_mem_cols = self._normalize_optional_bool(
            "use_explicit_ffn_down_stage_mem_cols",
            use_explicit_ffn_down_stage_mem_cols,
        )
        self.qkv_projection_mode = self._normalize_qkv_projection_mode(
            qkv_projection_mode
        )

        self._validate_configuration()

        self.w_o_proj = None
        self.weight_up_proj = None
        self.weight_down_proj = None
        self.qkv_proj_weight = None
        self.qkv_proj_bias = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_bias = None
        self.v_proj_bias = None
        if static_weights:
            self.w_o_proj = torch.zeros(
                (self.embed_sz, self.embed_sz), dtype=torch.bfloat16
            ).T
            self.weight_up_proj = torch.zeros(
                (self.embed_sz, self.ffn_intermediate_size), dtype=torch.bfloat16
            ).T
            self.weight_down_proj = torch.zeros(
                (self.ffn_intermediate_size, self.embed_sz), dtype=torch.bfloat16
            ).T

        if ln_weight is not None:
            if ln1_weight is None:
                ln1_weight = ln_weight
            if ln2_weight is None:
                ln2_weight = ln_weight
        self.ln1_weight = (
            ln1_weight
            if ln1_weight is not None
            else torch.ones(self.embed_sz, dtype=torch.bfloat16)
        )
        self.ln2_weight = (
            ln2_weight
            if ln2_weight is not None
            else torch.ones(self.embed_sz, dtype=torch.bfloat16)
        )

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    @staticmethod
    def _normalize_optional_positive_int(name: str, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires {name} to be an integer > 0 (got {value})"
            )
        normalized = int(value)
        if normalized <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires {name} > 0 (got {normalized})"
            )
        return normalized

    @staticmethod
    def _normalize_optional_bool(name: str, value: bool | None) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires {name} to be a bool or None (got {value})"
            )
        return value

    @classmethod
    def _normalize_qkv_projection_mode(cls, value: str) -> str:
        normalized = str(value)
        if normalized not in cls._QKV_PROJECTION_MODES:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires qkv_projection_mode to be one of "
                f"{cls._QKV_PROJECTION_MODES} (got {value!r})"
            )
        return normalized

    def _validate_configuration(self):
        requested_topology = topology_from_fields(
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            d=self.d,
            seq_tile=self.seq_tile,
            kv_seq_tile=self.kv_seq_tile,
            emb_tile=self.emb_tile,
            ffn_tile=self.ffn_tile,
            parallel_seq=self.parallel_seq,
            parallel_heads=self.parallel_heads,
            proj_acc_depth=self.proj_acc_depth,
            o_proj_acc_group_size=1,
            parallel_ffn=self.nB_tiles_distributed,
            ffn_intermediate_size=self.ffn_intermediate_size,
        )
        supported_topologies = load_supported_encoder_pipeline_topology_keys()
        placement = load_encoder_pipeline_topology_placements().get(
            requested_topology.key
        )
        if self.d != 64:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline only supports d=64 today (got {self.d})"
            )
        if self.parallel_seq <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires parallel_seq > 0 (got {self.parallel_seq})"
            )
        if self.parallel_heads <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires parallel_heads > 0 (got {self.parallel_heads})"
            )
        if self.nB_tiles_distributed <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires nB_tiles_distributed > 0 "
                f"(got {self.nB_tiles_distributed})"
            )
        if self.seq_len % self.seq_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires seq_len divisible by seq_tile "
                f"({self.seq_len} % {self.seq_tile} != 0)"
            )
        if (self.seq_len // self.seq_tile) % self.parallel_seq != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires num_q_seq_blocks divisible by parallel_seq "
                f"({self.seq_len // self.seq_tile} % {self.parallel_seq} != 0)"
            )
        if self.embed_sz % self.emb_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile to divide embed_sz "
                f"({self.embed_sz} % {self.emb_tile} != 0)"
            )
        if self.proj_acc_depth != self.embed_sz // self.emb_tile:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile * proj_acc_depth == embed_sz "
                f"({self.emb_tile} * {self.proj_acc_depth} != {self.embed_sz})"
            )
        if self.ffn_tile % 16 != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_tile divisible by 16 "
                f"({self.ffn_tile} % 16 != 0)"
            )
        if self.ffn_intermediate_size % self.ffn_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_intermediate_size divisible by ffn_tile "
                f"({self.ffn_intermediate_size} % {self.ffn_tile} != 0)"
            )
        if self.o_proj_acc_group_size <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires o_proj_acc_group_size > 0 "
                f"(got {self.o_proj_acc_group_size})"
            )
        if self.ffn_down_acc_group_size <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_down_acc_group_size > 0 "
                f"(got {self.ffn_down_acc_group_size})"
            )
        if self.parallel_seq > 1 and self.parallel_heads > 1:
            if self.o_proj_acc_group_size not in (1, self.parallel_heads):
                raise AIEOperatorConstraintError(
                    "encoder_pipeline seq-par currently supports "
                    "o_proj_acc_group_size of 1 or parallel_heads "
                    f"(got {self.o_proj_acc_group_size}, parallel_heads={self.parallel_heads})"
                )
        elif self.o_proj_acc_group_size != 1:
            raise AIEOperatorConstraintError(
                "encoder_pipeline currently supports o_proj_acc_group_size=1 "
                "outside the seq-par multi-head path "
                f"(got {self.o_proj_acc_group_size})"
            )
        if self.parallel_seq > 1 and self.nB_tiles_distributed > 1:
            if self.ffn_down_acc_group_size not in (1, self.nB_tiles_distributed):
                raise AIEOperatorConstraintError(
                    "encoder_pipeline seq-par currently supports "
                    "ffn_down_acc_group_size of 1 or nB_tiles_distributed "
                    f"(got {self.ffn_down_acc_group_size}, nB_tiles_distributed={self.nB_tiles_distributed})"
                )
        elif self.ffn_down_acc_group_size != 1:
            raise AIEOperatorConstraintError(
                "encoder_pipeline currently supports ffn_down_acc_group_size=1 "
                "outside the seq-par multi-branch path "
                f"(got {self.ffn_down_acc_group_size})"
            )
        if requested_topology.key not in supported_topologies:
            raise AIEOperatorConstraintError(
                "encoder_pipeline currently supports only hardcoded placement "
                f"topologies {sorted(supported_topologies)} "
                f"(got placement key {requested_topology.key})"
            )
        sequence_parallel = (
            None if placement is None else placement.get("sequence_parallel")
        )
        if self.use_transport_groups is not None:
            transport_groups = (
                None
                if sequence_parallel is None
                else sequence_parallel.get("transport_groups")
            )
            if transport_groups is None or len(transport_groups) <= 1:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline only supports use_transport_groups as an explicit "
                    "design-space dimension on sequence-parallel placements with "
                    "multiple transport groups"
                )
        if self.use_unified_qr_split is not None:
            unified_qr_split = (
                None
                if sequence_parallel is None
                else sequence_parallel.get("unified_qr_split")
            )
            if unified_qr_split is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline only supports use_unified_qr_split as an "
                    "explicit design-space dimension on placements that define "
                    "unified_qr_split"
                )
        if self.use_explicit_o_proj_stage_mem_cols is not None:
            explicit_cols = (
                None
                if sequence_parallel is None
                else sequence_parallel.get("lane_o_proj_stage_mem_cols")
            )
            if explicit_cols is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline only supports use_explicit_o_proj_stage_mem_cols "
                    "as an explicit design-space dimension on placements that define "
                    "lane_o_proj_stage_mem_cols"
                )
        if self.use_explicit_ffn_down_acc_mem_cols is not None:
            explicit_cols = (
                None
                if sequence_parallel is None
                else sequence_parallel.get("lane_ffn_down_acc_mem_cols")
            )
            if explicit_cols is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline only supports use_explicit_ffn_down_acc_mem_cols "
                    "as an explicit design-space dimension on placements that define "
                    "lane_ffn_down_acc_mem_cols"
                )
        if self.use_explicit_ffn_down_stage_mem_cols is not None:
            explicit_cols = (
                None
                if sequence_parallel is None
                else sequence_parallel.get("lane_ffn_down_stage_mem_cols")
            )
            if explicit_cols is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline only supports use_explicit_ffn_down_stage_mem_cols "
                    "as an explicit design-space dimension on placements that define "
                    "lane_ffn_down_stage_mem_cols"
                )
        if (
            self.use_fused_replayed_addnorm is True
            and not self._supports_addnorm_replay_fastpath()
        ):
            raise AIEOperatorConstraintError(
                "encoder_pipeline only supports use_fused_replayed_addnorm=True "
                "for the current winner topologies"
            )
        if self.qkv_projection_mode == "staged_hidden_states":
            if sequence_parallel is not None and (
                self.parallel_heads > 2 or self.nB_tiles_distributed != 1
            ):
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection currently requires parallel_heads <= 2 and "
                    "nB_tiles_distributed=1 "
                    f"(got parallel_heads={self.parallel_heads}, "
                    f"nB_tiles_distributed={self.nB_tiles_distributed})"
                )
            if self.parallel_heads > 2:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states QKV projection currently "
                    "supports parallel_heads <= 2 "
                    f"(got parallel_heads={self.parallel_heads})"
                )

    def _resolve_layout_design_space(self) -> dict[str, int | bool]:
        large_activation_tile = (
            self.emb_tile >= 128
            or self.seq_tile * self.emb_tile * np.dtype(bfloat16).itemsize > 8192
        )
        resolved = {
            "weight_forward_depth": (
                1 if large_activation_tile else (2 if self.ffn_tile <= 64 else 1)
            ),
            "o_proj_fifo_depth": 1 if large_activation_tile else 2,
            "ffn_replay_fifo_depth": 1 if large_activation_tile else 2,
            "ffn_up_consumer_depth": 1 if large_activation_tile else 2,
            "ffn_up_out_depth": 1 if large_activation_tile else 2,
            "ffn_down_output_producer_depth": 1 if large_activation_tile else 2,
        }
        for field_name in self._OPTIONAL_POSITIVE_LAYOUT_FIELDS:
            override_value = getattr(self, field_name)
            if override_value is not None:
                resolved[field_name] = override_value
        resolved["use_fused_replayed_addnorm"] = self._resolve_addnorm_replay_fastpath()
        for field_name in self._OPTIONAL_BOOL_DATA_MOVEMENT_FIELDS:
            override_value = getattr(self, field_name)
            if override_value is not None:
                resolved[field_name] = override_value
        return resolved

    def _resolve_addnorm_replay_fastpath(self) -> bool:
        if self.use_fused_replayed_addnorm is not None:
            return self.use_fused_replayed_addnorm
        disable_fastpath = os.environ.get("IRON_ENCODER_DISABLE_FASTADDNORM") == "1"
        return self._supports_addnorm_replay_fastpath() and not disable_fastpath

    def _artifact_stem(
        self, prefix: str, include_ln_weight_digests: bool = True
    ) -> str:
        layout_design_space = self._resolve_layout_design_space()
        use_addnorm_replay_fastpath = layout_design_space["use_fused_replayed_addnorm"]
        identity_parts = [
            prefix,
            self.num_heads,
            self.seq_len,
            self.d,
            self.seq_tile,
            self.kv_seq_tile,
            self.emb_tile,
            self.ffn_tile,
            self.parallel_seq,
            self.parallel_heads,
            self.proj_acc_depth,
            self.o_proj_acc_group_size,
            self.ffn_down_acc_group_size,
            self.nB_tiles_distributed,
            self.ffn_intermediate_size,
            self.qkv_projection_mode,
            int(use_addnorm_replay_fastpath),
            layout_design_space["weight_forward_depth"],
            layout_design_space["o_proj_fifo_depth"],
            layout_design_space["ffn_replay_fifo_depth"],
            layout_design_space["ffn_up_consumer_depth"],
            layout_design_space["ffn_up_out_depth"],
            layout_design_space["ffn_down_output_producer_depth"],
            layout_design_space.get("use_transport_groups"),
            layout_design_space.get("use_unified_qr_split"),
        ]
        if include_ln_weight_digests:
            identity_parts.extend(
                [
                    self._tensor_digest(self.ln1_weight),
                    self._tensor_digest(self.ln2_weight),
                ]
            )
        identity = "|".join(map(str, identity_parts))
        digest = hashlib.blake2s(identity.encode(), digest_size=6).hexdigest()
        return (
            f"{prefix}_{self.num_heads}h_{self.seq_len}s_{self.emb_tile}e_"
            f"{self.ffn_tile}f_"
            f"{self.parallel_seq}ps_{self.parallel_heads}ph_{self.proj_acc_depth}pa_"
            f"{self.o_proj_acc_group_size}g_{self.ffn_down_acc_group_size}fg_"
            f"{self.nB_tiles_distributed}pf_fa{int(use_addnorm_replay_fastpath)}_"
            f"wf{layout_design_space['weight_forward_depth']}_"
            f"of{layout_design_space['o_proj_fifo_depth']}_"
            f"fr{layout_design_space['ffn_replay_fifo_depth']}_"
            f"uc{layout_design_space['ffn_up_consumer_depth']}_"
            f"uo{layout_design_space['ffn_up_out_depth']}_"
            f"do{layout_design_space['ffn_down_output_producer_depth']}_"
            f"tg{_format_optional_bool_token(layout_design_space.get('use_transport_groups'))}_"
            f"uq{_format_optional_bool_token(layout_design_space.get('use_unified_qr_split'))}_"
            f"ops{_format_optional_bool_token(layout_design_space.get('use_explicit_o_proj_stage_mem_cols'))}_"
            f"fda{_format_optional_bool_token(layout_design_space.get('use_explicit_ffn_down_acc_mem_cols'))}_"
            f"fds{_format_optional_bool_token(layout_design_space.get('use_explicit_ffn_down_stage_mem_cols'))}_{digest}"
        )

    def _supports_addnorm_replay_fastpath(self) -> bool:
        return (
            self.seq_tile == 32
            and self.kv_seq_tile == 64
            and self.emb_tile == 96
            and self.proj_acc_depth == 8
            and (
                (
                    self.parallel_seq == 4
                    and self.parallel_heads == 1
                    and self.nB_tiles_distributed == 1
                )
                or (
                    self.parallel_seq == 2
                    and self.parallel_heads == 2
                    and self.nB_tiles_distributed == 2
                )
            )
        )

    @staticmethod
    def _tensor_digest(tensor: torch.Tensor) -> str:
        tensor_bytes = (
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
        return hashlib.blake2s(tensor_bytes, digest_size=4).hexdigest()

    @staticmethod
    def combine_qkv_projection_parameters(
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        value_weight: torch.Tensor,
        query_bias: torch.Tensor | None = None,
        key_bias: torch.Tensor | None = None,
        value_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        combined_weight = torch.cat(
            (query_weight, key_weight, value_weight),
            dim=0,
        ).T.contiguous()
        if query_bias is None and key_bias is None and value_bias is None:
            combined_bias = None
        else:
            if query_bias is None or key_bias is None or value_bias is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline combined QKV projection requires either "
                    "all biases or no biases"
                )
            combined_bias = torch.cat(
                (query_bias, key_bias, value_bias),
                dim=0,
            ).contiguous()
        return combined_weight, combined_bias

    @staticmethod
    def split_combined_qkv_projection_parameters(
        combined_weight: torch.Tensor,
        combined_bias: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if combined_weight.ndim != 2:
            raise AIEOperatorConstraintError(
                "encoder_pipeline combined QKV weight must be rank-2 "
                f"(got shape {tuple(combined_weight.shape)})"
            )
        input_dim, combined_dim = combined_weight.shape
        if combined_dim != 3 * input_dim:
            raise AIEOperatorConstraintError(
                "encoder_pipeline combined QKV weight must have shape "
                f"({input_dim}, {3 * input_dim}) "
                f"(got {tuple(combined_weight.shape)})"
            )
        if combined_bias is not None and tuple(combined_bias.shape) != (combined_dim,):
            raise AIEOperatorConstraintError(
                "encoder_pipeline combined QKV bias must have shape "
                f"({combined_dim},) (got {tuple(combined_bias.shape)})"
            )

        q_weight, k_weight, v_weight = torch.split(combined_weight, input_dim, dim=1)
        if combined_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = torch.split(combined_bias, input_dim, dim=0)
        return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias

    @staticmethod
    def pack_projected_qkv_rows(
        projected_qkv: torch.Tensor,
        *,
        seq_len: int,
        embed_sz: int,
    ) -> torch.Tensor:
        expected_shape = (seq_len, 3 * embed_sz)
        if tuple(projected_qkv.shape) != expected_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline projected QKV tensor must have shape "
                f"{expected_shape} (got {tuple(projected_qkv.shape)})"
            )
        return (
            projected_qkv.view(seq_len, 3, embed_sz)
            .permute(1, 0, 2)
            .contiguous()
            .view(3 * seq_len, embed_sz)
        )

    @staticmethod
    def pack_qkv_tensors(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len: int,
        num_heads: int,
        d: int,
    ) -> torch.Tensor:
        expected_head_major_shape = (num_heads, seq_len, d)
        expected_seq_major_shape = (seq_len, num_heads, d)

        def _to_seq_major(name: str, tensor: torch.Tensor) -> torch.Tensor:
            shape = tuple(tensor.shape)
            if shape == expected_head_major_shape:
                return tensor.transpose(0, 1).contiguous()
            if shape == expected_seq_major_shape:
                return tensor.contiguous()
            raise AIEOperatorConstraintError(
                "encoder_pipeline expected "
                f"{name} to have shape {expected_head_major_shape} or "
                f"{expected_seq_major_shape} (got {shape})"
            )

        q_rows = _to_seq_major("q", q).view(seq_len, num_heads * d)
        k_rows = _to_seq_major("k", k).view(seq_len, num_heads * d)
        v_rows = _to_seq_major("v", v).view(seq_len, num_heads * d)
        return torch.cat((q_rows, k_rows, v_rows), dim=0).contiguous()

    @staticmethod
    def pack_q_projection_bias_tiles(
        q_bias: torch.Tensor,
        *,
        seq_tile: int,
        num_heads: int,
        d: int,
    ) -> torch.Tensor:
        expected_shape = (num_heads * d,)
        if tuple(q_bias.shape) != expected_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline Q bias must have shape "
                f"{expected_shape} (got {tuple(q_bias.shape)})"
            )
        return (
            q_bias.view(num_heads, 1, d)
            .expand(num_heads, seq_tile, d)
            .contiguous()
            .view(num_heads * seq_tile, d)
        )

    @staticmethod
    def pack_k_projection_bias_tiles(
        k_bias: torch.Tensor,
        *,
        kv_seq_tile: int,
        num_heads: int,
        d: int,
    ) -> torch.Tensor:
        expected_shape = (num_heads * d,)
        if tuple(k_bias.shape) != expected_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline K bias must have shape "
                f"{expected_shape} (got {tuple(k_bias.shape)})"
            )
        return (
            k_bias.view(num_heads, d, 1)
            .expand(num_heads, d, kv_seq_tile)
            .contiguous()
            .view(num_heads * d, kv_seq_tile)
        )

    @staticmethod
    def pack_v_projection_bias_tiles(
        v_bias: torch.Tensor,
        *,
        kv_seq_tile: int,
        num_heads: int,
        d: int,
    ) -> torch.Tensor:
        expected_shape = (num_heads * d,)
        if tuple(v_bias.shape) != expected_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline V bias must have shape "
                f"{expected_shape} (got {tuple(v_bias.shape)})"
            )
        return (
            v_bias.view(num_heads, 1, d)
            .expand(num_heads, kv_seq_tile, d)
            .contiguous()
            .view(num_heads * kv_seq_tile, d)
        )

    def _build_design_callback_kwargs(
        self,
        *,
        kernel_archive: str,
        ln1_weight_file,
        ln2_weight_file,
        q_proj_bias_file,
        k_proj_bias_file,
        v_proj_bias_file,
    ) -> dict:
        layout_design_space = self._resolve_layout_design_space()
        return {
            "heads": self.num_heads,
            "seq_len": self.seq_len,
            "d": self.d,
            "seq_tile": self.seq_tile,
            "kv_seq_tile": self.kv_seq_tile,
            "emb_tile": self.emb_tile,
            "ffn_tile": self.ffn_tile,
            "parallel_seq": self.parallel_seq,
            "parallel_heads": self.parallel_heads,
            "proj_acc_depth": self.proj_acc_depth,
            "o_proj_acc_group_size": self.o_proj_acc_group_size,
            "ffn_down_acc_group_size": self.ffn_down_acc_group_size,
            "nB_tiles_distributed": self.nB_tiles_distributed,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "qkv_projection_mode": self.qkv_projection_mode,
            "emulate_bf16_mmul_with_bfp16": True,
            "kernel_archive": kernel_archive,
            "ln1_weight_file": ln1_weight_file,
            "ln2_weight_file": ln2_weight_file,
            "q_proj_bias_file": q_proj_bias_file,
            "k_proj_bias_file": k_proj_bias_file,
            "v_proj_bias_file": v_proj_bias_file,
            "trace_size": 0,
            "weight_forward_depth": layout_design_space["weight_forward_depth"],
            "o_proj_fifo_depth": layout_design_space["o_proj_fifo_depth"],
            "ffn_replay_fifo_depth": layout_design_space["ffn_replay_fifo_depth"],
            "ffn_up_consumer_depth": layout_design_space["ffn_up_consumer_depth"],
            "ffn_up_out_depth": layout_design_space["ffn_up_out_depth"],
            "ffn_down_output_producer_depth": layout_design_space[
                "ffn_down_output_producer_depth"
            ],
            "use_fused_replayed_addnorm": layout_design_space[
                "use_fused_replayed_addnorm"
            ],
            "use_transport_groups": layout_design_space.get("use_transport_groups"),
            "use_unified_qr_split": layout_design_space.get("use_unified_qr_split"),
            "use_explicit_o_proj_stage_mem_cols": layout_design_space.get(
                "use_explicit_o_proj_stage_mem_cols"
            ),
            "use_explicit_ffn_down_acc_mem_cols": layout_design_space.get(
                "use_explicit_ffn_down_acc_mem_cols"
            ),
            "use_explicit_ffn_down_stage_mem_cols": layout_design_space.get(
                "use_explicit_ffn_down_stage_mem_cols"
            ),
            "use_runtime_ln_weights": True,
        }

    def get_artifacts(self, prefix: str = "encoder_pipeline"):
        operator_dir = Path(__file__).parent
        xclbin_file_base = self._artifact_stem(prefix, include_ln_weight_digests=False)
        insts_file_base = self._artifact_stem(prefix, include_ln_weight_digests=True)

        ln1_weight_file_name = (
            self.context.build_dir / f"{insts_file_base}_ln1_weight_{self.embed_sz}.npy"
        )
        ln2_weight_file_name = (
            self.context.build_dir / f"{insts_file_base}_ln2_weight_{self.embed_sz}.npy"
        )
        np.save(ln1_weight_file_name, torch_to_numpy(self.ln1_weight))
        np.save(ln2_weight_file_name, torch_to_numpy(self.ln2_weight))
        q_proj_bias_file_name = None
        k_proj_bias_file_name = None
        v_proj_bias_file_name = None
        if self.qkv_projection_mode == "staged_hidden_states":
            static_b_q, static_b_k, static_b_v = (
                self._resolve_staged_projection_static_bias_tiles()
            )
            if static_b_q is not None:
                q_proj_bias_file_name = (
                    self.context.build_dir / f"{insts_file_base}_q_proj_bias.npy"
                )
                np.save(q_proj_bias_file_name, static_b_q)
            if static_b_k is not None:
                k_proj_bias_file_name = (
                    self.context.build_dir / f"{insts_file_base}_k_proj_bias.npy"
                )
                np.save(k_proj_bias_file_name, static_b_k)
            if static_b_v is not None:
                v_proj_bias_file_name = (
                    self.context.build_dir / f"{insts_file_base}_v_proj_bias.npy"
                )
                np.save(v_proj_bias_file_name, static_b_v)

        base_dir = self.context.base_dir
        mm_source = str(base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(base_dir / "aie_kernels" / "aie2p" / "softmax.cc")
        mha_source = str(base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        passthrough_source = str(
            base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        convert_copy_source = str(
            base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
        )
        add_source = str(base_dir / "aie_kernels" / "generic" / "add.cc")
        encoder_source = str(base_dir / "aie_kernels" / "aie2p" / "encoder.cc")

        mm_defines_rowmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.kv_seq_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        mm_defines_colmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.d}",
            f"-DDIM_N={self.kv_seq_tile}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DB_COL_MAJ",
        ]
        mm_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_rowmaj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_rowmaj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_rowmaj",
            "zero_bf16": "zero_bf16_rowmaj",
            "zero_scalar_bf16": "zero_scalar_bf16_rowmaj",
        }
        mm_o_proj_defines = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.d}",
            f"-DDIM_N={self.emb_tile}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
            "-DGENERATE_MATMUL_INIT_KERNELS",
        ]
        mm_o_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_o_proj",
            "matmul_init_bf16_bf16": "matmul_init_bf16_bf16_o_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_o_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_o_proj",
            "zero_bf16": "zero_bf16_o_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_o_proj",
        }
        mm_q_proj_defines = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
            "-DGENERATE_MATMUL_INIT_KERNELS",
        ]
        mm_q_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_q_proj",
            "matmul_init_bf16_bf16": "matmul_init_bf16_bf16_q_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_q_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_q_proj",
            "zero_bf16": "zero_bf16_q_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_q_proj",
        }
        mm_k_proj_defines = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.kv_seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
            "-DGENERATE_MATMUL_INIT_KERNELS",
            "-DC_COL_MAJ",
        ]
        mm_k_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_k_proj",
            "matmul_init_bf16_bf16": "matmul_init_bf16_bf16_k_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_k_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_k_proj",
            "zero_bf16": "zero_bf16_k_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_k_proj",
        }
        mm_v_proj_defines = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.kv_seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
            "-DGENERATE_MATMUL_INIT_KERNELS",
        ]
        mm_v_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_v_proj",
            "matmul_init_bf16_bf16": "matmul_init_bf16_bf16_v_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_v_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_v_proj",
            "zero_bf16": "zero_bf16_v_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_v_proj",
        }
        add_q_proj_rename_symbols = {
            "eltwise_add_bf16_tile_bias_matrix": "eltwise_add_bf16_tile_bias_matrix_q_proj",
        }
        add_kv_proj_rename_symbols = {
            "eltwise_add_bf16_tile_bias_matrix": "eltwise_add_bf16_tile_bias_matrix_kv_proj",
        }
        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.ffn_tile}",
        ]
        use_addnorm_replay_fastpath = self._resolve_layout_design_space()[
            "use_fused_replayed_addnorm"
        ]
        if use_addnorm_replay_fastpath:
            encoder_kernel_flags.append("-DBUILD_ADDNORM_REPLAY_FASTPATH")
        encoder_kernel_object_name = (
            f"{prefix}_encoder_{self.seq_tile}x{self.emb_tile}x{self.ffn_tile}_fastaddnorm.o"
            if use_addnorm_replay_fastpath
            else f"{prefix}_encoder_{self.seq_tile}x{self.emb_tile}x{self.ffn_tile}.o"
        )
        encoder_debug_value = os.environ.get("IRON_ENCODER_DEBUG_AIE_KERNELS")
        if encoder_debug_value is not None:
            encoder_kernel_flags.append(
                f"-DDEBUG_AIE_KERNELS={int(encoder_debug_value)}"
            )

        kernel_archive = (
            f"{xclbin_file_base}_fastaddnorm_kernels.a"
            if use_addnorm_replay_fastpath
            else f"{xclbin_file_base}_kernels.a"
        )
        xclbin_mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{xclbin_file_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="encoder_pipeline",
            tracked_paths=[
                operator_dir / "design.py",
                operator_dir / "op.py",
                operator_dir / "placements.py",
            ],
            callback_kwargs=self._build_design_callback_kwargs(
                kernel_archive=kernel_archive,
                ln1_weight_file=None,
                ln2_weight_file=None,
                q_proj_bias_file=q_proj_bias_file_name,
                k_proj_bias_file=k_proj_bias_file_name,
                v_proj_bias_file=v_proj_bias_file_name,
            ),
        )
        insts_mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{insts_file_base}.insts.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="encoder_pipeline",
            tracked_paths=[
                operator_dir / "design.py",
                operator_dir / "op.py",
                operator_dir / "placements.py",
            ],
            callback_kwargs=self._build_design_callback_kwargs(
                kernel_archive=kernel_archive,
                ln1_weight_file=ln1_weight_file_name,
                ln2_weight_file=ln2_weight_file_name,
                q_proj_bias_file=q_proj_bias_file_name,
                k_proj_bias_file=k_proj_bias_file_name,
                v_proj_bias_file=v_proj_bias_file_name,
            ),
        )

        kernel_archive_depends = [
            KernelObjectArtifact.new(
                f"{prefix}_mm_qk_{self.seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                extra_flags=mm_defines_colmaj,
                depends=[SourceArtifact.new(mm_source)],
            ),
            KernelObjectArtifact.new(
                f"{prefix}_mm_pv_rowmaj_{self.seq_tile}m_{self.kv_seq_tile}k_{self.d}n.o",
                extra_flags=mm_defines_rowmaj,
                depends=[SourceArtifact.new(mm_source)],
                rename_symbols=mm_rename_symbols,
            ),
            KernelObjectArtifact.new(
                f"{prefix}_mm_o_{self.seq_tile}m_{self.emb_tile}n_{self.d}k.o",
                extra_flags=mm_o_proj_defines,
                depends=[SourceArtifact.new(mm_source)],
                rename_symbols=mm_o_proj_rename_symbols,
            ),
            KernelObjectArtifact.new(
                f"{prefix}_softmax_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k.o",
                depends=[SourceArtifact.new(softmax_source)],
                extra_flags=[f"-DSM_VEC_LEN={self.kv_seq_tile}"],
            ),
            KernelObjectArtifact.new(
                f"{prefix}_mha_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k_causal0_0.o",
                depends=[SourceArtifact.new(mha_source)],
                extra_flags=[
                    "-DIS_CAUSAL=0",
                    f"-DVECTOR_LENGTH={min(self.seq_tile, self.kv_seq_tile)}",
                    f"-DSCALE_VECTOR_LENGTH={self.seq_tile}",
                ],
            ),
            KernelObjectArtifact.new(
                f"{prefix}_passThrough_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                extra_flags=["-DBIT_WIDTH=16"],
                depends=[SourceArtifact.new(passthrough_source)],
            ),
            KernelObjectArtifact.new(
                f"{prefix}_passThrough_o_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                extra_flags=["-DBIT_WIDTH=16"],
                depends=[SourceArtifact.new(passthrough_source)],
                rename_symbols={
                    "passThroughLine": "passThroughLine_o_proj",
                    "passThroughTile": "passThroughTile_o_proj",
                },
            ),
            KernelObjectArtifact.new(
                f"{prefix}_convert_copy.o",
                depends=[SourceArtifact.new(convert_copy_source)],
            ),
            KernelObjectArtifact.new(
                f"{prefix}_add_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                depends=[SourceArtifact.new(add_source)],
                rename_symbols={
                    "eltwise_add_bf16_scalar": "eltwise_add_bf16_scalar_o_proj",
                    "eltwise_add_bf16_vector": "eltwise_add_bf16_vector_o_proj",
                    "eltwise_add_f32_vector": "eltwise_add_f32_vector_o_proj",
                },
            ),
            KernelObjectArtifact.new(
                encoder_kernel_object_name,
                depends=[SourceArtifact.new(encoder_source)],
                extra_flags=encoder_kernel_flags,
            ),
        ]
        if self.qkv_projection_mode == "staged_hidden_states":
            kernel_archive_depends.extend(
                [
                    KernelObjectArtifact.new(
                        f"{prefix}_mm_q_proj_{self.seq_tile}m_{self.emb_tile}k_{self.d}n.o",
                        extra_flags=mm_q_proj_defines,
                        depends=[SourceArtifact.new(mm_source)],
                        rename_symbols=mm_q_proj_rename_symbols,
                    ),
                    KernelObjectArtifact.new(
                        f"{prefix}_mm_k_proj_{self.kv_seq_tile}m_{self.emb_tile}k_{self.d}n.o",
                        extra_flags=mm_k_proj_defines,
                        depends=[SourceArtifact.new(mm_source)],
                        rename_symbols=mm_k_proj_rename_symbols,
                    ),
                    KernelObjectArtifact.new(
                        f"{prefix}_mm_v_proj_{self.kv_seq_tile}m_{self.emb_tile}k_{self.d}n.o",
                        extra_flags=mm_v_proj_defines,
                        depends=[SourceArtifact.new(mm_source)],
                        rename_symbols=mm_v_proj_rename_symbols,
                    ),
                    KernelObjectArtifact.new(
                        f"{prefix}_add_q_proj_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                        depends=[SourceArtifact.new(add_source)],
                        rename_symbols=add_q_proj_rename_symbols,
                    ),
                    KernelObjectArtifact.new(
                        f"{prefix}_add_kv_proj_{self.kv_seq_tile}m_{self.kv_seq_tile}n_{self.d}k.o",
                        depends=[SourceArtifact.new(add_source)],
                        rename_symbols=add_kv_proj_rename_symbols,
                    ),
                ]
            )

        xclbin_artifact = XclbinArtifact.new(
            f"{xclbin_file_base}.xclbin",
            depends=[
                xclbin_mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=kernel_archive_depends,
                ),
            ],
            extra_flags=["--dynamic-objFifos"],
        )
        insts_artifact = InstsBinArtifact.new(
            f"{insts_file_base}.bin",
            depends=[insts_mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
            xclbin_input=xclbin_artifact,
            kernel_name=xclbin_artifact.kernel_name,
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def _uses_staged_hidden_state_projection(self) -> bool:
        return self.qkv_projection_mode == "staged_hidden_states"

    def _uses_staged_hidden_state_kv_cache(self) -> bool:
        return False

    def _or_buffer_shape(self):
        if self.parallel_seq > 1:
            ln1_stage_rows = self.seq_len
        else:
            ln1_stage_rows = self.proj_acc_depth * self.seq_tile
        or_rows_before_ln1_stage = (
            3 * self.seq_len
            if self._uses_staged_hidden_state_kv_cache()
            else (
                4 * self.seq_len
                if (
                    self._uses_staged_hidden_state_projection()
                    and self.parallel_seq > 1
                )
                else (
                    self.seq_len
                    if self._uses_staged_hidden_state_projection()
                    else 2 * self.seq_len
                )
            )
        )
        return (or_rows_before_ln1_stage + ln1_stage_rows, self.embed_sz)

    def _resolve_staged_projection_parameters(
        self,
        *,
        combined_weight: torch.Tensor | None = None,
        combined_bias: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        q_weight = self.q_proj_weight
        k_weight = self.k_proj_weight
        v_weight = self.v_proj_weight
        q_bias = self.q_proj_bias
        k_bias = self.k_proj_bias
        v_bias = self.v_proj_bias
        resolved_combined_weight = (
            self.qkv_proj_weight if combined_weight is None else combined_weight
        )
        resolved_combined_bias = (
            self.qkv_proj_bias if combined_bias is None else combined_bias
        )
        if q_weight is None or k_weight is None or v_weight is None:
            if resolved_combined_weight is None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode requires either "
                    "split Q/K/V projection weights or a combined QKV weight"
                )
            (
                q_weight,
                k_weight,
                v_weight,
                split_q_bias,
                split_k_bias,
                split_v_bias,
            ) = self.split_combined_qkv_projection_parameters(
                resolved_combined_weight,
                resolved_combined_bias,
            )
            if q_bias is None:
                q_bias = split_q_bias
            if k_bias is None:
                k_bias = split_k_bias
            if v_bias is None:
                v_bias = split_v_bias
        expected_weight_shape = (self.embed_sz, self.embed_sz)
        for name, tensor in (
            ("q_proj_weight", q_weight),
            ("k_proj_weight", k_weight),
            ("v_proj_weight", v_weight),
        ):
            if tuple(tensor.shape) != expected_weight_shape:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode requires "
                    f"{name} to have shape {expected_weight_shape} "
                    f"(got {tuple(tensor.shape)})"
                )
        expected_bias_shape = (self.embed_sz,)
        for name, tensor in (
            ("q_proj_bias", q_bias),
            ("k_proj_bias", k_bias),
            ("v_proj_bias", v_bias),
        ):
            if tensor is not None and tuple(tensor.shape) != expected_bias_shape:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode requires "
                    f"{name} to have shape {expected_bias_shape} "
                    f"(got {tuple(tensor.shape)})"
                )
        return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias

    def _pack_staged_attention_weights(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        w_o_weight: torch.Tensor,
    ) -> np.ndarray:
        return torch_to_numpy(
            torch.cat(
                (
                    q_weight,
                    k_weight,
                    v_weight,
                    w_o_weight.T.contiguous(),
                ),
                dim=0,
            )
        )

    def _resolve_staged_projection_static_attn_weights(self) -> np.ndarray | None:
        if (
            self.q_proj_weight is None
            and self.k_proj_weight is None
            and self.v_proj_weight is None
            and self.qkv_proj_weight is None
        ):
            return None
        if self.w_o_proj is None:
            return None
        q_weight, k_weight, v_weight, _, _, _ = (
            self._resolve_staged_projection_parameters()
        )
        return self._pack_staged_attention_weights(
            q_weight,
            k_weight,
            v_weight,
            self.w_o_proj,
        )

    def _resolve_staged_projection_static_bias_tiles(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if (
            self.q_proj_bias is None
            and self.k_proj_bias is None
            and self.v_proj_bias is None
            and self.qkv_proj_bias is None
        ):
            return None, None, None
        q_bias = self.q_proj_bias
        k_bias = self.k_proj_bias
        v_bias = self.v_proj_bias
        if (
            q_bias is None or k_bias is None or v_bias is None
        ) and self.qkv_proj_bias is not None:
            combined_bias = self.qkv_proj_bias
            expected_shape = (3 * self.embed_sz,)
            if tuple(combined_bias.shape) != expected_shape:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline combined QKV bias must have shape "
                    f"{expected_shape} (got {tuple(combined_bias.shape)})"
                )
            q_bias, k_bias, v_bias = torch.split(combined_bias, self.embed_sz)
        zero_bias = torch.zeros(self.embed_sz, dtype=torch.bfloat16)
        q_bias = zero_bias if q_bias is None else q_bias
        k_bias = zero_bias if k_bias is None else k_bias
        v_bias = zero_bias if v_bias is None else v_bias
        return (
            torch_to_numpy(q_bias),
            torch_to_numpy(k_bias),
            torch_to_numpy(v_bias),
        )

    def set_up_runtime(self):
        self.add_kernel(
            "encoder_pipeline",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )

        static_attn_weights = None
        static_w_o_proj = None
        if self._uses_staged_hidden_state_projection():
            static_attn_weights = self._resolve_staged_projection_static_attn_weights()
        elif self.w_o_proj is not None:
            static_w_o_proj = self.w_o_proj.T
            if isinstance(static_w_o_proj, torch.Tensor):
                static_w_o_proj = torch_to_numpy(static_w_o_proj)
        static_weights_up_proj = None
        if self.weight_up_proj is not None:
            static_weights_up_proj = self.weight_up_proj.T
            if isinstance(static_weights_up_proj, torch.Tensor):
                static_weights_up_proj = torch_to_numpy(static_weights_up_proj)
        static_weights_down_proj = None
        if self.weight_down_proj is not None:
            static_weights_down_proj = self.weight_down_proj.T
            if isinstance(static_weights_down_proj, torch.Tensor):
                static_weights_down_proj = torch_to_numpy(static_weights_down_proj)

        if self._uses_staged_hidden_state_projection():
            self.add_buffer(
                "W_ATTN",
                4 * self.embed_sz * self.embed_sz,
                static_data=static_attn_weights,
            )
            self.add_buffer("X", self.embed_sz * self.seq_len)
        else:
            self.add_buffer(
                "W_O", self.embed_sz * self.embed_sz, static_data=static_w_o_proj
            )
            self.add_buffer("QKV", 3 * self.embed_sz * self.seq_len)
        self.add_buffer("OR", int(np.prod(self._or_buffer_shape())))
        self.add_buffer("O", self.embed_sz * self.seq_len)
        self.add_buffer(
            "B_Up",
            self.embed_sz * self.ffn_intermediate_size,
            static_data=static_weights_up_proj,
        )
        self.add_buffer(
            "B_Down",
            self.ffn_intermediate_size * self.embed_sz,
            static_data=static_weights_down_proj,
        )
        self.buffer_aliases["O"] = "OR"
        if self._uses_staged_hidden_state_projection():
            self.add_to_runlist(
                "encoder_pipeline",
                "W_ATTN",
                "X",
                "OR",
                "B_Up",
                "B_Down",
            )
        else:
            self.add_to_runlist(
                "encoder_pipeline", "W_O", "QKV", "OR", "B_Up", "B_Down"
            )

    def _write_qkv_zero_copy(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        packed_qkv = self.pack_qkv_tensors(
            q,
            k,
            v,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            d=self.d,
        )
        qkv_view = self.buffer_view(
            "QKV",
            (3 * self.num_heads * self.seq_len * self.d,),
            dtype=bfloat16,
        )
        qkv_view[:] = torch_to_numpy(packed_qkv).reshape(-1)

    def _write_packed_qkv_zero_copy(self, qkv: torch.Tensor):
        qkv_np = torch_to_numpy(qkv).reshape(-1)
        qkv_view = self.buffer_view(
            "QKV",
            (3 * self.num_heads * self.seq_len * self.d,),
            dtype=bfloat16,
        )
        qkv_view[:] = qkv_np

    def _write_hidden_states_zero_copy(self, hidden_states: torch.Tensor):
        hidden_states_np = torch_to_numpy(hidden_states)
        if "X" in self.buffers:
            x_view = self.buffer_view(
                "X",
                (self.seq_len * self.embed_sz,),
                dtype=bfloat16,
            )
            x_view[:] = hidden_states_np.reshape(-1)
            return

        or_view = self.buffer_view(
            "OR",
            self._or_buffer_shape(),
            dtype=bfloat16,
        )
        or_view[: self.seq_len, :] = hidden_states_np

    def _write_staged_projection_runtime_parameters(
        self,
        *,
        q_weight: torch.Tensor | None = None,
        k_weight: torch.Tensor | None = None,
        v_weight: torch.Tensor | None = None,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        combined_weight: torch.Tensor | None = None,
        combined_bias: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
    ):
        (
            resolved_q_weight,
            resolved_k_weight,
            resolved_v_weight,
            resolved_q_bias,
            resolved_k_bias,
            resolved_v_bias,
        ) = self._resolve_staged_projection_parameters(
            combined_weight=combined_weight,
            combined_bias=combined_bias,
        )
        if q_weight is not None:
            resolved_q_weight = q_weight
        if k_weight is not None:
            resolved_k_weight = k_weight
        if v_weight is not None:
            resolved_v_weight = v_weight
        expected_weight_shape = (self.embed_sz, self.embed_sz)
        for name, tensor in (
            ("q_proj_weight", resolved_q_weight),
            ("k_proj_weight", resolved_k_weight),
            ("v_proj_weight", resolved_v_weight),
        ):
            if tuple(tensor.shape) != expected_weight_shape:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode requires "
                    f"{name} to have shape {expected_weight_shape} "
                    f"(got {tuple(tensor.shape)})"
                )
        expected_bias_shape = (self.embed_sz,)
        for name, tensor in (
            ("q_proj_bias", resolved_q_bias),
            ("k_proj_bias", resolved_k_bias),
            ("v_proj_bias", resolved_v_bias),
        ):
            if tensor is not None and tuple(tensor.shape) != expected_bias_shape:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode requires "
                    f"{name} to have shape {expected_bias_shape} "
                    f"(got {tuple(tensor.shape)})"
                )
        if any(bias is not None for bias in (q_bias, k_bias, v_bias, combined_bias)):
            raise AIEOperatorConstraintError(
                "encoder_pipeline staged_hidden_states mode requires Q/K/V biases "
                "to be assigned on the operator before compilation; runtime bias "
                "overrides are not supported"
            )
        resolved_w_o = self.w_o_proj if w_o is None else w_o
        expected_w_o_shape = (self.embed_sz, self.embed_sz)
        if resolved_w_o is None or tuple(resolved_w_o.shape) != expected_w_o_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline staged_hidden_states mode requires "
                f"w_o to have shape {expected_w_o_shape} "
                f"(got {None if resolved_w_o is None else tuple(resolved_w_o.shape)})"
            )
        if "W_ATTN" not in self.buffer_static_data:
            self.write_buffer(
                "W_ATTN",
                self._pack_staged_attention_weights(
                    resolved_q_weight,
                    resolved_k_weight,
                    resolved_v_weight,
                    resolved_w_o,
                ),
            )

    def project_hidden_states_to_packed_qkv(
        self,
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor | None = None,
        qkv_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(hidden_states.shape) != (self.seq_len, self.embed_sz):
            raise AIEOperatorConstraintError(
                "encoder_pipeline hidden_states must have shape "
                f"({self.seq_len}, {self.embed_sz}) "
                f"(got {tuple(hidden_states.shape)})"
            )
        resolved_weight = self.qkv_proj_weight if qkv_weight is None else qkv_weight
        resolved_bias = self.qkv_proj_bias if qkv_bias is None else qkv_bias
        if resolved_weight is None:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires qkv_proj_weight when projecting "
                "hidden states inside the operator boundary"
            )
        expected_weight_shape = (self.embed_sz, 3 * self.embed_sz)
        if tuple(resolved_weight.shape) != expected_weight_shape:
            raise AIEOperatorConstraintError(
                "encoder_pipeline combined QKV weight must have shape "
                f"{expected_weight_shape} (got {tuple(resolved_weight.shape)})"
            )
        if resolved_bias is not None and tuple(resolved_bias.shape) != (
            3 * self.embed_sz,
        ):
            raise AIEOperatorConstraintError(
                "encoder_pipeline combined QKV bias must have shape "
                f"({3 * self.embed_sz},) (got {tuple(resolved_bias.shape)})"
            )

        projected_qkv = torch.matmul(hidden_states, resolved_weight)
        if resolved_bias is not None:
            projected_qkv = projected_qkv + resolved_bias
        return self.pack_projected_qkv_rows(
            projected_qkv,
            seq_len=self.seq_len,
            embed_sz=self.embed_sz,
        )

    def _write_or_zero_copy(self, r: torch.Tensor | None):
        or_view = self.buffer_view("OR", self._or_buffer_shape(), dtype=bfloat16)
        or_view.fill(0)
        if self._uses_staged_hidden_state_projection():
            if r is not None:
                raise AIEOperatorConstraintError(
                    "encoder_pipeline staged_hidden_states mode derives the "
                    "initial residual from hidden_states internally; external "
                    "residual input is not supported"
                )
            return
        if r is not None:
            or_view[self.seq_len : 2 * self.seq_len, :] = torch_to_numpy(r)

    def forward(
        self,
        hidden_states_or_q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
        b_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ):
        if k is None and v is None:
            return self.forward_hidden_states(
                hidden_states_or_q,
                r=r,
                w_o=w_o,
                b_up=b_up,
                b_down=b_down,
            )
        if k is None or v is None:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline.forward(...) requires both k and v when "
                "using the legacy split-QKV compatibility path"
            )
        return self.forward_split_qkv(
            hidden_states_or_q,
            k,
            v,
            r=r,
            w_o=w_o,
            b_up=b_up,
            b_down=b_down,
        )

    def forward_split_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
        b_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ):
        if self.qkv_projection_mode != "packed_input":
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline.forward_split_qkv(...) is only supported when "
                "qkv_projection_mode='packed_input'"
            )
        b_up_shape = (
            b_up.shape
            if b_up is not None
            else (
                self.weight_up_proj.T.shape if self.weight_up_proj is not None else None
            )
        )
        b_down_shape = (
            b_down.shape
            if b_down is not None
            else (
                self.weight_down_proj.T.shape
                if self.weight_down_proj is not None
                else None
            )
        )
        if b_up_shape is None or b_down_shape is None:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: B_Up/B_Down must be provided unless static_weights=True"
            )

        expected_b_up = (self.embed_sz, self.ffn_intermediate_size)
        expected_b_down = (self.ffn_intermediate_size, self.embed_sz)
        if tuple(b_up_shape) != expected_b_up or tuple(b_down_shape) != expected_b_down:
            raise AIEOperatorConstraintError(
                f"AIEEncoderPipeline: expected B_Up shape {expected_b_up} and B_Down shape {expected_b_down}, "
                f"got {tuple(b_up_shape)} and {tuple(b_down_shape)}"
            )

        applicable = (
            q.shape[-1] == self.d
            and k.shape[-1] == self.d
            and v.shape[-1] == self.d
            and q.shape[-2] == self.seq_len
            and k.shape[-2] == self.seq_len
            and v.shape[-2] == self.seq_len
            and self.seq_len % 64 == 0
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: incompatible tensor shape(s)"
            )
        return self._execute_aie_operation(q, k, v, r, w_o, b_up, b_down)

    def _execute_aie_operation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor | None,
        w_o: torch.Tensor | None,
        b_up: torch.Tensor | None,
        b_down: torch.Tensor | None,
    ):
        self._write_qkv_zero_copy(q, k, v)
        self._write_or_zero_copy(r)
        if w_o is not None:
            self.write_buffer("W_O", torch_to_numpy(w_o))
        if b_up is not None:
            self.write_buffer("B_Up", torch_to_numpy(b_up))
        if b_down is not None:
            self.write_buffer("B_Down", torch_to_numpy(b_down))

        self.run_runlist()
        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        return numpy_to_torch(o_np)

    def forward_packed_qkv(
        self,
        qkv: torch.Tensor,
        r: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
        b_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ):
        if self.qkv_projection_mode != "packed_input":
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline.forward_packed_qkv(...) is only supported when "
                "qkv_projection_mode='packed_input'"
            )
        expected_qkv_shape = (3 * self.seq_len, self.embed_sz)
        if tuple(qkv.shape) != expected_qkv_shape:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: expected packed QKV shape "
                f"{expected_qkv_shape}, got {tuple(qkv.shape)}"
            )

        b_up_shape = (
            b_up.shape
            if b_up is not None
            else (
                self.weight_up_proj.T.shape if self.weight_up_proj is not None else None
            )
        )
        b_down_shape = (
            b_down.shape
            if b_down is not None
            else (
                self.weight_down_proj.T.shape
                if self.weight_down_proj is not None
                else None
            )
        )
        if b_up_shape is None or b_down_shape is None:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: B_Up/B_Down must be provided unless static_weights=True"
            )
        expected_b_up = (self.embed_sz, self.ffn_intermediate_size)
        expected_b_down = (self.ffn_intermediate_size, self.embed_sz)
        if tuple(b_up_shape) != expected_b_up or tuple(b_down_shape) != expected_b_down:
            raise AIEOperatorConstraintError(
                f"AIEEncoderPipeline: expected B_Up shape {expected_b_up} and B_Down shape {expected_b_down}, "
                f"got {tuple(b_up_shape)} and {tuple(b_down_shape)}"
            )

        self._write_packed_qkv_zero_copy(qkv)
        self._write_or_zero_copy(r)
        if w_o is not None:
            self.write_buffer("W_O", torch_to_numpy(w_o))
        if b_up is not None:
            self.write_buffer("B_Up", torch_to_numpy(b_up))
        if b_down is not None:
            self.write_buffer("B_Down", torch_to_numpy(b_down))

        self.run_runlist()
        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        return numpy_to_torch(o_np)

    def forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        r: torch.Tensor | None = None,
        qkv_weight: torch.Tensor | None = None,
        qkv_bias: torch.Tensor | None = None,
        q_weight: torch.Tensor | None = None,
        k_weight: torch.Tensor | None = None,
        v_weight: torch.Tensor | None = None,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
        b_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ):
        if tuple(hidden_states.shape) != (self.seq_len, self.embed_sz):
            raise AIEOperatorConstraintError(
                "encoder_pipeline hidden_states must have shape "
                f"({self.seq_len}, {self.embed_sz}) "
                f"(got {tuple(hidden_states.shape)})"
            )
        if self.qkv_projection_mode == "packed_input":
            packed_qkv = self.project_hidden_states_to_packed_qkv(
                hidden_states,
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
            )
            return self.forward_packed_qkv(
                packed_qkv,
                r=r,
                w_o=w_o,
                b_up=b_up,
                b_down=b_down,
            )
        if r is not None:
            raise AIEOperatorConstraintError(
                "encoder_pipeline staged_hidden_states mode derives the initial "
                "residual from hidden_states internally; do not pass r"
            )

        b_up_shape = (
            b_up.shape
            if b_up is not None
            else (
                self.weight_up_proj.T.shape if self.weight_up_proj is not None else None
            )
        )
        b_down_shape = (
            b_down.shape
            if b_down is not None
            else (
                self.weight_down_proj.T.shape
                if self.weight_down_proj is not None
                else None
            )
        )
        if b_up_shape is None or b_down_shape is None:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: B_Up/B_Down must be provided unless static_weights=True"
            )
        expected_b_up = (self.embed_sz, self.ffn_intermediate_size)
        expected_b_down = (self.ffn_intermediate_size, self.embed_sz)
        if tuple(b_up_shape) != expected_b_up or tuple(b_down_shape) != expected_b_down:
            raise AIEOperatorConstraintError(
                f"AIEEncoderPipeline: expected B_Up shape {expected_b_up} and B_Down shape {expected_b_down}, "
                f"got {tuple(b_up_shape)} and {tuple(b_down_shape)}"
            )

        self._write_or_zero_copy(None)
        self._write_hidden_states_zero_copy(hidden_states)
        self._write_staged_projection_runtime_parameters(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            combined_weight=qkv_weight,
            combined_bias=qkv_bias,
            w_o=w_o,
        )
        if w_o is not None and not self._uses_staged_hidden_state_projection():
            self.write_buffer("W_O", torch_to_numpy(w_o))
        if b_up is not None:
            self.write_buffer("B_Up", torch_to_numpy(b_up))
        if b_down is not None:
            self.write_buffer("B_Down", torch_to_numpy(b_down))

        self.run_runlist()
        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        return numpy_to_torch(o_np)
