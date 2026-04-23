# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import time

import torch
from ml_dtypes import bfloat16

from iron.common import AIEOperatorBase, AIEOperatorConstraintError
from iron.common.utils import torch_to_numpy
from iron.operators.dynamic_gemm.op import AIEDynamicGEMM

from iron.applications.transformer_layer.pattern.runlist.op import (
    _resolve_query_block_size,
)

LOGGER = logging.getLogger(__name__)

OFFLOAD_GEMM_OPERATOR_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_scores",
    "attn_output",
    "output_proj",
    "up_proj",
    "down_proj",
)
_STATIC_WEIGHT_OPERATORS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "up_proj",
    "down_proj",
}
_WEIGHT_BUFFER_BY_OPERATOR = {
    "q_proj": "q_weight",
    "k_proj": "k_weight",
    "v_proj": "v_weight",
    "output_proj": "attn_output_weight",
    "up_proj": "ffn_up_weight",
    "down_proj": "ffn_down_weight",
}
_OFFLOAD_SHARED_GEMM_DEFAULTS = {
    "num_aie_columns": 8,
    "b_col_maj": False,
    "c_col_maj": False,
    "prio_accuracy": False,
    "emulate_bf16_mmul_with_bfp16": True,
}
_OFFLOAD_TILE_MEMORY_TARGET_BYTES = 64 * 1024
_OFFLOAD_TILE_M_CANDIDATES = (64, 32, 16, 8)
_OFFLOAD_TILE_KN_CANDIDATES = (256, 192, 160, 128, 96, 80, 64, 48, 32, 24, 16, 8)


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _require_supported_num_aie_columns(num_aie_columns: int) -> None:
    if int(num_aie_columns) != int(_OFFLOAD_SHARED_GEMM_DEFAULTS["num_aie_columns"]):
        raise ValueError(
            "AIETransformerOffload uses a single shared GEMM topology per "
            "operator instance with num_aie_columns=8"
        )


def _pick_shared_offload_topology(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    *,
    query_block_size: int,
) -> dict[str, int]:
    head_dim = hidden_size // num_heads
    m_values = (seq_len, query_block_size)
    k_values = (hidden_size, intermediate_size, seq_len, head_dim)
    n_values = (hidden_size, intermediate_size, seq_len, head_dim)

    def _supports_row_shape(tile_m: int) -> bool:
        full_m = 4 * tile_m
        return all(
            value % tile_m == 0 and not (value % full_m != 0 and value < full_m)
            for value in m_values
        )

    tile_m = next(
        candidate
        for candidate in _OFFLOAD_TILE_M_CANDIDATES
        if _supports_row_shape(candidate)
    )

    valid_tile_k = [
        candidate
        for candidate in _OFFLOAD_TILE_KN_CANDIDATES
        if all(value % candidate == 0 for value in k_values)
    ]
    valid_tile_n = [
        candidate
        for candidate in _OFFLOAD_TILE_KN_CANDIDATES
        if all(value % candidate == 0 for value in n_values)
    ]
    if not valid_tile_k or not valid_tile_n:
        raise ValueError(
            "Unable to choose a shared offload GEMM tile_k/tile_n that divides "
            "all logical GEMM dimensions for this workload"
        )

    best_choice: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for tile_k in valid_tile_k:
        for tile_n in valid_tile_n:
            tile_bytes = 2 * (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n)
            if tile_bytes > _OFFLOAD_TILE_MEMORY_TARGET_BYTES:
                continue
            score = (
                tile_bytes,
                min(tile_k, tile_n),
                max(tile_k, tile_n),
            )
            if best_score is None or score > best_score:
                best_choice = (tile_k, tile_n)
                best_score = score

    if best_choice is None:
        # Fall back to the largest legal pair even if it cannot reach the target.
        best_choice = (max(valid_tile_k), max(valid_tile_n))

    tile_k, tile_n = best_choice
    return {
        "tile_m": int(tile_m),
        "tile_k": int(tile_k),
        "tile_n": int(tile_n),
    }


def default_offload_operator_config(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    *,
    workload_variant: str = "encoder_bert",
    num_aie_columns: int = 8,
) -> dict[str, dict[str, object]]:
    _require_supported_num_aie_columns(num_aie_columns)
    if workload_variant not in {"encoder_bert", "decoder_gpt2"}:
        raise ValueError(f"Unsupported offload workload_variant: {workload_variant}")

    head_dim = hidden_size // num_heads
    query_block_size = _resolve_query_block_size(seq_len, num_heads)
    shared_topology = _pick_shared_offload_topology(
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        query_block_size=query_block_size,
    )

    def _gemm_config(
        M: int,
        K: int,
        N: int,
        *,
        use_static_weight: bool,
        query_block: int | None = None,
    ) -> dict[str, object]:
        config = {
            **_OFFLOAD_SHARED_GEMM_DEFAULTS,
            **shared_topology,
            "M": M,
            "K": K,
            "N": N,
            "use_static_weight": use_static_weight,
        }
        if query_block is not None:
            config["query_block_size"] = query_block
            config["num_heads"] = num_heads
        return config

    return {
        "q_proj": _gemm_config(
            seq_len, hidden_size, hidden_size, use_static_weight=True
        ),
        "k_proj": _gemm_config(
            seq_len, hidden_size, hidden_size, use_static_weight=True
        ),
        "v_proj": _gemm_config(
            seq_len, hidden_size, hidden_size, use_static_weight=True
        ),
        "attn_scores": _gemm_config(
            query_block_size,
            head_dim,
            seq_len,
            use_static_weight=False,
            query_block=query_block_size,
        ),
        "attn_output": _gemm_config(
            query_block_size,
            seq_len,
            head_dim,
            use_static_weight=False,
            query_block=query_block_size,
        ),
        "output_proj": _gemm_config(
            seq_len, hidden_size, hidden_size, use_static_weight=True
        ),
        "up_proj": _gemm_config(
            seq_len, hidden_size, intermediate_size, use_static_weight=True
        ),
        "down_proj": _gemm_config(
            seq_len, intermediate_size, hidden_size, use_static_weight=True
        ),
    }


def resolve_offload_operator_config(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    *,
    workload_variant: str = "encoder_bert",
    num_aie_columns: int = 8,
    operator_config: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    resolved = {
        name: dict(config)
        for name, config in default_offload_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            workload_variant=workload_variant,
            num_aie_columns=num_aie_columns,
        ).items()
    }
    if operator_config is not None:
        for name, overrides in operator_config.items():
            if name not in resolved:
                raise ValueError(f"Unsupported offload operator config: {name}")
            resolved[name].update(overrides)

    query_block_size_values = {
        int(config.get("query_block_size", config["M"]))
        for name, config in resolved.items()
        if name in {"attn_scores", "attn_output"}
    }
    if len(query_block_size_values) != 1:
        raise ValueError(
            "Offload attention GEMMs require a shared query_block_size between "
            "attn_scores and attn_output"
        )
    query_block_size = next(iter(query_block_size_values))
    resolved["attn_scores"]["query_block_size"] = query_block_size
    resolved["attn_output"]["query_block_size"] = query_block_size
    resolved["attn_scores"]["M"] = query_block_size
    resolved["attn_output"]["M"] = query_block_size

    for name, config in resolved.items():
        config["num_aie_columns"] = int(
            _OFFLOAD_SHARED_GEMM_DEFAULTS["num_aie_columns"]
        )
        for key, value in _OFFLOAD_SHARED_GEMM_DEFAULTS.items():
            config.setdefault(key, value)
        if bool(config.get("c_col_maj", False)):
            raise ValueError("Offload GEMM operators require c_col_maj=False")
        if bool(config.get("b_col_maj", False)):
            raise ValueError("Offload GEMM operators require b_col_maj=False")
    return resolved


class AIETransformerOffload(AIEOperatorBase):
    """Transformer-layer pattern that offloads only GEMMs to the NPU."""

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_aie_columns: int = 8,
        ln1_weight=None,
        ln2_weight=None,
        workload_variant: str = "encoder_bert",
        operator_config: dict[str, dict[str, object]] | None = None,
        context=None,
    ):
        _require_supported_num_aie_columns(num_aie_columns)
        if hidden_size % num_heads != 0:
            raise AIEOperatorConstraintError(
                "hidden_size must be divisible by num_heads"
            )
        if workload_variant not in {"encoder_bert", "decoder_gpt2"}:
            raise ValueError(
                f"Unsupported offload workload_variant: {workload_variant}"
            )

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_aie_columns = int(num_aie_columns)
        self.workload_variant = workload_variant
        self.head_dim = hidden_size // num_heads
        self.operator_config = resolve_offload_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            workload_variant=workload_variant,
            num_aie_columns=num_aie_columns,
            operator_config=operator_config,
        )
        self.query_block_size = int(
            self.operator_config["attn_scores"]["query_block_size"]
        )
        self.query_block_count = max(1, seq_len // self.query_block_size)

        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.shared_gemm_xclbin = None
        self.shared_gemm_runtime_kernel_name = "offload_gemm"
        self.shared_gemm_runtime_operator = None
        self.q_proj_insts = None
        self.k_proj_insts = None
        self.v_proj_insts = None
        self.attn_scores_insts = None
        self.attn_output_insts = None
        self.output_proj_insts = None
        self.up_proj_insts = None
        self.down_proj_insts = None
        self.reset_buffer_names = ()
        self.enable_benchmark_buffer_reset = False
        self.host_output_buffer_names = ("output",)
        self._runtime_breakdown_enabled = False
        self._runtime_breakdown: dict[str, float | int] = {}

        AIEOperatorBase.__init__(self, context=context)

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self._runtime_breakdown_enabled = bool(enabled)
        if enabled:
            self.reset_runtime_breakdown()

    def reset_runtime_breakdown(self) -> None:
        self._runtime_breakdown = {
            "npu_gemm_time_sec": 0.0,
            "host_attention_time_sec": 0.0,
            "host_elementwise_time_sec": 0.0,
            "transfer_time_sec": 0.0,
            "launch_count": 0,
            "bytes_written": 0,
            "bytes_read": 0,
        }

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = dict(self._runtime_breakdown)
        summary["query_block_size"] = int(self.query_block_size)
        summary["query_block_count"] = int(self.query_block_count)
        return summary

    def _record_transfer_write(
        self, *, started: float, tensor: torch.Tensor | None
    ) -> None:
        if not self._runtime_breakdown_enabled:
            return
        self._runtime_breakdown["transfer_time_sec"] = float(
            self._runtime_breakdown.get("transfer_time_sec", 0.0)
        ) + (time.perf_counter() - started)
        self._runtime_breakdown["bytes_written"] = int(
            self._runtime_breakdown.get("bytes_written", 0)
        ) + _tensor_nbytes(tensor)

    def _record_transfer_read(
        self, *, started: float, tensor: torch.Tensor | None
    ) -> None:
        if not self._runtime_breakdown_enabled:
            return
        self._runtime_breakdown["transfer_time_sec"] = float(
            self._runtime_breakdown.get("transfer_time_sec", 0.0)
        ) + (time.perf_counter() - started)
        self._runtime_breakdown["bytes_read"] = int(
            self._runtime_breakdown.get("bytes_read", 0)
        ) + _tensor_nbytes(tensor)

    def _buffer_name_for_weight(self, operator_name: str) -> str:
        return _WEIGHT_BUFFER_BY_OPERATOR[operator_name]

    def _logical_gemm(
        self,
        operator_name: str,
        *,
        skip_add_to_list: bool = True,
    ) -> AIEDynamicGEMM:
        config = self.operator_config[operator_name]
        return AIEDynamicGEMM(
            int(config["M"]),
            int(config["K"]),
            int(config["N"]),
            use_static_weight=bool(config.get("use_static_weight", False)),
            tile_m=int(config["tile_m"]),
            tile_k=int(config["tile_k"]),
            tile_n=int(config["tile_n"]),
            num_aie_columns=int(config["num_aie_columns"]),
            b_col_maj=bool(config.get("b_col_maj", False)),
            c_col_maj=bool(config.get("c_col_maj", False)),
            prio_accuracy=bool(config.get("prio_accuracy", False)),
            emulate_bf16_mmul_with_bfp16=bool(
                config.get("emulate_bf16_mmul_with_bfp16", True)
            ),
            context=self.context,
            skip_add_to_list=skip_add_to_list,
        )

    def _shared_xclbin_prefix(self) -> str:
        variant = "decoder" if self.workload_variant == "decoder_gpt2" else "encoder"
        return f"{variant}_offload_shared_gemm_"

    def _inst_prefix(self, operator_name: str) -> str:
        variant = "decoder" if self.workload_variant == "decoder_gpt2" else "encoder"
        return f"{variant}_offload_{operator_name}_"

    def set_up_artifacts(self):
        shared_operator = self._logical_gemm("q_proj")
        shared_xclbin = shared_operator.get_runtime_xclbin_artifact(
            prefix=self._shared_xclbin_prefix()
        )
        shared_xclbin.extra_flags += [
            "--xclbin-instance-name=offload_gemm",
            "--xclbin-kernel-id=0x801",
        ]
        shared_xclbin.kernel_name = self.shared_gemm_runtime_kernel_name
        self.shared_gemm_xclbin = shared_xclbin
        self.shared_gemm_runtime_operator = shared_operator

        artifacts = [self.shared_gemm_xclbin]
        for operator_name in OFFLOAD_GEMM_OPERATOR_NAMES:
            prototype = self._logical_gemm(operator_name)
            insts = prototype.get_insts_artifact(
                prefix=self._inst_prefix(operator_name),
                xclbin_input=self.shared_gemm_xclbin,
                kernel_name=self.shared_gemm_runtime_kernel_name,
            )
            setattr(self, f"{operator_name}_insts", insts)
            artifacts.append(insts)

        self.add_artifacts(artifacts)

    def _weight_tensor(self, operator_name: str) -> torch.Tensor | None:
        return {
            "q_proj": self.q_weight,
            "k_proj": self.k_weight,
            "v_proj": self.v_weight,
            "output_proj": self.attn_output_weight,
            "up_proj": self.ffn_up_weight,
            "down_proj": self.ffn_down_weight,
        }.get(operator_name)

    def _validate_required_weights(self) -> None:
        required_weights = (
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
            "up_proj",
            "down_proj",
        )
        missing = [
            operator_name
            for operator_name in required_weights
            if self._weight_tensor(operator_name) is None
        ]
        if missing:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload is missing required weights for: "
                + ", ".join(missing)
            )
        if self.ln1_weight is None or self.ln2_weight is None:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload requires ln1_weight and ln2_weight"
            )

    def _max_dynamic_a_elems(self) -> int:
        return max(
            int(config["M"]) * int(config["K"])
            for config in self.operator_config.values()
        )

    def _max_dynamic_b_elems(self) -> int:
        dynamic_ops = (
            self.operator_config[name]
            for name in OFFLOAD_GEMM_OPERATOR_NAMES
            if not bool(self.operator_config[name].get("use_static_weight", False))
        )
        return max(int(config["K"]) * int(config["N"]) for config in dynamic_ops)

    def _max_dynamic_c_elems(self) -> int:
        return max(
            int(config["M"]) * int(config["N"])
            for config in self.operator_config.values()
        )

    def set_up_runtime(self):
        self._validate_required_weights()

        for operator_name in OFFLOAD_GEMM_OPERATOR_NAMES:
            self.add_kernel(
                operator_name,
                self.shared_gemm_xclbin,
                self.shared_gemm_runtime_kernel_name,
                getattr(self, f"{operator_name}_insts"),
            )

        self.add_buffer("input", self.seq_len * self.hidden_size)
        self.add_buffer("output", self.seq_len * self.hidden_size)
        self.add_buffer("gemm_a", self._max_dynamic_a_elems())
        self.add_buffer("gemm_b_dynamic", self._max_dynamic_b_elems())
        self.add_buffer("gemm_c", self._max_dynamic_c_elems())

        for operator_name in _STATIC_WEIGHT_OPERATORS:
            weight = self._weight_tensor(operator_name)
            if weight is None:
                continue
            config = self.operator_config[operator_name]
            self.add_buffer(
                self._buffer_name_for_weight(operator_name),
                int(config["K"]) * int(config["N"]),
                static_data=torch_to_numpy(weight).reshape(-1),
            )

        # These entries are not executed through the base runlist path. They exist so
        # runtime BO allocation can see the real simultaneous kernel arguments and keep
        # `gemm_a`, `gemm_b_dynamic`, and `gemm_c` on distinct BOs.
        for operator_name in _STATIC_WEIGHT_OPERATORS:
            self.add_to_runlist(
                operator_name,
                "gemm_a",
                self._buffer_name_for_weight(operator_name),
                "gemm_c",
            )
        for operator_name in ("attn_scores", "attn_output"):
            self.add_to_runlist(
                operator_name,
                "gemm_a",
                "gemm_b_dynamic",
                "gemm_c",
            )

    def _zero_bias(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(weight)

    def _run_gemm(
        self,
        operator_name: str,
        a_tensor: torch.Tensor,
        *,
        b_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.operator_config[operator_name]
        output_shape = (int(config["M"]), int(config["N"]))
        contiguous_a = a_tensor.contiguous()
        started = time.perf_counter()
        self.write_buffer("gemm_a", contiguous_a)
        self._record_transfer_write(started=started, tensor=contiguous_a)
        if b_tensor is None:
            b_buffer = self._buffer_name_for_weight(operator_name)
        else:
            contiguous_b = b_tensor.contiguous()
            started = time.perf_counter()
            self.write_buffer("gemm_b_dynamic", contiguous_b)
            self._record_transfer_write(started=started, tensor=contiguous_b)
            b_buffer = "gemm_b_dynamic"
        started = time.perf_counter()
        self.run_kernel_once(operator_name, "gemm_a", b_buffer, "gemm_c")
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["npu_gemm_time_sec"] = float(
                self._runtime_breakdown.get("npu_gemm_time_sec", 0.0)
            ) + (time.perf_counter() - started)
            self._runtime_breakdown["launch_count"] = int(
                self._runtime_breakdown.get("launch_count", 0)
            ) + 1
        started = time.perf_counter()
        output = self.read_buffer_as_torch("gemm_c", output_shape, dtype=bfloat16)
        self._record_transfer_read(started=started, tensor=output)
        return output

    def _blocked_attention(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        attn_output = torch.empty_like(v_heads)
        scale = math.sqrt(1.0 / float(self.head_dim))

        for block_index in range(self.query_block_count):
            q_start = block_index * self.query_block_size
            q_end = q_start + self.query_block_size
            q_block = q_heads[:, q_start:q_end, :].contiguous()

            for head_index in range(self.num_heads):
                scores = self._run_gemm(
                    "attn_scores",
                    q_block[head_index],
                    b_tensor=k_heads[head_index].transpose(-2, -1).contiguous(),
                )
                scores = scores.float() * scale
                if causal:
                    host_started = time.perf_counter()
                    query_positions = torch.arange(q_start, q_end, device=scores.device)
                    key_positions = torch.arange(self.seq_len, device=scores.device)
                    causal_mask = key_positions.unsqueeze(
                        0
                    ) > query_positions.unsqueeze(1)
                    scores = scores.masked_fill(causal_mask, float("-inf"))
                    if self._runtime_breakdown_enabled:
                        self._runtime_breakdown["host_attention_time_sec"] = float(
                            self._runtime_breakdown.get("host_attention_time_sec", 0.0)
                        ) + (time.perf_counter() - host_started)
                host_started = time.perf_counter()
                attn_probs = torch.nn.functional.softmax(scores, dim=-1).to(
                    torch.bfloat16
                )
                if self._runtime_breakdown_enabled:
                    self._runtime_breakdown["host_attention_time_sec"] = float(
                        self._runtime_breakdown.get("host_attention_time_sec", 0.0)
                    ) + (time.perf_counter() - host_started)
                attn_output[head_index, q_start:q_end, :] = self._run_gemm(
                    "attn_output",
                    attn_probs,
                    b_tensor=v_heads[head_index].contiguous(),
                )

        return attn_output

    def _run_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        q = self._run_gemm("q_proj", input_tensor)
        k = self._run_gemm("k_proj", input_tensor)
        v = self._run_gemm("v_proj", input_tensor)

        q_heads = q.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k_heads = k.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v_heads = v.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        attn_heads = self._blocked_attention(
            q_heads,
            k_heads,
            v_heads,
            causal=False,
        )
        attn_output = (
            attn_heads.transpose(0, 1).contiguous().view(self.seq_len, self.hidden_size)
        )
        projected = self._run_gemm("output_proj", attn_output)
        host_started = time.perf_counter()
        residual_hidden_states = projected + input_tensor
        hidden_states = torch.nn.functional.layer_norm(
            residual_hidden_states,
            (self.hidden_size,),
            self.ln1_weight,
            self._zero_bias(self.ln1_weight),
        ).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        intermediate = self._run_gemm("up_proj", hidden_states)
        host_started = time.perf_counter()
        intermediate = torch.nn.functional.gelu(intermediate).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        ffn_output = self._run_gemm("down_proj", intermediate)
        host_started = time.perf_counter()
        output = torch.nn.functional.layer_norm(
            ffn_output + hidden_states,
            (self.hidden_size,),
            self.ln2_weight,
            self._zero_bias(self.ln2_weight),
        ).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        return output

    def _run_decoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        host_started = time.perf_counter()
        attn_input = torch.nn.functional.layer_norm(
            input_tensor,
            (self.hidden_size,),
            self.ln1_weight,
            self._zero_bias(self.ln1_weight),
        ).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        q = self._run_gemm("q_proj", attn_input)
        k = self._run_gemm("k_proj", attn_input)
        v = self._run_gemm("v_proj", attn_input)

        q_heads = q.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k_heads = k.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v_heads = v.view(self.seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        attn_heads = self._blocked_attention(
            q_heads,
            k_heads,
            v_heads,
            causal=True,
        )
        attn_output = (
            attn_heads.transpose(0, 1).contiguous().view(self.seq_len, self.hidden_size)
        )
        projected = self._run_gemm("output_proj", attn_output)
        residual_hidden_states = projected + input_tensor
        host_started = time.perf_counter()
        ffn_input = torch.nn.functional.layer_norm(
            residual_hidden_states,
            (self.hidden_size,),
            self.ln2_weight,
            self._zero_bias(self.ln2_weight),
        ).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        intermediate = self._run_gemm("up_proj", ffn_input)
        host_started = time.perf_counter()
        intermediate = torch.nn.functional.gelu(intermediate).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        ffn_output = self._run_gemm("down_proj", intermediate)
        host_started = time.perf_counter()
        output = (ffn_output + residual_hidden_states).to(torch.bfloat16)
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["host_elementwise_time_sec"] = float(
                self._runtime_breakdown.get("host_elementwise_time_sec", 0.0)
            ) + (time.perf_counter() - host_started)
        return output

    def _run_transformer_layer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.workload_variant == "decoder_gpt2":
            return self._run_decoder(input_tensor)
        return self._run_encoder(input_tensor)

    def run_runlist(self):
        started = time.perf_counter()
        transfer_started = time.perf_counter()
        input_tensor = self.read_buffer_as_torch(
            "input",
            (self.seq_len, self.hidden_size),
            dtype=bfloat16,
        )
        self._record_transfer_read(started=transfer_started, tensor=input_tensor)
        output_tensor = self._run_transformer_layer(input_tensor)
        output_tensor = output_tensor.contiguous()
        transfer_started = time.perf_counter()
        self.write_buffer("output", output_tensor)
        self._record_transfer_write(started=transfer_started, tensor=output_tensor)
        elapsed_sec = time.perf_counter() - started
        if self._runtime_breakdown_enabled:
            self._runtime_breakdown["total_wall_time_sec"] = elapsed_sec
        return elapsed_sec

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._run_transformer_layer(input_tensor)
