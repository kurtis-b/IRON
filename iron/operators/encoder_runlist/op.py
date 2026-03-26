# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16
import math

from iron.common import AIEOperatorBase
from iron.operators.gemm.op import AIEGEMM
from iron.operators.softmax.op import AIESoftmax
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.gelu.op import AIEGELU
from iron.operators.transpose.op import AIETranspose
from iron.common.utils import torch_to_numpy


def _layer_norm_no_bias(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=weight,
        bias=None,
        eps=eps,
    )


class AIEEncoderRunlist(AIEOperatorBase):
    """
    AIE-accelerated post-projection encoder layer using a runlist-based implementation.

    The operator starts from supplied Q/K/V activations and a residual input:
    1. K transpose (for attention score calculations)
    2. Attention score calculations (GEMM per Q/K heads)
    3. Attention score scaling (Multiplication per attention score)
    4. Attention weight calculations (Softmax per attention score)
    5. Output head calculations (GEMM per attention weights/V heads)
    6. Output projection (GEMM with O weight)
    7. Residual connection (Eltwise add)
    8. Layer normalization
    9. Up projection (GEMM with Up projection weight)
    10. Activation function (GeLU)
    11. Down projection (GEMM with Down projection weight)
    12. Residual connection (Eltwise add)
    13. Layer normalization
    """

    component_order = (
        "k_transpose",
        "attn_scores",
        "attn_scale",
        "attn_softmax",
        "attn_output",
        "out_proj",
        "add1",
        "ln1",
        "ffn_up",
        "gelu",
        "ffn_down",
        "add2",
        "ln2",
    )

    @staticmethod
    def _resolve_k_transpose_num_channels(seq_len, tile_rows):
        max_channels = 2
        for num_channels in range(max_channels, 0, -1):
            if seq_len % (num_channels * tile_rows) == 0:
                return num_channels
        raise ValueError(
            f"K transpose requires seq_len to be divisible by tile_rows={tile_rows}; got seq_len={seq_len}"
        )

    @staticmethod
    def _resolve_mha_num_pipelines(seq_len, block_rows=64, max_pipelines=8):
        max_feasible = max(1, seq_len // block_rows)
        return min(max_pipelines, max_feasible)

    @staticmethod
    def _resolve_short_seq_tile(seq_len, scale_factor=8):
        for tile in (64, 32, 16, 8):
            if seq_len >= tile * scale_factor and seq_len % tile == 0:
                return tile
        raise ValueError(
            f"Unable to find a short-sequence tile for seq_len={seq_len} "
            f"with scale_factor={scale_factor}"
        )

    @classmethod
    def _resolve_short_seq_gemm_tile_m(cls, seq_len):
        return cls._resolve_short_seq_tile(seq_len, scale_factor=8)

    @classmethod
    def _resolve_seq_gemm_tile_m(cls, seq_len):
        if seq_len >= 16384:
            return 16
        return cls._resolve_short_seq_gemm_tile_m(seq_len)

    @staticmethod
    def _resolve_query_block_size(seq_len):
        if seq_len >= 16384:
            return 1024
        return seq_len

    def _runtime_rows(self):
        return self.query_block_size if self.uses_query_blocking else self.seq_len

    def _runtime_act_size(self):
        return self._runtime_rows() * self.hidden_size

    def _runtime_ffn_size(self):
        return self._runtime_rows() * self.intermediate_size

    def _runtime_attn_size(self):
        return self._runtime_rows() * self.seq_len * self.num_heads

    @classmethod
    def _resolve_attn_scores_tile_n(cls, seq_len, num_aie_columns):
        return cls._resolve_short_seq_tile(seq_len, scale_factor=num_aie_columns)

    def _resolve_embedding_tile(self):
        if self.hidden_size % self.num_aie_columns != 0:
            raise ValueError(
                "AIEEncoderRunlist requires hidden_size to be divisible by num_aie_columns; "
                f"got hidden_size={self.hidden_size}, num_aie_columns={self.num_aie_columns}"
            )
        return self.hidden_size // self.num_aie_columns

    def _resolve_projection_tile_n(self):
        embedding_tile = self._resolve_embedding_tile()
        if embedding_tile % 2 != 0:
            raise ValueError(
                "AIEEncoderRunlist requires the per-column embedding tile to be even; "
                f"got embedding_tile={embedding_tile}"
            )
        return embedding_tile // 2

    def _resolve_ffn_tile(self):
        if self.intermediate_size % (self.num_aie_columns * 4) != 0:
            raise ValueError(
                "AIEEncoderRunlist requires intermediate_size to be divisible by "
                f"num_aie_columns*4; got intermediate_size={self.intermediate_size}, "
                f"num_aie_columns={self.num_aie_columns}"
            )
        return self.intermediate_size // (self.num_aie_columns * 4)

    def _resolve_projection_tiling(self):
        embedding_tile = self._resolve_embedding_tile()
        projection_tile_n = self._resolve_projection_tile_n()
        ffn_tile = self._resolve_ffn_tile()

        if self.hidden_size == 768 and self.intermediate_size == 3072:
            return {
                "transpose_n": embedding_tile,
                "out_proj_tile_k": embedding_tile,
                "out_proj_tile_n": projection_tile_n,
                "up_proj_tile_k": projection_tile_n,
                "up_proj_tile_n": ffn_tile,
                "down_proj_tile_k": ffn_tile,
                "down_proj_tile_n": projection_tile_n,
            }

        if self.hidden_size == 1024 and self.intermediate_size == 4096:
            return {
                "transpose_n": embedding_tile,
                "out_proj_tile_k": 64,
                "out_proj_tile_n": 64,
                "up_proj_tile_k": 64,
                "up_proj_tile_n": 64,
                "down_proj_tile_k": 64,
                "down_proj_tile_n": 64,
            }

        raise ValueError(
            "AIEEncoderRunlist currently supports only the 768/3072 and "
            "1024/4096 encoder families; got "
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
        )

    def _resolve_k_transpose_layout(self):
        projection_tiling = self._resolve_projection_tiling()
        tile_rows = 64
        tile_cols = projection_tiling["transpose_n"]
        num_aie_columns = self.num_aie_columns

        if self.seq_len >= 16384:
            tile_rows = 128
            tile_cols = 64
            num_aie_columns = max(
                candidate
                for candidate in range(self.num_aie_columns, 0, -1)
                if self.hidden_size % (candidate * tile_cols) == 0
            )

        num_channels = self._resolve_k_transpose_num_channels(self.seq_len, tile_rows)

        return {
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "tile_shuffle": 8,
            "num_aie_columns": num_aie_columns,
            "num_channels": num_channels,
        }

    @staticmethod
    def _normalize_qkv_tensor(
        tensor: torch.Tensor,
        *,
        name: str,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        if tensor.dim() == 2:
            expected_shape = (seq_len, hidden_size)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"AIEEncoderRunlist.forward expects {name} to have shape "
                    f"{expected_shape}; got shape={tuple(tensor.shape)}"
                )
            return tensor.contiguous()
        if tensor.dim() == 3:
            expected_shape = (num_heads, seq_len, head_dim)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"AIEEncoderRunlist.forward expects {name} to have shape "
                    f"{expected_shape} or ({seq_len}, {hidden_size}); got shape={tuple(tensor.shape)}"
                )
            return tensor.permute(1, 0, 2).reshape(seq_len, hidden_size).contiguous()
        raise ValueError(
            f"AIEEncoderRunlist.forward expects {name} to be a 2D or 3D tensor; "
            f"got shape={tuple(tensor.shape)}"
        )

    @staticmethod
    def _pack_k_for_transpose(
        k_matrix: torch.Tensor,
        *,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
    ) -> torch.Tensor:
        return (
            k_matrix.view(seq_len, num_heads, head_dim)
            .permute(0, 2, 1)
            .reshape(seq_len, hidden_size)
            .contiguous()
        )

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=8,
        use_pip_ffn=False,
        use_pip_addnorm=False,
        use_pip_mha=False,
        use_pip_an_ffn=False,
        ln1_weight=None,
        ln2_weight=None,
        use_static_runtime_weights=True,
        context=None,
        skip_add_to_list=False,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_aie_columns = num_aie_columns

        # Derived dimensions
        self.head_dim = hidden_size // num_heads
        self.query_block_size = self._resolve_query_block_size(self.seq_len)
        self.uses_query_blocking = self.query_block_size < self.seq_len

        # Weights to be set by user
        self.attn_output_weight = None
        self.ln1_weight = ln1_weight
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln2_weight = ln2_weight
        self.use_pip_ffn = False
        self.use_pip_addnorm = False
        self.use_pip_mha = False
        self.use_pip_an_ffn = False
        self.use_static_runtime_weights = use_static_runtime_weights
        # The stitched encoder backend is stable when its stage handles are
        # loaded during runtime preparation and then reused. Lazy per-dispatch
        # loads are prone to XRT host-runtime crashes.
        self.lazy_kernel_loading = False

        # Artifacts created by set_up_artifacts() - one per layer
        self.combined_xclbin = None
        self.component_artifacts = {}
        # Output projection
        self.out_proj_xclbin = None
        self.out_proj_insts = None
        # Attention
        if self.use_pip_mha:
            self.mha_num_pipelines = self._resolve_mha_num_pipelines(self.seq_len)
            self.mha_xclbin = None
            self.mha_insts = None
        else:
            self.k_transpose_layout = self._resolve_k_transpose_layout()
            self.k_transpose_num_channels = self.k_transpose_layout["num_channels"]
            self.k_transpose_xclbin = None
            self.k_transpose_insts = None
            self.attn_scores_xclbin = None
            self.attn_scores_insts = None
            self.attn_scale_xclbin = None
            self.attn_scale_insts = None
            self.attn_softmax_xclbin = None
            self.attn_softmax_insts = None
            self.attn_output_xclbin = None
            self.attn_output_runtime_xclbin = None
            self.attn_output_insts = None
        if self.use_pip_an_ffn:
            self.anffn_xclbin = None
            self.anffn_insts = None
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm
                self.add_norm1_xclbin = None
                self.add_norm1_insts = None
                self.add_norm2_xclbin = None
                self.add_norm2_insts = None
            else:
                # Residual connection
                self.add_xclbin = None
                self.add_insts = None
                # Layer normalization
                self.ln1_xclbin = None
                self.ln1_insts = None
                self.ln2_xclbin = None
                self.ln2_insts = None
            if self.use_pip_ffn:
                # Pipelined FFN
                self.ffn_xclbin = None
                self.ffn_insts = None
            else:
                # Up projection
                self.up_proj_xclbin = None
                self.up_proj_insts = None
                # GeLU activation
                self.gelu_xclbin = None
                self.gelu_insts = None
                # Down projection
                self.down_proj_xclbin = None
                self.down_proj_insts = None

        AIEOperatorBase.__init__(
            self,
            context=context,
            skip_add_to_list=skip_add_to_list,
        )

    def _runtime_weight_static_data(self, tensor, *, transpose=False):
        if not self.use_static_runtime_weights or tensor is None:
            return None
        weight_tensor = tensor.T if transpose else tensor
        return torch_to_numpy(weight_tensor)

    def write_runtime_weights(self):
        self.write_buffer(
            "out_proj_weight",
            torch_to_numpy(self.attn_output_weight.T.contiguous()),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.write_buffer("ln1_weight", torch_to_numpy(self.ln1_weight))
        self.write_buffer(
            "ffn_up_weight",
            torch_to_numpy(self.ffn_up_weight.T.contiguous()),
        )
        self.write_buffer(
            "ffn_down_weight",
            torch_to_numpy(self.ffn_down_weight.T.contiguous()),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.write_buffer("ln2_weight", torch_to_numpy(self.ln2_weight))

    def assign_weights(
        self,
        *,
        w_o: torch.Tensor,
        b_up: torch.Tensor,
        b_down: torch.Tensor,
        ln1_weight: torch.Tensor | None = None,
        ln2_weight: torch.Tensor | None = None,
    ) -> None:
        self.attn_output_weight = w_o.contiguous()
        self.ffn_up_weight = b_up.contiguous()
        self.ffn_down_weight = b_down.contiguous()
        if ln1_weight is not None:
            self.ln1_weight = ln1_weight.contiguous()
        if ln2_weight is not None:
            self.ln2_weight = ln2_weight.contiguous()

    @classmethod
    def _normalize_component_names(
        cls, names: tuple[str, ...] | list[str] | None
    ) -> tuple[str, ...]:
        normalized = cls.component_order if names is None else tuple(names)
        if not normalized:
            raise ValueError("AIEEncoderRunlist component selection cannot be empty")
        unknown = [name for name in normalized if name not in cls.component_order]
        if unknown:
            raise ValueError(
                "Unknown encoder_runlist component(s): " + ", ".join(sorted(unknown))
            )
        deduped = []
        seen = set()
        for name in normalized:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return tuple(deduped)

    def prepare_component_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q_matrix = self._normalize_qkv_tensor(
            q,
            name="Q",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k_matrix = self._normalize_qkv_tensor(
            k,
            name="K",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        v_matrix = self._normalize_qkv_tensor(
            v,
            name="V",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        if r.dim() != 2 or tuple(r.shape) != (self.seq_len, self.hidden_size):
            raise ValueError(
                f"AIEEncoderRunlist expects R to have shape "
                f"({self.seq_len}, {self.hidden_size}); got shape={tuple(r.shape)}"
            )
        q_heads = q_matrix.view(self.seq_len, self.num_heads, self.head_dim)
        k_heads = k_matrix.view(self.seq_len, self.num_heads, self.head_dim)
        v_heads = v_matrix.view(self.seq_len, self.num_heads, self.head_dim)
        k_input_matrix = self._pack_k_for_transpose(
            k_matrix,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )
        k_transposed = (
            k_input_matrix.view(self.seq_len, self.head_dim, self.num_heads)
            .permute(1, 2, 0)
            .contiguous()
            .view(self.hidden_size, self.seq_len)
        )
        return {
            "q_matrix": q_matrix,
            "k_matrix": k_matrix,
            "k_input_matrix": k_input_matrix,
            "v_matrix": v_matrix,
            "k_transposed": k_transposed,
            "query_heads": q_heads,
            "key_heads": k_heads,
            "value_heads": v_heads,
            "residual": r.contiguous(),
        }

    def expected_component_outputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        *,
        eps: float,
        names: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        requested_names = self._normalize_component_names(names)
        return {
            name: self.validation_component_case(
                name,
                q,
                k,
                v,
                r,
                eps=eps,
            )["expected"]
            for name in requested_names
        }

    def validation_component_case(
        self,
        name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        *,
        eps: float,
    ) -> dict[str, object]:
        component_name = self._normalize_component_names((name,))[0]
        if (
            self.attn_output_weight is None
            or self.ffn_up_weight is None
            or self.ffn_down_weight is None
        ):
            raise RuntimeError(
                "AIEEncoderRunlist requires out_proj, ffn_up, and ffn_down weights before component validation is computed."
            )

        prepared = self.prepare_component_inputs(q, k, v, r)
        runtime_rows = self._runtime_rows()
        q_heads = prepared["query_heads"][:runtime_rows]
        k_heads = prepared["key_heads"]
        v_heads = prepared["value_heads"]
        residual = prepared["residual"][:runtime_rows]

        if component_name == "k_transpose":
            return {
                "args": (prepared["k_input_matrix"],),
                "expected": prepared["k_transposed"],
            }

        attn_scores = torch.einsum("shd,thd->hst", q_heads, k_heads).contiguous()
        if component_name == "attn_scores":
            return {
                "args": (
                    q_heads,
                    k_heads.permute(2, 1, 0).contiguous(),
                ),
                "expected": attn_scores,
            }

        attn_scale = attn_scores * (self.head_dim**-0.5)
        attn_scale_matrix = attn_scale.view(self.num_heads * runtime_rows, self.seq_len)
        if component_name == "attn_scale":
            return {
                "args": (attn_scores.unsqueeze(0),),
                "expected": attn_scale.unsqueeze(0),
            }

        attn_softmax = torch.softmax(
            attn_scale.to(torch.float32),
            dim=-1,
        ).to(torch.bfloat16)
        attn_softmax_matrix = attn_softmax.view(
            self.num_heads * runtime_rows, self.seq_len
        )
        if component_name == "attn_softmax":
            return {
                "args": (attn_scale_matrix,),
                "expected": attn_softmax_matrix,
            }

        attn_output = torch.einsum("hst,thd->shd", attn_softmax, v_heads).contiguous()
        if component_name == "attn_output":
            return {
                "args": (
                    attn_softmax,
                    v_heads,
                ),
                "expected": attn_output,
            }

        attn_output_matrix = attn_output.view(runtime_rows, self.hidden_size)
        out_proj = F.linear(attn_output_matrix, self.attn_output_weight)
        if component_name == "out_proj":
            return {
                "args": (attn_output_matrix,),
                "expected": out_proj,
            }

        add1 = out_proj + residual
        if component_name == "add1":
            return {
                "args": (
                    out_proj.view(1, -1),
                    residual.view(1, -1),
                ),
                "expected": add1.view(1, -1),
            }

        ln1 = _layer_norm_no_bias(add1, self.ln1_weight, eps)
        if component_name == "ln1":
            return {
                "args": (add1,),
                "expected": ln1,
            }

        ffn_up = F.linear(ln1, self.ffn_up_weight)
        if component_name == "ffn_up":
            return {
                "args": (ln1,),
                "expected": ffn_up,
            }

        gelu = F.gelu(ffn_up)
        if component_name == "gelu":
            return {
                "args": (ffn_up,),
                "expected": gelu,
            }

        ffn_down = F.linear(gelu, self.ffn_down_weight)
        if component_name == "ffn_down":
            return {
                "args": (gelu,),
                "expected": ffn_down,
            }

        add2 = ffn_down + ln1
        if component_name == "add2":
            return {
                "args": (
                    ffn_down.view(1, -1),
                    ln1.view(1, -1),
                ),
                "expected": add2.view(1, -1),
            }

        ln2 = _layer_norm_no_bias(add2, self.ln2_weight, eps)
        return {
            "args": (add2,),
            "expected": ln2,
        }

    def make_validation_component_operators(
        self,
        *,
        context,
        names: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, object]:
        requested_names = self._normalize_component_names(names)
        runtime_rows = self._runtime_rows()
        runtime_act_size = self._runtime_act_size()
        runtime_ffn_size = self._runtime_ffn_size()
        runtime_attn_size = self._runtime_attn_size()
        seq_gemm_tile_m = self._resolve_seq_gemm_tile_m(self.seq_len)
        attn_scores_tile_n = self._resolve_attn_scores_tile_n(
            self.seq_len, self.num_aie_columns
        )
        projection_tiling = self._resolve_projection_tiling()
        eltwise_mul_tile_size = runtime_attn_size // (self.num_aie_columns * 2)
        eltwise_add_tile_size = runtime_act_size // (self.num_aie_columns * 2)
        gelu_tile_size = runtime_ffn_size // (self.num_aie_columns * 2)

        operators: dict[str, object] = {}
        for name in requested_names:
            if name == "k_transpose":
                operators[name] = AIETranspose(
                    M=self.seq_len,
                    N=self.hidden_size,
                    num_aie_columns=self.k_transpose_layout["num_aie_columns"],
                    num_channels=self.k_transpose_layout["num_channels"],
                    m=self.k_transpose_layout["tile_rows"],
                    n=self.k_transpose_layout["tile_cols"],
                    s=self.k_transpose_layout["tile_shuffle"],
                    context=context,
                )
            elif name == "attn_scores":
                operators[name] = AIEGEMM(
                    M=runtime_rows,
                    K=self.head_dim,
                    N=self.seq_len,
                    tile_m=seq_gemm_tile_m,
                    tile_k=64,
                    tile_n=attn_scores_tile_n,
                    num_aie_columns=self.num_aie_columns,
                    batch_A=(self.num_heads, 1),
                    batch_B=(self.num_heads, 1),
                    batch_C=(self.num_heads, 0),
                    prio_accuracy=False,
                    emulate_bf16_mmul_with_bfp16=True,
                    context=context,
                )
            elif name == "attn_scale":
                operators[name] = AIEElementwiseMul(
                    size=runtime_attn_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_mul_tile_size),
                        eltwise_mul_tile_size,
                    ),
                    scalar_broadcast=math.sqrt(1.0 / self.head_dim),
                    context=context,
                )
            elif name == "attn_softmax":
                operators[name] = AIESoftmax(
                    rows=runtime_rows * self.num_heads,
                    cols=self.seq_len,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    context=context,
                )
            elif name == "attn_output":
                operators[name] = AIEGEMM(
                    M=runtime_rows,
                    K=self.seq_len,
                    N=self.head_dim,
                    tile_m=seq_gemm_tile_m,
                    tile_k=64,
                    tile_n=8,
                    num_aie_columns=8,
                    batch_A=(self.num_heads, 0),
                    batch_B=(self.num_heads, 1),
                    batch_C=(self.num_heads, 1),
                    prio_accuracy=False,
                    emulate_bf16_mmul_with_bfp16=True,
                    context=context,
                )
            elif name == "out_proj":
                operators[name] = AIEGEMM(
                    M=runtime_rows,
                    K=self.hidden_size,
                    N=self.hidden_size,
                    use_static_weight=True,
                    tile_m=seq_gemm_tile_m,
                    tile_k=projection_tiling["out_proj_tile_k"],
                    tile_n=projection_tiling["out_proj_tile_n"],
                    num_aie_columns=self.num_aie_columns,
                    prio_accuracy=False,
                    emulate_bf16_mmul_with_bfp16=True,
                    context=context,
                )
                operators[name].weight = self.attn_output_weight.contiguous()
            elif name in {"add1", "add2"}:
                operators[name] = AIEElementwiseAdd(
                    size=runtime_act_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_add_tile_size),
                        eltwise_add_tile_size,
                    ),
                    context=context,
                )
            elif name == "ln1":
                operators[name] = AIELayerNorm(
                    size=runtime_act_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln1_weight,
                    context=context,
                )
            elif name == "ffn_up":
                operators[name] = AIEGEMM(
                    M=runtime_rows,
                    K=self.hidden_size,
                    N=self.intermediate_size,
                    use_static_weight=True,
                    tile_m=seq_gemm_tile_m,
                    tile_k=projection_tiling["up_proj_tile_k"],
                    tile_n=projection_tiling["up_proj_tile_n"],
                    num_aie_columns=self.num_aie_columns,
                    prio_accuracy=False,
                    emulate_bf16_mmul_with_bfp16=True,
                    context=context,
                )
                operators[name].weight = self.ffn_up_weight.contiguous()
            elif name == "gelu":
                operators[name] = AIEGELU(
                    size=runtime_ffn_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
                    context=context,
                )
            elif name == "ffn_down":
                operators[name] = AIEGEMM(
                    M=runtime_rows,
                    K=self.intermediate_size,
                    N=self.hidden_size,
                    use_static_weight=True,
                    tile_m=seq_gemm_tile_m,
                    tile_k=projection_tiling["down_proj_tile_k"],
                    tile_n=projection_tiling["down_proj_tile_n"],
                    num_aie_columns=self.num_aie_columns,
                    prio_accuracy=False,
                    emulate_bf16_mmul_with_bfp16=True,
                    context=context,
                )
                operators[name].weight = self.ffn_down_weight.contiguous()
            elif name == "ln2":
                operators[name] = AIELayerNorm(
                    size=runtime_act_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln2_weight,
                    context=context,
                )
        return operators

    def set_up_artifacts(self):
        """Set up artifacts for the encoder runlist using 13 individual stages."""
        self.component_artifacts = {}
        artifacts = []

        kernel_id = 0x801

        eltwise_mul_tile_size = (self.seq_len * self.seq_len * self.num_heads) // (
            self.num_aie_columns * 2
        )
        if not self.use_pip_addnorm or not self.use_pip_an_ffn:
            eltwise_add_tile_size = (self.seq_len * self.hidden_size) // (
                self.num_aie_columns * 2
            )
        if not self.use_pip_ffn or not self.use_pip_an_ffn:
            gelu_tile_size = (self.seq_len * self.intermediate_size) // (
                self.num_aie_columns * 2
            )

        prefix_base = "encoder_runlist_qkvr_"
        seq_gemm_tile_m = self._resolve_seq_gemm_tile_m(self.seq_len)
        attn_scores_tile_n = self._resolve_attn_scores_tile_n(
            self.seq_len, self.num_aie_columns
        )
        projection_tiling = self._resolve_projection_tiling()
        runtime_rows = self._runtime_rows()
        runtime_act_size = self._runtime_act_size()
        runtime_ffn_size = self._runtime_ffn_size()
        # Output projection kernel
        out_proj = AIEGEMM(  # L1 utilization = 54 KB with double buffering
            M=runtime_rows,
            K=self.hidden_size,
            N=self.hidden_size,
            tile_m=seq_gemm_tile_m,
            tile_k=projection_tiling["out_proj_tile_k"],
            tile_n=projection_tiling["out_proj_tile_n"],
            num_aie_columns=self.num_aie_columns,
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
            skip_add_to_list=True,
        )
        self.out_proj_xclbin, self.out_proj_insts = out_proj.get_artifacts(
            prefix=f"{prefix_base}out_proj_"
        )
        self.out_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_out_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.out_proj_xclbin.kernel_name = "encoder_out_proj"
        artifacts.append(self.out_proj_insts)
        kernel_id += 1

        if self.use_pip_mha:
            self.mha_xclbin, self.mha_insts = AIEMHA(
                num_heads=self.num_heads,
                seq_len=self.seq_len,
                d=self.head_dim,
                num_KV_heads=self.num_heads,
                num_of_pipelines=self.mha_num_pipelines,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}mha_")
            self.mha_xclbin.xclbin_input = None
            self.mha_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_mha",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.mha_xclbin.kernel_name = "encoder_mha"
            artifacts.append(self.mha_insts)
            kernel_id += 1
            next_dep = self.mha_xclbin
        else:
            # K Transpose kernel (transpose K matrix)
            k_transpose = AIETranspose(
                M=self.seq_len,
                N=self.hidden_size,
                num_aie_columns=self.k_transpose_layout["num_aie_columns"],
                num_channels=self.k_transpose_layout["num_channels"],
                m=self.k_transpose_layout["tile_rows"],
                n=self.k_transpose_layout["tile_cols"],
                s=self.k_transpose_layout["tile_shuffle"],
                skip_add_to_list=True,
            )
            self.k_transpose_xclbin, self.k_transpose_insts = k_transpose.get_artifacts(
                prefix=f"{prefix_base}k_transpose_"
            )
            self.k_transpose_xclbin.xclbin_input = None
            self.k_transpose_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_k_transpose",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.k_transpose_xclbin.kernel_name = "encoder_k_transpose"
            artifacts.append(self.k_transpose_insts)
            kernel_id += 1

            # Attention score calculations (GEMM for Q*K^T, batched across heads)
            self.attn_scores_xclbin, self.attn_scores_insts = AIEGEMM(
                M=runtime_rows,
                K=self.head_dim,
                N=self.seq_len,
                tile_m=seq_gemm_tile_m,
                tile_k=64,
                tile_n=attn_scores_tile_n,
                num_aie_columns=self.num_aie_columns,
                batch_A=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_C=(self.num_heads, 0),  # Batch across heads, batch dim first
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_scores_")
            self.attn_scores_xclbin.xclbin_input = self.k_transpose_xclbin
            self.attn_scores_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_scores",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_scores_xclbin.kernel_name = "encoder_attn_scores"
            self.attn_scores_xclbin.depends += [
                self.k_transpose_xclbin,
            ]
            artifacts.append(self.attn_scores_insts)
            kernel_id += 1

            # Attention score scaling (Multiplication per attention score)
            self.attn_scale_xclbin, self.attn_scale_insts = AIEElementwiseMul(
                size=runtime_rows * self.seq_len * self.num_heads,
                num_aie_columns=self.num_aie_columns,
                num_channels=2,
                tile_size=min(
                    math.gcd(4096, eltwise_mul_tile_size), eltwise_mul_tile_size
                ),
                scalar_broadcast=math.sqrt(1.0 / self.head_dim),
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_scale_")
            self.attn_scale_xclbin.xclbin_input = self.attn_scores_xclbin
            self.attn_scale_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_scale",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_scale_xclbin.kernel_name = "encoder_attn_scale"
            self.attn_scale_xclbin.depends += [self.attn_scores_xclbin]
            artifacts.append(self.attn_scale_insts)
            kernel_id += 1

            # Attention weight calculations (Softmax per attention score)
            self.attn_softmax_xclbin, self.attn_softmax_insts = AIESoftmax(
                rows=runtime_rows * self.num_heads,
                cols=self.seq_len,
                num_aie_columns=self.num_aie_columns,
                num_channels=2,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_softmax_")
            self.attn_softmax_xclbin.xclbin_input = self.attn_scale_xclbin
            self.attn_softmax_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_softmax",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_softmax_xclbin.kernel_name = "encoder_attn_softmax"
            self.attn_softmax_xclbin.depends += [self.attn_scale_xclbin]
            artifacts.append(self.attn_softmax_insts)
            kernel_id += 1

            # Output head calculations (GEMM per attention weights/V heads, batched across heads)
            attn_output_gemm = AIEGEMM(
                M=runtime_rows,
                K=self.seq_len,
                N=self.head_dim,
                tile_m=seq_gemm_tile_m,
                tile_k=64,
                tile_n=8,
                num_aie_columns=8,
                batch_A=(self.num_heads, 0),  # Batch across heads, batch dim first
                batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_C=(self.num_heads, 1),  # Batch across heads, batch dim first
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
                skip_add_to_list=True,
            )
            self.attn_output_xclbin, self.attn_output_insts = (
                attn_output_gemm.get_artifacts(prefix=f"{prefix_base}attn_output_")
            )
            self.attn_output_runtime_xclbin = (
                attn_output_gemm.get_runtime_xclbin_artifact(
                    prefix=f"{prefix_base}attn_output_runtime_"
                )
            )
            self.attn_output_xclbin.xclbin_input = self.attn_softmax_xclbin
            self.attn_output_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_output",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_output_xclbin.kernel_name = "encoder_attn_output"
            self.attn_output_xclbin.depends += [
                self.attn_softmax_xclbin,
            ]
            self.attn_output_runtime_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_output",
            ]
            self.attn_output_runtime_xclbin.kernel_name = "encoder_attn_output"
            artifacts.append(self.attn_output_runtime_xclbin)
            artifacts.append(self.attn_output_insts)
            kernel_id += 1
            next_dep = self.attn_output_xclbin

        self.out_proj_xclbin.xclbin_input = next_dep
        self.out_proj_xclbin.depends += [next_dep]
        next_dep = self.out_proj_xclbin

        if self.use_pip_an_ffn:
            aie_anffn_config = {
                "emulate_bf16_mmul_with_bfp16": True,
                "nA_tiles_distributed": 4,
                "nB_tiles_distributed": 2,
                "stage_only": None,
                "gelu_stage": 1,
            }
            # Second Layer normalization kernel
            self.anffn_xclbin, self.anffn_insts = AIEFFNAN(
                M=runtime_rows,
                K=self.hidden_size,
                N=self.intermediate_size,
                tile_m=16,
                tile_k=128,
                tile_n=96,
                down_proj_depth=6,
                num_aie_columns=self.num_aie_columns,
                ln1_weight=self.ln1_weight,
                ln2_weight=self.ln2_weight,
                debug_mode=-1,
                **aie_anffn_config,
            ).get_artifacts(prefix=f"{prefix_base}anffn_")
            self.anffn_xclbin.xclbin_input = next_dep
            self.anffn_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_anffn",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.anffn_xclbin.kernel_name = "encoder_anffn"
            self.anffn_xclbin.depends += [next_dep]
            artifacts.append(self.anffn_xclbin)
            artifacts.append(self.anffn_insts)
            # Store final xclbin
            self.combined_xclbin = self.anffn_xclbin
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm kernel
                self.add_norm1_xclbin, self.add_norm1_insts = AIEAddAndNorm(
                    size=runtime_act_size,
                    num_aie_columns=self.num_aie_columns,
                    tile_size=self.hidden_size,
                    weights=self.ln1_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_norm1_")
                self.add_norm1_xclbin.xclbin_input = next_dep
                self.add_norm1_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add_norm1",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_norm1_xclbin.kernel_name = "encoder_add_norm1"
                self.add_norm1_xclbin.depends += [next_dep]
                artifacts.append(self.add_norm1_insts)
                next_dep = self.add_norm1_xclbin
                kernel_id += 1
            else:
                # Residual connection kernel (Eltwise add)
                self.add_xclbin, self.add_insts = AIEElementwiseAdd(
                    size=runtime_act_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size
                    ),
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_")
                self.add_xclbin.xclbin_input = next_dep
                self.add_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_xclbin.kernel_name = "encoder_add"
                self.add_xclbin.depends += [next_dep]
                artifacts.append(self.add_xclbin)
                artifacts.append(self.add_insts)
                kernel_id += 1

                # Layer normalization kernel
                self.ln1_xclbin, self.ln1_insts = AIELayerNorm(
                    size=runtime_act_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln1_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ln1_")
                self.ln1_xclbin.xclbin_input = None
                self.ln1_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ln1",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ln1_xclbin.kernel_name = "encoder_ln1"
                self.ln1_xclbin.depends += [next_dep]
                artifacts.append(self.ln1_insts)
                next_dep = self.ln1_xclbin
                kernel_id += 1
            if self.use_pip_ffn:
                aie_ffn_config = {
                    "b_col_maj": False,
                    "c_col_maj": False,
                    "emulate_bf16_mmul_with_bfp16": True,
                    "n_a_tiles_distributed": 4,
                    "n_b_tiles_distributed": 4,
                    "stage_only": None,
                    "gelu_stage": 1,
                }
                self.ffn_xclbin, self.ffn_insts = AIEFFN(
                    M=runtime_rows,
                    K=self.hidden_size,
                    N=self.intermediate_size,
                    tile_m=64,
                    tile_k=48,
                    tile_n=96,
                    down_proj_depth=8,
                    num_aie_columns=self.num_aie_columns,
                    **aie_ffn_config,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ffn_")
                self.ffn_xclbin.xclbin_input = next_dep
                self.ffn_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ffn",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ffn_xclbin.kernel_name = "encoder_ffn"
                self.ffn_xclbin.depends += [next_dep]
                artifacts.append(self.ffn_insts)
                next_dep = self.ffn_xclbin
                kernel_id += 1
            else:
                # Up projection (GEMM with Up projection weight)
                self.up_proj_xclbin, self.up_proj_insts = (
                    AIEGEMM(  # L1 utilization = 54 KB with double buffering
                        M=runtime_rows,
                        K=self.hidden_size,
                        N=self.intermediate_size,
                        tile_m=seq_gemm_tile_m,
                        tile_k=projection_tiling["up_proj_tile_k"],
                        tile_n=projection_tiling["up_proj_tile_n"],
                        num_aie_columns=self.num_aie_columns,
                        prio_accuracy=False,
                        emulate_bf16_mmul_with_bfp16=True,
                        skip_add_to_list=True,
                    ).get_artifacts(prefix=f"{prefix_base}up_proj_")
                )
                self.up_proj_xclbin.xclbin_input = None
                self.up_proj_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_up_proj",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.up_proj_xclbin.kernel_name = "encoder_up_proj"
                self.up_proj_xclbin.depends += [next_dep]
                artifacts.append(self.up_proj_insts)
                kernel_id += 1

                # Activation function (GeLU)
                self.gelu_xclbin, self.gelu_insts = AIEGELU(
                    size=runtime_ffn_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}gelu_")
                self.gelu_xclbin.xclbin_input = self.up_proj_xclbin
                self.gelu_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_gelu",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.gelu_xclbin.kernel_name = "encoder_gelu"
                self.gelu_xclbin.depends += [self.up_proj_xclbin]
                artifacts.append(self.gelu_insts)
                kernel_id += 1

                # Down projection (GEMM with Down projection weight)
                self.down_proj_xclbin, self.down_proj_insts = (
                    AIEGEMM(  # L1 utilization = 54 KB with double buffering
                        M=runtime_rows,
                        K=self.intermediate_size,
                        N=self.hidden_size,
                        tile_m=seq_gemm_tile_m,
                        tile_k=projection_tiling["down_proj_tile_k"],
                        tile_n=projection_tiling["down_proj_tile_n"],
                        num_aie_columns=self.num_aie_columns,
                        prio_accuracy=False,
                        emulate_bf16_mmul_with_bfp16=True,
                        skip_add_to_list=True,
                    ).get_artifacts(prefix=f"{prefix_base}down_proj_")
                )
                self.down_proj_xclbin.xclbin_input = None
                self.down_proj_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_down_proj",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.down_proj_xclbin.kernel_name = "encoder_down_proj"
                self.down_proj_xclbin.depends += [self.gelu_xclbin]
                artifacts.append(self.down_proj_insts)
                next_dep = self.down_proj_xclbin
                kernel_id += 1
            if self.use_pip_addnorm:
                # Second Pipelined add & norm kernel
                self.add_norm2_xclbin, self.add_norm2_insts = AIEAddAndNorm(
                    size=runtime_act_size,
                    num_aie_columns=self.num_aie_columns,
                    tile_size=self.hidden_size,
                    weights=self.ln2_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_norm2_")
                self.add_norm2_xclbin.xclbin_input = next_dep
                self.add_norm2_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add_norm2",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_norm2_xclbin.kernel_name = "encoder_add_norm2"
                self.add_norm2_xclbin.depends += [
                    next_dep,
                ]
                artifacts.append(self.add_norm2_xclbin)
                artifacts.append(self.add_norm2_insts)
                # Store final xclbin
                self.combined_xclbin = self.add_norm2_xclbin
            else:
                # Second Layer normalization kernel
                self.ln2_xclbin, self.ln2_insts = AIELayerNorm(
                    size=runtime_act_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln2_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ln2_")
                self.ln2_xclbin.xclbin_input = None
                self.ln2_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ln2",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ln2_xclbin.kernel_name = "encoder_ln2"
                self.ln2_xclbin.depends += [next_dep]
                artifacts.append(self.ln2_xclbin)
                artifacts.append(self.ln2_insts)
                # Store final xclbin
                self.combined_xclbin = self.ln2_xclbin

        stage_kernel_pairs = [
            (self.out_proj_insts, self.out_proj_xclbin.kernel_name),
            (
                self.mha_insts if self.use_pip_mha else None,
                self.mha_xclbin.kernel_name if self.use_pip_mha else None,
            ),
            (
                None if self.use_pip_mha else self.k_transpose_insts,
                None if self.use_pip_mha else self.k_transpose_xclbin.kernel_name,
            ),
            (
                None if self.use_pip_mha else self.attn_scores_insts,
                None if self.use_pip_mha else self.attn_scores_xclbin.kernel_name,
            ),
            (
                None if self.use_pip_mha else self.attn_scale_insts,
                None if self.use_pip_mha else self.attn_scale_xclbin.kernel_name,
            ),
            (
                None if self.use_pip_mha else self.attn_softmax_insts,
                None if self.use_pip_mha else self.attn_softmax_xclbin.kernel_name,
            ),
            (
                None if self.use_pip_mha else self.attn_output_insts,
                None if self.use_pip_mha else self.attn_output_xclbin.kernel_name,
            ),
            (
                self.anffn_insts if self.use_pip_an_ffn else None,
                self.anffn_xclbin.kernel_name if self.use_pip_an_ffn else None,
            ),
            (
                (
                    None
                    if self.use_pip_an_ffn or not self.use_pip_addnorm
                    else self.add_norm1_insts
                ),
                (
                    None
                    if self.use_pip_an_ffn or not self.use_pip_addnorm
                    else self.add_norm1_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or self.use_pip_addnorm else self.ln1_insts,
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_addnorm
                    else self.ln1_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or self.use_pip_addnorm else self.add_insts,
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_addnorm
                    else self.add_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or not self.use_pip_ffn else self.ffn_insts,
                (
                    None
                    if self.use_pip_an_ffn or not self.use_pip_ffn
                    else self.ffn_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or self.use_pip_ffn else self.up_proj_insts,
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_ffn
                    else self.up_proj_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or self.use_pip_ffn else self.gelu_insts,
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_ffn
                    else self.gelu_xclbin.kernel_name
                ),
            ),
            (
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_ffn
                    else self.down_proj_insts
                ),
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_ffn
                    else self.down_proj_xclbin.kernel_name
                ),
            ),
            (
                (
                    None
                    if self.use_pip_an_ffn or not self.use_pip_addnorm
                    else self.add_norm2_insts
                ),
                (
                    None
                    if self.use_pip_an_ffn or not self.use_pip_addnorm
                    else self.add_norm2_xclbin.kernel_name
                ),
            ),
            (
                None if self.use_pip_an_ffn or self.use_pip_addnorm else self.ln2_insts,
                (
                    None
                    if self.use_pip_an_ffn or self.use_pip_addnorm
                    else self.ln2_xclbin.kernel_name
                ),
            ),
        ]
        for insts_artifact, kernel_name in stage_kernel_pairs:
            if insts_artifact is not None:
                insts_artifact.kernel_name = kernel_name

        if not self.use_pip_mha and not self.use_pip_an_ffn:
            self.component_artifacts = {
                "k_transpose": (self.k_transpose_xclbin, self.k_transpose_insts),
                "attn_scores": (self.attn_scores_xclbin, self.attn_scores_insts),
                "attn_scale": (self.attn_scale_xclbin, self.attn_scale_insts),
                "attn_softmax": (self.attn_softmax_xclbin, self.attn_softmax_insts),
                "attn_output": (self.attn_output_xclbin, self.attn_output_insts),
                "out_proj": (self.out_proj_xclbin, self.out_proj_insts),
                "add1": (self.add_xclbin, self.add_insts),
                "ln1": (self.ln1_xclbin, self.ln1_insts),
                "ffn_up": (self.up_proj_xclbin, self.up_proj_insts),
                "gelu": (self.gelu_xclbin, self.gelu_insts),
                "ffn_down": (self.down_proj_xclbin, self.down_proj_insts),
                "add2": (self.add_xclbin, self.add_insts),
                "ln2": (self.ln2_xclbin, self.ln2_insts),
            }

        self.add_artifacts(artifacts)
        logging.info(f"Finished setting up {len(artifacts)} encoder runlist artifacts.")

    def set_up_runtime(self):
        """Set up runtime buffers and kernels for all 13 layers."""
        act_size = self.seq_len * self.hidden_size
        runtime_rows = self._runtime_rows()
        runtime_act_size = self._runtime_act_size()
        runtime_ffn_size = self._runtime_ffn_size()
        runtime_attn_size = self._runtime_attn_size()

        # Post-projection activations and residual input
        self.add_buffer("Q", runtime_act_size)
        self.add_buffer("K", act_size)
        self.add_buffer("V", act_size)
        self.add_buffer("R", runtime_act_size)

        # Weight buffers
        self.add_buffer(
            "out_proj_weight",
            self.hidden_size * self.hidden_size,
            static_data=self._runtime_weight_static_data(
                self.attn_output_weight,
                transpose=True,
            ),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.add_buffer(
                "ln1_weight",
                self.hidden_size,
                static_data=self._runtime_weight_static_data(self.ln1_weight),
            )
        self.add_buffer(
            "ffn_up_weight",
            self.hidden_size * self.intermediate_size,
            static_data=self._runtime_weight_static_data(
                self.ffn_up_weight,
                transpose=True,
            ),
        )
        self.add_buffer(
            "ffn_down_weight",
            self.intermediate_size * self.hidden_size,
            static_data=self._runtime_weight_static_data(
                self.ffn_down_weight,
                transpose=True,
            ),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.add_buffer(
                "ln2_weight",
                self.hidden_size,
                static_data=self._runtime_weight_static_data(self.ln2_weight),
            )

        # Intermediate buffers for all layers
        if self.use_pip_mha:
            self.add_buffer("mha_output", runtime_act_size)  # After layer 5
        else:
            # Keep K^T on a distinct BO. It stays live across the score/softmax
            # path while V must remain intact until the later attention-output
            # GEMM, so these same-sized buffers cannot share an allocation.
            self.add_buffer("k_transposed", act_size + 1)  # After K transpose
            self.add_buffer("attn_scores_output", runtime_attn_size)  # After layer 2
            self.add_buffer("attn_scaled_output", runtime_attn_size)  # After layer 3
            self.add_buffer("attn_weights_output", runtime_attn_size)  # After layer 4
            # Query-blocked long-sequence execution keeps one query block live at
            # a time. Within that block, reuse the attention-score BO across the
            # score/scale/softmax path to further reduce host BO pressure.
            if self.seq_len >= 16384:
                self.buffer_aliases["attn_scaled_output"] = "attn_scores_output"
                self.buffer_aliases["attn_weights_output"] = "attn_scores_output"
            else:
                self.buffer_aliases["attn_weights_output"] = "attn_scores_output"
            # Keep attention output on its own BO. On the 64x768x3072x12 surface,
            # it is the same size as the attention score/weight buffers, and
            # reusing that BO makes attn_output effectively in-place.
            self.add_buffer("attn_heads_output", runtime_act_size + 1)  # After layer 5
        self.add_buffer("output_proj_output", runtime_act_size)  # After layer 6
        if not self.use_pip_an_ffn:
            if self.use_pip_addnorm:
                self.add_buffer("add_norm1_output", runtime_act_size)  # After layer 7
            else:
                self.add_buffer(
                    "ln1_norm_output", runtime_act_size
                )  # After residual add 1
                self.add_buffer("ln1_output", runtime_act_size)  # After layer norm 1
            if self.use_pip_ffn:
                self.add_buffer("ffn_output", runtime_act_size)  # After layer 9-12
            else:
                self.add_buffer("up_proj_output", runtime_ffn_size)  # After layer 9
                self.add_buffer("gelu_output", runtime_ffn_size)  # After layer 10
                self.add_buffer("down_proj_output", runtime_act_size)  # After layer 11
            if not self.use_pip_addnorm:
                self.add_buffer(
                    "ln2_norm_output", runtime_act_size
                )  # After residual add 2

        # Output buffer
        self.add_buffer("output", runtime_act_size)
        logging.info(
            f"Finished setting up {len(self.buffers)} encoder runlist runtime buffers."
        )

        # Add kernels for all layers
        self.add_kernel(
            "encoder_out_proj",
            self.out_proj_xclbin,
            self.out_proj_xclbin.kernel_name,
            self.out_proj_insts,
        )
        if self.use_pip_mha:
            self.add_kernel(
                "encoder_mha",
                self.mha_xclbin,
                self.mha_xclbin.kernel_name,
                self.mha_insts,
            )
        else:
            self.add_kernel(
                "encoder_k_transpose",
                self.k_transpose_xclbin,
                self.k_transpose_xclbin.kernel_name,
                self.k_transpose_insts,
            )
            self.add_kernel(
                "encoder_attn_scores",
                self.attn_scores_xclbin,
                self.attn_scores_xclbin.kernel_name,
                self.attn_scores_insts,
            )
            self.add_kernel(
                "encoder_attn_scale",
                self.attn_scale_xclbin,
                self.attn_scale_xclbin.kernel_name,
                self.attn_scale_insts,
            )
            self.add_kernel(
                "encoder_attn_softmax",
                self.attn_softmax_xclbin,
                self.attn_softmax_xclbin.kernel_name,
                self.attn_softmax_insts,
            )
            self.add_kernel(
                "encoder_attn_output",
                self.attn_output_runtime_xclbin or self.attn_output_xclbin,
                (
                    self.attn_output_runtime_xclbin.kernel_name
                    if self.attn_output_runtime_xclbin is not None
                    else self.attn_output_xclbin.kernel_name
                ),
                self.attn_output_insts,
            )
        if self.use_pip_an_ffn:
            self.add_kernel(
                "encoder_anffn",
                self.anffn_xclbin,
                self.anffn_xclbin.kernel_name,
                self.anffn_insts,
            )
        else:
            if self.use_pip_addnorm:
                self.add_kernel(
                    "encoder_add_norm1",
                    self.add_norm1_xclbin,
                    self.add_norm1_xclbin.kernel_name,
                    self.add_norm1_insts,
                )
            else:
                self.add_kernel(
                    "encoder_ln1",
                    self.ln1_xclbin,
                    self.ln1_xclbin.kernel_name,
                    self.ln1_insts,
                )
                self.add_kernel(
                    "encoder_add",
                    self.add_xclbin,
                    self.add_xclbin.kernel_name,
                    self.add_insts,
                )
            if self.use_pip_ffn:
                self.add_kernel(
                    "encoder_ffn",
                    self.ffn_xclbin,
                    self.ffn_xclbin.kernel_name,
                    self.ffn_insts,
                )
            else:
                self.add_kernel(
                    "encoder_up_proj",
                    self.up_proj_xclbin,
                    self.up_proj_xclbin.kernel_name,
                    self.up_proj_insts,
                )
                self.add_kernel(
                    "encoder_gelu",
                    self.gelu_xclbin,
                    self.gelu_xclbin.kernel_name,
                    self.gelu_insts,
                )
                self.add_kernel(
                    "encoder_down_proj",
                    self.down_proj_xclbin,
                    self.down_proj_xclbin.kernel_name,
                    self.down_proj_insts,
                )
            if self.use_pip_addnorm:
                self.add_kernel(
                    "encoder_add_norm2",
                    self.add_norm2_xclbin,
                    self.add_norm2_xclbin.kernel_name,
                    self.add_norm2_insts,
                )
            else:
                self.add_kernel(
                    "encoder_ln2",
                    self.ln2_xclbin,
                    self.ln2_xclbin.kernel_name,
                    self.ln2_insts,
                )
        logging.info(
            f"Finished setting up {len(self.kernels)} encoder runlist runtime kernels."
        )

        # Build runlist for all layers
        if self.use_pip_mha:
            self.add_to_runlist("encoder_mha", "Q", "K", "V", "mha_output")
            next_output = "mha_output"
        else:
            # Transpose K matrix
            self.add_to_runlist("encoder_k_transpose", "K", "k_transposed")
            # Attention score calculations
            self.add_to_runlist(
                "encoder_attn_scores", "Q", "k_transposed", "attn_scores_output"
            )
            # Attention score scaling
            self.add_to_runlist(
                "encoder_attn_scale",
                "attn_scores_output",
                "attn_scaled_output",
            )
            # Attention weight calculations (Softmax)
            self.add_to_runlist(
                "encoder_attn_softmax", "attn_scaled_output", "attn_weights_output"
            )
            # Output head calculations
            self.add_to_runlist(
                "encoder_attn_output",
                "attn_weights_output",
                "V",
                "attn_heads_output",
            )
            next_output = "attn_heads_output"
        # Output projection
        self.add_to_runlist(
            "encoder_out_proj",
            next_output,
            "out_proj_weight",
            "output_proj_output",
        )
        if self.use_pip_an_ffn:
            # Pipelined AN-FFN, 2nd input is for residual connection
            self.add_to_runlist(
                "encoder_anffn",
                "output_proj_output",
                "R",
                "ffn_up_weight",
                "ffn_down_weight",
                "output",
            )
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm, 2nd input is for residual connection
                self.add_to_runlist(
                    "encoder_add_norm1",
                    "output_proj_output",
                    "R",
                    "add_norm1_output",
                )
                next_output = "add_norm1_output"
            else:
                # Residual connection then layer normalization
                self.add_to_runlist(
                    "encoder_add", "output_proj_output", "R", "ln1_norm_output"
                )
                self.add_to_runlist("encoder_ln1", "ln1_norm_output", "ln1_output")
                next_output = "ln1_output"
            if self.use_pip_ffn:
                # Pipelined FFN
                self.add_to_runlist(
                    "encoder_ffn",
                    next_output,
                    "ffn_up_weight",
                    "ffn_down_weight",
                    "ffn_output",
                )
                next_output = "ffn_output"
            else:
                # Up projection
                self.add_to_runlist(
                    "encoder_up_proj", next_output, "ffn_up_weight", "up_proj_output"
                )
                # GeLU activation
                self.add_to_runlist("encoder_gelu", "up_proj_output", "gelu_output")
                # Down projection
                self.add_to_runlist(
                    "encoder_down_proj",
                    "gelu_output",
                    "ffn_down_weight",
                    "down_proj_output",
                )
                next_output = "down_proj_output"
            if self.use_pip_addnorm:
                # Second Pipelined add & norm, 2nd input is for residual connection
                self.add_to_runlist(
                    "encoder_add_norm2", next_output, "add_norm1_output", "output"
                )
            else:
                # Residual connection then layer normalization
                self.add_to_runlist(
                    "encoder_add", next_output, "ln1_output", "ln2_norm_output"
                )
                self.add_to_runlist("encoder_ln2", "ln2_norm_output", "output")

        logging.info(
            f"Finished setting up {len(self.runlist)} encoder runlist dispatches."
        )

    def forward(self, q, k, v, residual, attention_mask=None):
        """
        Forward pass through the post-projection encoder layer.

        Args:
            q: Query tensor of shape (seq_len, hidden_size)
            k: Key tensor of shape (seq_len, hidden_size)
            v: Value tensor of shape (seq_len, hidden_size)
            residual: Residual tensor of shape (seq_len, hidden_size)
            attention_mask: Optional attention mask (not used for now)

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        expected_shape = (self.seq_len, self.hidden_size)
        q_matrix = self._normalize_qkv_tensor(
            q,
            name="Q",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k_matrix = self._normalize_qkv_tensor(
            k,
            name="K",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        v_matrix = self._normalize_qkv_tensor(
            v,
            name="V",
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        if residual.dim() != 2:
            raise ValueError(
                f"AIEEncoderRunlist.forward expects R to be a 2D tensor of shape "
                f"{expected_shape}; got shape={tuple(residual.shape)}"
            )
        if tuple(residual.shape) != expected_shape:
            raise ValueError(
                f"AIEEncoderRunlist.forward expects R to have shape "
                f"{expected_shape}; got shape={tuple(residual.shape)}"
            )

        uses_query_blocking = getattr(self, "uses_query_blocking", False)
        query_block_size = getattr(self, "query_block_size", self.seq_len)
        expected_size = self.seq_len * self.hidden_size

        # Flatten inputs for AIE processing
        k_flat = self._pack_k_for_transpose(
            k_matrix,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        ).view(-1)
        v_flat = v_matrix.view(-1)
        assert k_flat.shape[0] == expected_size
        assert v_flat.shape[0] == expected_size

        if not self.use_static_runtime_weights:
            self.write_runtime_weights()
        self.write_buffer("K", k_flat)
        self.write_buffer("V", v_flat)

        if not uses_query_blocking:
            q_flat = q_matrix.view(-1)
            residual_flat = residual.view(-1)
            # Verify input size matches expected dimensions
            assert q_flat.shape[0] == expected_size
            assert residual_flat.shape[0] == expected_size
            self.write_buffer("Q", q_flat)
            self.write_buffer("R", residual_flat)
            self.run_runlist()
            return self.read_buffer_as_torch(
                "output",
                (self.seq_len, self.hidden_size),
                dtype=bfloat16,
            ).view(expected_shape)

        block_outputs = []
        for block_start in range(0, self.seq_len, query_block_size):
            block_end = min(block_start + query_block_size, self.seq_len)
            current_rows = block_end - block_start
            q_block = q_matrix[block_start:block_end]
            residual_block = residual[block_start:block_end]
            # Runtime buffers are sized for the fixed query-block shape used to
            # compile the long-sequence path, so pad only the final short block.
            if current_rows != query_block_size:
                q_storage = q_matrix.new_zeros((query_block_size, self.hidden_size))
                residual_storage = residual.new_zeros(
                    (query_block_size, self.hidden_size)
                )
                q_storage[:current_rows] = q_block
                residual_storage[:current_rows] = residual_block
            else:
                q_storage = q_block.contiguous()
                residual_storage = residual_block.contiguous()

            self.write_buffer("Q", q_storage.view(-1))
            self.write_buffer("R", residual_storage.view(-1))
            self.run_runlist()
            block_output = self.read_buffer_as_torch(
                "output",
                (query_block_size, self.hidden_size),
                dtype=bfloat16,
            )
            block_outputs.append(block_output[:current_rows].contiguous())

        return torch.cat(block_outputs, dim=0).view(expected_shape)
